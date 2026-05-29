"""Causal reasoning over the user's canvas graph.

For a question like "if X increases, what happens to Y?":

  1. We operate on the *canvas* graph (the user's CLD) rather than
     pulling from Neo4j, because the user is asking about *their*
     constructed model.
  2. Enumerate all simple directed paths from X to Y up to a depth cap.
  3. For each path, multiply edge signs to get a net sign and damp
     strength by length × weight.
  4. Run Tarjan SCC to find feedback loops; mark any path whose nodes
     overlap a non-trivial SCC.
  5. Hand the top-K paths + loop summary to the narrator LLM, which
     writes a 2-3 sentence plain-English explanation citing dominant
     paths and (if any) reinforcing/balancing loops.

The deterministic stage (1-4) is fully testable; only the narration
depends on LLM output.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config import Config
from .llm_router import LLMRouter
from .schemas import CanvasState, CausalPath, CausalQueryRequest, CausalQueryResponse


LOG = logging.getLogger("graph_rag.causal")


# ── Graph helpers ──────────────────────────────────────────────────────

def _build_adj(canvas: CanvasState) -> Tuple[Dict[str, List[Tuple[str, str, int]]], Dict[str, str]]:
    """Return adjacency: src → [(tgt, polarity, weight)] and id→label."""
    adj: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    labels: Dict[str, str] = {}
    for n in canvas.nodes:
        labels[n.id] = n.label
        adj.setdefault(n.id, [])
    for e in canvas.edges:
        # canvas edges may carry a "weight" via description; treat absent as 3.
        weight = 3
        adj[e.source].append((e.target, e.polarity, weight))
    return adj, labels


def _enumerate_paths(
    adj: Dict[str, List[Tuple[str, str, int]]],
    src: str,
    tgt: str,
    max_depth: int = 5,
    max_paths: int = 20,
) -> List[List[Tuple[str, str, int]]]:
    """All simple paths src→tgt up to max_depth. Each path is
    [(node, polarity_into_node, weight_into_node), ...] with the source
    represented as ('+', 0) sentinel at index 0."""
    results: List[List[Tuple[str, str, int]]] = []
    if src == tgt:
        return results
    stack: List[Tuple[str, List[Tuple[str, str, int]], set]] = [(src, [(src, "+", 0)], {src})]
    while stack and len(results) < max_paths:
        node, path, visited = stack.pop()
        if len(path) > max_depth:
            continue
        for nxt, sign, w in adj.get(node, []):
            if nxt in visited:
                continue
            new_path = path + [(nxt, sign, w)]
            if nxt == tgt:
                results.append(new_path)
            else:
                stack.append((nxt, new_path, visited | {nxt}))
    return results


def _path_signature(path: List[Tuple[str, str, int]]) -> Tuple[str, float]:
    """Return (net_sign, strength) for a path."""
    # net sign: '+' if even number of '-' edges else '-'
    n_neg = sum(1 for (_n, s, _w) in path[1:] if s == "-")
    net = "+" if n_neg % 2 == 0 else "-"
    # strength: product of (weight/5) damped by 0.6 per hop beyond 1
    strength = 1.0
    for (_n, _s, w) in path[1:]:
        ww = (w or 3) / 5.0
        strength *= ww
    strength *= (0.6 ** max(0, len(path) - 2))
    return net, max(0.0, min(1.0, strength))


# Tarjan SCC for loop detection ----------------------------------------

def _tarjan_scc(adj: Dict[str, List[Tuple[str, str, int]]]) -> List[List[str]]:
    """Return SCCs as lists of node ids; size>=2 (or self-loop) are
    feedback loops in CLD terms."""
    index = {}
    lowlink = {}
    on_stack = set()
    stack: List[str] = []
    sccs: List[List[str]] = []
    counter = [0]

    def strong(v: str):
        index[v] = lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w, _s, _ww in adj.get(v, []):
            if w not in index:
                strong(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    # Iterative version (Python recursion limit safety) — but our canvas
    # is tiny so recursion is fine; bump limit just in case.
    import sys
    sys.setrecursionlimit(max(1000, sys.getrecursionlimit()))
    for v in list(adj.keys()):
        if v not in index:
            strong(v)
    return sccs


def _classify_loop(adj, scc: List[str]) -> Optional[Tuple[str, str]]:
    """Return (loop_id, type) where type is 'R' (reinforcing) or 'B'
    (balancing). Determined by multiplying signs around the loop."""
    if len(scc) < 2:
        # check for self-loop
        if not any(t == scc[0] for (t, _s, _w) in adj.get(scc[0], [])):
            return None
    # Find a simple cycle inside scc via DFS
    nodes_set = set(scc)
    start = scc[0]
    stack = [(start, [start], "+")]
    while stack:
        v, path, accum = stack.pop()
        for w, s, _ww in adj.get(v, []):
            if w not in nodes_set:
                continue
            new_sign = "+" if accum == s else "-"
            if w == start and len(path) >= 2:
                ltype = "R" if new_sign == "+" else "B"
                return (f"L{abs(hash(tuple(sorted(scc)))) % 10000}", ltype)
            if w not in path:
                stack.append((w, path + [w], new_sign))
    return None


# ── Public API ────────────────────────────────────────────────────────

class CausalReasoner:
    def __init__(self, cfg: Config, router: LLMRouter):
        self.cfg = cfg
        self.router = router

    def answer(self, req: CausalQueryRequest) -> CausalQueryResponse:
        if not req.target:
            # Future extension: rank-top-K reachable nodes.
            return CausalQueryResponse(
                summary="Specify a target node to compute the effect on.",
                paths=[],
            )

        adj, labels = _build_adj(req.canvas)
        paths_raw = _enumerate_paths(adj, req.source, req.target, max_depth=5, max_paths=20)

        # Loop detection
        sccs = _tarjan_scc(adj)
        loop_meta: Dict[str, str] = {}                # node → loop_id
        loop_types: Dict[str, str] = {}               # loop_id → 'R'|'B'
        for scc in sccs:
            if len(scc) >= 2:
                cl = _classify_loop(adj, scc)
                if cl:
                    lid, ltype = cl
                    loop_types[lid] = ltype
                    for n in scc:
                        loop_meta[n] = lid

        # Build CausalPath objects
        paths: List[CausalPath] = []
        for p in paths_raw:
            net, strength = _path_signature(p)
            node_ids = [n for (n, _s, _w) in p]
            edge_signs = [s for (_n, s, _w) in p[1:]]
            # Path touches a loop if any internal node is in a loop
            loop_id = None
            for nid in node_ids[1:-1]:
                if nid in loop_meta:
                    loop_id = loop_meta[nid]
                    break
            paths.append(CausalPath(
                nodes=node_ids,
                labels=[labels.get(nid, nid) for nid in node_ids],
                signs=edge_signs,
                net_sign=net,
                strength=round(strength, 3),
                loop_id=loop_id,
            ))

        # Sort by strength × consistency-bonus (paths that agree with majority sign)
        if paths:
            majority_pos = sum(1 for p in paths if p.net_sign == "+")
            majority_neg = len(paths) - majority_pos
            majority_sign = "+" if majority_pos >= majority_neg else "-"
            paths.sort(key=lambda p: (p.net_sign == majority_sign, p.strength), reverse=True)
            paths = paths[: req.max_paths]
            # Flip net direction if user asked about a "decrease"
            net_direction = self._aggregate_direction(paths, asked=req.direction)
        else:
            net_direction = None

        loops_involved = sorted({
            f"{lid} ({loop_types.get(lid, '?')})"
            for p in paths if p.loop_id for lid in [p.loop_id]
        })

        # Narrate
        summary = self._narrate(
            req=req,
            paths=paths,
            loops_involved=loops_involved,
            labels=labels,
            net_direction=net_direction,
        )
        return CausalQueryResponse(
            summary=summary,
            net_direction=net_direction,
            paths=paths,
            loops_involved=loops_involved,
        )

    # internal --------------------------------------------------------------

    @staticmethod
    def _aggregate_direction(paths: List[CausalPath], asked: str) -> Optional[str]:
        """Combine top paths into one net direction relative to the asked change."""
        if not paths:
            return None
        pos = sum(p.strength for p in paths if p.net_sign == "+")
        neg = sum(p.strength for p in paths if p.net_sign == "-")
        if abs(pos - neg) < 0.05:
            return "ambiguous"
        net_pos = pos > neg
        # If user asked about a *decrease* at source, flip the sense.
        if asked == "decrease":
            net_pos = not net_pos
        return "increase" if net_pos else "decrease"

    def _narrate(self, req, paths, loops_involved, labels, net_direction) -> str:
        if not paths:
            return (
                f"No directed causal path from “{labels.get(req.source, req.source)}” "
                f"to “{labels.get(req.target, req.target)}” exists in the current diagram."
            )
        _spec, client = self.router.get("causal_narrator")
        sys_prompt = (
            "You are a system-dynamics tutor. Given a question about a causal "
            "diagram and the dominant signed paths between two variables, "
            "write a clear 2–3 sentence explanation. Mention reinforcing (R) or "
            "balancing (B) loops by id only when present. Avoid jargon. "
            "Do not fabricate paths or loops that are not in the input."
        )
        user_payload = {
            "question": {
                "source": labels.get(req.source, req.source),
                "target": labels.get(req.target, req.target),
                "direction": req.direction,
            },
            "net_direction": net_direction,
            "paths": [
                {
                    "labels": p.labels,
                    "signs": p.signs,
                    "net_sign": p.net_sign,
                    "strength": p.strength,
                    "loop_id": p.loop_id,
                }
                for p in paths
            ],
            "loops_involved": loops_involved,
        }
        from src.llm_client import call_with_retries
        try:
            return call_with_retries(client, sys_prompt, json.dumps(user_payload), retries=1).strip()
        except Exception as e:
            LOG.warning("Narrator LLM failed (%s); falling back to template", e)
            # Deterministic fallback
            top = paths[0]
            arrow = " → ".join(top.labels)
            loops_note = (
                f" This effect is amplified by loop {top.loop_id}."
                if top.loop_id and top.net_sign == "+" else
                f" Loop {top.loop_id} counteracts this effect." if top.loop_id else ""
            )
            return (
                f"A {req.direction} in “{labels.get(req.source)}” tends to "
                f"{net_direction} “{labels.get(req.target)}” via the path: "
                f"{arrow}.{loops_note}"
            )
