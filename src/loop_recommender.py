"""Background feedback-loop recommender for the Modify-tab Diagram Assistant.

Unlike the KG-backed node recommender (`recommender.py`), this component looks
ONLY at the diagram the user already has on screen. It answers two questions:

  1. Is there a feedback loop *missing* that could be closed using mostly the
     variables already present (by adding a few edges)?
  2. Could a small, sensible extension (≤ a couple of new nodes) introduce a
     reinforcing or balancing loop that reveals an important dynamic?

It is deliberately conservative. The LLM proposes at most one loop; a
deterministic validation step then confirms the proposed edges actually form a
single directed cycle and re-derives the loop's reinforcing/balancing type from
edge-sign parity (so the badge the user sees is never the model's guess). If the
proposal doesn't validate, isn't confident, adds too many new nodes, or repeats
a loop the user already dismissed, the service returns ``found = False`` and the
UI stays silent.

The output is a single COMPOSITE ``add_loop`` mutation — accept once and all the
missing nodes + edges are applied atomically.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .llm_router import LLMRouter
from .llm_client import extract_json
from .schemas import (
    CanvasState,
    LoopRecommendRequest,
    LoopRecommendResponse,
    PendingMutation,
)

LOG = logging.getLogger("graph_rag.loop_recommender")

# Don't bother the model until there's enough structure to reason about, and
# require the proposal to be reasonably confident + small.
MIN_NODES = 3
MIN_CONFIDENCE = 0.7
MAX_NEW_NODES = 2
MAX_CYCLE_LEN = 8

SUBSYSTEMS = {
    "Demographics", "Land Use", "Housing", "Transportation",
    "Infrastructure", "Energy", "Utilities", "Environment",
    "Economy", "Governance", "Finance", "Health", "Social", "Others",
}

SYSTEM_PROMPT = f"""You are a system-dynamics expert embedded in a causal-loop \
diagram (CLD) editor. You are given the user's CURRENT diagram only — you have \
NO external knowledge base to draw from. Your job is to spot ONE feedback loop \
that is genuinely missing and worth adding.

A feedback loop is a closed directed cycle of signed edges. Its type is:
  • REINFORCING (R): an EVEN number of negative links — change amplifies itself.
  • BALANCING  (B): an ODD number of negative links — change is counteracted.

Strongly prefer loops that close using variables ALREADY on the canvas by \
adding only edges. You MAY introduce at most {MAX_NEW_NODES} new bridging \
variables, but only when doing so reveals an important, realistic dynamic.

BE CONSERVATIVE. Most of the time the right answer is "nothing worth adding". \
Only propose a loop you are highly confident is (a) causally plausible in the \
real world, (b) not already present, and (c) genuinely informative about the \
system's dynamics. If there is no such loop, return {{"found": false}}.

Respond with ONLY a JSON object, no markdown:
{{
  "found": true,
  "confidence": 0.0-1.0,        // your honest confidence this loop is valuable
  "loop_type": "R" | "B",       // your read; the system re-checks it
  "headline": "short title, e.g. 'Reinforcing displacement spiral'",
  "rationale": "1-3 sentences: what the loop is and why it matters",
  "new_nodes": [                // omit or [] if none needed
    {{"label": "...", "subsystem": "one of the 14 canonical subsystems", "description": "..."}}
  ],
  "edges": [                    // the FULL cycle, in order; each source/target is a node LABEL
    {{"source": "label A", "target": "label B", "polarity": "+"|"-", "label": "short verb", "description": "..."}}
  ]
}}

The "edges" array MUST list every edge of the cycle so it closes back on itself \
(the last edge's target equals the first edge's source). Reuse exact existing \
labels where possible."""


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _summarise_canvas(canvas: CanvasState) -> str:
    id_to_label = {n.id: n.label for n in canvas.nodes}
    lines = [f"Variables ({len(canvas.nodes)}):"]
    for n in canvas.nodes[:120]:
        lines.append(f"  - {n.label}  [{n.subsystem or n.category or '?'}]")
    lines.append("")
    lines.append(f"Existing signed edges ({len(canvas.edges)}):")
    if not canvas.edges:
        lines.append("  (none yet)")
    for e in canvas.edges[:200]:
        lines.append(
            f"  - {id_to_label.get(e.source, e.source)} {e.polarity}→ "
            f"{id_to_label.get(e.target, e.target)}"
        )
    return "\n".join(lines)


def _loop_signature(ordered_labels: List[str], polarities: List[str]) -> str:
    """Stable, rotation-invariant signature from a cycle's labels + signs.

    Label-based (not id-based) so the SAME conceptual loop dedupes across
    calls even though new-node ids are minted fresh each time. Rotated to start
    at the lexicographically smallest normalized label; direction is preserved.
    """
    norm = [_normalize(l) for l in ordered_labels]
    n = len(norm)
    if n == 0:
        return ""
    start = min(range(n), key=lambda i: norm[i])
    rot_labels = [norm[(start + i) % n] for i in range(n)]
    rot_pols = [polarities[(start + i) % n] for i in range(n)]
    return "|".join(f"{rot_labels[i]}{rot_pols[i]}" for i in range(n))


class LoopRecommender:
    def __init__(self, cfg: Config, router: LLMRouter):
        self.cfg = cfg
        self.router = router

    # ── public ─────────────────────────────────────────────────────────
    def recommend(self, req: LoopRecommendRequest) -> LoopRecommendResponse:
        canvas = req.canvas
        if len(canvas.nodes) < MIN_NODES:
            return LoopRecommendResponse(found=False)

        try:
            data = self._propose(canvas)
        except Exception as e:
            LOG.warning("loop recommender LLM call failed: %s", e)
            return LoopRecommendResponse(found=False)

        if not data or not data.get("found"):
            return LoopRecommendResponse(found=False)
        if float(data.get("confidence", 0.0)) < MIN_CONFIDENCE:
            return LoopRecommendResponse(found=False)

        built = self._validate_and_build(canvas, data)
        if built is None:
            return LoopRecommendResponse(found=False)

        signature, loop_type, mutation, headline, rationale = built
        if signature in set(req.exclude_signatures or []):
            return LoopRecommendResponse(found=False)

        return LoopRecommendResponse(
            found=True,
            loop_type=loop_type,
            headline=headline,
            rationale=rationale,
            signature=signature,
            mutation=mutation,
        )

    # ── LLM proposal ───────────────────────────────────────────────────
    def _propose(self, canvas: CanvasState) -> Optional[Dict[str, Any]]:
        _spec, client = self.router.get("loop_recommender")
        user = (
            "Here is the current diagram. Propose at most one missing feedback "
            "loop, or return {\"found\": false}.\n\n" + _summarise_canvas(canvas)
        )
        raw = client.complete(SYSTEM_PROMPT, user)
        obj = extract_json(raw)
        return obj if isinstance(obj, dict) else None

    # ── deterministic validation + mutation assembly ───────────────────
    def _validate_and_build(
        self, canvas: CanvasState, data: Dict[str, Any],
    ) -> Optional[Tuple[str, str, PendingMutation, str, str]]:
        edges_in = data.get("edges") or []
        if not isinstance(edges_in, list) or len(edges_in) < 2:
            return None

        # Resolve every label to a node id, minting ids for declared/implied
        # new nodes. `existing_by_norm` maps normalized label -> existing id.
        existing_by_norm = {_normalize(n.label): n.id for n in canvas.nodes}
        # Declared new nodes (label -> spec)
        declared_new: Dict[str, Dict[str, Any]] = {}
        for nn in (data.get("new_nodes") or []):
            lbl = (nn.get("label") or "").strip()
            if lbl:
                declared_new[_normalize(lbl)] = nn

        label_to_id: Dict[str, str] = {}
        new_nodes_payload: List[Dict[str, Any]] = []
        id_to_label: Dict[str, str] = {n.id: n.label for n in canvas.nodes}

        def resolve(label: str) -> Optional[str]:
            norm = _normalize(label)
            if not norm:
                return None
            if norm in label_to_id:
                return label_to_id[norm]
            if norm in existing_by_norm:                       # already on canvas
                label_to_id[norm] = existing_by_norm[norm]
                return label_to_id[norm]
            # Need a new node. Cap how many we invent.
            if len(new_nodes_payload) >= MAX_NEW_NODES:
                return None
            spec = declared_new.get(norm, {})
            subsystem = spec.get("subsystem") if spec.get("subsystem") in SUBSYSTEMS else "Others"
            nid = f"node_{uuid.uuid4().hex[:8]}"
            display = (spec.get("label") or label).strip()
            new_nodes_payload.append({
                "id": nid,
                "label": display,
                "subsystem": subsystem,
                "description": spec.get("description") or "",
            })
            label_to_id[norm] = nid
            id_to_label[nid] = display
            return nid

        # Build resolved cycle edges.
        resolved: List[Dict[str, Any]] = []
        for e in edges_in:
            sid = resolve(e.get("source", ""))
            tid = resolve(e.get("target", ""))
            pol = e.get("polarity")
            if not sid or not tid or sid == tid or pol not in ("+", "-"):
                return None
            resolved.append({
                "source": sid, "target": tid, "polarity": pol,
                "label": (e.get("label") or ("increases" if pol == "+" else "decreases")),
                "description": e.get("description") or "",
            })

        # Validate it's exactly one simple directed cycle: k distinct nodes,
        # k edges, every node in/out degree 1.
        nodes_in_cycle = {x["source"] for x in resolved} | {x["target"] for x in resolved}
        if len(resolved) != len(nodes_in_cycle) or len(nodes_in_cycle) > MAX_CYCLE_LEN:
            return None
        succ: Dict[str, Dict[str, Any]] = {}
        indeg: Dict[str, int] = {n: 0 for n in nodes_in_cycle}
        for x in resolved:
            if x["source"] in succ:        # >1 outgoing → not a simple cycle
                return None
            succ[x["source"]] = x
            indeg[x["target"]] += 1
        if any(indeg[n] != 1 for n in nodes_in_cycle) or len(succ) != len(nodes_in_cycle):
            return None

        # Walk the cycle to get a canonical node order + per-edge polarity.
        start = resolved[0]["source"]
        order_ids: List[str] = [start]
        order_pols: List[str] = []
        cur = start
        for _ in range(len(nodes_in_cycle)):
            edge = succ.get(cur)
            if edge is None:
                return None
            order_pols.append(edge["polarity"])
            nxt = edge["target"]
            if nxt == start:
                break
            order_ids.append(nxt)
            cur = nxt
        if len(order_ids) != len(nodes_in_cycle) or len(order_pols) != len(nodes_in_cycle):
            return None

        # Effective polarity per edge = canvas polarity if the edge already
        # exists, else the proposed polarity. Parity over THOSE gives the
        # real loop sign once applied.
        existing_edge_pol = {(e.source, e.target): e.polarity for e in canvas.edges}
        new_links: List[Dict[str, Any]] = []
        neg = 0
        for x in resolved:
            key = (x["source"], x["target"])
            eff_pol = existing_edge_pol.get(key, x["polarity"])
            if eff_pol == "-":
                neg += 1
            if key not in existing_edge_pol:               # only stage what's missing
                new_links.append(x)

        # Must actually add something, and must form a real cycle (≥1 new edge).
        if not new_links and not new_nodes_payload:
            return None
        if not new_links:
            return None

        loop_type = "R" if neg % 2 == 0 else "B"
        order_labels = [id_to_label.get(i, i) for i in order_ids]
        signature = _loop_signature(order_labels, order_pols)

        headline = (data.get("headline") or "").strip() or (
            ("Reinforcing" if loop_type == "R" else "Balancing") + " loop"
        )
        rationale = (data.get("rationale") or "").strip()

        mutation = PendingMutation(
            id=f"m_{uuid.uuid4().hex[:8]}",
            op="add_loop",
            payload={
                "loop_type": loop_type,
                "rationale": rationale,
                "signature": signature,
                "nodes": new_nodes_payload,
                "links": new_links,
                # ordered labels for a readable summary on the client
                "cycle_labels": order_labels,
            },
            summary=headline,
        )
        return signature, loop_type, mutation, headline, rationale
