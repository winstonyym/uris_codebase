"""Loop-aware Q&A assistant for the Visualise tab.

Distinct from `chat_agent.py` in three ways:

  1. **Read-only** — no tools, no mutations. The user is exploring,
     not editing.
  2. **Scope-bound** — the model only sees the canvas nodes/edges that
     are *currently visible* on screen (after subsystem & loop filters).
     The system prompt forbids it from inventing or referencing anything
     outside that scope.
  3. **Causal-path traced** — for "what is the effect of X on Y?"
     questions we deterministically enumerate signed paths (just like
     `causal.py` does) and inject the result into the prompt, so the
     LLM's job is reduced to *narrating* the answer rather than
     inferring it. This keeps responses honest to the user's actual
     diagram.

The streaming protocol matches `chat_agent.respond_stream` so the SSE
endpoint can be a near-identical wrapper:

    yield ("delta", text_chunk)
    ...
    yield ("done", reply_string)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import Config
from .llm_router import LLMRouter
from .schemas import (
    CanvasState,
    ChatMessage,
    LoopChatRequest,
    SelectionScope,
)
from .causal import _build_adj, _enumerate_paths, _path_signature, _tarjan_scc, _classify_loop


LOG = logging.getLogger("graph_rag.loop_chat")


SYSTEM_PROMPT_HEADER = """\
You are a system-dynamics tutor whose ONLY job is to help the user
understand the variables, edges and feedback loops they currently have
on screen. You must obey the following rules at all times:

  1. Treat the "Visible variables" and "Visible signed edges" lists as
     the ENTIRE world. Never reference a variable, edge or loop that is
     not in those lists, even if the user mentions one — instead, tell
     the user that variable / edge / loop is not part of their current
     view of the diagram.

  2. For "what is the effect of A on B?" questions:
       a. Confirm BOTH A and B are in Visible variables. If either is
          missing, say so plainly and stop.
       b. If a "Causal path trace" block is provided below, USE IT
          verbatim — do not invent paths. Explain the net direction
          (+ → A increases ⇒ B tends to increase, − → B tends to
          decrease) and cite the dominant 1-2 paths by walking through
          the listed nodes.
       c. If a path passes through a feedback loop in "Visible loops",
          name it (e.g. "loop R1 reinforces this", "loop B2 dampens it")
          but only if the loop appears in that list.

  3. Be concise. Aim for 2-3 sentences. Plain English, minimal jargon.

  4. NEVER suggest changes to the diagram. NEVER call tools. This is a
     read-only Q&A. If asked to add or remove anything, say "I can only
     describe what's already on your diagram — switch to the Modify
     tab to make edits."
"""


# Recognise effect questions so we can pre-trace paths before calling
# the LLM. Forgiving regex — false positives are fine because the trace
# block is just additional context.
_EFFECT_PATTERNS = [
    r"effect of (.+?) on (.+?)(\?|$|\.|,)",
    r"if (.+?) (?:increase|decrease|goes up|goes down|rises|falls).*on (.+?)(\?|$|\.|,)",
    r"how does (.+?) affect (.+?)(\?|$|\.|,)",
    r"what happens to (.+?) (?:if|when) (.+?)(\?|$|\.|,)",  # B if A → swap
]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _try_parse_effect_question(text: str, label_to_id: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Return (source_id, target_id) if the message is an effect-style
    question we can trace; else None.

    Variable mentions are matched against canvas labels via normalised
    substring containment, so "rent burden" matches the node labelled
    "Rent burden".
    """
    text_l = text.lower()
    for pat in _EFFECT_PATTERNS:
        m = re.search(pat, text_l)
        if not m:
            continue
        a_raw, b_raw = m.group(1).strip(), m.group(2).strip()
        # last pattern swaps order (B if A)
        if "what happens to" in pat:
            a_raw, b_raw = b_raw, a_raw
        a_norm, b_norm = _normalize(a_raw), _normalize(b_raw)
        a_id = _label_lookup(a_norm, label_to_id)
        b_id = _label_lookup(b_norm, label_to_id)
        if a_id and b_id and a_id != b_id:
            return a_id, b_id
    return None


def _label_lookup(norm_query: str, label_to_id: Dict[str, str]) -> Optional[str]:
    if not norm_query:
        return None
    # Exact match first
    if norm_query in label_to_id:
        return label_to_id[norm_query]
    # Substring fallback — pick the shortest matching label so "rent"
    # prefers "Rent burden" over "Affordable rent regulation reform".
    candidates = sorted(
        [(label, kid) for label, kid in label_to_id.items() if norm_query in label],
        key=lambda x: len(x[0]),
    )
    return candidates[0][1] if candidates else None


def _filter_visible(canvas: CanvasState, scope: SelectionScope) -> Tuple[List[Any], List[Any]]:
    """Apply the user's current subsystem + loop filters to derive the
    visible subset of nodes & edges."""
    # Subsystem filter
    if scope.active_subsystem and scope.active_subsystem != "ALL":
        nodes_visible = [n for n in canvas.nodes if (n.subsystem or "") == scope.active_subsystem]
    else:
        nodes_visible = list(canvas.nodes)
    # Then loop filter — if a loop is selected, restrict further to its members.
    if scope.selected_loop_id and scope.selected_loop_id != "ALL" and scope.selected_loop_link_pairs:
        loop_node_ids = set()
        for pair in scope.selected_loop_link_pairs:
            if len(pair) == 2:
                loop_node_ids.add(pair[0]); loop_node_ids.add(pair[1])
        nodes_visible = [n for n in nodes_visible if n.id in loop_node_ids]
    # If the client sent an explicit visible-ids list, intersect with it.
    if scope.visible_node_ids:
        allowed = set(scope.visible_node_ids)
        nodes_visible = [n for n in nodes_visible if n.id in allowed]
    visible_ids = {n.id for n in nodes_visible}
    edges_visible = [e for e in canvas.edges if e.source in visible_ids and e.target in visible_ids]
    return nodes_visible, edges_visible


def _summarise_scope(nodes_visible, edges_visible, scope: SelectionScope) -> str:
    if not nodes_visible:
        return (
            "Current view: EMPTY — no variables match the user's current "
            "loop / subsystem filters. Politely tell the user to widen "
            "their selection."
        )
    lines = []
    if scope.active_subsystem and scope.active_subsystem != "ALL":
        lines.append(f"Subsystem filter: {scope.active_subsystem}")
    else:
        lines.append("Subsystem filter: ALL")
    if scope.selected_loop_id and scope.selected_loop_id != "ALL":
        ltype = scope.selected_loop_type or "?"
        lines.append(
            f"Loop highlighted: {scope.selected_loop_id} "
            f"({'Reinforcing' if ltype == 'R' else 'Balancing' if ltype == 'B' else 'unknown'})"
            f"{' — ' + scope.selected_loop_name if scope.selected_loop_name else ''}"
        )
    else:
        lines.append("Loop highlighted: none (all loops in view)")
    if scope.selected_item_kind and scope.selected_item_label:
        lines.append(f"User has clicked on: {scope.selected_item_kind} \"{scope.selected_item_label}\"")
    lines.append("")
    lines.append(f"Visible variables ({len(nodes_visible)}):")
    for n in nodes_visible[:80]:
        lines.append(f"  - {n.label}  (id={n.id})")
    if len(nodes_visible) > 80:
        lines.append(f"  …and {len(nodes_visible) - 80} more.")
    lines.append("")
    lines.append(f"Visible signed edges ({len(edges_visible)}):")
    id_to_label = {n.id: n.label for n in nodes_visible}
    for e in edges_visible[:120]:
        lines.append(
            f"  - {id_to_label.get(e.source, e.source)} {e.polarity}→ "
            f"{id_to_label.get(e.target, e.target)}"
        )
    if len(edges_visible) > 120:
        lines.append(f"  …and {len(edges_visible) - 120} more.")
    return "\n".join(lines)


def _trace_paths_block(canvas, nodes_visible, edges_visible, src_id, tgt_id) -> str:
    """Run the same deterministic signed-path search the causal reasoner
    uses, then format the top 2 paths for the prompt."""
    # Build a CanvasState restricted to the visible set so the trace
    # respects the user's filter.
    from .schemas import CanvasState as CS, CanvasNode, CanvasEdge
    restricted = CS(
        nodes=[CanvasNode(**n.model_dump()) for n in nodes_visible],
        edges=[CanvasEdge(**e.model_dump()) for e in edges_visible],
    )
    adj, labels = _build_adj(restricted)
    if src_id not in adj or tgt_id not in adj:
        return ""
    paths = _enumerate_paths(adj, src_id, tgt_id, max_depth=5, max_paths=8)
    if not paths:
        return (
            "Causal path trace:\n"
            f"  No directed path from \"{labels.get(src_id, src_id)}\" to "
            f"\"{labels.get(tgt_id, tgt_id)}\" exists in the visible diagram."
        )
    # Loop detection on the visible subgraph
    sccs = _tarjan_scc(adj)
    loop_meta: Dict[str, str] = {}
    for scc in sccs:
        if len(scc) >= 2:
            cl = _classify_loop(adj, scc)
            if cl:
                lid, _ltype = cl
                for n in scc:
                    loop_meta[n] = lid
    rendered: List[str] = ["Causal path trace:"]
    rendered.append(f"  source: {labels.get(src_id, src_id)}")
    rendered.append(f"  target: {labels.get(tgt_id, tgt_id)}")
    # Sort paths by strength descending
    scored = sorted(
        ((p, _path_signature(p)) for p in paths),
        key=lambda x: x[1][1], reverse=True,
    )[:2]
    for i, (p, (net, strength)) in enumerate(scored, 1):
        arrow = ""
        for idx, (nid, sign, _w) in enumerate(p):
            if idx == 0:
                arrow = labels.get(nid, nid)
            else:
                arrow += f" {sign}→ {labels.get(nid, nid)}"
        loops_hit = sorted({loop_meta[n] for (n, _, _) in p if n in loop_meta})
        loops_str = f"  (passes through loop(s): {', '.join(loops_hit)})" if loops_hit else ""
        rendered.append(f"  path {i}: net {net}, strength {strength:.2f}")
        rendered.append(f"    {arrow}{loops_str}")
    return "\n".join(rendered)


def _build_system_prompt(canvas: CanvasState, scope: SelectionScope, last_user_msg: str) -> str:
    nodes_visible, edges_visible = _filter_visible(canvas, scope)
    scope_block = _summarise_scope(nodes_visible, edges_visible, scope)
    label_to_id = {_normalize(n.label): n.id for n in nodes_visible}
    parsed = _try_parse_effect_question(last_user_msg, label_to_id)
    trace_block = ""
    if parsed:
        src_id, tgt_id = parsed
        trace_block = _trace_paths_block(canvas, nodes_visible, edges_visible, src_id, tgt_id)
    # Visible loops list (just labels) — we cheat slightly: a "loop" here is
    # whatever the frontend sent in scope.selected_loop_id; we don't try to
    # recompute named loops because the canvas data carries them via
    # `links` pairs only when a single loop is selected.
    loop_block_lines = []
    if scope.selected_loop_id and scope.selected_loop_id != "ALL":
        loop_block_lines.append(
            f"Visible loops:\n  - {scope.selected_loop_id} "
            f"({'Reinforcing' if scope.selected_loop_type == 'R' else 'Balancing' if scope.selected_loop_type == 'B' else 'unknown'})"
        )
    else:
        loop_block_lines.append("Visible loops: ALL (no specific loop highlighted)")
    pieces = [SYSTEM_PROMPT_HEADER, "", scope_block, "", "\n".join(loop_block_lines)]
    if trace_block:
        pieces.extend(["", trace_block])
    return "\n".join(pieces)


# ── Public API ─────────────────────────────────────────────────────────

class LoopAssistant:
    def __init__(self, cfg: Config, router: LLMRouter):
        self.cfg = cfg
        self.router = router

    def respond_stream(self, req: LoopChatRequest) -> Iterator[Tuple[str, Any]]:
        """Stream chunks of the loop assistant's reply.

        Yields:
            ("delta", str) — token / chunk of text
            ("done", str)  — final assembled reply
        """
        last_user_msg = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            "",
        )
        system = _build_system_prompt(req.canvas, req.scope, last_user_msg)
        spec = self.router.spec("causal_narrator")  # reuse the narrator model
        try:
            if spec.backend == "anthropic":
                yield from self._stream_anthropic(spec, system, req.messages)
            else:
                yield from self._stream_openai(spec, system, req.messages)
        except Exception as e:
            LOG.exception("loop-chat stream failed")
            err = f"(loop-assistant error: {e})"
            yield ("delta", err)
            yield ("done", err)

    # ── OpenAI-compatible ─────────────────────────────────────────────

    def _stream_openai(self, spec, system: str, messages: List[ChatMessage]) -> Iterator[Tuple[str, Any]]:
        from openai import OpenAI
        api_key = (os.environ.get(spec.api_key_env or "OPENAI_API_KEY", "") or "").strip()
        kwargs: Dict[str, Any] = {"api_key": api_key or "EMPTY"}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        client = OpenAI(**kwargs)

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "system":
                continue
            msgs.append({"role": m.role, "content": m.content})

        stream = client.chat.completions.create(
            model=spec.model,
            messages=msgs,
            temperature=spec.temperature,
            stream=True,
        )
        buf: List[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            txt = getattr(delta, "content", None)
            if txt:
                buf.append(txt)
                yield ("delta", txt)
        yield ("done", "".join(buf).strip() or "(no response)")

    # ── Anthropic ─────────────────────────────────────────────────────

    def _stream_anthropic(self, spec, system: str, messages: List[ChatMessage]) -> Iterator[Tuple[str, Any]]:
        from anthropic import Anthropic
        raw_key = os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY", "") or ""
        api_key = raw_key.strip()
        client = Anthropic(api_key=api_key) if api_key else Anthropic()
        msg_history = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]
        resp = client.messages.create(
            model=spec.model,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            system=system,
            messages=msg_history,
        )
        out = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if out:
            yield ("delta", out)
        yield ("done", out.strip() or "(no response)")
