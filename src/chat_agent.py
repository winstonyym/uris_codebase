"""Tool-calling chat agent for the Modify-tab chat panel.

Capabilities, exposed as tools to BOTH the Anthropic and OpenAI-compatible
chat-completions APIs (DeepSeek, Qwen, Gemini-OpenAI-mode, vLLM, …):

  * add_node       — propose a new node on the canvas
  * remove_node    — remove a node and incident edges
  * add_edge       — propose a new signed edge
  * remove_edge    — remove an edge
  * list_canvas    — read-only inspection of current canvas state

The Diagram Assistant is deliberately scoped to STRUCTURE only. Causal
interpretation, "what-if" questions, and loop tracing belong to the
LoopAssistant on the Visualise tab (graph_rag/loop_chat.py).

Mutations are *staged* rather than applied: each tool call returns a
`PendingMutation` object that the frontend renders as a confirm/reject
card. The user clicks Accept → the React state mutates → the canvas
updates. This keeps undo history coherent and prevents the LLM from
making destructive edits unilaterally.

Backends:
  * `anthropic` — native Messages API tool-use.
  * `openai`    — chat-completions `tools` / `tool_calls`. Works for any
                  OpenAI-compatible server (the default `chat_agent`
                  model `deepseek-v4-flash` lives here).

Both backends support streaming via `respond_stream()`, which yields
("delta", text), ("mutation", PendingMutation) and ("done", ChatResponse)
events for the /chat/stream SSE endpoint.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import usage
from .config import Config
from .llm_router import LLMRouter
from .schemas import CanvasState, ChatMessage, ChatResponse, PendingMutation


LOG = logging.getLogger("graph_rag.chat_agent")

from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=False)


# Hard ceiling on how many *new* nodes one chat turn may stage. The user
# asked for "just one or two" — we both instruct the model (system prompt)
# and enforce it in _dispatch_tool so a chatty model can't blow past it.
MAX_NEW_NODES = 2

# Max agentic tool-use iterations per user message.
MAX_STEPS = 5


SYSTEM_PROMPT = f"""You are the Diagram Assistant inside a causal-loop diagram (CLD) editor.

YOUR SCOPE IS STRICTLY LIMITED TO STRUCTURAL EDITS. You may only:
  • Add a node (variable) to the canvas.
  • Remove a node (and its incident edges) from the canvas.
  • Add a signed directed edge between two existing nodes.
  • Remove an edge between two existing nodes.
  • Inspect the current canvas with `list_canvas` when you need context.

YOU MUST NOT:
  • Explain or interpret what the diagram MEANS.
  • Trace causal paths, predict effects ("what if X increases?"), describe
    feedback loops, or comment on relationships between variables.
  • Speculate about real-world dynamics.

If the user asks for any kind of interpretation, analysis, "what-if" question,
loop explanation, or anything that is not a structural edit, you MUST politely
decline in 1-2 sentences and direct them to the Loop Assistant on the
**Visualise** tab, e.g.:

    "I can only add or remove nodes and edges here. For questions about how
    variables affect each other, switch to the Visualise tab and ask the
    Loop Assistant."

When the user requests a valid mutation, ALWAYS use the matching tool — never
describe the change in prose alone. The tool's result is shown to the user as
a "pending diff" they can accept or reject. After calling tools, give a short
conversational reply (≤2 sentences) describing what you've staged.

IMPORTANT constraints:
  • Add AT MOST {MAX_NEW_NODES} new nodes per request. If the user asks for more,
    add the {MAX_NEW_NODES} most important ones and say so in your reply.
  • Strongly prefer connecting to / reusing variables ALREADY on the canvas
    rather than inventing new nodes. Only add a node when nothing suitable exists.
  • Choose edge polarity from common-sense system dynamics."""


def _build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "add_node",
            "description": (
                "Stage a new node (variable) to add to the canvas. "
                "Subsystem must be one of the 14 canonical values below; "
                "this drives the node's colour on the diagram."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "subsystem": {
                        "type": "string",
                        "description": "Canonical category the variable belongs to.",
                        "enum": [
                            "Demographics", "Land Use", "Housing", "Transportation",
                            "Infrastructure", "Energy", "Utilities", "Environment",
                            "Economy", "Governance", "Finance", "Health",
                            "Social", "Others",
                        ],
                    },
                    "description": {"type": "string"},
                },
                "required": ["label", "subsystem"],
            },
        },
        {
            "name": "remove_node",
            "description": "Stage removal of a node (and all its incident edges) by label or id.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "node": {"type": "string", "description": "Label or canvas id of the node to remove."},
                },
                "required": ["node"],
            },
        },
        {
            "name": "add_edge",
            "description": "Stage a new directed signed edge between two existing canvas nodes.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source node label or id."},
                    "target": {"type": "string", "description": "Target node label or id."},
                    "polarity": {"type": "string", "enum": ["+", "-"]},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target", "polarity"],
            },
        },
        {
            "name": "remove_edge",
            "description": "Stage removal of an edge by source+target labels/ids.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "target"],
            },
        },
        # `query_effect` was removed from the Diagram Assistant on
        # purpose: interpretation / "what-if" Q&A is the Loop Assistant's
        # job on the Visualise tab. The system prompt redirects users
        # there. The dispatcher case below is kept for defence in depth
        # (returns an error if the model ever tries to call it anyway).
        {
            "name": "list_canvas",
            "description": "Return the current canvas (read-only). Use sparingly; prefer reasoning from prior context.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]


def _openai_tools() -> List[Dict[str, Any]]:
    """Convert the Anthropic-style tool specs into OpenAI chat-completions
    `tools` format (`{"type":"function","function":{name,description,parameters}}`)."""
    out: List[Dict[str, Any]] = []
    for t in _build_tools():
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return out


def _normalize_label(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _resolve_node(token: str, canvas: CanvasState) -> Optional[str]:
    """Resolve a label-or-id token to a canvas node id."""
    if not token:
        return None
    # Direct id hit
    for n in canvas.nodes:
        if n.id == token:
            return n.id
    # Label match (normalised)
    t = _normalize_label(token)
    for n in canvas.nodes:
        if _normalize_label(n.label) == t:
            return n.id
    # Substring fallback
    for n in canvas.nodes:
        if t and t in _normalize_label(n.label):
            return n.id
    return None


class ChatAgent:
    def __init__(self, cfg: Config, router: LLMRouter, causal_reasoner=None):
        self.cfg = cfg
        self.router = router
        self.causal = causal_reasoner

    # ── public: non-streaming ──────────────────────────────────────────

    def respond(self, messages: List[ChatMessage], canvas: CanvasState) -> ChatResponse:
        """Non-streaming reply. Collapses respond_stream() into one result."""
        reply_parts: List[str] = []
        mutations: List[PendingMutation] = []
        final: Optional[ChatResponse] = None
        for kind, payload in self.respond_stream(messages, canvas):
            if kind == "delta":
                reply_parts.append(payload)
            elif kind == "mutation":
                mutations.append(payload)
            elif kind == "done":
                final = payload
        if final is not None:
            return final
        return ChatResponse(reply="".join(reply_parts).strip() or "Done.", mutations=mutations)

    # ── public: streaming ──────────────────────────────────────────────

    def respond_stream(
        self, messages: List[ChatMessage], canvas: CanvasState,
    ) -> Iterator[Tuple[str, Any]]:
        """Stream the agent's reply.

        Yields tuples:
          ("delta",    str)             — a chunk of assistant text
          ("mutation", PendingMutation) — a staged graph change
          ("done",     ChatResponse)    — final, full reply + all mutations

        Routes to the backend-specific implementation. OpenAI-compatible
        models get true token streaming; Anthropic models get a single
        text delta after the (non-streamed) tool-use loop completes.
        """
        spec = self.router.spec("chat_agent")
        canvas_summary = self._summarise_canvas(canvas)
        system = SYSTEM_PROMPT + "\n\nCurrent canvas:\n" + canvas_summary
        try:
            if spec.backend == "anthropic":
                yield from self._stream_anthropic(spec, system, messages, canvas)
            else:
                yield from self._stream_openai(spec, system, messages, canvas)
        except Exception as e:
            LOG.exception("chat respond_stream failed")
            yield ("delta", f"\n\n_(chat error: {e})_")
            yield ("done", ChatResponse(reply=f"(chat error: {e})", mutations=[]))

    # ── OpenAI-compatible backend (DeepSeek, Qwen, …) ───────────────────

    def _openai_client(self, spec):
        from openai import OpenAI
        api_key = (os.environ.get(spec.api_key_env or "OPENAI_API_KEY", "") or "").strip()
        kwargs: Dict[str, Any] = {"api_key": api_key or "EMPTY"}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        return OpenAI(**kwargs)

    def _stream_openai(
        self, spec, system: str, messages: List[ChatMessage], canvas: CanvasState,
    ) -> Iterator[Tuple[str, Any]]:
        client = self._openai_client(spec)
        tools = _openai_tools()

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "system":
                continue
            msgs.append({"role": m.role, "content": m.content})

        pending: List[PendingMutation] = []
        reply_parts: List[str] = []

        for _step in range(MAX_STEPS):
            create_kwargs: Dict[str, Any] = {
                "model": spec.model,
                "messages": msgs,
                "tools": tools,
                "temperature": spec.temperature,
                "stream": True,
                # Ask for a trailing usage chunk so the free-tier ledger can
                # charge real token counts instead of guessing from length.
                "stream_options": {"include_usage": True},
            }
            try:
                stream = client.chat.completions.create(**create_kwargs)
            except Exception as e:
                if "stream_options" not in str(e).lower():
                    raise
                LOG.info("chat: provider rejected stream_options; usage will be estimated")
                create_kwargs.pop("stream_options", None)
                stream = client.chat.completions.create(**create_kwargs)

            text_buf: List[str] = []
            # tool calls arrive fragmented across chunks, keyed by index
            tool_acc: Dict[int, Dict[str, str]] = {}
            finish_reason: Optional[str] = None
            saw_usage = False

            for chunk in stream:
                # The usage chunk carries no choices, so read it before the
                # empty-choices guard below skips the chunk entirely.
                if getattr(chunk, "usage", None):
                    usage.record_openai(chunk)
                    saw_usage = True
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if getattr(delta, "content", None):
                    text_buf.append(delta.content)
                    reply_parts.append(delta.content)
                    yield ("delta", delta.content)
                for tc in (getattr(delta, "tool_calls", None) or []):
                    idx = tc.index or 0
                    acc = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function.arguments:
                            acc["args"] += tc.function.arguments

            if not saw_usage:
                # Provider streamed without usage data — charge an estimate
                # rather than letting the turn go free.
                usage.record_estimate(json.dumps(msgs), "".join(text_buf))

            if not tool_acc:
                # No tool calls this turn → conversation is finished.
                break

            # Record the assistant turn (with its tool calls) so the model
            # has the full transcript on the next iteration.
            msgs.append({
                "role": "assistant",
                "content": "".join(text_buf) or None,
                "tool_calls": [
                    {
                        "id": acc["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": acc["name"], "arguments": acc["args"] or "{}"},
                    }
                    for i, acc in sorted(tool_acc.items())
                ],
            })

            # Dispatch each tool call, stream out any staged mutations.
            for i, acc in sorted(tool_acc.items()):
                try:
                    args = json.loads(acc["args"] or "{}")
                except Exception:
                    args = {}
                result, mutation = self._dispatch_tool(acc["name"], args, canvas, pending)
                if mutation:
                    pending.append(mutation)
                    yield ("mutation", mutation)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": acc["id"] or f"call_{i}",
                    "content": json.dumps(result),
                })
            # loop back so the model can react to the tool results

        reply = "".join(reply_parts).strip()
        if not reply:
            reply = (
                f"Staged {len(pending)} change{'s' if len(pending) != 1 else ''}."
                if pending else "Done."
            )
        yield ("done", ChatResponse(reply=reply, mutations=pending))

    # ── Anthropic backend ──────────────────────────────────────────────

    def _stream_anthropic(
        self, spec, system: str, messages: List[ChatMessage], canvas: CanvasState,
    ) -> Iterator[Tuple[str, Any]]:
        from anthropic import Anthropic
        raw_key = os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY", "") or ""
        api_key = raw_key.strip()
        client = Anthropic(api_key=api_key) if api_key else Anthropic()

        msg_history: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            msg_history.append({"role": m.role, "content": m.content})

        tools = _build_tools()
        pending: List[PendingMutation] = []
        reply_chunks: List[str] = []

        for _step in range(MAX_STEPS):
            resp = client.messages.create(
                model=spec.model,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                system=system,
                tools=tools,
                messages=msg_history,
            )
            usage.record_anthropic(resp)
            tool_results_for_next: List[Dict[str, Any]] = []
            assistant_blocks: List[Dict[str, Any]] = []

            for block in resp.content:
                if block.type == "text":
                    if block.text.strip():
                        reply_chunks.append(block.text)
                        # Anthropic path isn't token-streamed; emit the
                        # whole text block as one delta for UI uniformity.
                        yield ("delta", block.text)
                    assistant_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_blocks.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })
                    result, mutation = self._dispatch_tool(
                        block.name, block.input or {}, canvas, pending,
                    )
                    if mutation:
                        pending.append(mutation)
                        yield ("mutation", mutation)
                    tool_results_for_next.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            if assistant_blocks:
                msg_history.append({"role": "assistant", "content": assistant_blocks})
            if resp.stop_reason != "tool_use":
                break
            msg_history.append({"role": "user", "content": tool_results_for_next})

        reply = "\n".join(s for s in reply_chunks if s).strip()
        if not reply:
            reply = (
                f"Staged {len(pending)} change{'s' if len(pending) != 1 else ''}."
                if pending else "Done."
            )
        yield ("done", ChatResponse(reply=reply, mutations=pending))

    # internal --------------------------------------------------------------

    @staticmethod
    def _summarise_canvas(canvas: CanvasState) -> str:
        if not canvas.nodes:
            return "(empty)"
        lines = [f"Nodes ({len(canvas.nodes)}):"]
        for n in canvas.nodes[:60]:
            lines.append(f"  - {n.label}  [{n.category or '?'}]  id={n.id}")
        if len(canvas.nodes) > 60:
            lines.append(f"  …and {len(canvas.nodes) - 60} more.")
        lines.append(f"Edges ({len(canvas.edges)}):")
        # Resolve labels for readability
        id_to_label = {n.id: n.label for n in canvas.nodes}
        for e in canvas.edges[:80]:
            lines.append(f"  - {id_to_label.get(e.source, e.source)} {e.polarity}→ {id_to_label.get(e.target, e.target)}")
        if len(canvas.edges) > 80:
            lines.append(f"  …and {len(canvas.edges) - 80} more.")
        return "\n".join(lines)

    def _dispatch_tool(
        self,
        name: str,
        args: Dict[str, Any],
        canvas: CanvasState,
        pending: List[PendingMutation],
    ) -> Tuple[Dict[str, Any], Optional[PendingMutation]]:
        """Return (tool_result_for_model, optional pending mutation).

        `pending` is the list of mutations staged so far this turn — used
        to enforce the MAX_NEW_NODES cap.
        """
        if name == "list_canvas":
            payload = {
                "nodes": [n.model_dump() for n in canvas.nodes],
                "edges": [e.model_dump() for e in canvas.edges],
            }
            return payload, None

        if name == "add_node":
            label = (args.get("label") or "").strip()
            if not label:
                return {"error": "label is required"}, None
            # Enforce the per-turn new-node ceiling.
            already = sum(1 for m in pending if m.op == "add_node")
            if already >= MAX_NEW_NODES:
                return {
                    "error": (
                        f"Node limit reached — at most {MAX_NEW_NODES} new nodes "
                        f"per request. Connect to existing canvas nodes instead."
                    )
                }, None
            # Subsystem replaces the old `category` field on the canvas
            # — it must come from the 14-value enum advertised by the
            # tool above. Falls back to "Others" if the model forgets,
            # which still resolves to a valid colour client-side.
            mut = PendingMutation(
                id=f"m_{uuid.uuid4().hex[:8]}",
                op="add_node",
                payload={
                    "id": f"node_{uuid.uuid4().hex[:8]}",
                    "label": label,
                    "subsystem": args.get("subsystem") or "Others",
                    "description": args.get("description") or "",
                },
                summary=f"Add node “{label}”",
            )
            return {"status": "staged", "mutation_id": mut.id}, mut

        if name == "remove_node":
            nid = _resolve_node(args.get("node", ""), canvas)
            if not nid:
                return {"error": f"node not found: {args.get('node')!r}"}, None
            label = next((n.label for n in canvas.nodes if n.id == nid), nid)
            mut = PendingMutation(
                id=f"m_{uuid.uuid4().hex[:8]}",
                op="remove_node",
                payload={"id": nid},
                summary=f"Remove node “{label}” (and incident edges)",
            )
            return {"status": "staged", "mutation_id": mut.id}, mut

        if name == "add_edge":
            sid = _resolve_node(args.get("source", ""), canvas)
            tid = _resolve_node(args.get("target", ""), canvas)
            polarity = args.get("polarity")
            if not sid or not tid:
                return {"error": "source or target not on canvas"}, None
            if polarity not in ("+", "-"):
                return {"error": "polarity must be '+' or '-'"}, None
            slabel = next((n.label for n in canvas.nodes if n.id == sid), sid)
            tlabel = next((n.label for n in canvas.nodes if n.id == tid), tid)
            mut = PendingMutation(
                id=f"m_{uuid.uuid4().hex[:8]}",
                op="add_edge",
                payload={
                    "source": sid,
                    "target": tid,
                    "polarity": polarity,
                    "label": args.get("label") or "",
                    "description": args.get("description") or "",
                },
                summary=f"Add edge {slabel} {polarity}→ {tlabel}",
            )
            return {"status": "staged", "mutation_id": mut.id}, mut

        if name == "remove_edge":
            sid = _resolve_node(args.get("source", ""), canvas)
            tid = _resolve_node(args.get("target", ""), canvas)
            if not sid or not tid:
                return {"error": "source or target not on canvas"}, None
            slabel = next((n.label for n in canvas.nodes if n.id == sid), sid)
            tlabel = next((n.label for n in canvas.nodes if n.id == tid), tid)
            mut = PendingMutation(
                id=f"m_{uuid.uuid4().hex[:8]}",
                op="remove_edge",
                payload={"source": sid, "target": tid},
                summary=f"Remove edge {slabel} → {tlabel}",
            )
            return {"status": "staged", "mutation_id": mut.id}, mut

        if name == "query_effect":
            # Defensive refusal — the tool is no longer advertised to the
            # model in `_build_tools`, but if a stale model call sneaks
            # through we tell it explicitly that this scope belongs to
            # the Loop Assistant on the Visualise tab.
            return {
                "error": (
                    "Interpretation / what-if questions belong to the Loop "
                    "Assistant on the Visualise tab — the Diagram Assistant "
                    "only edits structure (add / remove nodes and edges)."
                )
            }, None

        return {"error": f"unknown tool {name!r}"}, None
