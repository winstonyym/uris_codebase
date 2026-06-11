"""FastAPI entrypoint for the Graph-RAG service.

Run:
    cd graph_rag
    uvicorn graph_rag.app:app --reload --port 8000

The service depends on:
  * Neo4j running on $NEO4J_URI (default bolt://localhost:7687)
  * OPENAI_API_KEY in the environment (for embeddings)
  * ANTHROPIC_API_KEY in the environment (for default LLM components)

All three can be overridden via models.yaml + env.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# Ensure relative-package imports work even when uvicorn launches us oddly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.causal import CausalReasoner
from src.chat_agent import ChatAgent
from src.loop_chat import LoopAssistant
from src import r2_uploader
from src.config import load_config
from src.embeddings import make_embedder
from src.llm_router import LLMRouter
from src.neo4j_client import Neo4jClient
from src.recommender import Recommender
from src.schemas import (
    CausalQueryRequest,
    CausalQueryResponse,
    ChatRequest,
    ChatResponse,
    LogRequest,
    LoopChatRequest,
    LoopDescribeRequest,
    LoopDescribeResponse,
    SuggestRequest,
    SuggestResponse,
)

from dotenv import load_dotenv  # noqa: E402
# Important: override=False so a key already exported in the user's shell
# wins over a possibly-stale value in .env (avoids the classic
# "I rotated my key but the service still uses the old one" 401).
load_dotenv(override=False)

LOG = logging.getLogger("graph_rag.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")


# ── Lifespan: build all singletons once and reuse them ───────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_path = os.environ.get("GRAPH_RAG_MODELS_YAML") or None
    cfg = load_config(cfg_path)
    LOG.info("Loaded config from %s", cfg.models_yaml_path)

    neo = Neo4jClient(
        uri=cfg.settings.neo4j_uri,
        user=cfg.settings.neo4j_user,
        password=cfg.settings.neo4j_password,
        database=cfg.settings.neo4j_database,
    )
    embedder = make_embedder(cfg.components["embedder"])
    router = LLMRouter(cfg)
    causal = CausalReasoner(cfg, router)
    recommender = Recommender(cfg, neo, embedder, router)
    chat = ChatAgent(cfg, router, causal_reasoner=causal)
    loop_chat = LoopAssistant(cfg, router)

    # Warm the in-memory KG indices + caches eagerly so the first /suggest
    # call doesn't pay the load cost. Falls back to lazy-load if Neo4j is
    # unreachable at startup — the recommender's _ensure_indices retries.
    try:
        recommender.warm()
    except Exception as e:
        LOG.warning("Recommender warm-up failed (%s); will retry lazily", e)

    app.state.cfg = cfg
    app.state.neo = neo
    app.state.embedder = embedder
    app.state.router = router
    app.state.recommender = recommender
    app.state.causal = causal
    app.state.chat = chat
    app.state.loop_chat = loop_chat
    LOG.info("Graph-RAG service ready.")
    try:
        yield
    finally:
        neo.close()


app = FastAPI(title="Graph-RAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # dev — tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
def root() -> Dict[str, str]:
    """Root health probe — Vercel pings this to confirm the function is live."""
    return {"status": "ok"}


@app.get("/health")
def health() -> Dict[str, Any]:
    cfg = app.state.cfg
    return {
        "status": "ok",
        "neo4j": cfg.settings.neo4j_uri,
        "components": {k: v.model for k, v in cfg.components.items()},
        "r2": r2_uploader.status(),
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    """Visibility into cache hit-rates and pre-loaded index sizes."""
    return app.state.recommender.stats()


# ── Activity logging ──────────────────────────────────────────────────
#
# One JSON document per (user, session). Browser clients post batches;
# this endpoint accumulates them into a single, pretty-printed JSON
# object — session metadata at the top (including the beta access key id)
# and an `events` array — so the file previews cleanly in the Cloudflare
# R2 dashboard instead of being raw JSONL.

import re as _re   # local alias so we don't reshuffle the imports above
from datetime import datetime as _dt, timezone as _tz

# On Vercel the only writable path is /tmp — anywhere else is read-only,
# so /tmp/logs is the canonical local target. The R2 mirror keeps the
# data durable across the ephemeral function lifecycle.
_LOGS_DIR = Path("/tmp/logs")
_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_slug(s: str, limit: int = 60) -> str:
    return _re.sub(r"[^A-Za-z0-9_-]+", "_", s or "")[:limit] or "anon"


def _load_session_doc(path: Path, r2_key: str) -> Dict[str, Any]:
    """Return this session's accumulated JSON doc.

    Prefers the local /tmp copy; if absent (e.g. a cold serverless
    instance), seeds from the existing R2 object so events accumulated by
    earlier instances aren't lost. Falls back to a fresh doc. Never raises.
    """
    for source in (
        lambda: path.read_bytes() if path.exists() else None,
        lambda: r2_uploader.read_object(r2_key) if r2_uploader.is_enabled() else None,
    ):
        try:
            raw = source()
            if raw:
                doc = json.loads(raw)
                if isinstance(doc, dict) and isinstance(doc.get("events"), list):
                    return doc
        except Exception:
            LOG.debug("session doc load failed, continuing", exc_info=True)
    return {"events": []}


@app.post("/log")
def log_events(req: LogRequest, background: BackgroundTasks) -> Dict[str, Any]:
    """Accumulate a batch of activity events into this session's JSON log.

    Two-tier durability:
      1. Local read-modify-write (sync, in-request) under /tmp — the
         working copy for this instance, seeded from R2 when cold.
      2. Cloudflare R2 upload (background, best-effort) — uploads the
         whole updated JSON document under
         `sessions/<YYYY-MM-DD>/<user>_<session>.json`. Never blocks the
         response; if R2 is unreachable or unconfigured, the local path
         keeps working.

    The document is a single JSON object: session metadata at the top
    (variant + beta access key id included) and an `events` array.
    """
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    fname = f"{_safe_slug(req.user_id)}_{_safe_slug(req.session_id, 80)}.json"
    path = _LOGS_DIR / fname
    r2_key = r2_uploader.build_session_key(req.user_id, req.session_id, req.started_at)

    prior = _load_session_doc(path, r2_key)
    events = prior.get("events") or []
    for ev in req.events:
        events.append({
            "timestamp": ev.timestamp,
            "event":     ev.event,
            "payload":   ev.payload,
        })

    # Rebuild as an ordered document: session metadata first (so the beta
    # access key id is right at the top of the R2 preview), events last.
    doc = {
        "session_id":    req.session_id,
        "user_id":       req.user_id,
        "name":          req.name,
        "variant":       req.variant,
        "secure":        req.secure,
        "access_key_id": req.access_key_id,   # beta access key (never the raw key)
        "started_at":    req.started_at,
        "updated_at":    _dt.now(_tz.utc).isoformat(),
        "event_count":   len(events),
        "events":        events,
    }

    try:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        LOG.exception("/log write failed for %s", path)
        raise HTTPException(status_code=500, detail=f"log write failed: {e}")

    # Mirror to R2 — async, never blocks the response, never raises.
    r2_synced = False
    if r2_uploader.is_enabled():
        background.add_task(r2_uploader.upload_file, path, r2_key)
        r2_synced = True

    return {
        "ok": True,
        "written": len(req.events),
        "event_count": doc["event_count"],
        "file": fname,
        "r2_synced": r2_synced,
    }


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest) -> SuggestResponse:
    try:
        return app.state.recommender.suggest(req)
    except Exception as e:
        LOG.exception("/suggest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/loop-describe", response_model=LoopDescribeResponse)
def loop_describe(req: LoopDescribeRequest) -> LoopDescribeResponse:
    """Name + describe a single feedback loop the frontend just detected.

    The loop's *type* (R / B) is computed deterministically on the client
    from edge-sign parity and passed in — the LLM only writes prose, it
    never re-classifies. Uses the `loop_describer` component (deepseek-v4-flash
    by default). If the LLM call or JSON parse fails, we fall back to a
    deterministic description so the UI always has something to show.
    """
    type_word = "reinforcing" if req.type == "R" else "balancing"

    # Render the cycle as a readable signed walk, e.g.
    #   New house price -→ Real family demand  (the arrow carries the sign)
    walk_lines = []
    for e in req.edges:
        sign = "+" if e.polarity == "+" else "−"
        walk_lines.append(f"  {e.source_label} {sign}→ {e.target_label}")
    walk = "\n".join(walk_lines) if walk_lines else "  (no edges)"
    n_neg = sum(1 for e in req.edges if e.polarity == "-")

    system = (
        "You are a system-dynamics expert. Given the edges of ONE feedback "
        "loop in a causal loop diagram, write a short title and a 1-2 "
        "sentence plain-English explanation of how the loop behaves. "
        "The loop's type is already decided for you — do not contradict it. "
        "A REINFORCING (R) loop amplifies change; a BALANCING (B) loop "
        "counteracts it. Respond ONLY with a JSON object of the form "
        '{"name": "...", "description": "..."}. No markdown, no extra keys.'
    )
    user = (
        f"Loop id: {req.loop_id}\n"
        f"Loop type: {req.type} ({type_word}; it has {n_neg} negative link(s))\n"
        f"Edges (in cycle order):\n{walk}\n\n"
        "Write the JSON now."
    )

    # Deterministic fallback used if anything below fails.
    fallback_name = f"{type_word.capitalize()} loop"
    if req.edges:
        nodes_in_order = [req.edges[0].source_label] + [e.target_label for e in req.edges[:-1]]
        fallback_desc = (
            f"A {type_word} loop linking "
            + " → ".join(nodes_in_order)
            + (". Change in any variable propagates around the loop and "
               + ("amplifies itself." if req.type == "R" else "is counteracted."))
        )
    else:
        fallback_desc = f"A {type_word} feedback loop."

    try:
        from src.llm_client import extract_json
        _spec, client = app.state.router.get("loop_describer")
        raw = client.complete(system, user)
        data = extract_json(raw)
        name = str(data.get("name") or fallback_name).strip()
        description = str(data.get("description") or fallback_desc).strip()
        return LoopDescribeResponse(name=name, description=description, type=req.type)
    except Exception as e:
        LOG.warning("/loop-describe falling back to deterministic text: %s", e)
        return LoopDescribeResponse(name=fallback_name, description=fallback_desc, type=req.type)


@app.post("/causal-query", response_model=CausalQueryResponse)
def causal_query(req: CausalQueryRequest) -> CausalQueryResponse:
    try:
        return app.state.causal.answer(req)
    except Exception as e:
        LOG.exception("/causal-query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Non-streaming chat. Returns the assistant's text reply plus any
    pending mutations the user can accept/reject from the UI."""
    try:
        return app.state.chat.respond(req.messages, req.canvas)
    except Exception as e:
        LOG.exception("/chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Token-streaming chat over SSE.

    Event sequence:
      • ``thinking`` — fired immediately so the UI shows a spinner.
      • ``delta``    — a chunk of assistant text (may fire many times).
      • ``mutation`` — a staged graph change (add/remove node/edge); the
                       frontend renders a confirm card for each.
      • ``done``     — final ChatResponse JSON (full reply + all mutations)
                       plus total latency.
      • ``error``    — emitted with a message if the pipeline fails.

    The chat agent's tool-use loop is synchronous, so we run the whole
    generator on a worker thread and bridge each yielded item back onto
    the asyncio event loop via a queue.
    """
    import anyio

    async def event_gen():
        t0 = time.perf_counter()
        yield {"event": "thinking", "data": "{}"}

        send, receive = anyio.create_memory_object_stream(64)

        def produce():
            # Runs on a worker thread — drives the sync generator and
            # pushes each ("kind", payload) onto the stream.
            try:
                for kind, payload in app.state.chat.respond_stream(
                    req.messages, req.canvas,
                ):
                    if kind == "done":
                        data = payload.model_dump_json()
                    elif kind == "mutation":
                        data = payload.model_dump_json()
                    else:  # delta
                        data = json.dumps({"text": payload})
                    anyio.from_thread.run(send.send, (kind, data))
            except Exception as e:  # pragma: no cover
                LOG.exception("/chat/stream failed")
                anyio.from_thread.run(send.send, ("error", str(e)))
            finally:
                anyio.from_thread.run(send.aclose)

        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.to_thread.run_sync, produce)
            async for kind, data in receive:
                yield {"event": kind, "data": data}

        yield {"event": "done_meta", "data": str(int((time.perf_counter() - t0) * 1000))}

    return EventSourceResponse(event_gen())


@app.post("/loop-chat/stream")
async def loop_chat_stream(req: LoopChatRequest, request: Request):
    """Scope-bound, read-only chat for the Visualise tab.

    The generator in `loop_chat.LoopAssistant.respond_stream` yields
    ("delta", text) and ("done", final_text). We re-emit them as SSE
    events of the same names; the frontend parses ``delta`` as
    ``{text: "..."}`` for symmetry with /chat/stream.
    """
    import anyio

    async def event_gen():
        t0 = time.perf_counter()
        yield {"event": "thinking", "data": "{}"}
        send, receive = anyio.create_memory_object_stream(64)

        def produce():
            try:
                for kind, payload in app.state.loop_chat.respond_stream(req):
                    if kind == "delta":
                        data = json.dumps({"text": payload})
                    else:  # done
                        data = json.dumps({"reply": payload})
                    anyio.from_thread.run(send.send, (kind, data))
            except Exception as e:
                LOG.exception("/loop-chat/stream failed")
                anyio.from_thread.run(send.send, ("error", str(e)))
            finally:
                anyio.from_thread.run(send.aclose)

        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.to_thread.run_sync, produce)
            async for kind, data in receive:
                yield {"event": kind, "data": data}

        yield {"event": "done_meta", "data": str(int((time.perf_counter() - t0) * 1000))}

    return EventSourceResponse(event_gen())


@app.post("/suggest/stream")
async def suggest_stream(req: SuggestRequest, request: Request):
    """Progressive SSE stream of /suggest results.

    Event sequence:
      • ``thinking``    — fired immediately so the UI can show a spinner.
      • ``vector``      — fast deterministic result (target ≤100 ms warm,
                          ≤200 ms cold). Frontend should render ghosts
                          on receipt of this event.
      • ``rerank``      — *optional* follow-up after the LLM rerank
                          finishes. Same schema as ``vector``; frontend
                          should patch α values in place.
      • ``done``        — final signal carrying total latency in ms.
      • ``error``       — emitted with a message if anything blew up.

    The whole pipeline runs in a worker thread so the asyncio event-loop
    isn't blocked by Neo4j or LLM calls.
    """
    import anyio

    async def event_gen():
        t0 = time.perf_counter()
        yield {"event": "thinking", "data": "{}"}
        try:
            # Drive the synchronous generator on a thread so the loop is
            # free for the next request / heartbeat.
            it = await anyio.to_thread.run_sync(
                lambda: list(app.state.recommender.stages(req))
            )
        except Exception as e:
            LOG.exception("/suggest/stream failed")
            yield {"event": "error", "data": str(e)}
            return
        for kind, resp in it:
            yield {"event": kind, "data": resp.model_dump_json()}
        yield {"event": "done", "data": str(int((time.perf_counter() - t0) * 1000))}

    return EventSourceResponse(event_gen())
