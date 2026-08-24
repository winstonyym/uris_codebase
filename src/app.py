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
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# Ensure relative-package imports work even when uvicorn launches us oddly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src import usage
from src.auth import ClerkUser, allowed_origins, conversation_id, require_user
from src.usage import QuotaExceeded, get_ledger
from src.causal import CausalReasoner
from src.chat_agent import ChatAgent
from src.loop_chat import LoopAssistant
from src.loop_recommender import LoopRecommender
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
    LoopRecommendRequest,
    LoopRecommendResponse,
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
    loop_recommender = LoopRecommender(cfg, router)

    # Warm the in-memory KG indices + caches, but DO NOT block the lifespan on
    # it. The warm runs a full Neo4j query to load every KG embedding, which can
    # take many seconds (cold Aura connection + transfer). If we awaited it here
    # the ASGI app wouldn't accept ANY request — even /health — until it
    # finished, which is exactly the "backend feels dead on first load" symptom.
    #
    # Instead we kick it off on a daemon thread so the app reports ready
    # immediately. `_ensure_indices()` is idempotent and lock-guarded, so the
    # first real /suggest (or an explicit /warm ping) safely waits on / re-runs
    # it if the background thread hasn't finished yet.
    def _bg_warm():
        try:
            recommender.warm()
            LOG.info("Background warm-up complete.")
        except Exception as e:
            LOG.warning("Recommender warm-up failed (%s); will retry lazily", e)

    threading.Thread(target=_bg_warm, name="kg-warm", daemon=True).start()

    app.state.cfg = cfg
    app.state.neo = neo
    app.state.embedder = embedder
    app.state.router = router
    app.state.recommender = recommender
    app.state.causal = causal
    app.state.chat = chat
    app.state.loop_chat = loop_chat
    app.state.loop_recommender = loop_recommender
    LOG.info("Graph-RAG service ready.")
    try:
        yield
    finally:
        neo.close()


app = FastAPI(title="Graph-RAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Set ALLOWED_ORIGINS on the deployment to narrow this to your own
    # frontends; falls back to "*" so an unconfigured deploy still works.
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    # Authorization carries the Clerk session token, X-Session-Id names the
    # conversation the free-tier budget is charged against.
    allow_headers=["*"],
    expose_headers=["X-Usage-Used", "X-Usage-Limit", "X-Usage-Remaining"],
)


# ── Auth + free-tier quota ────────────────────────────────────────────
#
# Every endpoint below that can reach an LLM depends on `billed_caller`,
# which (a) verifies the Clerk session token and (b) checks the caller
# hasn't already spent this conversation's token budget. Endpoints that
# cost nothing — health, warm, stats — stay open so the keep-warm cron and
# the pre-sign-in landing page keep working.


class Caller:
    """A verified user plus the conversation their tokens are charged to."""

    __slots__ = ("user", "conversation")

    def __init__(self, user: ClerkUser, conversation: str):
        self.user = user
        self.conversation = conversation

    @property
    def user_id(self) -> str:
        return self.user.user_id


def billed_caller(request: Request, user: ClerkUser = Depends(require_user)) -> Caller:
    """Verify the caller and admit them under the free tier, or raise."""
    conversation = conversation_id(request, user)
    get_ledger().admit(user.user_id, conversation)
    return Caller(user, conversation)


@app.exception_handler(QuotaExceeded)
async def _quota_exceeded_handler(request: Request, exc: QuotaExceeded) -> JSONResponse:
    """Turn a blown budget into a 429 the frontend can render verbatim."""
    body: Dict[str, Any] = {
        "error": "quota_exceeded",
        "reason": exc.reason,
        "detail": exc.message,
    }
    body.update(exc.detail)
    return JSONResponse(status_code=429, content=body)


def _usage_headers(acc: usage.UsageAccumulator) -> Dict[str, str]:
    """Budget meter for a just-completed request, from the committed total."""
    m = usage.meter(acc.conversation_total, acc.total)
    return {
        "X-Usage-Used": str(m["used_tokens"]),
        "X-Usage-Limit": str(m["limit_tokens"]),
        "X-Usage-Remaining": str(m["remaining_tokens"] if m["remaining_tokens"] is not None else -1),
    }


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


@app.api_route("/warm", methods=["GET", "POST"])
def warm() -> Dict[str, Any]:
    """Explicit wake-up call for cold serverless instances.

    Synchronously does the expensive one-time work so the user's FIRST real
    request is fast: (1) builds the KG indices (idempotent — no-op if the
    background warm already finished), and (2) instantiates every LLM client so
    the first chat/suggest/recommend call doesn't pay client construction.

    Designed to be pinged from the Welcome page on mount AND by an external
    keep-warm pinger. Never raises — partial warmth is reported, not thrown, so
    a flaky Neo4j/LLM never turns a warm-up into a 500.
    """
    t0 = time.perf_counter()
    result: Dict[str, Any] = {"ok": True, "steps": {}}

    # 1) KG indices (the slow part: a full Neo4j embedding query).
    s = time.perf_counter()
    try:
        app.state.recommender.warm()
        result["steps"]["indices"] = {"ok": True, "ms": int((time.perf_counter() - s) * 1000)}
    except Exception as e:
        result["ok"] = False
        result["steps"]["indices"] = {"ok": False, "error": str(e), "ms": int((time.perf_counter() - s) * 1000)}

    # 2) Pre-create LLM clients so the first completion skips client init.
    #    This builds the client objects (cheap); it does not spend tokens.
    clients: Dict[str, bool] = {}
    for comp in ("chat_agent", "loop_recommender", "causal_narrator",
                 "loop_describer", "recommender_reranker"):
        try:
            app.state.router.get(comp)
            clients[comp] = True
        except Exception as e:
            clients[comp] = False
            LOG.debug("warm: could not pre-create %s client: %s", comp, e)
    result["steps"]["llm_clients"] = clients

    result["total_ms"] = int((time.perf_counter() - t0) * 1000)
    return result


@app.api_route("/db-ping", methods=["GET", "POST"])
def db_ping() -> Dict[str, Any]:
    """Lightweight Neo4j keep-alive.

    Runs a trivial ``RETURN 1`` over the existing Bolt connection so an idle
    Aura Free instance doesn't auto-pause. It deliberately does NOT load the KG
    indices (that path reads from the R2 cache and never touches Neo4j), and it
    is intentionally NOT written to the activity log. Called once a day by the
    keep-backend-warm GitHub Actions workflow. Never raises.
    """
    t0 = time.perf_counter()
    try:
        with app.state.neo.session() as s:
            value = s.run("RETURN 1 AS ok").single()["ok"]
        return {"ok": True, "value": value, "ms": int((time.perf_counter() - t0) * 1000)}
    except Exception as e:
        LOG.warning("/db-ping failed: %s", e)
        return {"ok": False, "error": str(e), "ms": int((time.perf_counter() - t0) * 1000)}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    """Visibility into cache hit-rates and pre-loaded index sizes."""
    return app.state.recommender.stats()


@app.get("/usage")
def usage_meter(request: Request, user: ClerkUser = Depends(require_user)) -> Dict[str, Any]:
    """How much of this conversation's free budget is left.

    The frontend polls this after each turn to draw its meter. Requires a
    valid session token but never 429s — you can always ask how broke you
    are. `backing` reports whether counters are shared (redis) or
    per-process (in-process), so a misconfigured deploy is visible here.
    """
    return get_ledger().snapshot(user.user_id, conversation_id(request, user))


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
def log_events(
    req: LogRequest,
    background: BackgroundTasks,
    user: ClerkUser = Depends(require_user),
) -> Dict[str, Any]:
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
        # Identity now comes from the verified Clerk token, not from the
        # body — a participant can't relabel someone else's session.
        "clerk_user_id": user.user_id,
        "clerk_email":   user.email,
        "access_key_id": req.access_key_id,   # legacy beta key id, if any
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
def suggest(
    req: SuggestRequest,
    response: Response,
    caller: Caller = Depends(billed_caller),
) -> SuggestResponse:
    try:
        with get_ledger().track(caller.user_id, caller.conversation) as acc:
            out = app.state.recommender.suggest(req)
    except Exception as e:
        LOG.exception("/suggest failed")
        raise HTTPException(status_code=500, detail=str(e))
    response.headers.update(_usage_headers(acc))
    return out


@app.post("/loop-describe", response_model=LoopDescribeResponse)
def loop_describe(
    req: LoopDescribeRequest,
    caller: Caller = Depends(billed_caller),
) -> LoopDescribeResponse:
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
        with get_ledger().track(caller.user_id, caller.conversation):
            raw = client.complete(system, user)
        data = extract_json(raw)
        name = str(data.get("name") or fallback_name).strip()
        description = str(data.get("description") or fallback_desc).strip()
        return LoopDescribeResponse(name=name, description=description, type=req.type)
    except Exception as e:
        LOG.warning("/loop-describe falling back to deterministic text: %s", e)
        return LoopDescribeResponse(name=fallback_name, description=fallback_desc, type=req.type)


@app.post("/loop-recommend", response_model=LoopRecommendResponse)
def loop_recommend(
    req: LoopRecommendRequest,
    caller: Caller = Depends(billed_caller),
) -> LoopRecommendResponse:
    """Background feedback-loop recommender for the Modify-tab Diagram Assistant.

    Inspects ONLY the current canvas (never the KG) and, when it's highly
    confident, returns a single composite ``add_loop`` mutation completing a
    reinforcing/balancing cycle. Returns ``found = False`` otherwise so the UI
    stays silent. Never raises — a failure is just "no recommendation".
    """
    try:
        with get_ledger().track(caller.user_id, caller.conversation):
            return app.state.loop_recommender.recommend(req)
    except Exception as e:
        LOG.warning("/loop-recommend failed softly: %s", e)
        return LoopRecommendResponse(found=False)


@app.post("/causal-query", response_model=CausalQueryResponse)
def causal_query(
    req: CausalQueryRequest,
    response: Response,
    caller: Caller = Depends(billed_caller),
) -> CausalQueryResponse:
    try:
        with get_ledger().track(caller.user_id, caller.conversation) as acc:
            out = app.state.causal.answer(req)
    except Exception as e:
        LOG.exception("/causal-query failed")
        raise HTTPException(status_code=500, detail=str(e))
    response.headers.update(_usage_headers(acc))
    return out


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    response: Response,
    caller: Caller = Depends(billed_caller),
) -> ChatResponse:
    """Non-streaming chat. Returns the assistant's text reply plus any
    pending mutations the user can accept/reject from the UI."""
    try:
        with get_ledger().track(caller.user_id, caller.conversation) as acc:
            out = app.state.chat.respond(req.messages, req.canvas)
    except Exception as e:
        LOG.exception("/chat failed")
        raise HTTPException(status_code=500, detail=str(e))
    response.headers.update(_usage_headers(acc))
    return out


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    caller: Caller = Depends(billed_caller),
):
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

    ledger = get_ledger()

    async def event_gen():
        t0 = time.perf_counter()
        yield {"event": "thinking", "data": "{}"}

        send, receive = anyio.create_memory_object_stream(64)
        # Created out here, bound inside the worker thread: the LLM clients
        # push their token counts into it from wherever they run.
        acc = usage.UsageAccumulator()

        def produce():
            # Runs on a worker thread — drives the sync generator and
            # pushes each ("kind", payload) onto the stream.
            try:
                with usage.bind(acc):
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

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(anyio.to_thread.run_sync, produce)
                async for kind, data in receive:
                    yield {"event": kind, "data": data}
        finally:
            # Charge the budget even if the client hung up mid-stream —
            # those tokens were still spent. Nothing may be yielded from a
            # finally block during generator close, so the meter event goes
            # after it, on the normal path only.
            acc.conversation_total = ledger.commit(caller.user_id, caller.conversation, acc)

        yield {"event": "usage", "data": json.dumps(usage.meter(acc.conversation_total, acc.total))}
        yield {"event": "done_meta", "data": str(int((time.perf_counter() - t0) * 1000))}

    return EventSourceResponse(event_gen())


@app.post("/loop-chat/stream")
async def loop_chat_stream(
    req: LoopChatRequest,
    request: Request,
    caller: Caller = Depends(billed_caller),
):
    """Scope-bound, read-only chat for the Visualise tab.

    The generator in `loop_chat.LoopAssistant.respond_stream` yields
    ("delta", text) and ("done", final_text). We re-emit them as SSE
    events of the same names; the frontend parses ``delta`` as
    ``{text: "..."}`` for symmetry with /chat/stream.
    """
    import anyio

    ledger = get_ledger()

    async def event_gen():
        t0 = time.perf_counter()
        yield {"event": "thinking", "data": "{}"}
        send, receive = anyio.create_memory_object_stream(64)
        acc = usage.UsageAccumulator()

        def produce():
            try:
                with usage.bind(acc):
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

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(anyio.to_thread.run_sync, produce)
                async for kind, data in receive:
                    yield {"event": kind, "data": data}
        finally:
            acc.conversation_total = ledger.commit(caller.user_id, caller.conversation, acc)

        yield {"event": "usage", "data": json.dumps(usage.meter(acc.conversation_total, acc.total))}
        yield {"event": "done_meta", "data": str(int((time.perf_counter() - t0) * 1000))}

    return EventSourceResponse(event_gen())


@app.post("/suggest/stream")
async def suggest_stream(
    req: SuggestRequest,
    request: Request,
    caller: Caller = Depends(billed_caller),
):
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

    ledger = get_ledger()

    async def event_gen():
        t0 = time.perf_counter()
        yield {"event": "thinking", "data": "{}"}
        acc = usage.UsageAccumulator()

        def run_stages():
            with usage.bind(acc):
                return list(app.state.recommender.stages(req))

        try:
            # Drive the synchronous generator on a thread so the loop is
            # free for the next request / heartbeat.
            it = await anyio.to_thread.run_sync(run_stages)
        except Exception as e:
            LOG.exception("/suggest/stream failed")
            ledger.commit(caller.user_id, caller.conversation, acc)
            yield {"event": "error", "data": str(e)}
            return
        acc.conversation_total = ledger.commit(caller.user_id, caller.conversation, acc)
        for kind, resp in it:
            yield {"event": kind, "data": resp.model_dump_json()}
        yield {"event": "usage", "data": json.dumps(usage.meter(acc.conversation_total, acc.total))}
        yield {"event": "done", "data": str(int((time.perf_counter() - t0) * 1000))}

    return EventSourceResponse(event_gen())
