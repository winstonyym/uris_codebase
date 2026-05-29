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

from neo4j import GraphDatabase
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.causal import CausalReasoner
from src.chat_agent import ChatAgent
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
    SuggestRequest,
    SuggestResponse,
)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=False)

LOG = logging.getLogger("graph_rag.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")

# driver = GraphDatabase.driver(
#     os.environ["NEO4J_URI"],
#     auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
# )

# driver.verify_connectivity()

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
def health_check():
    return {"status": "ok"}

@app.get("/health")
def health() -> Dict[str, Any]:
    cfg = app.state.cfg
    return {
        "status": "ok",
        "neo4j": cfg.settings.neo4j_uri,
        "components": {k: v.model for k, v in cfg.components.items()},
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    """Visibility into cache hit-rates and pre-loaded index sizes."""
    return app.state.recommender.stats()


# ── Activity logging ──────────────────────────────────────────────────
#
# One JSONL file per (user, session) under graph_rag/logs/. Browser
# clients post batches; this endpoint appends each event on its own
# line so the files are tail-friendly and trivially parseable.

import re as _re   # local alias so we don't reshuffle the imports above

_LOGS_DIR = Path("/tmp/logs")
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _safe_slug(s: str, limit: int = 60) -> str:
    return _re.sub(r"[^A-Za-z0-9_-]+", "_", s or "")[:limit] or "anon"


@app.post("/log")
def log_events(req: LogRequest) -> Dict[str, Any]:
    """Append a batch of activity events to this session's JSONL log file.

    Each event is written as a single JSON object on its own line, with
    the session metadata duplicated alongside so the file is self-
    describing even if a downstream tool processes one event at a time.
    """
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    fname = f"{_safe_slug(req.user_id)}_{_safe_slug(req.session_id, 80)}.jsonl"
    path = _LOGS_DIR / fname
    # Open in append mode — multiple concurrent writers would interleave
    # whole lines safely on POSIX as long as each write is under PIPE_BUF.
    try:
        with path.open("a", encoding="utf-8") as f:
            for ev in req.events:
                f.write(json.dumps({
                    "timestamp":  ev.timestamp,
                    "session_id": req.session_id,
                    "user_id":    req.user_id,
                    "name":       req.name,
                    "event":      ev.event,
                    "payload":    ev.payload,
                }, ensure_ascii=False) + "\n")
    except OSError as e:
        LOG.exception("/log write failed for %s", path)
        raise HTTPException(status_code=500, detail=f"log write failed: {e}")
    return {"ok": True, "written": len(req.events), "file": fname}


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest) -> SuggestResponse:
    try:
        return app.state.recommender.suggest(req)
    except Exception as e:
        LOG.exception("/suggest failed")
        raise HTTPException(status_code=500, detail=str(e))


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
