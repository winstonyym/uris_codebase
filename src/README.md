# graph_rag — recommendation, causal Q&A, and chat for the CLD editor

A FastAPI service that turns the Neo4j-backed knowledge graph (built
by `knowledge_graph_workflow/build_kg.py`) into three capabilities the
React frontend consumes:

| Endpoint                 | Purpose                                                                |
| ------------------------ | ---------------------------------------------------------------------- |
| `POST /suggest`          | Suggest new nodes (and edges) to add near a query node on the canvas. |
| `POST /suggest/stream`   | Same as `/suggest` over SSE — used by the live ghost-suggestion rail. |
| `POST /causal-query`     | "If X increases, what happens to Y?" — signed-path + LLM narration.    |
| `POST /chat`             | Tool-calling chat agent that stages diff-style mutations.              |
| `POST /chat/stream`      | SSE variant of `/chat`.                                                 |
| `GET  /health`           | Quick smoke-check.                                                      |

## Architecture

```
React  ──HTTP/SSE──▶  FastAPI (graph_rag.app)
                         │
                         ├── recommender.py  ◀──▶  Neo4j vector index + Cypher
                         ├── causal.py       (signed-path + loop detection)
                         ├── chat_agent.py   (Anthropic tool-use)
                         │
                         └── llm_router.py   ──▶  utils/llm_client.py (Anthropic / OpenAI)
                                              ──▶  embeddings.py     (OpenAI text-embedding-3-small)
```

Every LLM/embedding choice flows through `models.yaml`'s `components:`
section, so you can swap, say, `chat_agent` to Sonnet 4.6 by editing
one line — nothing else needs to change.

## Setup

```bash
# 1. Python deps
pip install -r graph_rag/requirements.txt

# 2. Start Neo4j (one-time)
cd graph_db && docker compose up -d && cd ..

# 3. Ingest the knowledge graph (embeds nodes, populates Neo4j)
export OPENAI_API_KEY=sk-...
python -m graph_rag.ingest

# 4. Run the service
export ANTHROPIC_API_KEY=sk-ant-...
./graph_rag/run_server.sh        # or: uvicorn graph_rag.app:app --reload --port 8000
```

Re-running `python -m graph_rag.ingest` after regenerating
`merged_kg.json` is safe: nodes whose embedding text hasn't changed
keep their existing vector, so the call is essentially free in steady
state.

## Configuring models per component

In `knowledge_graph_workflow/models.yaml`:

```yaml
components:
  embedder:
    provider: openai
    model: text-embedding-3-small
  recommender_reranker:
    model: claude-haiku-4.5     # cheap, fast
  chat_agent:
    model: claude-sonnet-4.6    # ← bump this for higher-quality chat
  causal_narrator:
    model: claude-haiku-4.5
```

Models listed under `models:` in the same file are the catalogue; any
component can reference one of those names.

## Frontend wiring

Set `VITE_GRAPH_RAG_URL=http://localhost:8000` in `.env.local`
(defaults to that value). The React side calls the API via
`src/lib/graphRag.js`.
