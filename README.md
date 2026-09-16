# URIS Mapping Platform

# Overview

We present URIS, an open science platform for building, exploring, and sharing causal loop diagrams (CLDs) grounded in a literature-derived knowledge graph. The knowledge graph contains 44,413 variables and 48,912 signed causal links (35,536 positive, 12,930 negative, 446 unspecified) extracted from 3,534 research papers, with each node and edge traceable to its source publications. This repository contains the Graph-RAG backend service that powers the platform. It combines vector search over the knowledge graph with large language models to (i) suggest related variables and links as users build a diagram, (ii) detect and recommend reinforcing and balancing feedback loops, (iii) answer "if X increases, what happens to Y?" questions using deterministic signed-path tracing, and (iv) save diagrams and publish them to a public gallery. The repository includes scripts to ingest the knowledge graph, run the service locally, and reproduce the deployed configuration.

# System Requirements

## Hardware Requirements

To run the service, users require only a standard computer with enough RAM to hold the in-memory knowledge graph index (~44k embeddings of 512 dimensions, ~90 MB) and the bundled `merged_kg.json` (~48 MB). Having >4GB RAM would be preferable.

The platform was implemented on a MacBook Pro, <CHIP> with <RAM> RAM and <STORAGE> storage.

## Software Requirements

### OS Requirements

The package development version is tested on MacOS operating systems. The implementation would work on the following systems:

Linux: Ubuntu 22.04  
Mac OSX:  
Windows:  

Before running the service, users should have Python version 3.10 or higher (development used Python 3.12, see `.python-version`).

### External Services

| Service | Required | Purpose |
| --- | --- | --- |
| Neo4j (local Docker or Aura) | Yes | Stores the knowledge graph and its vector index |
| OpenAI API key | Yes | Node and query embeddings (`text-embedding-3-small`, 512-d) |
| LLM API key(s) | Yes | Models assigned in `models.yaml` (DeepSeek and Anthropic by default) |
| Redis | Optional | Persistent response and embedding cache (falls back to in-process LRU) |
| Cloudflare R2 | Optional | Cached index artifact, saved diagrams, and the public gallery |
| Clerk | Optional | User authentication (can be disabled for local use) |

# Installation Guide & Dependencies

1. Clone the code repository either with CLI or by accessing the code base
2. Navigate to the local folder location
3. Open up a terminal/command prompt and install the dependencies

```
$ git clone https://github.com/winstonyym/uris_codebase.git
$ cd ./uris_codebase
$ uv sync
```

or, without `uv`:

```
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your credentials

```
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=<password>
OPENAI_API_KEY=<key>
ANTHROPIC_API_KEY=<key>
DEEPSEEK_API_KEY=<key>
CLERK_AUTH_DISABLED=1        # local development only
```

5. Installation completed

(Optional) Model choices for each component (embedder, reranker, chat agent, causal narrator, loop describer, loop recommender) are set in the `components:` section of `models.yaml`. Changing one line swaps the model for that component.

# Repository Structure
- `merged_kg.json` (Knowledge graph of nodes, signed edges, and paper provenance [^1])
- `models.yaml` (Model catalogue, per-component model assignment, and retrieval settings)
- `src` (FastAPI service)
  - `app.py` (API entrypoint and routes)
  - `ingest.py` (Loads `merged_kg.json` into Neo4j and computes node embeddings)
  - `recommender.py` (Vector search, graph expansion, and LLM reranking for node suggestions)
  - `causal.py` (Signed-path enumeration and feedback loop detection)
  - `loop_recommender.py` (Proposes missing feedback loops on the user's diagram)
  - `chat_agent.py` / `loop_chat.py` (Diagram editing assistant and read-only loop Q&A assistant)
  - `diagrams.py` (Private saved diagrams and public gallery)
  - `auth.py` / `usage.py` / `cache.py` (Authentication, free-tier quotas, and caching)
- `scripts` (Utility scripts, e.g. seeding the cached index artifact)
- `test_diagrams.py` / `test_quota.py` (Offline tests, no external services needed)
- `COLD_START.md` (Deployment notes for serverless hosting)
- `.github/workflows` (Scheduled keep-warm job for the deployed backend)

# Quickstart

## To build the knowledge graph database
1) Start a Neo4j instance and set `NEO4J_URI` / `NEO4J_PASSWORD`
2) run `python -m src.ingest` in command line (use `--kg <path>` for a different graph file)

Re-running ingest is safe: nodes whose text has not changed keep their existing embedding.

## To run the service
1) run `uvicorn src.app:app --reload --port 8000` in command line
2) check the service at `http://localhost:8000/health`

| Endpoint | Purpose |
| --- | --- |
| `POST /suggest`, `POST /suggest/stream` | Suggest nodes and edges near a selected variable |
| `POST /loop-recommend` | Recommend a missing feedback loop for the current diagram |
| `POST /loop-describe` | Describe a feedback loop in plain language |
| `POST /causal-query` | Trace the effect of one variable on another |
| `POST /chat/stream` | Diagram editing assistant (proposes changes for the user to accept) |
| `POST /loop-chat/stream` | Q&A about the loops currently shown |
| `GET/POST /diagrams`, `GET /gallery` | Save, publish, and browse diagrams |
| `GET /health`, `GET /stats`, `POST /warm` | Health check, graph statistics, and warm-up |

## To run the tests
run `python test_diagrams.py` and `python test_quota.py` in command line


<br>
