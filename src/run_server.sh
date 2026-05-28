#!/usr/bin/env bash
# Convenience script to run the Graph-RAG FastAPI service.
# Usage:
#   ./run_server.sh                # uses defaults
#   PORT=8001 ./run_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "▶ Starting Graph-RAG on http://$HOST:$PORT"
echo "  • models.yaml: ${GRAPH_RAG_MODELS_YAML:-knowledge_graph_workflow/models.yaml}"
echo "  • Neo4j:       ${NEO4J_URI:-bolt://localhost:7687}"
echo
exec python -m uvicorn graph_rag.app:app --host "$HOST" --port "$PORT" --reload
