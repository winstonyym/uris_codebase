#!/usr/bin/env python
"""Seed (or refresh) the R2-cached KG index artifact.

Run this ONCE locally after deploying or re-ingesting the knowledge graph. It
builds the in-memory index from Neo4j (no serverless time limit here) and
uploads the compressed artifact to R2, so production cold starts load that
~55 MB object instead of streaming the whole KG from Aura.

Why you need it on Hobby: the first serverless cold start with an empty cache
would do the slow Neo4j build itself, which can exceed Vercel's 60 s function
limit and get killed *before* it uploads the artifact — leaving every cold start
rebuilding. Seeding from your machine sidesteps that.

Usage:
    cd backend
    # ensure the same env the service uses (Neo4j + R2 + embedder creds):
    #   NEO4J_URI / NEO4J_PASSWORD, R2_ACCOUNT_ID / R2_ACCESS_KEY_ID /
    #   R2_SECRET_ACCESS_KEY / R2_BUCKET_NAME, OPENAI_API_KEY
    python scripts/build_index_cache.py

    # Force a rebuild even if an artifact already exists / to bump the version:
    KG_INDEX_VERSION=v2 python scripts/build_index_cache.py

After a re-ingest, bump KG_INDEX_VERSION (here AND in the Vercel env) so stale
artifacts are never served, then re-run this script.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make `src` importable whether run from backend/ or backend/scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=False)

from src.config import load_config            # noqa: E402
from src.embeddings import make_embedder      # noqa: E402
from src.llm_router import LLMRouter          # noqa: E402
from src.neo4j_client import Neo4jClient      # noqa: E402
from src.recommender import Recommender       # noqa: E402
from src import r2_uploader                   # noqa: E402


def main() -> int:
    # Always rebuild from Neo4j (ignore any existing cached artifact), then
    # the recommender uploads the fresh artifact as part of warm().
    os.environ["KG_INDEX_REBUILD"] = "1"

    cfg = load_config(os.environ.get("GRAPH_RAG_MODELS_YAML") or None)
    print(f"• models.yaml: {cfg.models_yaml_path}")
    print(f"• R2 enabled:  {r2_uploader.is_enabled()}  ({r2_uploader.status()})")
    if not r2_uploader.is_enabled():
        print("✗ R2 is not configured (need R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / "
              "R2_SECRET_ACCESS_KEY / R2_BUCKET_NAME). Aborting.")
        return 2

    neo = Neo4jClient(
        uri=cfg.settings.neo4j_uri, user=cfg.settings.neo4j_user,
        password=cfg.settings.neo4j_password, database=cfg.settings.neo4j_database,
    )
    embedder = make_embedder(cfg.components["embedder"])
    router = LLMRouter(cfg)
    rec = Recommender(cfg, neo, embedder, router)

    # KG_INDEX_REBUILD only forces _try_load_index_from_r2 to return False (so
    # we skip the cached artifact and rebuild from Neo4j). The subsequent
    # _save_index_to_r2 upload always runs, so the fresh artifact is published.
    print("• Building index from Neo4j (this is the slow part)…")
    t0 = time.perf_counter()
    rec.warm()                       # builds + uploads to R2
    dt = time.perf_counter() - t0

    stats = rec.stats()
    print(f"✓ Done in {dt:.1f}s — {stats['kg_nodes']} nodes indexed.")
    print(f"  Artifact key: {rec._index_artifact_key()}")
    print("  Production cold starts will now load this from R2.")
    neo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
