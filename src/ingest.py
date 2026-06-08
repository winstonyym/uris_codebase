"""Load merged_kg.json into Neo4j and (re)compute node embeddings.

Idempotent — re-running after merged_kg.json is regenerated keeps the
graph in sync. Nodes whose label+description hash is unchanged keep
their existing embedding to avoid extra OpenAI calls.

Usage:
    cd graph_rag
    python -m graph_rag.ingest                    # uses ../knowledge_graph_workflow/models.yaml
    python -m graph_rag.ingest --kg path/to.json  # explicit path

Environment:
    OPENAI_API_KEY                                # required by default embedder
    NEO4J_URI / NEO4J_PASSWORD                    # optional overrides
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# When run as `python -m graph_rag.ingest` Python adds the parent dir to
# sys.path automatically. When run as a plain script we add it manually.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.embeddings import make_embedder
from src.neo4j_client import Neo4jClient
from src.cache import KVCache


LOG = logging.getLogger("graph_rag.ingest")

from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=False)

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _emb_cache_key(text: str, model: str, dim: int) -> str:
    """Cache key for a node embedding. Includes the model id and output
    dimensionality so changing either (e.g. 1536 → 512) never reuses a
    stale vector of the wrong shape."""
    return _hash_text(f"{model}|{dim}|{text}")


def _embedding_text(node: Dict[str, Any]) -> str:
    """Build the string we embed for a KG node — label + aliases +
    category prefix. Description is rarely populated in merged_kg.json
    today but we include it if present."""
    parts = [node.get("label", "")]
    aliases = node.get("aliases") or []
    for a in aliases:
        if a and a.lower() != node.get("label", "").lower():
            parts.append(a)
    cat = node.get("category")
    if cat:
        parts.append(f"[{cat}]")
    desc = node.get("description")
    if desc:
        parts.append(desc)
    return " | ".join(p for p in parts if p)


def _load_kg(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _build_rows(
    kg: Dict[str, Any],
    embedder,
    existing_hashes: Dict[str, str] | None = None,
    existing_embeddings: Dict[str, List[float]] | None = None,
    emb_cache: "KVCache | None" = None,
    model: str = "",
    dim: int = 0,
) -> List[Dict[str, Any]]:
    """Materialise node rows for upsert, re-embedding only what we must.

    Embeddings are sourced in priority order:
      1. Unchanged node already in Neo4j (skip when the graph was cleared).
      2. Redis / Upstash cache (KVCache), keyed by model+dim+text.
      3. OpenAI API — and the result is written back to the cache so future
         runs (even after the graph is cleared) reuse it instead of paying
         for another embedding call.
    """
    existing_hashes = existing_hashes or {}
    existing_embeddings = existing_embeddings or {}

    rows: List[Dict[str, Any]] = []
    to_embed_texts: List[str] = []
    to_embed_idx: List[int] = []
    n_cache_hits = 0

    for n in kg["nodes"]:
        text = _embedding_text(n)
        h = _hash_text(text)
        prov_json = json.dumps(n.get("provenance") or [])

        row = {
            "id": n["id"],
            "label": n["label"],
            "category": n.get("category", "other"),
            "aliases": n.get("aliases") or [],
            "description": n.get("description") or "",
            "provenance_json": prov_json,
            "_emb_text_hash": h,
            "embedding": None,  # filled below
        }

        # 1) Reuse an unchanged node's embedding straight from Neo4j.
        if existing_hashes.get(n["id"]) == h and existing_embeddings.get(n["id"]):
            row["embedding"] = existing_embeddings[n["id"]]
        else:
            # 2) Reuse from the Redis/Upstash cache (right dimensionality only).
            cached = emb_cache.get(_emb_cache_key(text, model, dim)) if emb_cache else None
            if cached and (not dim or len(cached) == dim):
                row["embedding"] = cached
                n_cache_hits += 1
            else:
                # 3) Embed via the API below.
                to_embed_idx.append(len(rows))
                to_embed_texts.append(text)
        rows.append(row)

    if emb_cache is not None:
        LOG.info("Embedding cache hits: %d / %d nodes", n_cache_hits, len(kg["nodes"]))

    # Batch the OpenAI call. Chunks of 96 keep us well under rate limits.
    if to_embed_texts:
        LOG.info("Embedding %d node(s) via API …", len(to_embed_texts))
        BATCH = 96
        for start in range(0, len(to_embed_texts), BATCH):
            batch_texts = to_embed_texts[start : start + BATCH]
            batch_idx = to_embed_idx[start : start + BATCH]
            vecs = embedder.embed(batch_texts)
            for txt, idx, vec in zip(batch_texts, batch_idx, vecs):
                rows[idx]["embedding"] = vec
                # Persist to Redis/Upstash so the next run reuses it.
                if emb_cache is not None:
                    emb_cache.set(_emb_cache_key(txt, model, dim), vec)
            LOG.info("  … embedded %d / %d", min(start + BATCH, len(to_embed_texts)), len(to_embed_texts))
    else:
        LOG.info("No new embeddings needed — all sourced from Neo4j/cache.")

    return rows


def _fetch_existing(neo: Neo4jClient) -> tuple[Dict[str, str], Dict[str, List[float]]]:
    """Pull hashes + embeddings for nodes already in Neo4j so we can skip
    redundant embedding calls on re-ingest."""
    q = "MATCH (n:KGNode) RETURN n.id AS id, n._emb_text_hash AS h, n.embedding AS e"
    hashes: Dict[str, str] = {}
    embs: Dict[str, List[float]] = {}
    with neo.session() as s:
        for rec in s.run(q):
            if rec["h"]:
                hashes[rec["id"]] = rec["h"]
            if rec["e"]:
                embs[rec["id"]] = rec["e"]
    return hashes, embs


def _build_edge_rows(kg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for e in kg["edges"]:
        rows.append({
            "source_id": e["source_id"],
            "target_id": e["target_id"],
            "sign": e["sign"],
            "weight": int(e.get("weight", 1)),
            "provenance_json": json.dumps(e.get("provenance") or []),
        })
    return rows


# ── Top-K precompute (L4) ──────────────────────────────────────────────
#
# For each KG node, compute its top-K candidate neighbours offline by
# combining the same vector + graph signal the recommender uses at query
# time. We deliberately DO NOT run an LLM rerank here — that would slow
# ingest dramatically and the deterministic ranking is what the
# recommender falls back to anyway when scores are confident.
#
# The output is stored as a JSON blob on each node (`top_candidates`),
# which graph_rag/recommender.py picks up automatically.


def _precompute_top_candidates(
    kg: Dict[str, Any],
    rows: List[Dict[str, Any]],
    top_k: int = 12,
    chunk: int = 512,
) -> List[Dict[str, Any]]:
    """Return [{id, top}, …] ready to SET on each node.

    Uses the *just-built* in-memory data (rows + edges from the KG file)
    so we don't pay a Neo4j round-trip. The O(N²) vector pass is done as a
    chunked block matrix-multiply (B×N at a time) so it stays memory-bounded
    and BLAS-fast even at tens of thousands of nodes — a full N×N matrix is
    never materialised.
    """
    import numpy as np

    # ── build the normalised embedding matrix ────────────────────────
    ids: List[str] = []
    vecs: List[List[float]] = []
    label_by_id: Dict[str, str] = {}
    category_by_id: Dict[str, str] = {}
    for r in rows:
        if not r.get("embedding"):
            continue
        ids.append(r["id"])
        vecs.append(r["embedding"])
        label_by_id[r["id"]] = r.get("label", r["id"])
        category_by_id[r["id"]] = r.get("category", "other")
    if not ids:
        return []
    N = len(ids)
    M = np.asarray(vecs, dtype=np.float32)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    idx_of = {kid: i for i, kid in enumerate(ids)}

    # ── 1-hop neighbour map with sign/weight ─────────────────────────
    neighbours: Dict[str, Dict[str, Dict[str, Any]]] = {kid: {} for kid in ids}
    for e in kg["edges"]:
        a, b = e["source_id"], e["target_id"]
        sign = e.get("sign", "+")
        w = int(e.get("weight", 1))
        if a in neighbours:
            cur = neighbours[a].get(b)
            if (cur is None) or (w > cur["weight"]):
                neighbours[a][b] = {"id": b, "sign": sign, "weight": w, "direction": "outgoing"}
        if b in neighbours:
            cur = neighbours[b].get(a)
            if (cur is None) or (w > cur["weight"]):
                neighbours[b][a] = {"id": a, "sign": sign, "weight": w, "direction": "incoming"}

    n_pool = min(top_k * 2 + 1, N)  # +1 so we can drop self before trimming

    # ── For each node, rank candidates by 0.6·graph + 0.4·vector ────
    out: List[Dict[str, Any]] = []
    for start in range(0, N, chunk):
        block = M[start : start + chunk]            # B×dim
        sims_block = block @ M.T                     # B×N cosine sims
        for bi in range(block.shape[0]):
            kid = ids[start + bi]
            sims = sims_block[bi]
            score: Dict[str, Dict[str, Any]] = {}

            # graph neighbours (always included)
            for other in neighbours.get(kid, {}).values():
                oid = other["id"]
                j = idx_of.get(oid)
                if j is None:
                    continue
                score[oid] = {
                    "id": oid,
                    "label": label_by_id.get(oid, oid),
                    "category": category_by_id.get(oid),
                    "graph_weight": other["weight"] / 5.0,
                    "vector_score": float(sims[j]),
                    "hops": 1,
                    "direction": other["direction"],
                    "sign": other["sign"],
                    "weight": other["weight"],
                }

            # purely-semantic high scorers (cap so the property doesn't bloat)
            top_idx = np.argpartition(-sims, n_pool - 1)[:n_pool]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            for j in top_idx:
                oid = ids[int(j)]
                if oid == kid or oid in score:
                    continue
                score[oid] = {
                    "id": oid,
                    "label": label_by_id.get(oid, oid),
                    "category": category_by_id.get(oid),
                    "graph_weight": 0.0,
                    "vector_score": float(sims[int(j)]),
                    "hops": 99,
                    "direction": None,
                    "sign": None,
                    "weight": 0,
                }

            ranked = sorted(
                score.values(),
                key=lambda c: 0.6 * c["graph_weight"] + 0.4 * c["vector_score"],
                reverse=True,
            )[:top_k]
            out.append({"id": kid, "top": json.dumps(ranked)})
        LOG.info("  … top-K %d / %d nodes", min(start + chunk, N), N)
    return out


def _write_top_candidates(neo: Neo4jClient, rows: List[Dict[str, Any]], batch: int = 2000) -> int:
    if not rows:
        return 0
    q = (
        "UNWIND $rows AS r "
        "MATCH (n:KGNode {id: r.id}) "
        "SET n.top_candidates = r.top "
        "RETURN count(n) AS n"
    )
    total = 0
    with neo.session() as s:
        for start in range(0, len(rows), batch):
            total += s.run(q, rows=rows[start : start + batch]).single()["n"]
    return total


def run(cfg: Config, kg_path: Path, clear: bool = True, precompute: bool = True) -> Dict[str, Any]:
    if not kg_path.exists():
        raise FileNotFoundError(kg_path)
    LOG.info("Loading KG from %s", kg_path)
    kg = _load_kg(kg_path)
    LOG.info("KG has %d nodes, %d edges", len(kg.get("nodes", [])), len(kg.get("edges", [])))

    emb_spec = cfg.components["embedder"]
    embedder = make_embedder(emb_spec)
    dim = embedder.dimensions

    # Embedding cache — persisted in Redis/Upstash when cache.redis_url (or
    # REDIS_URL) is set; otherwise an in-process LRU. Lets re-ingests reuse
    # embeddings without re-calling OpenAI, even after the graph is cleared.
    cc = cfg.settings.cache
    emb_cache = KVCache(
        redis_url=cc.redis_url, namespace="emb",
        ttl=cc.embedding_ttl_sec, lru_size=cc.lru_size,
    )

    neo = Neo4jClient(
        uri=cfg.settings.neo4j_uri,
        user=cfg.settings.neo4j_user,
        password=cfg.settings.neo4j_password,
        database=cfg.settings.neo4j_database,
    )
    try:
        if clear:
            # Drop the vector index first so a dimensionality change (e.g.
            # 1536 → 512) takes effect, then wipe all nodes + relationships.
            LOG.info("Clearing graph + dropping vector index %r …", cfg.settings.vector_index_name)
            neo.drop_vector_index(cfg.settings.vector_index_name)
            removed = neo.clear_graph()
            LOG.info("Removed %d existing nodes (and their edges)", removed)

        LOG.info("Ensuring schema (vector index dim=%d) …", dim)
        neo.ensure_schema(cfg.settings.vector_index_name, dim)

        # After a clear there's nothing to reuse from Neo4j — rely on the cache.
        if clear:
            existing_hashes, existing_embs = {}, {}
        else:
            existing_hashes, existing_embs = _fetch_existing(neo)

        rows = _build_rows(
            kg, embedder, existing_hashes, existing_embs,
            emb_cache=emb_cache, model=emb_spec.model, dim=dim,
        )

        # Strip helper field before upsert
        for r in rows:
            r.pop("_emb_text_hash", None)

        n_nodes = neo.upsert_nodes(rows)
        LOG.info("Upserted %d nodes", n_nodes)

        edge_rows = _build_edge_rows(kg)
        n_edges = neo.upsert_edges(edge_rows)
        LOG.info("Upserted %d edges", n_edges)

        # Write hashes back so future (non-clearing) runs can skip unchanged
        # embeddings.
        rehash_rows = [
            {"id": n["id"], "h": _hash_text(_embedding_text(n))} for n in kg["nodes"]
        ]
        with neo.session() as s:
            for start in range(0, len(rehash_rows), 5000):
                s.run(
                    "UNWIND $rows AS r MATCH (n:KGNode {id:r.id}) SET n._emb_text_hash = r.h",
                    rows=rehash_rows[start : start + 5000],
                )

        # ── Top-K precompute (L4) ───────────────────────────────────────
        n_pre = 0
        if precompute:
            LOG.info("Pre-computing top-K candidates per KG node …")
            t0 = time.perf_counter()
            precomputed = _precompute_top_candidates(kg, rows, top_k=12)
            n_pre = _write_top_candidates(neo, precomputed)
            LOG.info(
                "Wrote top_candidates for %d nodes (%.1f s)",
                n_pre, time.perf_counter() - t0,
            )
        else:
            LOG.info("Skipping top-K precompute (--no-precompute).")

        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "embedded": sum(1 for r in rows if r["embedding"]),
            "embedding_dim": dim,
            "precomputed": n_pre,
            "cleared": bool(clear),
            "embedding_cache": emb_cache.stats(),
        }
    finally:
        neo.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser(description="Clear Neo4j and ingest merged_kg.json.")
    parser.add_argument("--models-yaml", type=Path, default=None,
                        help="Path to models.yaml (default: ./models.yaml)")
    parser.add_argument("--kg", type=Path, default=None,
                        help="Path to merged_kg.json (default: from models.yaml)")
    parser.add_argument("--no-clear", dest="clear", action="store_false",
                        help="Do NOT wipe the graph first (default: wipe nodes + edges).")
    parser.add_argument("--no-precompute", dest="precompute", action="store_false",
                        help="Skip the per-node top-K candidate precompute.")
    parser.set_defaults(clear=True, precompute=True)
    args = parser.parse_args()

    cfg = load_config(args.models_yaml)
    kg_path = args.kg or (
        (cfg.models_yaml_path.parent / cfg.settings.merged_kg_path).resolve()
    )
    stats = run(cfg, kg_path, clear=args.clear, precompute=args.precompute)
    print(json.dumps({"ingest": stats}, indent=2))


if __name__ == "__main__":
    main()
