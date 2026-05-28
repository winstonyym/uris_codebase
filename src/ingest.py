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


LOG = logging.getLogger("graph_rag.ingest")

from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=False)

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


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
) -> List[Dict[str, Any]]:
    """Materialise node rows for upsert, only re-embedding what changed."""
    existing_hashes = existing_hashes or {}
    existing_embeddings = existing_embeddings or {}

    rows: List[Dict[str, Any]] = []
    to_embed_texts: List[str] = []
    to_embed_idx: List[int] = []

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

        if existing_hashes.get(n["id"]) == h and existing_embeddings.get(n["id"]):
            row["embedding"] = existing_embeddings[n["id"]]
        else:
            to_embed_idx.append(len(rows))
            to_embed_texts.append(text)
        rows.append(row)

    # Batch the OpenAI call — default batch size is fine for 361 nodes.
    if to_embed_texts:
        LOG.info("Embedding %d new/changed nodes …", len(to_embed_texts))
        # Batch in chunks of 96 to stay well under rate limits.
        BATCH = 96
        for start in range(0, len(to_embed_texts), BATCH):
            batch_texts = to_embed_texts[start : start + BATCH]
            batch_idx = to_embed_idx[start : start + BATCH]
            vecs = embedder.embed(batch_texts)
            for idx, vec in zip(batch_idx, vecs):
                rows[idx]["embedding"] = vec
    else:
        LOG.info("All node embeddings already up-to-date.")

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
) -> List[Dict[str, Any]]:
    """Return [{id, top_candidates_json}, …] ready to SET on each node.

    Uses the *just-built* in-memory data (rows + edges from the KG file)
    so we don't pay a Neo4j round-trip. Scales linearly in #edges and
    O(N²) in #nodes for the vector pass, which is fine up to ~50k.
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
    M = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    M = M / norms
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

    # ── For each node, rank candidates by 0.6·graph + 0.4·vector ────
    out: List[Dict[str, Any]] = []
    for kid in ids:
        i = idx_of[kid]
        sims = M @ M[i]                          # cosine to every other node
        # Build a candidate score dict
        score: Dict[str, Dict[str, Any]] = {}
        for other in neighbours.get(kid, {}).values():
            oid = other["id"]
            if oid not in idx_of:
                continue
            j = idx_of[oid]
            gw = other["weight"] / 5.0
            vs = float(sims[j])
            score[oid] = {
                "id": oid,
                "label": label_by_id.get(oid, oid),
                "category": category_by_id.get(oid),
                "graph_weight": gw,
                "vector_score": vs,
                "hops": 1,
                "direction": other["direction"],
                "sign": other["sign"],
                "weight": other["weight"],
            }
        # Also add purely-semantic high scorers that aren't graph neighbours
        # (cap candidates so the property doesn't bloat).
        top_idx = np.argpartition(-sims, min(top_k * 2, len(sims) - 1))[: top_k * 2]
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
        )[: top_k]
        out.append({"id": kid, "top": json.dumps(ranked)})
    return out


def _write_top_candidates(neo: Neo4jClient, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    q = (
        "UNWIND $rows AS r "
        "MATCH (n:KGNode {id: r.id}) "
        "SET n.top_candidates = r.top "
        "RETURN count(n) AS n"
    )
    with neo.session() as s:
        return s.run(q, rows=rows).single()["n"]


def run(cfg: Config, kg_path: Path) -> Dict[str, int]:
    if not kg_path.exists():
        raise FileNotFoundError(kg_path)
    LOG.info("Loading KG from %s", kg_path)
    kg = _load_kg(kg_path)

    embedder = make_embedder(cfg.components["embedder"])
    neo = Neo4jClient(
        uri=cfg.settings.neo4j_uri,
        user=cfg.settings.neo4j_user,
        password=cfg.settings.neo4j_password,
        database=cfg.settings.neo4j_database,
    )
    try:
        LOG.info("Ensuring schema (vector index dim=%d) …", embedder.dimensions)
        neo.ensure_schema(cfg.settings.vector_index_name, embedder.dimensions)

        existing_hashes, existing_embs = _fetch_existing(neo)
        rows = _build_rows(kg, embedder, existing_hashes, existing_embs)

        # Strip helper field before upsert
        for r in rows:
            r.pop("_emb_text_hash", None)

        # Re-attach the hash separately so we can store it on the node:
        # easier to do via a second SET pass than restructuring upsert.
        n_nodes = neo.upsert_nodes(rows)
        LOG.info("Upserted %d nodes", n_nodes)

        edge_rows = _build_edge_rows(kg)
        n_edges = neo.upsert_edges(edge_rows)
        LOG.info("Upserted %d edges", n_edges)

        # Write hashes back so future runs can skip unchanged embeddings.
        rehash_rows = []
        for n in kg["nodes"]:
            text = _embedding_text(n)
            rehash_rows.append({"id": n["id"], "h": _hash_text(text)})
        with neo.session() as s:
            s.run(
                "UNWIND $rows AS r MATCH (n:KGNode {id:r.id}) SET n._emb_text_hash = r.h",
                rows=rehash_rows,
            )

        # ── Top-K precompute (L4) ───────────────────────────────────────
        LOG.info("Pre-computing top-K candidates per KG node …")
        t0 = time.perf_counter()
        precomputed = _precompute_top_candidates(kg, rows, top_k=12)
        n_pre = _write_top_candidates(neo, precomputed)
        LOG.info(
            "Wrote top_candidates for %d nodes (%.0f ms)",
            n_pre, (time.perf_counter() - t0) * 1000,
        )

        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "embedded": sum(1 for r in rows if r["embedding"]),
            "precomputed": n_pre,
        }
    finally:
        neo.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-yaml", type=Path, default=None,
                        help="Path to models.yaml (default: ./models.yaml)")
    parser.add_argument("--kg", type=Path, default=None,
                        help="Path to merged_kg.json (default: ./merged_kg_path)")
    args = parser.parse_args()

    cfg = load_config(args.models_yaml)
    kg_path = args.kg or (
        (cfg.models_yaml_path.parent / cfg.settings.merged_kg_path).resolve()
    )
    stats = run(cfg, kg_path)
    print(json.dumps({"ingest": stats}, indent=2))


if __name__ == "__main__":
    main()
