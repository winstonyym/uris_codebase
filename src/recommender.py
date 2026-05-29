"""Hybrid recommender — fast path.

Latency goals: <100 ms warm (cached), <150 ms cold (uncached, no LLM),
<1.5 s cold-with-LLM-rerank — but rerank is now optional and runs in the
background via /suggest/stream, so the user-facing latency is the first
two bullets.

The big architectural shift vs. the first version:

  1.  All KG node embeddings live in a single in-process numpy matrix,
      built once at startup. Cosine similarity is a BLAS call, sub-ms
      for 100k vectors. No Neo4j round-trip per query.

  2.  All 1- and 2-hop neighbour info lives in a dict[id → list[…]]
      assembled once at startup. Same idea: no per-query Cypher.

  3.  The N+1 Cypher pattern for "edges from candidate to canvas
      nodes" is collapsed into one batched Cypher per request.

  4.  Three tiers of caching, all optional:
        • Embedding LRU (text → vector). Hit-rate ≈100% within a session.
        • Pre-canvas response LRU (query+ranking, no canvas filter).
          Keyed by query hash. Survives canvas edits.
        • Semantic cache (vec → pre-canvas response). When a fresh query
          embeds within ~0.97 cosine of a cached one, we reuse its
          pre-canvas ranking. Pays one embedding API call only.

  5.  LLM rerank is gated by *score margin*, not just absolute top
      score. If the top candidates differ by enough that reranking
      can't reasonably flip the order, we skip the LLM entirely.

  6.  A `stages()` generator drives the SSE endpoint — it yields a
      first "vector" partial within ~50 ms, then optionally a "rerank"
      patch when the LLM finishes.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .cache import KVCache, SemanticCache, canvas_signature, hash_text
from .config import Config
from .embeddings import EmbedderProtocol
from .llm_router import LLMRouter
from .neo4j_client import Neo4jClient
from .schemas import (
    CanvasNode,
    CanvasState,
    ProvenanceSnippet,
    SuggestRequest,
    SuggestResponse,
    SuggestedCanvasEdge,
    SuggestedEdgeFromNew,
    SuggestedNode,
)


LOG = logging.getLogger("graph_rag.recommender")


# ── Tunables not worth exposing in yaml yet ────────────────────────────

_MATCH_CONFIDENCE_FLOOR = 0.62


# ── helpers ────────────────────────────────────────────────────────────

def _canvas_node_text(n: CanvasNode) -> str:
    parts = [n.label or ""]
    if n.category:
        parts.append(f"[{n.category}]")
    if n.description:
        parts.append(n.description)
    return " | ".join(p for p in parts if p)


def _normalize_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _coerce_provenance(raw: Any) -> List[ProvenanceSnippet]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    out: List[ProvenanceSnippet] = []
    for p in raw[:5]:
        out.append(ProvenanceSnippet(
            paper_id=p.get("paper_id", ""),
            page=p.get("page"),
            figure_id=p.get("figure_id"),
            source=p.get("source"),
            quote=p.get("source_quote"),
        ))
    return out


# ── core ───────────────────────────────────────────────────────────────

@dataclass
class _Candidate:
    """Scoring container — `as_suggestion` converts to the wire schema."""
    id: str
    label: str
    category: Optional[str]
    vector_score: float = 0.0
    graph_weight: float = 0.0
    hops: int = 99
    rel_chain: List[Dict[str, Any]] = field(default_factory=list)
    raw_score: float = 0.0
    alpha: float = 0.0
    rationale: Optional[str] = None


# Item shape stored in the *pre-canvas* response cache. We keep just
# what the canvas filter step needs so payloads are tiny.
def _candidate_to_cacheable(c: _Candidate) -> Dict[str, Any]:
    return {
        "id": c.id, "label": c.label, "category": c.category,
        "vector_score": c.vector_score, "graph_weight": c.graph_weight,
        "hops": c.hops, "rel_chain": c.rel_chain,
        "raw_score": c.raw_score, "alpha": c.alpha, "rationale": c.rationale,
    }

def _candidate_from_cacheable(d: Dict[str, Any]) -> _Candidate:
    return _Candidate(
        id=d["id"], label=d["label"], category=d.get("category"),
        vector_score=float(d.get("vector_score", 0.0)),
        graph_weight=float(d.get("graph_weight", 0.0)),
        hops=int(d.get("hops", 99)),
        rel_chain=list(d.get("rel_chain") or []),
        raw_score=float(d.get("raw_score", 0.0)),
        alpha=float(d.get("alpha", 0.0)),
        rationale=d.get("rationale"),
    )


class Recommender:
    """Fast, cached recommender. Build one per service instance and reuse."""

    def __init__(self, cfg: Config, neo: Neo4jClient, embedder: EmbedderProtocol, router: LLMRouter):
        self.cfg = cfg
        self.neo = neo
        self.embedder = embedder
        self.router = router

        # Pre-loaded indices (filled by _ensure_indices, lazy on first use).
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, embedder.dimensions), dtype=np.float32)
        self._labels: Dict[str, str] = {}
        self._categories: Dict[str, str] = {}
        self._aliases_norm: Dict[str, List[str]] = {}    # normalised aliases per id
        self._alias_index: Dict[str, str] = {}            # normalised label/alias → kg_id
        self._neighbours: Dict[str, List[Dict[str, Any]]] = {}   # id → 1-hop entries
        self._precomputed_top: Dict[str, List[Dict[str, Any]]] = {}
        self._node_provenance: Dict[str, Any] = {}        # id → raw provenance list
        self._loaded = False
        self._load_lock = threading.Lock()

        # Caches
        c = cfg.settings.cache
        self._emb_cache = KVCache(
            redis_url=c.redis_url, namespace="emb",
            ttl=c.embedding_ttl_sec, lru_size=c.lru_size,
        )
        self._pre_cache = KVCache(
            redis_url=c.redis_url, namespace="pre",
            ttl=c.response_ttl_sec, lru_size=c.lru_size,
        )
        self._sem_cache = SemanticCache(
            dim=embedder.dimensions,
            threshold=c.semantic_threshold,
            maxsize=c.semantic_max_entries,
        )

    # ── index loading ──────────────────────────────────────────────────

    def _ensure_indices(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            self._build_indices()
            self._loaded = True
            LOG.info(
                "Indices loaded: %d nodes, %d neighbour groups, %d precomputed (%.0f ms)",
                len(self._ids), len(self._neighbours), len(self._precomputed_top),
                (time.perf_counter() - t0) * 1000,
            )

    def _build_indices(self) -> None:
        q_nodes = """
        MATCH (n:KGNode)
        RETURN n.id          AS id,
               n.label       AS label,
               n.category    AS category,
               n.aliases     AS aliases,
               n.embedding   AS embedding,
               n.provenance  AS provenance,
               n.top_candidates AS top_candidates
        """
        ids: List[str] = []
        vecs: List[List[float]] = []
        with self.neo.session() as s:
            for r in s.run(q_nodes):
                if not r["embedding"]:
                    continue
                kid = r["id"]
                ids.append(kid)
                vecs.append(r["embedding"])
                self._labels[kid] = r["label"]
                self._categories[kid] = r["category"]
                aliases = list(r["aliases"] or [])
                norm_aliases = [_normalize_label(a) for a in aliases if a]
                norm_label = _normalize_label(r["label"])
                self._aliases_norm[kid] = sorted({norm_label, *norm_aliases})
                for a in self._aliases_norm[kid]:
                    if a and a not in self._alias_index:
                        self._alias_index[a] = kid
                # Node-level provenance — stored as a JSON string by ingest.
                # Parsed once here so every suggested node can carry its
                # source papers/figures.
                if r["provenance"]:
                    try:
                        self._node_provenance[kid] = json.loads(r["provenance"])
                    except Exception:
                        self._node_provenance[kid] = []
                if r["top_candidates"]:
                    try:
                        self._precomputed_top[kid] = json.loads(r["top_candidates"])
                    except Exception:
                        pass

        self._ids = ids
        self._id_to_idx = {kid: i for i, kid in enumerate(ids)}
        if vecs:
            # Defensive matrix build:
            #   1. Coerce to float32 (driver may hand us floats wrapped in
            #      Python ints / objects).
            #   2. Strip any NaN/Inf from the raw vectors — a single bad row
            #      poisons every matmul score with NaN, which in turn makes
            #      rerank-skip always fail (everything looks ambiguous) and
            #      the LLM rerank runs on every request, defeating the
            #      whole fast-path optimisation.
            #   3. Replace zero-norm rows with the unit identity so the
            #      division below never produces inf / NaN.
            m = np.asarray(vecs, dtype=np.float32)
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            zero_rows = (norms.ravel() == 0)
            if zero_rows.any():
                LOG.warning(
                    "Embedding matrix had %d zero-norm row(s); they will be inert in similarity search.",
                    int(zero_rows.sum()),
                )
            norms = np.where(norms == 0, 1.0, norms)
            self._matrix = (m / norms).astype(np.float32)
            # Final paranoia pass — should be a no-op given the above.
            self._matrix = np.nan_to_num(
                self._matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Hard assert at startup that the matrix is finite. If this ever
            # trips, there's a real bug somewhere (corrupt embeddings stored
            # in Neo4j, dtype mismatch, etc.). We log diagnostics instead of
            # raising so the service still boots.
            if not np.isfinite(self._matrix).all():
                bad = (~np.isfinite(self._matrix)).any(axis=1).sum()
                LOG.error(
                    "Embedding matrix STILL has %d non-finite row(s) after "
                    "cleanup — replacing with zeros. Investigate the stored "
                    "embeddings in Neo4j.", int(bad),
                )
                self._matrix = np.where(
                    np.isfinite(self._matrix), self._matrix, 0.0,
                ).astype(np.float32)
            else:
                LOG.info(
                    "Embedding matrix verified finite: shape=%s, dtype=%s",
                    self._matrix.shape, self._matrix.dtype,
                )
        else:
            self._matrix = np.zeros((0, self.embedder.dimensions), dtype=np.float32)

        # 1-hop edges. We synthesise both directions for fast lookup, but
        # remember the actual edge direction in each entry so callers can
        # still tell "incoming" from "outgoing".
        q_edges = """
        MATCH (a:KGNode)-[r:CAUSES]->(b:KGNode)
        RETURN a.id AS a, b.id AS b,
               r.sign AS sign, r.weight AS weight, r.provenance AS prov
        """
        with self.neo.session() as s:
            for rec in s.run(q_edges):
                a, b = rec["a"], rec["b"]
                entry_out = {
                    "id": b, "direction": "outgoing",
                    "sign": rec["sign"], "weight": int(rec["weight"]),
                    "prov": rec["prov"],
                }
                entry_in  = {
                    "id": a, "direction": "incoming",
                    "sign": rec["sign"], "weight": int(rec["weight"]),
                    "prov": rec["prov"],
                }
                self._neighbours.setdefault(a, []).append(entry_out)
                self._neighbours.setdefault(b, []).append(entry_in)

    def warm(self) -> None:
        """Call from FastAPI lifespan to load indices eagerly at startup."""
        self._ensure_indices()

    def stats(self) -> Dict[str, Any]:
        self._ensure_indices()
        return {
            "kg_nodes":      len(self._ids),
            "neighbour_set": len(self._neighbours),
            "precomputed":   len(self._precomputed_top),
            "caches": {
                "embedding": self._emb_cache.stats(),
                "pre_canvas": self._pre_cache.stats(),
                "semantic":  self._sem_cache.stats(),
            },
        }

    # ── public entrypoints ─────────────────────────────────────────────

    def suggest(self, req: SuggestRequest) -> SuggestResponse:
        """Synchronous one-shot. Skips LLM rerank by default; the LLM
        only runs when the deterministic scores are too close to call.
        """
        self._ensure_indices()
        t0 = time.perf_counter()
        anchor_id, match_conf, candidates, emb = self._get_pre_canvas_candidates(req.query_node)
        # Decide whether to run the LLM synchronously. Default policy:
        # only when scores are ambiguous.
        run_llm = self._should_run_rerank(candidates)
        if run_llm and req.include_new_nodes:
            try:
                self._llm_rerank(req.query_node, candidates[: req.max_suggestions * 2])
            except Exception as e:
                LOG.warning("LLM rerank failed (%s); using deterministic scores", e)
        resp = self._project_response(
            req=req,
            anchor_id=anchor_id,
            match_conf=match_conf,
            candidates=candidates,
        )
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        return resp

    def stages(self, req: SuggestRequest) -> Iterator[Tuple[str, Any]]:
        """Generator used by /suggest/stream.

        Yields:
            ("vector", SuggestResponse)  — fast path, no LLM, ~30-150ms
            ("rerank", SuggestResponse)  — optional, after LLM polish

        The endpoint serialises each tuple as an SSE event.
        """
        self._ensure_indices()
        t0 = time.perf_counter()
        anchor_id, match_conf, candidates, emb = self._get_pre_canvas_candidates(req.query_node)
        first = self._project_response(
            req=req,
            anchor_id=anchor_id,
            match_conf=match_conf,
            candidates=candidates,
        )
        first.latency_ms = int((time.perf_counter() - t0) * 1000)
        yield ("vector", first)

        # Skip the LLM unless the deterministic scores leave room for doubt.
        if not req.include_new_nodes:
            return
        if not self._should_run_rerank(candidates):
            return
        try:
            self._llm_rerank(req.query_node, candidates[: req.max_suggestions * 2])
        except Exception as e:
            LOG.warning("LLM rerank failed (%s); keeping deterministic ranking", e)
            return
        second = self._project_response(
            req=req,
            anchor_id=anchor_id,
            match_conf=match_conf,
            candidates=candidates,
        )
        second.latency_ms = int((time.perf_counter() - t0) * 1000)
        yield ("rerank", second)

    # ── pipeline pieces ────────────────────────────────────────────────

    def _get_pre_canvas_candidates(
        self, query: CanvasNode,
    ) -> Tuple[Optional[str], float, List[_Candidate], Optional[List[float]]]:
        """Resolve the query node → (anchor_id, confidence, candidates).

        Caches at three layers:
          1. Embedding (text → vector) keyed by canonical query text.
          2. Pre-canvas response keyed by the same hash.
          3. Semantic cache (embedding → pre-canvas response) for queries
             that aren't textually identical but are semantically close.
        """
        text = _canvas_node_text(query)
        text_key = hash_text("v1", text)

        # 1) Exact response cache
        cached = self._pre_cache.get(text_key)
        if cached:
            anchor = cached.get("anchor_id")
            conf   = float(cached.get("match_conf", 0.0))
            cands  = [_candidate_from_cacheable(d) for d in cached.get("candidates", [])]
            return anchor, conf, cands, None

        # 2) Embedding cache, else call the API
        emb = self._emb_cache.get(text_key)
        if emb is None:
            emb = self.embedder.embed([text])[0]
            self._emb_cache.set(text_key, emb)

        # 3) Semantic cache — reuse a near-identical query's response
        sem_hit = self._sem_cache.get(emb)
        if sem_hit:
            _, cached2 = sem_hit
            self._pre_cache.set(text_key, cached2)
            anchor = cached2.get("anchor_id")
            conf   = float(cached2.get("match_conf", 0.0))
            cands  = [_candidate_from_cacheable(d) for d in cached2.get("candidates", [])]
            return anchor, conf, cands, emb

        # Cold path: in-memory vector search + neighbour expansion
        anchor_id, match_conf, candidates = self._rank(emb, query)

        # Store both caches
        payload = {
            "anchor_id":  anchor_id,
            "match_conf": match_conf,
            "candidates": [_candidate_to_cacheable(c) for c in candidates],
        }
        self._pre_cache.set(text_key, payload)
        self._sem_cache.set(emb, payload)
        return anchor_id, match_conf, candidates, emb

    def _rank(
        self, emb: List[float], query: CanvasNode,
    ) -> Tuple[Optional[str], float, List[_Candidate]]:
        """Pure ranking step on the in-memory indices."""

        # ── Vector search (numpy) ───────────────────────────────────
        if self._matrix.shape[0] == 0:
            return None, 0.0, []
        q = np.asarray(emb, dtype=np.float32)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        qnorm = float(np.linalg.norm(q))
        if qnorm == 0.0:
            # Query embedding is unusable — fall back to "no anchor".
            return None, 0.0, []
        q = q / qnorm
        # Apple's Accelerate BLAS (the default backend on macOS) raises
        # spurious divide-by-zero / overflow / invalid RuntimeWarnings on
        # *subnormal* float32 operations during matmul, even when both the
        # matrix and the query vector are perfectly finite (which we
        # asserted at index-build time). Suppress those false positives
        # here, then post-validate the result.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self._matrix @ q
        # Guard against any residual nasties from heterogeneous data.
        sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)
        K = min(self.cfg.settings.top_k_vector, sims.shape[0])
        # argpartition for top-K is O(n) rather than the O(n log n) of a sort.
        idx = np.argpartition(-sims, K - 1)[:K]
        idx = idx[np.argsort(-sims[idx])]
        # Build vector-hit list (kg_id, score)
        vec_hits: List[Tuple[str, float]] = []
        for i in idx:
            kid = self._ids[i]
            vec_hits.append((kid, float(sims[i])))

        # ── Resolve the query to a KG anchor ────────────────────────
        # First: exact alias match (cheap, very precise)
        norm_q = _normalize_label(query.label)
        anchor_id = self._alias_index.get(norm_q)
        match_conf = 1.0 if anchor_id else 0.0
        if anchor_id is None:
            # Fall back to top vector hit if confident enough
            if vec_hits and vec_hits[0][1] >= _MATCH_CONFIDENCE_FLOOR:
                anchor_id, match_conf = vec_hits[0]

        # ── Graph-neighbour expansion (in-memory dict lookup) ──────
        candidates: Dict[str, _Candidate] = {}
        if anchor_id:
            for entry in self._neighbours.get(anchor_id, []):
                kid = entry["id"]
                if kid == anchor_id:
                    continue
                cand = candidates.setdefault(kid, _Candidate(
                    id=kid, label=self._labels.get(kid, kid),
                    category=self._categories.get(kid),
                ))
                w = entry["weight"] / 5.0
                if w > cand.graph_weight:
                    cand.graph_weight = w
                    cand.hops = 1
                    cand.rel_chain = [{
                        "sign":   entry["sign"],
                        "weight": entry["weight"],
                        "prov":   entry["prov"],
                        "direction": entry["direction"],
                    }]

        # Add purely-semantic vector candidates (not necessarily connected
        # to the anchor)
        for kid, score in vec_hits:
            if anchor_id and kid == anchor_id:
                continue
            cand = candidates.setdefault(kid, _Candidate(
                id=kid, label=self._labels.get(kid, kid),
                category=self._categories.get(kid),
            ))
            cand.vector_score = max(cand.vector_score, score)

        # ── Combine and rank — KG-connected candidates always win ──────
        #
        # The previous formula (0.6·graph + 0.4·vector) let a high pure-
        # semantic match outrank a genuine KG neighbour. We now use two
        # disjoint score bands so that ANY candidate with a real edge to
        # the anchor in the Neo4j graph ranks above EVERY pure-vector
        # match:
        #
        #   • KG-connected  → raw_score in [0.50, 1.00]
        #   • Vector-only   → raw_score in [0.00, 0.49]
        #
        # This makes the recommender prioritise the knowledge graph's
        # actual causal structure, falling back to semantic matches only
        # to fill the remaining slots.
        for c in candidates.values():
            gw = 0.0 if (c.graph_weight != c.graph_weight) else c.graph_weight  # NaN-safe
            vs = 0.0 if (c.vector_score != c.vector_score) else c.vector_score
            c.graph_weight, c.vector_score = gw, vs
            if gw > 0.0:
                # KG-connected band: weight by edge strength, nudged by
                # semantic similarity.
                c.raw_score = 0.50 + 0.50 * (0.75 * gw + 0.25 * vs)
            else:
                # Pure-vector band, hard-capped below the connected band.
                c.raw_score = 0.49 * vs
            c.alpha = max(0.05, min(0.95, c.raw_score))

        # Sort: connected first (by score), then vector-only (by score).
        ordered = sorted(
            candidates.values(),
            key=lambda c: (c.graph_weight > 0.0, c.raw_score),
            reverse=True,
        )
        return anchor_id, float(match_conf), ordered

    def _project_response(
        self,
        req: SuggestRequest,
        anchor_id: Optional[str],
        match_conf: float,
        candidates: List[_Candidate],
    ) -> SuggestResponse:
        """Take the canvas-independent ranking and produce a wire response
        filtered/shaped by the user's *current* canvas state."""
        canvas_by_label = {_normalize_label(n.label): n for n in req.canvas.nodes}

        # Drop anything already on the canvas (by normalised label)
        novel = [
            c for c in candidates
            if c.id != anchor_id and _normalize_label(c.label) not in canvas_by_label
        ][: max(req.max_suggestions * 2, 6)]

        # Build new-node suggestions, with per-candidate canvas edge proposals
        # batched into ONE Cypher call (replaces the previous N+1 pattern).
        edges_by_cand = self._batch_edges_to_canvas(
            [c.id for c in novel[: req.max_suggestions]],
            req.canvas,
        )

        new_nodes_out: List[SuggestedNode] = []
        if req.include_new_nodes:
            for c in novel[: req.max_suggestions]:
                # Every suggested node carries its *node-level* provenance
                # (which papers / figures the variable was extracted from).
                # Fall back to the connecting edge's provenance only if the
                # node itself has none recorded.
                provenance = _coerce_provenance(self._node_provenance.get(c.id))
                if not provenance and c.rel_chain:
                    provenance = _coerce_provenance(c.rel_chain[0].get("prov"))
                new_nodes_out.append(SuggestedNode(
                    kg_id=c.id,
                    label=c.label,
                    category=c.category,
                    alpha=round(c.alpha, 3),
                    rationale=c.rationale,
                    provenance=provenance,
                    suggested_edges=edges_by_cand.get(c.id, []),
                ))

        # New edges between canvas-nodes — one Cypher for the lot
        new_edges: List[SuggestedCanvasEdge] = []
        if req.include_new_edges and anchor_id:
            new_edges = self._propose_canvas_edges(anchor_id, req.query_node, req.canvas)

        return SuggestResponse(
            query_kg_match=anchor_id,
            match_confidence=round(match_conf, 3),
            new_nodes=new_nodes_out,
            new_edges=new_edges,
            latency_ms=0,
        )

    # ── Canvas-aware Cypher (now batched) ──────────────────────────────

    def _batch_edges_to_canvas(
        self, kg_ids: List[str], canvas: CanvasState,
    ) -> Dict[str, List[SuggestedEdgeFromNew]]:
        """For a *batch* of candidate KG ids, find all canvas-touching
        edges in one Cypher call.

        This replaces the old per-candidate query that turned a single
        /suggest into N round-trips.
        """
        if not kg_ids or not canvas.nodes:
            return {}
        norm_to_id = {_normalize_label(n.label): n.id for n in canvas.nodes}

        q = """
        UNWIND $kg_ids AS kid
        MATCH (k:KGNode {id: kid})-[r:CAUSES]-(b:KGNode)
        WHERE toLower(b.label) IN $labels
           OR any(a IN b.aliases WHERE toLower(a) IN $labels)
        RETURN kid AS kid,
               b.label AS blabel,
               startNode(r).id AS src,
               endNode(r).id   AS tgt,
               r.sign AS sign, r.weight AS weight, r.provenance AS prov
        """
        labels_lc = list(norm_to_id.keys())
        with self.neo.session() as s:
            rows = s.run(q, kg_ids=list(kg_ids), labels=labels_lc).data()

        out: Dict[str, List[SuggestedEdgeFromNew]] = {kid: [] for kid in kg_ids}
        for rec in rows:
            canvas_id = norm_to_id.get(_normalize_label(rec["blabel"]))
            if not canvas_id:
                continue
            direction = "outgoing" if rec["src"] == rec["kid"] else "incoming"
            out[rec["kid"]].append(SuggestedEdgeFromNew(
                canvas_node_id=canvas_id,
                direction=direction,
                polarity=rec["sign"],
                weight=int(rec["weight"]),
                provenance=_coerce_provenance(rec["prov"]),
            ))
        return out

    def _propose_canvas_edges(
        self, anchor_id: str, query: CanvasNode, canvas: CanvasState,
    ) -> List[SuggestedCanvasEdge]:
        """Suggest KG-backed edges between nodes already on the canvas
        (excluding the query itself). One Cypher round-trip total.
        """
        existing = {(e.source, e.target) for e in canvas.edges}
        norm_to_id = {
            _normalize_label(n.label): n.id
            for n in canvas.nodes if n.id != query.id
        }
        if not norm_to_id:
            return []
        labels_lc = list(norm_to_id.keys())
        q = """
        UNWIND $labels AS lbl
        MATCH (b:KGNode)
        WHERE toLower(b.label) = lbl
           OR any(a IN b.aliases WHERE toLower(a) = lbl)
        WITH lbl, b
        MATCH (a:KGNode {id: $anchor})-[r:CAUSES]-(b)
        RETURN lbl, b.id AS kid,
               startNode(r).id AS src, endNode(r).id AS tgt,
               r.sign AS sign, r.weight AS weight, r.provenance AS prov
        """
        with self.neo.session() as s:
            rows = s.run(q, labels=labels_lc, anchor=anchor_id).data()

        proposals: List[SuggestedCanvasEdge] = []
        for rec in rows:
            canvas_id = norm_to_id.get(rec["lbl"])
            if not canvas_id:
                continue
            # Map KG src/tgt back to canvas ids
            src_canvas = query.id if rec["src"] == anchor_id else canvas_id
            tgt_canvas = canvas_id if rec["tgt"] == rec["kid"] else query.id
            if (src_canvas, tgt_canvas) in existing:
                continue
            weight = int(rec["weight"])
            alpha = round(min(0.95, 0.4 + 0.1 * weight), 3)
            proposals.append(SuggestedCanvasEdge(
                source=src_canvas, target=tgt_canvas,
                polarity=rec["sign"], weight=weight,
                alpha=alpha,
                rationale=f"Direct KG link (weight {weight})",
                provenance=_coerce_provenance(rec["prov"]),
            ))
        return proposals

    # ── Rerank decisions ───────────────────────────────────────────────

    def _should_run_rerank(self, candidates: List[_Candidate]) -> bool:
        """Skip the LLM when scores are confidently ordered."""
        if len(candidates) < 2:
            return False
        top_score = candidates[0].raw_score
        if top_score >= self.cfg.settings.skip_rerank_if_top_score_above:
            return False
        # Margin between top-1 and the next "interesting" candidate
        if len(candidates) >= 2:
            margin = candidates[0].raw_score - candidates[1].raw_score
            if margin >= self.cfg.settings.skip_rerank_if_margin_above:
                return False
        return True

    def _llm_rerank(self, query: CanvasNode, candidates: List[_Candidate]) -> None:
        if not candidates:
            return
        _spec, client = self.router.get("recommender_reranker")
        sys_prompt = (
            "You score candidate causal neighbours of a query variable in a "
            "system-dynamics model. Output JSON only."
        )
        cand_payload = [
            {"id": c.id, "label": c.label, "category": c.category,
             "graph_weight": round(c.graph_weight, 3),
             "vector_score": round(c.vector_score, 3),
             "hops": c.hops}
            for c in candidates
        ]
        user_prompt = json.dumps({
            "query": {
                "label": query.label, "category": query.category,
                "description": query.description,
            },
            "candidates": cand_payload,
            "task": (
                "For each candidate, score its plausibility as a direct causal "
                "neighbour of the query in [0,1] (`alpha`). Provide a 12-word "
                "rationale only for candidates scoring ≥ 0.4."
            ),
            "schema": {"scored": [{"id": "string", "alpha": "number 0..1", "rationale": "string|null"}]},
        })
        from src.llm_client import call_with_retries, extract_json
        raw = call_with_retries(client, sys_prompt, user_prompt, retries=1)
        data = extract_json(raw)
        scored = data.get("scored") or []
        by_id = {c.id: c for c in candidates}
        for row in scored:
            cand = by_id.get(row.get("id"))
            if not cand:
                continue
            try:
                a = float(row.get("alpha", cand.alpha))
                cand.alpha = max(0.05, min(0.95, a))
            except Exception:
                pass
            r = row.get("rationale")
            if isinstance(r, str) and r.strip():
                cand.rationale = r.strip()
