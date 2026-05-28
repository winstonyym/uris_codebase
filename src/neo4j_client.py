"""Thin Neo4j driver wrapper.

Exposes synchronous helpers used by the recommender, causal reasoner and
ingest pipeline. We use the synchronous driver because the FastAPI
service runs each request handler on a worker thread (sync routes), which
is simpler and faster for the small queries we issue.

All write queries are idempotent (MERGE-based) so ingest can be re-run
safely after merged_kg.json is regenerated.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase, Driver


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        # When NEO4J_AUTH=none (docker-compose default), pass auth=None.
        auth = (user, password) if password else None
        self._driver: Driver = GraphDatabase.driver(uri, auth=auth)
        self._database = database

    def close(self):
        self._driver.close()

    @contextmanager
    def session(self):
        with self._driver.session(database=self._database) as s:
            yield s

    # ── Schema / indexes ───────────────────────────────────────────────

    def ensure_schema(self, vector_index_name: str, dimensions: int):
        """Create node-id uniqueness + vector index if absent."""
        with self.session() as s:
            s.run(
                "CREATE CONSTRAINT kg_node_id IF NOT EXISTS "
                "FOR (n:KGNode) REQUIRE n.id IS UNIQUE"
            )
            # Vector index requires Neo4j 5.13+.
            s.run(
                f"""
                CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
                FOR (n:KGNode) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dim=int(dimensions),
            )

    # ── Writes ─────────────────────────────────────────────────────────

    def upsert_nodes(self, nodes: List[Dict[str, Any]]) -> int:
        """Bulk upsert KG nodes. Each dict needs id/label/category and may
        include aliases, description, embedding, provenance."""
        if not nodes:
            return 0
        q = """
        UNWIND $rows AS row
        MERGE (n:KGNode {id: row.id})
        SET n.label       = row.label,
            n.category    = row.category,
            n.aliases     = coalesce(row.aliases, []),
            n.description = coalesce(row.description, ''),
            n.provenance  = row.provenance_json,
            n.embedding   = row.embedding
        RETURN count(n) AS n
        """
        with self.session() as s:
            res = s.run(q, rows=list(nodes))
            return res.single()["n"]

    def upsert_edges(self, edges: List[Dict[str, Any]]) -> int:
        """Bulk upsert directed signed edges as :CAUSES relationships."""
        if not edges:
            return 0
        q = """
        UNWIND $rows AS row
        MATCH (a:KGNode {id: row.source_id})
        MATCH (b:KGNode {id: row.target_id})
        MERGE (a)-[r:CAUSES {sign: row.sign}]->(b)
        SET r.weight     = row.weight,
            r.provenance = row.provenance_json
        RETURN count(r) AS n
        """
        with self.session() as s:
            res = s.run(q, rows=list(edges))
            return res.single()["n"]

    # ── Reads ──────────────────────────────────────────────────────────

    def get_node(self, kg_id: str) -> Optional[Dict[str, Any]]:
        q = "MATCH (n:KGNode {id:$id}) RETURN n {.*} AS n"
        with self.session() as s:
            rec = s.run(q, id=kg_id).single()
            return dict(rec["n"]) if rec else None

    def vector_search(
        self,
        index_name: str,
        embedding: List[float],
        k: int = 25,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-K KG nodes by cosine similarity, excluding given ids."""
        exclude = list(exclude_ids or [])
        q = f"""
        CALL db.index.vector.queryNodes($index, $k, $emb)
        YIELD node, score
        WHERE NOT node.id IN $exclude
        RETURN node.id AS id, node.label AS label, node.category AS category,
               node.aliases AS aliases, node.description AS description,
               score
        ORDER BY score DESC
        """
        with self.session() as s:
            return [dict(r) for r in s.run(q, index=index_name, k=int(k), emb=embedding, exclude=exclude)]

    def neighbours_of(
        self,
        kg_id: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """1- and 2-hop neighbours of `kg_id`, returning paths with sign+weight."""
        q = """
        MATCH path = (a:KGNode {id:$id})-[rels:CAUSES*1..2]-(b:KGNode)
        WHERE a <> b
        WITH b, rels, length(path) AS hops,
             reduce(s = '+', r IN rels |
                 CASE WHEN s = r.sign THEN '+' ELSE '-' END) AS net_sign,
             reduce(w = 1.0, r IN rels | w * (r.weight / 5.0)) AS w
        RETURN DISTINCT
            b.id AS id, b.label AS label, b.category AS category,
            hops, net_sign, w AS weight,
            [r IN rels | { sign:r.sign, weight:r.weight, prov: r.provenance }] AS rel_chain
        ORDER BY hops ASC, weight DESC
        LIMIT 60
        """
        with self.session() as s:
            return [dict(r) for r in s.run(q, id=kg_id)]

    def edge_between(self, src_id: str, tgt_id: str) -> Optional[Dict[str, Any]]:
        """Return the KG edge (if any) between two KG nodes (either direction)."""
        q = """
        MATCH (a:KGNode {id:$a})-[r:CAUSES]->(b:KGNode {id:$b})
        RETURN r.sign AS sign, r.weight AS weight, r.provenance AS provenance
        """
        with self.session() as s:
            rec = s.run(q, a=src_id, b=tgt_id).single()
            if not rec:
                return None
            return dict(rec)
