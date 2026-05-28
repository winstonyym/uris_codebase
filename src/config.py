"""Config loader for graph_rag.

Reads the same `models.yaml` that build_kg.py uses, but additionally
parses the `components:` and `graph_rag:` sections introduced for the
Graph-RAG service. Each component (embedder, recommender_reranker,
chat_agent, causal_narrator) can be independently swapped to a different
model by editing models.yaml — nothing in this package hard-codes a
provider.

The loader is deliberately tolerant: a missing optional section falls
back to a documented default, so the file remains usable for build_kg.py
even if a user trims it down.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# Make ../knowledge_graph_workflow importable so we can reuse llm_client.
_KGW = Path(__file__).resolve().parent.parent / "knowledge_graph_workflow"
if str(_KGW) not in sys.path:
    sys.path.insert(0, str(_KGW))


DEFAULT_MODELS_YAML = (
    Path(__file__).resolve().parent.parent
    / "models.yaml"
)


@dataclass
class ComponentSpec:
    """One row in the `components:` section of models.yaml."""
    name: str
    model: str                         # name from `models:` catalogue OR raw model id
    temperature: float = 0.0
    # Embedder-only
    provider: str | None = None        # e.g. "openai"
    api_key_env: str | None = None
    dimensions: int | None = None
    base_url: str | None = None


@dataclass
class CacheSettings:
    redis_url: str | None = None
    embedding_ttl_sec: int = 86400
    response_ttl_sec: int = 300
    semantic_threshold: float = 0.97
    semantic_max_entries: int = 1024
    lru_size: int = 2048


@dataclass
class StreamSettings:
    progressive: bool = True
    skip_llm_if_score_above: float = 0.90


@dataclass
class GraphRagSettings:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    vector_index_name: str = "nodeLabelEmbedding"
    merged_kg_path: str = "../knowledge_graph_workflow/temp_graph/merged_kg.json"
    top_k_vector: int = 25
    top_k_final: int = 10
    max_hops: int = 2
    damping: float = 0.6
    skip_rerank_if_top_score_above: float = 0.92
    skip_rerank_if_margin_above:    float = 0.18
    cache:  CacheSettings  = field(default_factory=CacheSettings)
    stream: StreamSettings = field(default_factory=StreamSettings)


@dataclass
class Config:
    """Top-level config for the graph_rag service."""
    models_yaml_path: Path
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    components: Dict[str, ComponentSpec] = field(default_factory=dict)
    settings: GraphRagSettings = field(default_factory=GraphRagSettings)

    def model_spec_for(self, component: str):
        """Resolve `components.<component>.model` against the `models:`
        catalogue and return a `ModelSpec` ready for `make_client`.

        Falls back to a thin stub if the model name isn't in the catalogue,
        assuming an anthropic backend (the most common default).
        """
        from utils.llm_client import ModelSpec  # imported lazily — sys.path was set above

        if component not in self.components:
            raise KeyError(
                f"Component {component!r} missing in models.yaml `components:` section"
            )
        comp = self.components[component]
        model_row = self.models.get(comp.model)
        if model_row is None:
            # User likely typed a raw model id; assume anthropic default.
            return ModelSpec(
                name=comp.model,
                backend="anthropic",
                model=comp.model,
                api_key_env="ANTHROPIC_API_KEY",
                temperature=comp.temperature,
            )
        return ModelSpec(
            name=model_row["name"],
            backend=model_row["backend"],
            model=model_row["model"],
            base_url=model_row.get("base_url"),
            api_key_env=model_row.get("api_key_env", "OPENAI_API_KEY"),
            max_tokens=model_row.get("max_tokens", 4096),
            temperature=comp.temperature,
            reasoning_effort=model_row.get("reasoning_effort"),
            request_json_mode=model_row.get("request_json_mode", True),
        )


def load_config(path: Path | str | None = None) -> Config:
    p = Path(path) if path else DEFAULT_MODELS_YAML
    if not p.exists():
        raise FileNotFoundError(f"models.yaml not found at {p}")
    raw = yaml.safe_load(p.read_text()) or {}

    cfg = Config(models_yaml_path=p)

    # Catalogue
    for row in raw.get("models", []) or []:
        cfg.models[row["name"]] = row

    # Components
    for name, spec in (raw.get("components") or {}).items():
        cfg.components[name] = ComponentSpec(
            name=name,
            model=spec.get("model", ""),
            temperature=float(spec.get("temperature", 0.0)),
            provider=spec.get("provider"),
            api_key_env=spec.get("api_key_env"),
            dimensions=spec.get("dimensions"),
            base_url=spec.get("base_url"),
        )

    # Graph-RAG settings
    s = raw.get("graph_rag") or {}
    cache_cfg  = s.get("cache")  or {}
    stream_cfg = s.get("stream") or {}
    cfg.settings = GraphRagSettings(
        neo4j_uri=s.get("neo4j_uri", cfg.settings.neo4j_uri),
        neo4j_user=s.get("neo4j_user", cfg.settings.neo4j_user),
        neo4j_password=s.get("neo4j_password", cfg.settings.neo4j_password),
        neo4j_database=s.get("neo4j_database", cfg.settings.neo4j_database),
        vector_index_name=s.get("vector_index_name", cfg.settings.vector_index_name),
        merged_kg_path=s.get("merged_kg_path", cfg.settings.merged_kg_path),
        top_k_vector=int(s.get("top_k_vector", cfg.settings.top_k_vector)),
        top_k_final=int(s.get("top_k_final", cfg.settings.top_k_final)),
        max_hops=int(s.get("max_hops", cfg.settings.max_hops)),
        damping=float(s.get("damping", cfg.settings.damping)),
        skip_rerank_if_top_score_above=float(
            s.get("skip_rerank_if_top_score_above", cfg.settings.skip_rerank_if_top_score_above)
        ),
        skip_rerank_if_margin_above=float(
            s.get("skip_rerank_if_margin_above", cfg.settings.skip_rerank_if_margin_above)
        ),
        cache=CacheSettings(
            redis_url=cache_cfg.get("redis_url") or os.environ.get("REDIS_URL") or None,
            embedding_ttl_sec=int(cache_cfg.get("embedding_ttl_sec", 86400)),
            response_ttl_sec=int(cache_cfg.get("response_ttl_sec", 300)),
            semantic_threshold=float(cache_cfg.get("semantic_threshold", 0.97)),
            semantic_max_entries=int(cache_cfg.get("semantic_max_entries", 1024)),
            lru_size=int(cache_cfg.get("lru_size", 2048)),
        ),
        stream=StreamSettings(
            progressive=bool(stream_cfg.get("progressive", True)),
            skip_llm_if_score_above=float(stream_cfg.get("skip_llm_if_score_above", 0.90)),
        ),
    )

    # Allow env overrides for sensitive bits.
    if "NEO4J_URI" in os.environ:
        cfg.settings.neo4j_uri = os.environ["NEO4J_URI"]
    if "NEO4J_PASSWORD" in os.environ:
        cfg.settings.neo4j_password = os.environ["NEO4J_PASSWORD"]

    return cfg
