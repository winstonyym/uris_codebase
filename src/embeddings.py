"""Embedding client.

Currently supports OpenAI (or any OpenAI-compatible /v1/embeddings server
via base_url). The model is selected in models.yaml under
`components.embedder.model`. Adding Voyage AI or Cohere is a ~20-line
extension — see EmbedderProtocol below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Protocol

from .config import ComponentSpec


class EmbedderProtocol(Protocol):
    dimensions: int
    def embed(self, texts: List[str]) -> List[List[float]]: ...


@dataclass
class OpenAIEmbedder:
    spec: ComponentSpec

    def __post_init__(self):
        from openai import OpenAI
        api_key = os.environ.get(self.spec.api_key_env or "OPENAI_API_KEY", "EMPTY")
        kwargs = {"api_key": api_key}
        if self.spec.base_url:
            kwargs["base_url"] = self.spec.base_url
        self._client = OpenAI(**kwargs)

    @property
    def dimensions(self) -> int:
        return int(self.spec.dimensions or 1536)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # OpenAI API supports batching by passing a list.
        resp = self._client.embeddings.create(model=self.spec.model, input=texts)
        return [d.embedding for d in resp.data]


def make_embedder(spec: ComponentSpec) -> EmbedderProtocol:
    provider = (spec.provider or "openai").lower()
    if provider == "openai":
        return OpenAIEmbedder(spec)
    raise ValueError(
        f"Unsupported embedding provider: {spec.provider!r}. "
        "Add an implementation in graph_rag/embeddings.py."
    )
