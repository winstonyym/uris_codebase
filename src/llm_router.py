"""Resolves which LLM to call for a given component name.

We reuse `utils/llm_client.py` from the knowledge_graph_workflow package
so there's one place where Anthropic/OpenAI/etc. backends are
implemented. The router caches clients per component to avoid recreating
HTTP sessions on every request.

For the chat agent we *also* need raw Anthropic tool-use, which the
shared `complete()` interface doesn't expose. For that path we drop down
to the native SDK in chat_agent.py — but config still flows through here
so the model is configurable.
"""
from __future__ import annotations

from threading import RLock
from typing import Dict

from .config import Config


class LLMRouter:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._cache: Dict[str, object] = {}
        self._lock = RLock()

    def get(self, component: str):
        """Return an `LLMClient` (with `.complete(system, user)` interface)."""
        from src.llm_client import make_client
        with self._lock:
            if component not in self._cache:
                spec = self._cfg.model_spec_for(component)
                self._cache[component] = (spec, make_client(spec))
            return self._cache[component]

    def spec(self, component: str):
        return self._cfg.model_spec_for(component)
