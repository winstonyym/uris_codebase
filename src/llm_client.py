"""Thin multi-provider LLM client.

Two backends:
  * `openai`    — any OpenAI-compatible HTTP API. Set `base_url` to point at
                  DeepSeek, Qwen/DashScope, Together, Groq, vLLM, Ollama,
                  llama.cpp server, etc.
  * `anthropic` — native Anthropic Messages API.

Each backend implements the same `complete(system, user) -> str` interface.
We deliberately avoid provider-specific structured-output features
(`response_format`, tool calls) so the same prompt works everywhere; JSON
is requested in the prompt and validated downstream.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelSpec:
    """One row in the model grid."""
    name: str
    backend: str
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096           # ADD THIS LINE
    request_json_mode: bool = True
    vision: bool | None = None
    reasoning_effort: str | None = None
 
 
class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...
 
 
_REASONING_PATTERNS = ("gpt-5", "o1", "o3", "o4", "gemini-3")
 
 
def _is_reasoning_model(model: str) -> bool:
    """Heuristic: OpenAI reasoning models reject `temperature` and accept
    `reasoning_effort`. Includes the o-series and the gpt-5.x family."""
    m = model.lower()
    return any(p in m for p in _REASONING_PATTERNS)
 
 
def _build_openai_kwargs(spec: ModelSpec, base: dict) -> dict:
    """Add temperature/reasoning_effort/response_format to a base kwargs dict
    based on whether the target is a reasoning model."""
    kwargs = dict(base)
    is_reasoning = _is_reasoning_model(spec.model)
    if is_reasoning:
        # Reasoning models reject `temperature` and `max_tokens`. They use
        # `max_completion_tokens` instead (which counts both reasoning and
        # visible-output tokens).
        kwargs["max_completion_tokens"] = spec.max_tokens
        if spec.reasoning_effort:
            kwargs["reasoning_effort"] = spec.reasoning_effort
    else:
        kwargs["max_tokens"] = spec.max_tokens
        kwargs["temperature"] = spec.temperature
    if spec.request_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs
 

class OpenAICompatibleClient:
    """Works for OpenAI, DeepSeek, Qwen/DashScope (OpenAI-compat mode),
    Together, Groq, vLLM, Ollama, llama.cpp, etc."""
 
    def __init__(self, spec: ModelSpec):
        from openai import OpenAI  # imported lazily so the file imports cleanly without the lib
        self.spec = spec
        api_key = os.environ.get(spec.api_key_env, "EMPTY")  # local servers often don't need a key
        kwargs: dict = {"api_key": api_key}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        self.client = OpenAI(**kwargs)
 
    def complete(self, system: str, user: str) -> str:
        kwargs = _build_openai_kwargs(self.spec, {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        })
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # If a server rejects response_format, retry once without it.
            if "response_format" in kwargs and "response_format" in str(e).lower():
                kwargs.pop("response_format")
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""
 
class AnthropicClient:
    def __init__(self, spec: ModelSpec):
        from anthropic import Anthropic
        self.spec = spec
        api_key = os.environ.get(spec.api_key_env, "")
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.spec.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.spec.max_tokens,    # ADD THIS LINE
            temperature=self.spec.temperature,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def make_client(spec: ModelSpec) -> LLMClient:
    if spec.backend == "openai":
        return OpenAICompatibleClient(spec)
    if spec.backend == "anthropic":
        return AnthropicClient(spec)
    raise ValueError(f"Unknown backend: {spec.backend!r}")


# --- JSON extraction --------------------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """Best-effort JSON extraction from a model response.

    Tries: raw parse, ```json fenced block, first {...} balanced span,
    first [...] balanced span. Raises ValueError if nothing parses.

    Returns whatever JSON was extracted — usually a dict, but may be a
    list when the model returned an array.
    """
    text = text.strip()
    # 1. Direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. Fenced block
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # Generic balanced-span scan: find both candidate types, prefer
    # whichever starts EARLIEST in the text (i.e. the outermost
    # structure). Without this, a list of dicts in prose would have
    # the inner dict picked off before the list wrapper is considered.
    def _scan(open_ch: str, close_ch: str):
        start = text.find(open_ch)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_ch:
                    depth += 1
                elif text[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return start, json.loads(candidate)
                        except Exception:
                            break
            start = text.find(open_ch, start + 1)
        return None

    obj_hit = _scan("{", "}")
    arr_hit = _scan("[", "]")
    if obj_hit and arr_hit:
        # Earlier-starting structure wins (it's the outermost)
        return obj_hit[1] if obj_hit[0] <= arr_hit[0] else arr_hit[1]
    if obj_hit:
        return obj_hit[1]
    if arr_hit:
        return arr_hit[1]
    raise ValueError(f"No JSON object or array found in response:\n{text[:500]}")


def call_with_retries(
    client: LLMClient, system: str, user: str, retries: int = 2, backoff: float = 2.0
) -> str:
    """Retry transient failures with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return client.complete(system, user)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_exc}")