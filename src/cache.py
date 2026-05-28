"""Tiered cache for the Graph-RAG service.

Two layers, both web-deployable:

  * In-process LRU (always on, sub-microsecond hit). Lost on restart.
  * Redis            (optional). When `REDIS_URL` is set AND the host is
                     reachable, values are also persisted there with TTL.
                     If Redis goes away mid-flight, the cache silently
                     degrades to LRU-only — never raises.

Two cache *kinds*:

  * `KVCache`        — exact-key get/set/delete with TTL. Used for
                       embeddings (key = sha1(text)) and pre-canvas
                       response components (key = hash of query+request).

  * `SemanticCache`  — vector-keyed. `get(vec)` finds the most-similar
                       cached embedding above a configurable cosine
                       threshold and returns its value. Lives in-process
                       only — Redis round-trips on float-array payloads
                       would defeat the speed purpose.

Design constraints worth knowing:

  * JSON for all serialised payloads. We never store Pydantic models in
    the cache directly — callers convert with `model_dump()` before
    setting and re-validate on get.
  * Connection timeouts on Redis are 1 s by default; the service must
    never block on a flaky cache server.
  * Namespacing: every Redis key is prefixed `grag:<namespace>:` so
    multiple Graph-RAG instances can share a Redis without colliding.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import ssl
import threading
import time
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np


LOG = logging.getLogger("graph_rag.cache")


def _resolve_ca_bundle() -> Optional[str]:
    """Find a CA bundle on disk that we can hand to redis-py's TLS layer.

    Resolution order:
      1. SSL_CERT_FILE  — respects whatever the user already configured.
      2. REQUESTS_CA_BUNDLE — same idea, common alt env var.
      3. certifi.where() — bundled by openai/anthropic SDKs anyway, so
         this is almost always present in our process.
      4. ssl.get_default_verify_paths().cafile — system store, may exist
         on Linux but is usually empty on macOS.
    """
    for env in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        p = os.environ.get(env)
        if p and os.path.exists(p):
            return p
    try:
        import certifi
        p = certifi.where()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    p = ssl.get_default_verify_paths().cafile
    if p and os.path.exists(p):
        return p
    return None


# ──────────────────────────────────────────────────────────────────────
# In-process LRU with optional TTL. Thread-safe.
# ──────────────────────────────────────────────────────────────────────

class _LRU:
    def __init__(self, maxsize: int = 1024, default_ttl: float = 0.0):
        self._maxsize = max(1, int(maxsize))
        self._default_ttl = float(default_ttl)
        self._lock = threading.RLock()
        # value_tuple = (value, expiry_epoch_or_0)
        self._store: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._store.get(key)
            if row is None:
                return None
            value, exp = row
            if exp and exp < time.time():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = self._default_ttl if ttl is None else float(ttl)
        exp = time.time() + ttl if ttl > 0 else 0.0
        with self._lock:
            self._store[key] = (value, exp)
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._store), "maxsize": self._maxsize}


# ──────────────────────────────────────────────────────────────────────
# KVCache: LRU + optional Redis back-end.
# ──────────────────────────────────────────────────────────────────────

class KVCache:
    """Exact-key cache with optional Redis backing.

    Usage:
        cache = KVCache(redis_url=os.environ.get("REDIS_URL"),
                        namespace="emb", ttl=86400, lru_size=2048)
        cache.set("abc123", [0.1, 0.2, ...])
        cache.get("abc123")    # → list or None
    """

    def __init__(
        self,
        redis_url: Optional[str],
        namespace: str,
        ttl: int = 3600,
        lru_size: int = 1024,
    ):
        self.namespace = namespace
        self.ttl = int(ttl)
        self._lru = _LRU(maxsize=lru_size, default_ttl=ttl)
        self._redis = self._maybe_connect(redis_url)

    # Track whether we've already complained about a bad Redis URL this
    # process. Without this the message fires once per cache namespace
    # (emb, pre, …), which is annoying.
    _bad_url_logged: bool = False

    @staticmethod
    def _maybe_connect(redis_url: Optional[str]):
        if not redis_url:
            return None
        # Validate scheme up front. Redis' from_url() raises ValueError on
        # bare host:port, which used to spam the startup log once per
        # namespace. We silently skip Redis when the scheme is missing.
        valid_schemes = ("redis://", "rediss://", "unix://")
        if not any(redis_url.startswith(s) for s in valid_schemes):
            if not KVCache._bad_url_logged:
                LOG.warning(
                    "Redis URL %r has no recognised scheme (expected one of %s); "
                    "skipping Redis and using in-process LRU only.",
                    _redact(redis_url), valid_schemes,
                )
                KVCache._bad_url_logged = True
            return None
        try:
            import redis  # lazy import — package is optional at install-time
            connect_kwargs: dict = {
                "decode_responses": True,
                "socket_connect_timeout": 2.0,    # TLS handshake needs a bit more
                "socket_timeout": 2.0,
            }
            # rediss:// → TLS connection. macOS Python can't find system CAs,
            # so verification fails with "unable to get local issuer
            # certificate". Pointing redis-py at the certifi CA bundle (which
            # openai/anthropic already install) resolves it deterministically.
            if redis_url.startswith("rediss://"):
                ca_bundle = _resolve_ca_bundle()
                if ca_bundle:
                    connect_kwargs["ssl_ca_certs"] = ca_bundle
                # Escape hatch for restricted networks / self-signed certs.
                # Set REDIS_SSL_INSECURE=1 to skip verification entirely.
                if os.environ.get("REDIS_SSL_INSECURE", "").lower() in {"1", "true", "yes"}:
                    connect_kwargs["ssl_cert_reqs"] = None
                    LOG.warning(
                        "REDIS_SSL_INSECURE is set — Redis TLS certificate "
                        "verification is DISABLED for this process."
                    )
            client = redis.from_url(redis_url, **connect_kwargs)
            client.ping()
            LOG.info("KVCache: Redis connected (%s)", _redact(redis_url))
            return client
        except Exception as e:
            if not KVCache._bad_url_logged:
                LOG.warning("KVCache: Redis unavailable (%s); falling back to LRU-only", e)
                KVCache._bad_url_logged = True
            return None

    def _full_key(self, key: str) -> str:
        return f"grag:{self.namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        v = self._lru.get(key)
        if v is not None:
            return v
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._full_key(key))
        except Exception as e:
            LOG.warning("KVCache(%s) Redis get failed: %s", self.namespace, e)
            return None
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except Exception:
            return None
        # Promote into LRU so subsequent reads in this process are O(1)
        self._lru.set(key, value)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        eff_ttl = int(ttl if ttl is not None else self.ttl)
        self._lru.set(key, value, ttl=eff_ttl)
        if self._redis is None:
            return
        try:
            self._redis.set(self._full_key(key), json.dumps(value), ex=eff_ttl)
        except Exception as e:
            LOG.warning("KVCache(%s) Redis set failed: %s", self.namespace, e)

    def delete(self, key: str) -> None:
        self._lru.delete(key)
        if self._redis is not None:
            try:
                self._redis.delete(self._full_key(key))
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "namespace": self.namespace,
            "redis": bool(self._redis),
            "lru": self._lru.stats(),
        }


# ──────────────────────────────────────────────────────────────────────
# SemanticCache: in-process, cosine-nearest lookup.
# ──────────────────────────────────────────────────────────────────────

class SemanticCache:
    """Vector-keyed cache.

    On `get(embedding)`, returns the value of the cached entry whose key
    embedding has the highest cosine similarity with the query, provided
    that similarity is at least `threshold`. Otherwise returns None.

    Internally we store a normalised (unit-length) matrix so cosine
    similarity is just a dot product. For 512–4096 cached entries with
    1536-dim vectors, this is sub-millisecond on a laptop CPU.

    Lives in-process only. (Redis serialisation of float arrays adds
    latency that defeats the purpose of a "fast lookup" cache.)
    """

    def __init__(self, dim: int, threshold: float = 0.97, maxsize: int = 1024):
        self.dim = int(dim)
        self.threshold = float(threshold)
        self.maxsize = int(maxsize)
        self._lock = threading.RLock()
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._values: List[Any] = []
        self._fp_seen: set[str] = set()    # dedupe identical embeddings

    @staticmethod
    def _unit(v) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        n = float(np.linalg.norm(a))
        if n == 0.0:
            # Degenerate input — return the zero vector so dot products are
            # exactly 0 and this entry never wins a similarity lookup.
            return a
        return a / n

    @staticmethod
    def _fp(arr: np.ndarray) -> str:
        # Fingerprint for dedupe — first few elements are enough since
        # query texts producing identical embeddings are rare collisions.
        return hashlib.md5(arr.tobytes()).hexdigest()

    def get(self, embedding) -> Optional[Tuple[float, Any]]:
        """Returns (similarity, value) if there's a hit above threshold, else None."""
        with self._lock:
            if len(self._values) == 0:
                return None
            q = self._unit(embedding)
            sims = self._matrix @ q
            i = int(np.argmax(sims))
            sim = float(sims[i])
            if sim >= self.threshold:
                return sim, self._values[i]
            return None

    def set(self, embedding, value: Any) -> None:
        with self._lock:
            q = self._unit(embedding)
            fp = self._fp(q)
            if fp in self._fp_seen:
                return
            if self._matrix.shape[0] == 0:
                self._matrix = q[None, :]
            else:
                self._matrix = np.vstack([self._matrix, q[None, :]])
            self._values.append(value)
            self._fp_seen.add(fp)
            # FIFO eviction (oldest first) — acceptable for a cache that
            # exists to capture short-term repeated semantic queries.
            if len(self._values) > self.maxsize:
                self._matrix = self._matrix[1:]
                self._values = self._values[1:]

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._values), "maxsize": self.maxsize, "threshold": self.threshold}


# ──────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────

def hash_text(*parts: str) -> str:
    """Stable hash over an ordered sequence of strings/ints/etc."""
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def canvas_signature(nodes, edges) -> str:
    """Order-independent canvas fingerprint used as part of cache keys.

    Two canvases that have the same set of node ids and the same set of
    directed signed edges produce the same signature, regardless of how
    they were serialised.
    """
    node_part = sorted(n.get("id", "") for n in nodes)
    edge_part = sorted(
        f"{e.get('source','')}>{e.get('polarity','+')}>{e.get('target','')}"
        for e in edges
    )
    return hash_text("nodes", *node_part, "edges", *edge_part)


def _redact(url: str) -> str:
    """Mask the credentials portion of a Redis URL for log lines."""
    try:
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
    except Exception:
        pass
    return "redis://***"
