"""Free-tier token accounting for the Graph-RAG service.

Every LLM call this service makes spends the *operator's* API key, so a
public deployment needs a ceiling. The ceiling here is per **conversation**:

  * a signed-in user gets ``FREE_TIER_SESSION_TOKENS`` tokens (prompt +
    completion, summed across every model a request touches) for a given
    browser session, and
  * may open at most ``FREE_TIER_SESSIONS_PER_DAY`` conversations per UTC
    day. Without that second limit the first one is decorative — the
    session id is minted in the browser, so a reload would hand the user a
    fresh budget.

Two moving parts:

``UsageAccumulator`` + ``bind()``
    A :class:`contextvars.ContextVar` that the LLM clients push their
    per-call token counts into, so no call site has to thread a counter
    through five layers. ``anyio`` copies the context into worker threads,
    which is what makes this survive the ``to_thread.run_sync`` hops in
    ``app.py``.

``UsageLedger``
    The durable side. Redis when ``REDIS_URL`` is set — ``INCRBY`` and a
    small Lua script are atomic, so two parallel requests can't both slip
    past the cap. An in-process dict otherwise: correct for local dev,
    **leaky on serverless** (every cold start resets it and parallel
    instances don't share it). ``ledger.backing`` reports which one is live
    and ``/usage`` surfaces it, so a misconfigured deploy is visible rather
    than silently free.

Environment:
    REDIS_URL                     rediss://… from Upstash (or any Redis)
    FREE_TIER_SESSION_TOKENS      per-conversation cap      (default 75000)
    FREE_TIER_SESSIONS_PER_DAY    conversations per UTC day (default 5)
    FREE_TIER_DISABLED            set to 1 to lift the caps entirely
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Tuple

LOG = logging.getLogger("graph_rag.usage")

# ── Configuration ─────────────────────────────────────────────────────


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        LOG.warning("%s is not an integer; using default %d", name, default)
        return default


def session_token_cap() -> int:
    return _int_env("FREE_TIER_SESSION_TOKENS", 75_000)


def sessions_per_day_cap() -> int:
    return _int_env("FREE_TIER_SESSIONS_PER_DAY", 5)


def quotas_disabled() -> bool:
    return str(os.environ.get("FREE_TIER_DISABLED", "")).lower() in {"1", "true", "yes"}


# Conversation counters outlive a browser session but not by much; a day is
# plenty and keeps Redis from accumulating dead keys.
_CONV_TTL_S = 24 * 60 * 60
_DAY_TTL_S = 48 * 60 * 60


# ── The accumulator (per-request, in-memory) ──────────────────────────


@dataclass
class UsageAccumulator:
    """Tokens spent while serving one request.

    Mutated from whichever thread the LLM call lands on. The GIL makes the
    ``+=`` on ints safe enough for counters we only read after the work is
    done, and a lock guards the compound updates anyway.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    estimated: bool = False          # True if any call fell back to guessing
    # Conversation total *after* this request was committed. Filled in by
    # UsageLedger.track() so handlers can report the meter without a second
    # round trip to Redis.
    conversation_total: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int, *, estimated: bool = False) -> None:
        with self._lock:
            self.input_tokens += max(0, int(input_tokens or 0))
            self.output_tokens += max(0, int(output_tokens or 0))
            self.calls += 1
            self.estimated = self.estimated or estimated

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
            "llm_calls": self.calls,
            "estimated": self.estimated,
        }


_current: ContextVar[Optional[UsageAccumulator]] = ContextVar("graph_rag_usage", default=None)


@contextmanager
def bind(acc: Optional[UsageAccumulator] = None) -> Iterator[UsageAccumulator]:
    """Make ``acc`` the accumulator for this context (and any thread it's
    copied into). Yields it so callers can read the totals afterwards."""
    acc = acc if acc is not None else UsageAccumulator()
    token = _current.set(acc)
    try:
        yield acc
    finally:
        _current.reset(token)


def record(input_tokens: int, output_tokens: int, *, estimated: bool = False) -> None:
    """Called by the LLM clients after every completion. A no-op when
    nothing is bound, so library code stays usable outside a request."""
    acc = _current.get()
    if acc is not None:
        acc.add(input_tokens, output_tokens, estimated=estimated)


def record_openai(resp: Any) -> None:
    """Record usage from an OpenAI-compatible response or final stream chunk."""
    u = getattr(resp, "usage", None)
    if not u:
        return
    record(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


def record_anthropic(resp: Any) -> None:
    """Record usage from an Anthropic Messages response."""
    u = getattr(resp, "usage", None)
    if not u:
        return
    # Cache reads/writes are billed differently but still cost tokens, so
    # count them against the budget when the SDK reports them.
    input_tokens = (
        (getattr(u, "input_tokens", 0) or 0)
        + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        + (getattr(u, "cache_read_input_tokens", 0) or 0)
    )
    record(input_tokens, getattr(u, "output_tokens", 0) or 0)


def record_estimate(prompt_text: str, output_text: str) -> None:
    """Last-resort accounting when a provider streams without usage data.

    ~4 characters per token is the usual English rule of thumb. It is a
    guess, and `estimated` marks the request so the number isn't mistaken
    for billing truth — but a rough charge beats charging nothing.
    """
    record(len(prompt_text or "") // 4, len(output_text or "") // 4, estimated=True)


# ── Quota errors ──────────────────────────────────────────────────────


def meter(used: int, request_tokens: int = 0) -> Dict[str, Any]:
    """The shape the UI draws its budget bar from."""
    cap = session_token_cap()
    return {
        "used_tokens": used,
        "limit_tokens": cap,
        "remaining_tokens": max(0, cap - used) if cap > 0 else None,
        "request_tokens": request_tokens,
    }


class QuotaExceeded(Exception):
    """Raised by the ledger when a request would exceed the free tier."""

    def __init__(self, reason: str, message: str, detail: Dict[str, Any]):
        super().__init__(message)
        self.reason = reason          # "conversation_tokens" | "daily_conversations"
        self.message = message
        self.detail = detail


# ── The ledger (durable, shared) ──────────────────────────────────────


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# SADD-if-room, as one atomic step. Returns 1 when the session is allowed
# (already known, or newly admitted) and 0 when the daily cap is full.
_ADMIT_SESSION_LUA = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 1 then return 1 end
if redis.call('SCARD', KEYS[1]) >= tonumber(ARGV[2]) then return 0 end
redis.call('SADD', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return 1
"""


class UsageLedger:
    """Per-conversation token counters, shared across workers when possible."""

    def __init__(self, redis_url: Optional[str] = None):
        self._redis = self._connect(redis_url if redis_url is not None else os.environ.get("REDIS_URL"))
        self._admit = None
        if self._redis is not None:
            try:
                self._admit = self._redis.register_script(_ADMIT_SESSION_LUA)
            except Exception as e:                       # pragma: no cover
                LOG.warning("usage: could not register Lua script (%s); using non-atomic path", e)
        # In-process fallback: {key: (tokens, expires_at)} and {day_key: {sids}}
        self._mem_tokens: Dict[str, Tuple[int, float]] = {}
        self._mem_days: Dict[str, Tuple[set, float]] = {}
        self._mem_lock = threading.Lock()
        if self._redis is None:
            LOG.warning(
                "usage: no Redis (REDIS_URL unset or unreachable) — free-tier "
                "counters are per-process only. On serverless this means the "
                "cap resets on every cold start."
            )

    # -- connection ----------------------------------------------------

    @staticmethod
    def _connect(redis_url: Optional[str]):
        if not redis_url:
            return None
        if not any(redis_url.startswith(s) for s in ("redis://", "rediss://", "unix://")):
            LOG.warning("usage: REDIS_URL has no recognised scheme; counters stay in-process")
            return None
        try:
            import redis
            from .cache import _redact, _resolve_ca_bundle

            kwargs: Dict[str, Any] = {
                "decode_responses": True,
                "socket_connect_timeout": 2.0,
                "socket_timeout": 2.0,
            }
            if redis_url.startswith("rediss://"):
                ca = _resolve_ca_bundle()
                if ca:
                    kwargs["ssl_ca_certs"] = ca
                if os.environ.get("REDIS_SSL_INSECURE", "").lower() in {"1", "true", "yes"}:
                    kwargs["ssl_cert_reqs"] = None
            client = redis.from_url(redis_url, **kwargs)
            client.ping()
            LOG.info("usage: Redis connected (%s)", _redact(redis_url))
            return client
        except Exception as e:
            LOG.warning("usage: Redis unavailable (%s); counters stay in-process", e)
            return None

    @property
    def backing(self) -> str:
        return "redis" if self._redis is not None else "in-process"

    # -- keys ----------------------------------------------------------

    @staticmethod
    def _conv_key(user_id: str, session_id: str) -> str:
        return f"grag:quota:conv:{user_id}:{session_id}"

    @staticmethod
    def _day_key(user_id: str) -> str:
        return f"grag:quota:day:{user_id}:{_utc_day()}"

    # -- in-process helpers --------------------------------------------

    def _mem_get_tokens(self, key: str) -> int:
        with self._mem_lock:
            entry = self._mem_tokens.get(key)
            if not entry:
                return 0
            tokens, expires = entry
            if expires < time.time():
                self._mem_tokens.pop(key, None)
                return 0
            return tokens

    def _mem_add_tokens(self, key: str, n: int) -> int:
        with self._mem_lock:
            tokens, expires = self._mem_tokens.get(key, (0, 0.0))
            if expires < time.time():
                tokens = 0
            tokens += n
            self._mem_tokens[key] = (tokens, time.time() + _CONV_TTL_S)
            return tokens

    def _mem_admit(self, day_key: str, session_id: str, cap: int) -> bool:
        with self._mem_lock:
            sids, expires = self._mem_days.get(day_key, (set(), 0.0))
            if expires < time.time():
                sids, expires = set(), time.time() + _DAY_TTL_S
            if session_id in sids:
                self._mem_days[day_key] = (sids, expires)
                return True
            if len(sids) >= cap:
                self._mem_days[day_key] = (sids, expires)
                return False
            sids.add(session_id)
            self._mem_days[day_key] = (sids, expires)
            return True

    # -- public API ----------------------------------------------------

    def admit(self, user_id: str, session_id: str) -> None:
        """Check both caps before any tokens are spent. Raises QuotaExceeded.

        Called on the way *in* to every LLM endpoint. The conversation check
        is deliberately "have you already spent your budget?" rather than
        "will this request fit?" — we can't know a request's cost up front,
        so the last request of a conversation is allowed to overshoot the
        cap a little. Overshoot is bounded by the model's ``max_tokens``.
        """
        if quotas_disabled():
            return

        day_cap = sessions_per_day_cap()
        if day_cap > 0 and not self._admit_session(user_id, session_id, day_cap):
            raise QuotaExceeded(
                "daily_conversations",
                f"You've started your {day_cap} free conversations for today. "
                "They reset at midnight UTC.",
                {"conversations_limit": day_cap},
            )

        cap = session_token_cap()
        used = self.used(user_id, session_id)
        if cap > 0 and used >= cap:
            raise QuotaExceeded(
                "conversation_tokens",
                "This conversation has used up its free token budget. "
                "Start a new conversation to continue.",
                {"used_tokens": used, "limit_tokens": cap},
            )

    def _admit_session(self, user_id: str, session_id: str, cap: int) -> bool:
        day_key = self._day_key(user_id)
        if self._redis is None:
            return self._mem_admit(day_key, session_id, cap)
        try:
            if self._admit is not None:
                return bool(int(self._admit(keys=[day_key], args=[session_id, cap, _DAY_TTL_S])))
            # Non-atomic fallback if the script couldn't be registered.
            if self._redis.sismember(day_key, session_id):
                return True
            if int(self._redis.scard(day_key) or 0) >= cap:
                return False
            self._redis.sadd(day_key, session_id)
            self._redis.expire(day_key, _DAY_TTL_S)
            return True
        except Exception as e:
            # A flaky quota store must not take the service down. Fail open
            # on the *session* count (the token cap below still applies) and
            # make the degradation loud.
            LOG.warning("usage: session admission check failed (%s); allowing request", e)
            return True

    def used(self, user_id: str, session_id: str) -> int:
        key = self._conv_key(user_id, session_id)
        if self._redis is None:
            return self._mem_get_tokens(key)
        try:
            return int(self._redis.get(key) or 0)
        except Exception as e:
            LOG.warning("usage: read failed (%s); treating conversation as unused", e)
            return 0

    def commit(self, user_id: str, session_id: str, acc: UsageAccumulator) -> int:
        """Add this request's tokens to the conversation total."""
        if acc.total <= 0:
            return self.used(user_id, session_id)
        key = self._conv_key(user_id, session_id)
        if self._redis is None:
            return self._mem_add_tokens(key, acc.total)
        try:
            pipe = self._redis.pipeline()
            pipe.incrby(key, acc.total)
            pipe.expire(key, _CONV_TTL_S)
            total = int(pipe.execute()[0] or 0)
            return total
        except Exception as e:
            LOG.warning("usage: commit failed (%s); %d tokens not counted", e, acc.total)
            return self.used(user_id, session_id)

    def conversations_today(self, user_id: str) -> int:
        day_key = self._day_key(user_id)
        if self._redis is None:
            with self._mem_lock:
                sids, expires = self._mem_days.get(day_key, (set(), 0.0))
                return len(sids) if expires >= time.time() else 0
        try:
            return int(self._redis.scard(day_key) or 0)
        except Exception:
            return 0

    def snapshot(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """What the UI needs to draw a meter."""
        cap = session_token_cap()
        used = self.used(user_id, session_id)
        day_cap = sessions_per_day_cap()
        return {
            "enabled": not quotas_disabled(),
            "backing": self.backing,
            "session_id": session_id,
            "used_tokens": used,
            "limit_tokens": cap,
            "remaining_tokens": max(0, cap - used) if cap > 0 else None,
            "conversations_used": self.conversations_today(user_id),
            "conversations_limit": day_cap,
        }

    @contextmanager
    def track(self, user_id: str, session_id: str) -> Iterator[UsageAccumulator]:
        """Bind an accumulator for the request and commit it on the way out.

        Commits even when the handler raises, so a request that dies
        half-way through still pays for the tokens it burned.
        """
        acc = UsageAccumulator()
        with bind(acc):
            try:
                yield acc
            finally:
                acc.conversation_total = self.commit(user_id, session_id, acc)


_ledger: Optional[UsageLedger] = None
_ledger_lock = threading.Lock()


def get_ledger() -> UsageLedger:
    """Process-wide singleton. Built lazily so importing this module never
    opens a socket (matters for tests and for `python -c` smoke checks)."""
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = UsageLedger()
    return _ledger
