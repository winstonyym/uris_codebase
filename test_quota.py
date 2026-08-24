"""Focused checks on the free-tier plumbing.

Runs without Neo4j, Redis or a real Clerk instance: the ledger's
in-process backing is exercised directly, and the FastAPI wiring is
rebuilt in miniature so the dependency → exception-handler path can be
verified end to end.

    python3 test_quota.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("FREE_TIER_SESSION_TOKENS", "1000")
os.environ.setdefault("FREE_TIER_SESSIONS_PER_DAY", "2")
os.environ.pop("REDIS_URL", None)

from fastapi import Depends, FastAPI, Request, Response          # noqa: E402
from fastapi.responses import JSONResponse                        # noqa: E402
from fastapi.testclient import TestClient                         # noqa: E402

from src import usage                                             # noqa: E402
from src.auth import ClerkUser, conversation_id, require_user     # noqa: E402
from src.usage import QuotaExceeded, UsageLedger                  # noqa: E402

FAILURES = []


def check(label: str, cond: bool, extra: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  — ' + extra if extra and not cond else ''}")
    if not cond:
        FAILURES.append(label)


# ── 1. accumulator + contextvar ───────────────────────────────────────

print("\n[1] accumulator")


class _FakeOpenAIUsage:
    prompt_tokens = 120
    completion_tokens = 30


class _FakeOpenAIResp:
    usage = _FakeOpenAIUsage()


class _FakeAnthropicUsage:
    input_tokens = 200
    output_tokens = 50
    cache_read_input_tokens = 10
    cache_creation_input_tokens = 0


class _FakeAnthropicResp:
    usage = _FakeAnthropicUsage()


with usage.bind() as acc:
    usage.record_openai(_FakeOpenAIResp())
    usage.record_anthropic(_FakeAnthropicResp())
check("openai + anthropic usage accumulates", acc.total == 150 + 260, f"got {acc.total}")
check("call count", acc.calls == 2, f"got {acc.calls}")
check("not marked estimated", acc.estimated is False)

with usage.bind() as acc2:
    usage.record_estimate("x" * 400, "y" * 80)
check("estimate ≈ chars/4", acc2.total == 120, f"got {acc2.total}")
check("estimate flagged", acc2.estimated is True)

# Outside a bind, recording is a no-op rather than an error.
usage.record(10, 10)
check("record outside bind is inert", True)

# Threads inherit the binding (this is what makes anyio.to_thread work).
import threading                                                   # noqa: E402
import contextvars                                                 # noqa: E402

with usage.bind() as acc3:
    ctx = contextvars.copy_context()
    t = threading.Thread(target=lambda: ctx.run(usage.record, 7, 3))
    t.start()
    t.join()
check("worker thread charges the parent's accumulator", acc3.total == 10, f"got {acc3.total}")


# ── 2. ledger, in-process backing ─────────────────────────────────────

print("\n[2] ledger")

ledger = UsageLedger(redis_url=None)
check("falls back to in-process", ledger.backing == "in-process")

ledger.admit("user_a", "conv1")
spend = usage.UsageAccumulator()
spend.add(600, 0)
ledger.commit("user_a", "conv1", spend)
check("tokens recorded", ledger.used("user_a", "conv1") == 600)

ledger.admit("user_a", "conv1")          # 600 < 1000, still allowed
more = usage.UsageAccumulator()
more.add(500, 0)
ledger.commit("user_a", "conv1", more)
check("second turn allowed under cap, overshoot lands at 1100",
      ledger.used("user_a", "conv1") == 1100)

try:
    ledger.admit("user_a", "conv1")
    check("third turn blocked", False, "no QuotaExceeded raised")
except QuotaExceeded as e:
    check("third turn blocked", e.reason == "conversation_tokens", e.reason)

# A fresh conversation gets a fresh budget…
ledger.admit("user_a", "conv2")
check("second conversation admitted", True)
# …but only up to the daily cap of 2.
try:
    ledger.admit("user_a", "conv3")
    check("daily conversation cap enforced", False, "third conversation admitted")
except QuotaExceeded as e:
    check("daily conversation cap enforced", e.reason == "daily_conversations", e.reason)

# Another user is unaffected.
ledger.admit("user_b", "conv1")
check("budgets are per user", ledger.used("user_b", "conv1") == 0)

snap = ledger.snapshot("user_a", "conv1")
check("snapshot reports exhaustion", snap["remaining_tokens"] == 0, str(snap))
check("snapshot counts conversations", snap["conversations_used"] == 2, str(snap))

# track() commits on the way out, even when the handler raises.
try:
    with ledger.track("user_c", "conv1") as tracked:
        usage.record(40, 10)
        raise RuntimeError("boom")
except RuntimeError:
    pass
check("track() charges tokens spent before a failure",
      ledger.used("user_c", "conv1") == 50, str(ledger.used("user_c", "conv1")))
check("track() reports the running total", tracked.conversation_total == 50)

# The daily cap can be turned off entirely.
os.environ["FREE_TIER_DISABLED"] = "1"
ledger.admit("user_a", "conv1")           # would otherwise raise
check("FREE_TIER_DISABLED lifts the caps", True)
del os.environ["FREE_TIER_DISABLED"]


# ── 3. FastAPI wiring ─────────────────────────────────────────────────

print("\n[3] fastapi wiring")

os.environ["CLERK_AUTH_DISABLED"] = ""     # exercise the real 401 path
app = FastAPI()
test_ledger = UsageLedger(redis_url=None)


class Caller:
    def __init__(self, user: ClerkUser, conversation: str):
        self.user, self.conversation = user, conversation

    @property
    def user_id(self) -> str:
        return self.user.user_id


def billed_caller(request: Request, user: ClerkUser = Depends(require_user)) -> Caller:
    conversation = conversation_id(request, user)
    test_ledger.admit(user.user_id, conversation)
    return Caller(user, conversation)


@app.exception_handler(QuotaExceeded)
async def _quota(request: Request, exc: QuotaExceeded) -> JSONResponse:
    body = {"error": "quota_exceeded", "reason": exc.reason, "detail": exc.message}
    body.update(exc.detail)
    return JSONResponse(status_code=429, content=body)


@app.post("/chat")
def chat(response: Response, caller: Caller = Depends(billed_caller)):
    with test_ledger.track(caller.user_id, caller.conversation) as acc:
        usage.record(400, 100)
    m = usage.meter(acc.conversation_total, acc.total)
    response.headers["X-Usage-Remaining"] = str(m["remaining_tokens"])
    return {"ok": True, "conversation": caller.conversation}


client = TestClient(app)

r = client.post("/chat", json={})
check("no token → 401", r.status_code == 401, f"got {r.status_code}")
check("401 names the fix", "sign in" in r.json()["detail"].lower(), r.text)

r = client.post("/chat", json={}, headers={"Authorization": "Basic abc"})
check("non-bearer scheme → 401", r.status_code == 401, f"got {r.status_code}")

# With auth bypassed, the quota path is still live.
os.environ["CLERK_AUTH_DISABLED"] = "1"
h = {"Authorization": "Bearer whatever", "X-Session-Id": "sess-xyz"}
r = client.post("/chat", json={}, headers=h)
check("dev bypass lets the request through", r.status_code == 200, r.text)
check("conversation comes from X-Session-Id", r.json()["conversation"] == "sess-xyz", r.text)
check("usage header set", r.headers.get("X-Usage-Remaining") == "500", r.headers.get("X-Usage-Remaining"))

r = client.post("/chat", json={}, headers=h)          # total now 1000 = cap
check("second turn still allowed", r.status_code == 200, r.text)

r = client.post("/chat", json={}, headers=h)          # blocked before spending
check("cap → 429", r.status_code == 429, f"got {r.status_code}")
body = r.json()
check("429 body is machine-readable", body["error"] == "quota_exceeded"
      and body["reason"] == "conversation_tokens", r.text)
check("429 body carries the numbers", body["limit_tokens"] == 1000, r.text)

# A header full of junk can't escape the budget or poison a Redis key.
r = client.post("/chat", json={}, headers={"Authorization": "Bearer x",
                                           "X-Session-Id": "../../etc/passwd\n\r"})
check("session id is sanitised", r.json()["conversation"] == "....etcpasswd", r.text)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all checks passed")
