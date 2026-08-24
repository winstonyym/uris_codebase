"""Clerk authentication for the Graph-RAG service.

Every endpoint that can spend an LLM token is now behind a verified Clerk
session token. The browser gets one from ``getToken()`` and sends it as
``Authorization: Bearer <jwt>``; we verify the signature, expiry and
authorized party here before any model is called.

Verification runs one of two ways:

  * ``CLERK_JWT_KEY`` set — networkless. Paste the PEM public key from the
    Clerk dashboard (API keys → JWT public key). Preferred on serverless:
    no JWKS fetch on a cold start.
  * ``CLERK_SECRET_KEY`` set — the SDK fetches and caches your instance's
    JWKS. One extra round trip per cold instance, nothing after that.

Environment:
    CLERK_SECRET_KEY            sk_test_… / sk_live_…
    CLERK_JWT_KEY               (optional) PEM public key, networkless path
    CLERK_AUTHORIZED_PARTIES    comma-separated origins allowed to mint
                                tokens for this API, e.g.
                                "https://your-app.vercel.app,http://localhost:5173"
    CLERK_AUTH_DISABLED         set to 1 to bypass auth entirely — local
                                development only; the service refuses to
                                treat it as a production configuration.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request

LOG = logging.getLogger("graph_rag.auth")

_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9_:.-]")


@dataclass(frozen=True)
class ClerkUser:
    """The verified caller."""

    user_id: str                      # Clerk `sub` — stable, e.g. user_2ab…
    session_id: Optional[str]         # Clerk's own session id (`sid`)
    email: Optional[str]
    claims: Dict[str, Any]

    @property
    def is_anonymous(self) -> bool:
        return self.user_id.startswith("dev_")


def auth_disabled() -> bool:
    return str(os.environ.get("CLERK_AUTH_DISABLED", "")).lower() in {"1", "true", "yes"}


def authorized_parties() -> Optional[List[str]]:
    """Origins allowed to present tokens to this API.

    Clerk puts the minting origin in the token's ``azp`` claim; checking it
    is what stops a token issued for someone else's app (or a phishing
    origin running your publishable key) from working against your API.
    Returns None when unset, which skips the check — fine for a first
    deploy, worth setting before you go public.
    """
    raw = (os.environ.get("CLERK_AUTHORIZED_PARTIES") or "").strip()
    if not raw:
        return None
    parties = [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]
    return parties or None


def _bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def require_user(request: Request) -> ClerkUser:
    """FastAPI dependency: verify the Clerk session token or 401.

    Deliberately a sync ``def`` — ``verify_token`` may do a blocking JWKS
    fetch, and FastAPI runs sync dependencies in its threadpool rather than
    on the event loop.
    """
    if auth_disabled():
        LOG.debug("auth: CLERK_AUTH_DISABLED is set — request accepted unverified")
        return ClerkUser(user_id="dev_local", session_id=None, email=None, claims={})

    token = _bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Sign in to use this feature.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    secret_key = (os.environ.get("CLERK_SECRET_KEY") or "").strip()
    jwt_key = (os.environ.get("CLERK_JWT_KEY") or "").strip()
    if not secret_key and not jwt_key:
        # Fail closed. A deploy missing its Clerk config must not quietly
        # become an open proxy to the operator's API keys.
        LOG.error("auth: neither CLERK_SECRET_KEY nor CLERK_JWT_KEY is set")
        raise HTTPException(status_code=503, detail="Authentication is not configured.")

    try:
        from clerk_backend_api.security import verify_token
        from clerk_backend_api.security.types import VerifyTokenOptions

        claims = verify_token(
            token,
            VerifyTokenOptions(
                secret_key=secret_key or None,
                jwt_key=jwt_key or None,
                authorized_parties=authorized_parties(),
            ),
        )
    except ImportError:                                   # pragma: no cover
        LOG.exception("auth: clerk-backend-api is not installed")
        raise HTTPException(status_code=503, detail="Authentication is not configured.")
    except Exception as e:
        # Expired tokens are the common case and are not worth a stack trace.
        LOG.info("auth: token rejected (%s)", e)
        raise HTTPException(
            status_code=401,
            detail="Your session has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = str(claims.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Token is missing a subject claim.")

    return ClerkUser(
        user_id=user_id,
        session_id=claims.get("sid"),
        email=claims.get("email") or (claims.get("user") or {}).get("email"),
        claims=claims,
    )


def conversation_id(request: Request, user: ClerkUser) -> str:
    """Which conversation this request belongs to.

    The browser mints a per-visit session id (``src/lib/session.js``) and
    sends it as ``X-Session-Id``; that's what the token budget is charged
    against. It's client-controlled, which is exactly why the ledger also
    caps how many conversations a user may open per day — see
    ``usage.UsageLedger.admit``.

    Falls back to Clerk's own session id, then to the user id, so a client
    that forgets the header still gets *a* budget rather than an unlimited
    one.
    """
    raw = (request.headers.get("x-session-id") or "").strip()
    if raw:
        cleaned = _SESSION_ID_RE.sub("", raw)[:80]
        if cleaned:
            return cleaned
    return str(user.session_id or user.user_id)


def allowed_origins() -> List[str]:
    """CORS allowlist, from ``ALLOWED_ORIGINS`` (comma-separated).

    Defaults to ``["*"]`` so an existing deploy doesn't break the moment
    this ships — but a wide-open API that now requires a bearer token is a
    very different thing from one that didn't, and narrowing this is still
    worth doing.
    """
    raw = (os.environ.get("ALLOWED_ORIGINS") or "").strip()
    if not raw:
        LOG.warning("ALLOWED_ORIGINS is unset — CORS stays open to '*'")
        return ["*"]
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or ["*"]
