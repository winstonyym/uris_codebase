"""Cloudflare R2 (S3-compatible) uploader for activity logs.

Optional add-on to the local JSONL logger. When all required env vars
are present AND `boto3` is installed, every successful append to a
session's local log file is followed by a background upload of the
full JSONL to R2. If anything's missing the module is a no-op and the
local-only path keeps working — never breaks the request.

Required env vars:
    R2_ACCOUNT_ID         — bucket account; forms the S3 endpoint URL
    R2_ACCESS_KEY_ID      — S3 access key (R2 → Manage R2 → Manage Tokens)
    R2_SECRET_ACCESS_KEY  — matching secret
    R2_BUCKET_NAME        — destination bucket

Notes:
  * The `R2_TOKEN_VALUE` env var the user set is Cloudflare's *API*
    token (used for the Cloudflare REST control plane). The S3-compatible
    data plane that boto3 talks to uses the access key / secret pair
    above — the token itself isn't needed here. We leave the env var
    alone so it can authenticate other Cloudflare API calls later.
  * R2 doesn't support S3 append; we upload the WHOLE session file on
    every batch (PUT-with-overwrite). For typical sessions (KB–low MB)
    this is fine. If you ever need partial uploads, switch to writing
    each batch as a separate object under `sessions/<id>/batches/`.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("graph_rag.r2")


# ── lazy-init state ──────────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()
_disabled = False                  # latched True if env or boto3 are missing


def _required_env() -> Optional[dict]:
    """Return the four required env vars as a dict, or None if any is empty."""
    keys = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME")
    out: dict = {}
    for k in keys:
        v = (os.environ.get(k) or "").strip()
        if not v:
            return None
        out[k] = v
    return out


def _get_client():
    """Return a boto3 S3 client configured for R2, or None if disabled."""
    global _client, _disabled
    if _disabled:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        env = _required_env()
        if not env:
            LOG.info("R2 env vars missing — log sync to R2 disabled.")
            _disabled = True
            return None
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            LOG.info("boto3 not installed — R2 sync disabled.")
            _disabled = True
            return None
        try:
            endpoint = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
            _client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
                region_name="auto",  # R2 ignores the region but boto3 wants one
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=5,
                    read_timeout=10,
                ),
            )
            LOG.info(
                "R2 client ready (bucket=%s, endpoint=%s)",
                env["R2_BUCKET_NAME"], endpoint,
            )
            return _client
        except Exception as e:
            LOG.warning("R2 client init failed (%s); R2 sync disabled.", e)
            _disabled = True
            return None


# ── public API ──────────────────────────────────────────────────────

def is_enabled() -> bool:
    """True iff env vars present, boto3 installed, and client init succeeded."""
    return _get_client() is not None


def status() -> dict:
    """Diagnostic surface for /health & /stats."""
    enabled = is_enabled()
    bucket = (os.environ.get("R2_BUCKET_NAME") or "").strip()
    return {
        "enabled": enabled,
        "bucket": bucket if enabled else None,
        "account_id_set": bool((os.environ.get("R2_ACCOUNT_ID") or "").strip()),
        "credentials_set": bool(
            (os.environ.get("R2_ACCESS_KEY_ID") or "").strip()
            and (os.environ.get("R2_SECRET_ACCESS_KEY") or "").strip()
        ),
    }


def _safe(s: str, limit: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s or "")[:limit] or "anon"


def build_session_key(user_id: str, session_id: str, started_at: Optional[str] = None) -> str:
    """Object key under which this session's JSONL is stored.

    Layout: `sessions/<YYYY-MM-DD>/<user_id>_<session_id>.json`

    Stored as a single pretty-printed JSON document (not JSONL) so the
    object previews cleanly in the Cloudflare R2 dashboard.

    Date-partitioning by `started_at` (falls back to today UTC) makes
    the bucket browsable as it grows. The same key is reused on every
    batch upload for a given session, so the R2 object always reflects
    the latest server-side state.
    """
    if started_at:
        try:
            day = started_at.split("T", 1)[0]
            # Sanity-check it really looks like YYYY-MM-DD; fall back if not.
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
                day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"sessions/{day}/{_safe(user_id)}_{_safe(session_id)}.json"


def read_object(key: str) -> Optional[bytes]:
    """Fetch an object's bytes from R2, or None if missing/unavailable.

    Used to seed the per-session JSON document when this (possibly cold)
    instance has no local copy yet, so accumulated events survive across
    serverless instances. Never raises.
    """
    client = _get_client()
    if client is None:
        return None
    bucket = (os.environ.get("R2_BUCKET_NAME") or "").strip()
    if not bucket:
        return None
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except Exception as e:  # includes NoSuchKey
        LOG.debug("R2 get miss (%s): %s", key, e)
        return None


def upload_file(local_path: Path, key: str) -> bool:
    """Upload `local_path` to R2 under `key`. Overwrites if it exists.

    Returns True on success, False otherwise. NEVER raises — caller can
    safely fire-and-forget this from a background task.
    """
    client = _get_client()
    if client is None:
        return False
    bucket = (os.environ.get("R2_BUCKET_NAME") or "").strip()
    if not bucket:
        return False
    try:
        local_path = Path(local_path)
        if not local_path.exists():
            LOG.warning("R2 upload skipped — file gone: %s", local_path)
            return False
        with local_path.open("rb") as f:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=f.read(),
                # JSON (not NDJSON) so the object previews in the R2 dashboard.
                ContentType="application/json",
                # Cache-Control: telemetry shouldn't be served, but in
                # case anyone wires up a public view, keep it private.
                CacheControl="no-store",
            )
        LOG.debug("R2 PUT %s → s3://%s/%s", local_path.name, bucket, key)
        return True
    except Exception as e:
        LOG.warning("R2 upload failed (%s → %s): %s", local_path, key, e)
        return False
