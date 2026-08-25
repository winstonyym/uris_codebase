"""Saved diagrams: private per-user documents and the public gallery.

Layout in R2, which is the source of truth:

    diagrams/<user_id>/<diagram_id>.json    private working copies
    gallery/<publication_id>.json           published snapshots

Three properties this module is built around:

**Publishing copies, it does not flag.** A publication is an immutable
snapshot with its own id. If the public gallery pointed at the owner's live
document, editing would silently rewrite what every other user sees,
deleting would break their gallery, and there would be nothing stable to
cite. The cost is that republishing makes a new entry; that is the intended
behaviour for a tool whose gallery carries cited models.

**Ownership is structural.** The private prefix is keyed on the verified
Clerk `sub`, so a caller can only ever address their own documents — there
is no code path where a user-supplied id selects someone else's prefix.
Publication keys embed a *hash* of the owner id, never the id itself, so
the public bucket leaks nothing about who published what beyond the display
name the doc carries.

**Listings are derived, never authoritative.** Both listings come from an
R2 prefix listing and are cached in Redis via `KVCache`. Losing Redis costs
one slow listing, never a diagram — which preserves the invariant the rest
of this codebase already relies on (`cache.py` and `usage.py` both degrade
to in-process when Redis is gone). Because publications are immutable,
their cards cache indefinitely; private listings carry a short TTL and are
invalidated explicitly on every write. Without Redis that invalidation only
reaches the process that served the write, so a second serverless instance
can serve a listing up to `ttl` seconds stale — harmless for a gallery, and
the reason the TTL is 60s rather than an hour.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src import r2_uploader
from src.cache import KVCache

LOG = logging.getLogger("graph_rag.diagrams")

# ── Limits ────────────────────────────────────────────────────────────
#
# The node/link caps are not arbitrary: GalleryPanel's DiagramPreview draws
# one SVG line per link for every visible card, so an unbounded diagram
# locks the browser of everyone browsing the gallery, not just its author.

def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default


def max_nodes() -> int:            return _int_env("DIAGRAM_MAX_NODES", 2000)
def max_links() -> int:            return _int_env("DIAGRAM_MAX_LINKS", 5000)
def max_bytes() -> int:            return _int_env("DIAGRAM_MAX_BYTES", 1_000_000)
def max_per_user() -> int:         return _int_env("DIAGRAMS_PER_USER", 50)
def max_publications_day() -> int: return _int_env("PUBLICATIONS_PER_DAY", 5)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PUB_ID_RE = re.compile(r"^[0-9]{8}-[0-9a-f]{12}-[0-9a-f]{8}$")


class StoreError(Exception):
    """A diagram operation that failed for a reason the caller should see.

    Carries the HTTP status app.py should return, so the storage layer
    stays free of FastAPI imports while still expressing "this is a 404,
    not a 500".
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ── Caches ────────────────────────────────────────────────────────────

_listing_cache: Optional[KVCache] = None
_card_cache: Optional[KVCache] = None


def _listings() -> KVCache:
    global _listing_cache
    if _listing_cache is None:
        _listing_cache = KVCache(
            redis_url=os.environ.get("REDIS_URL"), namespace="diaglist", ttl=60, lru_size=256,
        )
    return _listing_cache


def _cards() -> KVCache:
    global _card_cache
    if _card_cache is None:
        # Publications are immutable, so a card can never go stale — the
        # only reason for any TTL at all is to reclaim space.
        _card_cache = KVCache(
            redis_url=os.environ.get("REDIS_URL"), namespace="diagcard", ttl=86400, lru_size=1024,
        )
    return _card_cache


# ── Keys ──────────────────────────────────────────────────────────────

def _check_id(value: str, what: str) -> str:
    value = (value or "").strip()
    if not _ID_RE.match(value):
        raise StoreError(400, f"{what} must be 1-64 characters of A-Z, a-z, 0-9, _ or -.")
    return value


def _owner_hash(user_id: str) -> str:
    """Stable 12-hex digest of a Clerk user id.

    Used in publication keys so a daily publish count is a prefix listing
    with no object reads, without putting the raw Clerk id in a key that
    anyone browsing the public gallery could enumerate.
    """
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _user_prefix(user_id: str) -> str:
    return f"diagrams/{_check_id(user_id, 'user id')}/"


def _diagram_key(user_id: str, diagram_id: str) -> str:
    return f"{_user_prefix(user_id)}{_check_id(diagram_id, 'diagram id')}.json"


def _publication_key(publication_id: str) -> str:
    pid = (publication_id or "").strip()
    if not _PUB_ID_RE.match(pid):
        raise StoreError(400, "Malformed publication id.")
    return f"gallery/{pid}.json"


def _new_diagram_id() -> str:
    return f"d{secrets.token_hex(10)}"


def _new_publication_id(user_id: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{day}-{_owner_hash(user_id)}-{secrets.token_hex(4)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Validation ────────────────────────────────────────────────────────

def validate_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Check an incoming {nodes, links, meta} body and return it normalised.

    Rejects dangling links rather than storing them: the frontend's
    `importDiagram` silently drops any link whose endpoints it can't
    resolve, so a diagram saved with broken links would come back from the
    gallery quietly missing edges, which is far harder to diagnose than a
    400 at save time.
    """
    if not isinstance(data, dict):
        raise StoreError(400, "data must be an object with `nodes` and `links`.")

    nodes = data.get("nodes")
    links = data.get("links") if data.get("links") is not None else data.get("edges")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise StoreError(400, "data.nodes and data.links must both be arrays.")
    if not nodes:
        raise StoreError(400, "A diagram needs at least one node.")
    if len(nodes) > max_nodes():
        raise StoreError(413, f"Too many nodes ({len(nodes)}); the limit is {max_nodes()}.")
    if len(links) > max_links():
        raise StoreError(413, f"Too many links ({len(links)}); the limit is {max_links()}.")

    seen: set = set()
    clean_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            raise StoreError(400, "Every node must be an object.")
        nid = str(n.get("id") or "").strip()
        label = str(n.get("label") or "").strip()
        if not nid or not label:
            raise StoreError(400, "Every node needs a non-empty `id` and `label`.")
        if nid in seen:
            raise StoreError(400, f"Duplicate node id: {nid}")
        seen.add(nid)
        clean_nodes.append({
            "id": nid,
            "label": label[:200],
            "subsystem": str(n.get("subsystem") or "Others")[:60],
            "description": str(n.get("description") or "")[:2000],
            **({"x": float(n["x"])} if isinstance(n.get("x"), (int, float)) else {}),
            **({"y": float(n["y"])} if isinstance(n.get("y"), (int, float)) else {}),
        })

    clean_links: List[Dict[str, Any]] = []
    for l in links:
        if not isinstance(l, dict):
            raise StoreError(400, "Every link must be an object.")
        s, t = str(l.get("source") or ""), str(l.get("target") or "")
        if s not in seen or t not in seen:
            raise StoreError(400, f"Link {s or '?'} → {t or '?'} references a node that isn't in the diagram.")
        polarity = "-" if l.get("polarity") == "-" else "+"
        clean_links.append({
            "source": s,
            "target": t,
            "polarity": polarity,
            "label": str(l.get("label") or ("decreases" if polarity == "-" else "increases"))[:120],
            "description": str(l.get("description") or "")[:2000],
            **({"evidence": str(l["evidence"])[:400]} if l.get("evidence") else {}),
        })

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else None
    out: Dict[str, Any] = {"nodes": clean_nodes, "links": clean_links}
    if meta:
        out["meta"] = meta
    return out


# ── Projection ────────────────────────────────────────────────────────

def _card(doc: Dict[str, Any], *, public: bool) -> Dict[str, Any]:
    """The listing view of a document: everything a gallery card needs, no graph.

    `subsystems` are the raw authored values; the frontend already runs
    them through `normalizeSubsystem` in `itemSubsystems`, so mapping here
    would just duplicate that (and drift from it).
    """
    data = doc.get("data") or {}
    nodes = data.get("nodes") or []
    subsystems = sorted({str(n.get("subsystem") or "Others") for n in nodes})
    card = {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "source": doc.get("source"),
        "note": doc.get("note"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "node_count": len(nodes),
        "link_count": len(data.get("links") or []),
        "subsystems": subsystems,
    }
    if public:
        # Attribution is the display name the publisher chose. The Clerk id
        # stays in the stored doc for ownership checks and never ships.
        card["author"] = doc.get("author") or "Anonymous"
        card["published_at"] = doc.get("created_at")
    else:
        card["visibility"] = doc.get("visibility", "private")
        card["publications"] = doc.get("publications") or []
    return card


def _get_many(keys: List[str]) -> List[Dict[str, Any]]:
    """Fetch several objects concurrently, dropping any that have vanished.

    Sequential GETs are what makes a cold listing feel broken on
    serverless; boto3 clients are thread-safe, so a small pool is the whole
    fix.
    """
    if not keys:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(keys))) as pool:
        docs = list(pool.map(r2_uploader.read_json, keys))
    return [d for d in docs if isinstance(d, dict)]


# ── Private diagrams ──────────────────────────────────────────────────

def list_mine(user_id: str) -> List[Dict[str, Any]]:
    """This user's saved diagrams, newest first, as cards."""
    cache_key = f"mine:{_owner_hash(user_id)}"
    hit = _listings().get(cache_key)
    if hit is not None:
        return hit

    entries = r2_uploader.list_objects(_user_prefix(user_id))
    entries.sort(key=lambda e: e["last_modified"], reverse=True)
    cards = [_card(d, public=False) for d in _get_many([e["key"] for e in entries])]
    cards.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
    _listings().set(cache_key, cards)
    return cards


def _invalidate_mine(user_id: str) -> None:
    _listings().delete(f"mine:{_owner_hash(user_id)}")


def get_mine(user_id: str, diagram_id: str) -> Dict[str, Any]:
    doc = r2_uploader.read_json(_diagram_key(user_id, diagram_id))
    if not isinstance(doc, dict):
        raise StoreError(404, "Diagram not found.")
    return doc


def save_mine(
    user_id: str,
    *,
    diagram_id: Optional[str],
    title: str,
    data: Dict[str, Any],
    source: Optional[str] = None,
    note: Optional[str] = None,
    author: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or overwrite one of this user's diagrams. Returns the stored doc."""
    title = (title or "").strip()
    if not title:
        raise StoreError(400, "A title is required.")
    clean = validate_payload(data)

    existing: Optional[Dict[str, Any]] = None
    if diagram_id:
        did = _check_id(diagram_id, "diagram id")
        existing = r2_uploader.read_json(_diagram_key(user_id, did))
    else:
        did = _new_diagram_id()

    # The cap counts *new* objects only, so overwriting your own diagram
    # keeps working at the limit — otherwise hitting the cap would lock you
    # out of editing what you already have. Note this checks `existing`,
    # not whether an id was supplied: a client that invents its own ids
    # must not be able to slip past the cap by always sending one.
    if existing is None and len(r2_uploader.list_objects(_user_prefix(user_id))) >= max_per_user():
        raise StoreError(
            409,
            f"You've reached the limit of {max_per_user()} saved diagrams. "
            "Delete one to make room.",
        )

    now = _now()
    doc = {
        "id": did,
        "owner_id": user_id,
        "author": (author or (existing or {}).get("author") or "").strip()[:80] or None,
        "title": title[:200],
        "source": (source or "").strip()[:300] or None,
        "note": (note or "").strip()[:2000] or None,
        "visibility": "private",
        "created_at": (existing or {}).get("created_at") or now,
        "updated_at": now,
        "publications": (existing or {}).get("publications") or [],
        "data": clean,
    }

    written = r2_uploader.put_json(_diagram_key(user_id, did), doc)
    if written > max_bytes():
        # Roll back rather than leave an oversized object behind. Checking
        # after serialising is the only way to know the real size, and R2
        # has no conditional put to lean on.
        try:
            r2_uploader.delete_object(_diagram_key(user_id, did))
        except Exception:
            LOG.warning("could not roll back oversized diagram %s", did, exc_info=True)
        raise StoreError(413, f"Diagram is {written} bytes; the limit is {max_bytes()}.")

    _invalidate_mine(user_id)
    return doc


def delete_mine(user_id: str, diagram_id: str) -> None:
    """Delete a private diagram. Published snapshots of it are left alone.

    That is deliberate: a publication is an independent citable object, and
    deleting your working copy should not silently retract something others
    may already have imported. Unpublish is a separate, explicit act.
    """
    key = _diagram_key(user_id, diagram_id)
    if r2_uploader.read_json(key) is None:
        raise StoreError(404, "Diagram not found.")
    r2_uploader.delete_object(key)
    _invalidate_mine(user_id)


# ── Public gallery ────────────────────────────────────────────────────

def _published_today(user_id: str) -> int:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return len(r2_uploader.list_objects(f"gallery/{day}-{_owner_hash(user_id)}-"))


def publish(user_id: str, diagram_id: str, *, author: Optional[str] = None) -> Dict[str, Any]:
    """Snapshot one of this user's diagrams into the public gallery."""
    cap = max_publications_day()
    if cap > 0 and _published_today(user_id) >= cap:
        raise StoreError(429, f"You can publish {cap} diagrams per day. Try again tomorrow.")

    src = get_mine(user_id, diagram_id)
    pub_id = _new_publication_id(user_id)
    now = _now()
    pub = {
        "id": pub_id,
        "owner_id": user_id,                     # for ownership checks; never in a card
        "author": (author or src.get("author") or "").strip()[:80] or "Anonymous",
        "title": src.get("title"),
        "source": src.get("source"),
        "note": src.get("note"),
        "visibility": "public",
        "status": "listed",                      # flip to "hidden" to unlist without deleting
        "published_from": diagram_id,
        "created_at": now,
        "updated_at": now,
        "data": src.get("data") or {},
    }
    r2_uploader.put_json(_publication_key(pub_id), pub)

    # Record the publication on the working copy so the owner's card can
    # show "published" without a second listing.
    src.setdefault("publications", [])
    src["publications"] = [*src["publications"], {"id": pub_id, "published_at": now}][-20:]
    src["updated_at"] = now
    try:
        r2_uploader.put_json(_diagram_key(user_id, diagram_id), src)
    except Exception:
        # The snapshot is the thing that matters and it is already stored.
        LOG.warning("publish: could not back-reference %s on the source doc", pub_id, exc_info=True)

    _invalidate_mine(user_id)
    _listings().delete("gallery:index")
    return pub


def unpublish(user_id: str, publication_id: str, *, is_admin: bool = False) -> None:
    """Remove a publication. Only its owner (or an admin) may do so."""
    key = _publication_key(publication_id)
    doc = r2_uploader.read_json(key)
    if not isinstance(doc, dict):
        raise StoreError(404, "Publication not found.")
    # Checked against the stored doc, not the key — the owner hash in the
    # key is for cheap counting, not for authorisation.
    if not is_admin and doc.get("owner_id") != user_id:
        raise StoreError(403, "You can only unpublish your own diagrams.")
    r2_uploader.delete_object(key)
    _cards().delete(publication_id)
    _listings().delete("gallery:index")
    _invalidate_mine(user_id)


def list_public(limit: int = 24, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """A page of the public gallery, newest first, plus the total count.

    The listing (keys + timestamps) is cached and paginated *before* any
    object is read, so a gallery of a thousand publications still costs one
    listing plus `limit` small GETs — and those GETs are themselves cached
    forever, because a publication never changes.
    """
    limit = max(1, min(int(limit or 24), 100))
    offset = max(0, int(offset or 0))

    entries = _listings().get("gallery:index")
    if entries is None:
        entries = [
            {"key": e["key"], "last_modified": e["last_modified"]}
            for e in r2_uploader.list_objects("gallery/")
            if e["key"].endswith(".json")
        ]
        entries.sort(key=lambda e: e["last_modified"], reverse=True)
        _listings().set("gallery:index", entries)

    total = len(entries)
    page = entries[offset:offset + limit]

    cards: List[Dict[str, Any]] = []
    misses: List[str] = []
    for e in page:
        pid = e["key"].rsplit("/", 1)[-1][:-5]
        hit = _cards().get(pid)
        if hit is not None:
            cards.append(hit)
        else:
            misses.append(e["key"])

    for doc in _get_many(misses):
        if doc.get("status") == "hidden":
            continue
        card = _card(doc, public=True)
        _cards().set(str(doc.get("id")), card)
        cards.append(card)

    cards.sort(key=lambda c: c.get("published_at") or "", reverse=True)
    return cards, total


def get_public(publication_id: str) -> Dict[str, Any]:
    """One publication, with its full graph. Hidden entries read as absent."""
    doc = r2_uploader.read_json(_publication_key(publication_id))
    if not isinstance(doc, dict) or doc.get("status") == "hidden":
        raise StoreError(404, "Publication not found.")
    # owner_id is an internal field; strip it on the way out.
    return {k: v for k, v in doc.items() if k != "owner_id"}
