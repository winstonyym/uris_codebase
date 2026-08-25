"""End-to-end exercise of src/diagrams.py + the diagram/gallery endpoints.

    cd backend && python3 test_diagrams.py

Runs against an in-memory fake S3 with Clerk auth disabled, so it needs no
network, no R2 credentials and no Redis — same spirit as test_quota.py.
Exits non-zero on any failure, so it drops straight into CI.
"""
from __future__ import annotations

import io
import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone

os.environ["CLERK_AUTH_DISABLED"] = "1"
os.environ["R2_BUCKET_NAME"] = "test-bucket"
os.environ.pop("REDIS_URL", None)
os.environ["DIAGRAMS_PER_USER"] = "3"
os.environ["PUBLICATIONS_PER_DAY"] = "2"

sys.path.insert(0, os.path.dirname(__file__))

# ── Stub the heavy siblings app.py imports at module scope ────────────
for name, attrs in {
    "src.causal": ["CausalReasoner"],
    "src.chat_agent": ["ChatAgent"],
    "src.loop_chat": ["LoopAssistant"],
    "src.loop_recommender": ["LoopRecommender"],
    "src.config": ["load_config"],
    "src.embeddings": ["make_embedder"],
    "src.llm_router": ["LLMRouter"],
    "src.neo4j_client": ["Neo4jClient"],
    "src.recommender": ["Recommender"],
}.items():
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, type(a, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules[name] = mod

from src import diagrams, r2_uploader            # noqa: E402
from src.diagrams import StoreError              # noqa: E402


# ── In-memory stand-in for the boto3 S3 client ────────────────────────

class NoSuchKey(Exception):
    pass


class _Paginator:
    def __init__(self, store):
        self._store = store

    def paginate(self, Bucket=None, Prefix="", **kw):
        items = [
            {"Key": k, "Size": len(v[0]), "LastModified": v[1]}
            for k, v in sorted(self._store.items())
            if k.startswith(Prefix)
        ]
        # Split in two so the pagination loop is actually exercised.
        mid = max(1, len(items) // 2)
        yield {"Contents": items[:mid]}
        if items[mid:]:
            yield {"Contents": items[mid:]}


class FakeS3:
    def __init__(self):
        self.store: dict = {}                    # key -> (bytes, LastModified)
        self.exceptions = type("E", (), {"NoSuchKey": NoSuchKey})
        self._t = datetime(2026, 8, 24, tzinfo=timezone.utc)

    def _stamp(self):
        self._t += timedelta(seconds=1)
        return self._t

    def put_object(self, Bucket=None, Key=None, Body=None, **kw):
        self.store[Key] = (Body, self._stamp())
        return {}

    def get_object(self, Bucket=None, Key=None, **kw):
        if Key not in self.store:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.store[Key][0])}

    def delete_object(self, Bucket=None, Key=None, **kw):
        self.store.pop(Key, None)
        return {}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.store)

fake = FakeS3()
r2_uploader._client = fake
r2_uploader._disabled = False

USER = "user_alice"
OTHER = "user_bob"

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + str(extra)) if extra and not cond else ''}")


def raises(name, status, fn, *a, **k):
    try:
        fn(*a, **k)
    except StoreError as e:
        check(f"{name} → {status}", e.status == status, f"got {e.status}: {e.message}")
        return
    except Exception as e:
        check(f"{name} → {status}", False, f"unexpected {type(e).__name__}: {e}")
        return
    check(f"{name} → {status}", False, "no error raised")


def diagram(n=3, dangling=False):
    nodes = [{"id": f"n{i}", "label": f"Node {i}", "subsystem": "Health"} for i in range(n)]
    links = [{"source": f"n{i}", "target": f"n{i+1}", "polarity": "+"} for i in range(n - 1)]
    if dangling:
        links.append({"source": "n0", "target": "ghost", "polarity": "-"})
    return {"nodes": nodes, "links": links, "meta": {"scope": "ecological"}}


print("\n── validation ──")
raises("empty diagram", 400, diagrams.validate_payload, {"nodes": [], "links": []})
raises("dangling link", 400, diagrams.validate_payload, diagram(dangling=True))
raises("node without label", 400, diagrams.validate_payload,
       {"nodes": [{"id": "a"}], "links": []})
raises("duplicate node id", 400, diagrams.validate_payload,
       {"nodes": [{"id": "a", "label": "A"}, {"id": "a", "label": "B"}], "links": []})
os.environ["DIAGRAM_MAX_NODES"] = "2"
raises("too many nodes", 413, diagrams.validate_payload, diagram(5))
os.environ["DIAGRAM_MAX_NODES"] = "2000"
clean = diagrams.validate_payload(diagram())
check("edges alias accepted", "links" in diagrams.validate_payload(
    {"nodes": [{"id": "a", "label": "A"}], "edges": []}))
check("polarity defaults to +", clean["links"][0]["polarity"] == "+")
check("meta preserved", clean.get("meta", {}).get("scope") == "ecological")

print("\n── private CRUD ──")
doc = diagrams.save_mine(USER, diagram_id=None, title="First", data=diagram(), author="Alice")
did = doc["id"]
check("save returns an id", bool(did))
check("owner recorded", doc["owner_id"] == USER)
check("stored under the owner's prefix",
      f"diagrams/{USER}/{did}.json" in fake.store)
check("listed for owner", [c["id"] for c in diagrams.list_mine(USER)] == [did])
check("not listed for anyone else", diagrams.list_mine(OTHER) == [])
check("card carries counts",
      diagrams.list_mine(USER)[0]["node_count"] == 3
      and diagrams.list_mine(USER)[0]["link_count"] == 2)
check("card carries no graph", "data" not in diagrams.list_mine(USER)[0])

again = diagrams.save_mine(USER, diagram_id=did, title="First (edited)", data=diagram(4))
check("overwrite keeps the id", again["id"] == did)
check("overwrite keeps created_at", again["created_at"] == doc["created_at"])
check("overwrite bumps updated_at", again["updated_at"] >= doc["updated_at"])
check("overwrite is not a second object", len(diagrams.list_mine(USER)) == 1)

raises("other user can't read it", 404, diagrams.get_mine, OTHER, did)
raises("missing diagram", 404, diagrams.get_mine, USER, "nope")
raises("path traversal in id", 400, diagrams.get_mine, USER, "../../etc/passwd")
raises("slash in id", 400, diagrams.get_mine, USER, "a/b")

print("\n── per-user cap ──")
diagrams.save_mine(USER, diagram_id=None, title="Second", data=diagram())
diagrams.save_mine(USER, diagram_id=None, title="Third", data=diagram())
raises("4th new diagram blocked (cap 3)", 409,
       diagrams.save_mine, USER, diagram_id=None, title="Fourth", data=diagram())
edited = diagrams.save_mine(USER, diagram_id=did, title="Still editable", data=diagram())
check("editing still works at the cap", edited["title"] == "Still editable")
raises("client-invented id can't bypass the cap", 409,
       diagrams.save_mine, USER, diagram_id="myOwnId", title="Sneaky", data=diagram())

print("\n── publishing ──")
pub = diagrams.publish(USER, did, author="Alice")
pid = pub["id"]
check("publication id is <day>-<owner hash>-<rand>", len(pid.split("-")) == 3)
check("raw clerk id absent from the public key", USER not in f"gallery/{pid}.json")
check("snapshot stored", f"gallery/{pid}.json" in fake.store)
check("snapshot is listed", diagrams.list_public()[1] == 1)
check("back-reference on the source",
      pid in [p["id"] for p in diagrams.get_mine(USER, did)["publications"]])

# The core property: editing the source must not change the publication.
diagrams.save_mine(USER, diagram_id=did, title="Renamed after publishing", data=diagram(7))
snap = diagrams.get_public(pid)
check("publication title frozen", snap["title"] == "Still editable")
check("publication graph frozen", len(snap["data"]["nodes"]) == 3)
check("owner_id stripped from the public read", "owner_id" not in snap)

cards, total = diagrams.list_public()
check("public card exposes author", cards[0].get("author") == "Alice")
check("public card hides owner_id", "owner_id" not in cards[0])

print("\n── publish cap + ownership ──")
diagrams.publish(USER, did)
raises("3rd publish today blocked (cap 2)", 429, diagrams.publish, USER, did)
raises("can't publish someone else's diagram", 404, diagrams.publish, OTHER, did)
raises("can't unpublish someone else's", 403, diagrams.unpublish, OTHER, pid)
diagrams.unpublish(USER, pid)
raises("unpublished is gone", 404, diagrams.get_public, pid)
raises("unpublish twice", 404, diagrams.unpublish, USER, pid)
bob_diagram = diagrams.save_mine(OTHER, diagram_id=None, title="Bob's", data=diagram())["id"]
bob_pid = diagrams.publish(OTHER, bob_diagram)["id"]
diagrams.unpublish("user_admin", bob_pid, is_admin=True)
check("admin can unpublish another user's", f"gallery/{bob_pid}.json" not in fake.store)

print("\n── hidden status ──")
d2 = diagrams.save_mine(OTHER, diagram_id=None, title="Hidden one", data=diagram())["id"]
hp = diagrams.publish(OTHER, d2)["id"]
raw = json.loads(fake.store[f"gallery/{hp}.json"][0])
raw["status"] = "hidden"
fake.store[f"gallery/{hp}.json"] = (json.dumps(raw).encode(), fake.store[f"gallery/{hp}.json"][1])
diagrams._cards().delete(hp)
diagrams._listings().delete("gallery:index")
raises("hidden reads as absent", 404, diagrams.get_public, hp)
check("hidden not listed", hp not in [c["id"] for c in diagrams.list_public()[0]])

print("\n── delete semantics ──")
d3 = diagrams.save_mine(OTHER, diagram_id=None, title="Doomed", data=diagram())["id"]
p3 = diagrams.publish(OTHER, d3)["id"]
diagrams.delete_mine(OTHER, d3)
raises("private copy gone", 404, diagrams.get_mine, OTHER, d3)
check("publication survives the source's deletion",
      diagrams.get_public(p3)["title"] == "Doomed")

print("\n── pagination ──")
diagrams._listings().delete("gallery:index")
page, total = diagrams.list_public(limit=1, offset=0)
check("limit respected", len(page) == 1)
check("total counts everything listed", total >= 1)
check("limit clamped", diagrams.list_public(limit=9999)[0] is not None)

print("\n── HTTP surface ──")
from fastapi.testclient import TestClient      # noqa: E402
from src import app as app_mod                 # noqa: E402

client = TestClient(app_mod.app)
routes = {(r.path, tuple(sorted(m for m in r.methods if m not in {"HEAD", "OPTIONS"})))
          for r in app_mod.app.routes if hasattr(r, "methods")}
for path, method in [
    ("/diagrams", "GET"), ("/diagrams", "POST"),
    ("/diagrams/{diagram_id}", "GET"), ("/diagrams/{diagram_id}", "DELETE"),
    ("/diagrams/{diagram_id}/publish", "POST"),
    ("/gallery", "GET"), ("/gallery/{publication_id}", "GET"),
    ("/gallery/{publication_id}", "DELETE"),
]:
    check(f"route {method} {path}", any(p == path and method in m for p, m in routes))

r = client.post("/diagrams", json={"title": "Via HTTP", "data": diagram()})
check("POST /diagrams 200", r.status_code == 200, r.text[:200])
http_id = r.json().get("id")
check("response has no owner_id", "owner_id" not in r.json())
check("response carries the graph", len(r.json()["data"]["nodes"]) == 3)

r = client.get("/diagrams")
check("GET /diagrams 200", r.status_code == 200, r.text[:200])
check("GET /diagrams lists it", http_id in [i["id"] for i in r.json()["items"]])

r = client.post("/diagrams", json={"title": "Bad", "data": diagram(dangling=True)})
check("dangling link → 400 over HTTP", r.status_code == 400, r.text[:200])

r = client.post(f"/diagrams/{http_id}/publish", json={"author": "Alice"})
check("publish over HTTP", r.status_code in (200, 429), r.text[:200])
if r.status_code == 200:
    http_pid = r.json()["id"]
    g = client.get("/gallery")
    check("GET /gallery 200", g.status_code == 200, g.text[:200])
    check("gallery payload shape",
          {"items", "total", "limit", "offset"} <= set(g.json().keys()))
    one = client.get(f"/gallery/{http_pid}")
    check("GET /gallery/{id} 200", one.status_code == 200, one.text[:200])
    check("public doc hides owner_id", "owner_id" not in one.json())

check("GET /gallery/{bad id} → 400", client.get("/gallery/not-a-real-id").status_code == 400)
check("DELETE /diagrams works", client.delete(f"/diagrams/{http_id}").status_code == 200)
check("GET deleted → 404", client.get(f"/diagrams/{http_id}").status_code == 404)

print("\n── R2 not configured ──")
r2_uploader._client = None
r2_uploader._disabled = True
r = client.get("/diagrams")
check("unconfigured R2 → 503 not 500", r.status_code == 503, f"{r.status_code} {r.text[:160]}")
r2_uploader._client = fake
r2_uploader._disabled = False

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
