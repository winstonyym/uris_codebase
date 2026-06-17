# Backend cold-start & warm-up

The Graph-RAG backend runs on Vercel **serverless**, and on boot it loads the
entire knowledge graph (every KG embedding) from Neo4j Aura. That load is the
dominant cold-start cost, so the **first** request after an instance spins up
can take many seconds. This doc explains the moving parts and how to keep the
service responsive — especially on the **Hobby** plan.

## What the code already does

1. **Readiness is decoupled from the KG load.** The FastAPI lifespan no longer
   blocks on `recommender.warm()`; it runs it on a daemon thread. The app
   accepts requests (e.g. `/health`, `/warm`) immediately instead of hanging
   until the KG finishes loading. (`src/app.py`)

2. **`/warm` endpoint** (`GET` or `POST`). Synchronously builds the KG indices
   (idempotent — a no-op if the background warm already finished) and
   pre-creates every LLM client. Never raises; it reports partial warmth
   instead. This is the wake-up call.

3. **The Welcome page warms early.** The frontend pings `POST /warm` on
   Welcome-page *mount* (and again on Start), so the cold start overlaps with
   the time the user spends reading the page. (`frontend/src/components/LandingPage.jsx`)

## The big lever: R2-cached index artifact

The dominant cold-start cost is `_build_indices()` streaming the whole KG from
Neo4j Aura — ~44k node rows (each with a 512-dim embedding) plus ~49k edges —
and building a ~90 MB numpy matrix. The textual metadata is also in the bundled
`merged_kg.json`, but the **embeddings live only in Neo4j**.

So the recommender now caches the fully-built index as a single compressed
artifact in **R2** (`src/recommender.py`):

- On warm, `_try_load_index_from_r2()` does one ~55 MB object GET (float16
  matrix + JSON metadata) and rehydrates the index in a second or two — no Aura
  scan. R2 egress is free.
- On a miss (empty/stale cache, or `KG_INDEX_REBUILD` set), it falls back to the
  Neo4j build and `_save_index_to_r2()` uploads a fresh artifact for next time.
- The artifact key embeds a version + the embedding dimensionality:
  `kg-index/<KG_INDEX_VERSION>/index_d<dim>.npz`.

### Seed the cache once (important on Hobby)

The *first* serverless cold start with an empty cache would still do the slow
Neo4j build itself — which can exceed Hobby's 60 s limit and get killed before
it uploads, leaving every cold start rebuilding. Avoid that by seeding from your
machine (no time limit):

```bash
cd backend
python scripts/build_index_cache.py     # builds from Neo4j, uploads to R2
```

### Invalidate after a re-ingest

When you re-ingest the KG, the embeddings change. Bump the version so a stale
artifact is never served, in **both** places, then re-seed:

1. Set `KG_INDEX_VERSION=v2` in the Vercel project env (and your local env).
2. `python scripts/build_index_cache.py`

(Or set `KG_INDEX_REBUILD=1` in Vercel to force a one-time rebuild without
changing the version.)

### Required env (runtime)

R2 must be reachable from the deployed function, so set these in the **Vercel
project env** (your local `.env` does not deploy): `R2_ACCOUNT_ID`,
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` (plus `REDIS_URL`
and the model API keys). Without R2 env the service still works — it just falls
back to the slow Neo4j build each cold start.

## What you should enable on Vercel (Hobby)

1. **Raise the function timeout.** A cold warm can exceed Hobby's default
   function duration and get killed mid-load, which leaves the instance cold
   forever. In the Vercel dashboard:
   **Project → Settings → Functions → Function Max Duration → 60s** (the Hobby
   maximum), and bump **Memory** if available. Doing this in the dashboard is
   safer than committing a `vercel.json`, because this repo has no committed
   Vercel function config — the build is configured in the dashboard, and a
   mismatched `vercel.json` `functions` glob would fail the deploy.

2. **Keep an instance warm.** Vercel Cron on Hobby only runs **once per day**,
   which can't keep a function hot. Use one of these instead:
   - **GitHub Actions** (included): `.github/workflows/keep-backend-warm.yml`
     pings `/warm` every 5 minutes. Set repo variable `BACKEND_URL` if the URL
     changes.
   - **External uptime pinger** (zero code): point UptimeRobot / cron-job.org /
     Better Uptime at `https://<backend>/warm` on a 5-minute interval.

## If you upgrade to Pro

On Pro you can replace the external pinger with a native Vercel Cron and use a
longer timeout. Add a `vercel.json` **only after** confirming your function
entry/root in the dashboard, e.g.:

```json
{
  "crons": [{ "path": "/warm", "schedule": "*/5 * * * *" }]
}
```

(Pro allows sub-daily cron schedules; Hobby will reject this at deploy time.)
