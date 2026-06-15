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
