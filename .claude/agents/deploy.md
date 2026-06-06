---
name: deploy
description: Deployment and infrastructure tasks — free-tier hosting (Supabase, local/free backend host, Vercel), Dockerfile, docker-compose. Use for changes scoped to deployment config.
---

You work only in `deploy/` and deployment-related config files: `backend/Dockerfile`, `backend/docker-compose.yml`. Never modify application code in `frontend/src/`, `backend/app/`, `ml/`, or `backend/sql/`.

> **Paid GCP infra was retired on 2026-06-02** (Cloud Run, Cloud SQL, Artifact Registry, GCS
> buckets deleted to avoid cost). The project targets a **free-tier-only** stack now. Do **not**
> recreate paid GCP resources without an explicit decision to spend money. Old GCP details are
> kept as a historical reference in `docs/deployment.md`.

## Architecture (free-tier)
```
Vercel (Frontend, free)        Backend + data (free)
───────────────────────        ─────────────────────
React + TypeScript  ──API──→  FastAPI: local docker-compose
                               (or free host: Render / Fly.io / HF Spaces)
                                   │
                      ┌────────────┼────────────┐
                      ▼            ▼            ▼
                 Supabase     local ml/model/  Gemini API
              Postgres+pgvector  BGE-M3 + LGBM  (AI Studio
              (free, 500MB)    (no GCS mount)    free tier)
              112K products
```

## Key config
- **DB:** Supabase free tier (Postgres + pgvector ≥ 0.7 for `halfvec` + HNSW). `DATABASE_URL`
  points at the Supabase pooler URL; or local docker-compose Postgres for dev.
- **Backend:** primary = local `docker-compose up` (`:8080`). Optional free host
  (Render / Fly.io / HF Spaces) — these idle-spin-down (cold start on first request).
- **Models:** load from local `ml/model/` (docker-compose mounts it at `/app/ml/model-gcs/`),
  **not** a GCS volume. ~2.1 GB BGE-M3 + the LightGBM pkl.
- **Gemini:** Google AI Studio **free-tier** key (`GEMINI_API_KEY`), not paid Vertex AI.
- **Dockerfile:** build context is project root (`..`) — COPY paths reference `backend/` and `ml/`.
- **Vercel:** static hosting, `VITE_API_URL` env var for the backend URL.

## Constraints
- Free-tier only — no Cloud Run / Cloud SQL / Artifact Registry / Cloud Build. If a free host is
  added, add its CD job to GitHub Actions then (see `TODO.md` Step 2).
- Model files come from the local `ml/model/` mount, not bundled in the image, not GCS.
- Mind the Supabase 500MB cap — the `halfvec` + HNSW index is the largest object; check
  `pg_database_size` before adding indexes (e.g. the hybrid-retrieval GIN index).
