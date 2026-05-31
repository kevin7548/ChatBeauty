# Deployment

Production runs on Google Cloud (backend) + Vercel (frontend). The reference script is
`deploy/setup-gcp.sh`; the container is `backend/Dockerfile`. Local dev uses
`backend/docker-compose.yml` (see [development.md](development.md)).

## Topology

```
Vercel (React SPA) ──API──▶ Cloud Run (FastAPI)
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
           Cloud SQL        GCS volume       Gemini API
       PostgreSQL+pgvector  BGE-M3 + LGBM    2.5 Flash
       112K products        (mounted RO)
```

## Container (`backend/Dockerfile`)
- `python:3.12-slim`; installs `build-essential` + `libgomp1` (LightGBM).
- Deps via Poetry (`--only main`); then `pip install -e ml/` so `item_ranker` is importable.
- Copies `backend/app/` only. **Models are not in the image** — they come from a GCS volume
  mounted at `/app/ml/model-gcs/`:
  - `retrieval/bge-m3-finetuned-20260202-120852/`
  - `reranking/lgbm_reranker_current_features_v1.pkl`
- `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1`.

## Cloud Run (`deploy/setup-gcp.sh`)
Steps: enable APIs → create Artifact Registry (`asia-northeast3`) → build via Cloud Build
(`gcloud builds submit`, no local Docker) → `gcloud run deploy`. Key flags:

| Flag | Value |
|---|---|
| `--memory` / `--cpu` | `4Gi` / `2` |
| `--min-instances` / `--max-instances` | `1` / `2` (min 1 prevents cold starts) |
| `--timeout` | `300` |
| `--allow-unauthenticated` | yes — public endpoint (no auth) |
| `--add-cloudsql-instances` | Cloud SQL via private socket `/cloudsql/...` |
| `--execution-environment` | `gen2` |
| `--add-volume` / `--add-volume-mount` | GCS bucket `chatbeauty-models` → `/app/ml/model-gcs` |

Post-deploy (from the script's "Next steps"): upload models
(`gcloud storage cp -r ml/model/* gs://chatbeauty-models/`), populate the DB via
`cloud-sql-proxy`, then `curl $SERVICE_URL/health`.

## Environment variables

| Variable | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` | explanation | required; service raises at import if unset |
| `DATABASE_URL` | retrieval/reranking | default `postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty` |
| `BGE_MODEL_PATH` | retrieval | default = GCS mount path |
| `RERANK_MODEL_PATH` | reranking | default = GCS mount path |
| `ALLOWED_ORIGINS` | CORS | comma-separated; set to the Vercel URL in prod |
| `DEBUG` | FastAPI | `true`/`false` |
| `DB_PASSWORD` | docker-compose only | local Postgres password |
| `VITE_API_URL` | frontend | Cloud Run URL (Vercel env) |

See `backend/.env.example` and `frontend/.env.example` for templates.

## Secret management
Today secrets reach Cloud Run as plain env vars: `setup-gcp.sh` passes `DATABASE_URL` and
`GEMINI_API_KEY` via `--set-env-vars` (the script expects `GEMINI_API_KEY` exported in your
shell). Locally they come from `backend/.env` (git-ignored). For production hardening,
GCP Secret Manager (mounted as env/volume) is the recommended future step — see
[Known limitations](architecture.md#known-limitations--future-work).

## Operations & observability
- **Logging:** `LatencyMiddleware` logs each request (`<METHOD> <path> completed in <ms>`)
  and the route logs the per-stage `latency` breakdown; these land in Cloud Logging.
  Responses carry an `X-Total-Latency-Ms` header.
- **Metrics:** Cloud Run's built-in dashboards (request count, latency, CPU/memory,
  instance count) cover the basics.
- **Alerting:** none configured yet (future work). If this section grows, split it into a
  dedicated `docs/observability.md`.

## Frontend (Vercel)
Static Vite build, framework defaults (no `vercel.json`). Set `VITE_API_URL` to the Cloud
Run service URL; ensure that origin is in the backend's `ALLOWED_ORIGINS`.
