# Deployment

> ⚠️ **The paid Google Cloud deployment was retired on 2026-06-02** (Cloud Run, Cloud SQL,
> Artifact Registry, and the GCS model buckets were deleted to avoid cost). ChatBeauty now
> targets a **free-tier-only** stack. The trained models are saved locally in `ml/model/`
> (nothing was lost). The original GCP setup is preserved at the bottom as a historical
> reference in case the project is ever re-platformed on paid infra.

The project is designed to run at **$0**: frontend on Vercel (free), database on Supabase
(free tier), the Gemini explanation step on a Google AI Studio free-tier key, and the backend
either **locally** (the existing `backend/docker-compose.yml`) or on a free container host.
Local dev runbook: [development.md](development.md).

## Free-tier topology

```
Vercel (React SPA, free) ──API──▶ FastAPI backend
                                  (local docker-compose, or free host:
                                   Render / Fly.io / HF Spaces)
                                       │
                       ┌───────────────┼────────────────┐
                       ▼               ▼                ▼
                  Supabase         local ml/model/   Gemini API
              Postgres+pgvector    BGE-M3 + LGBM      (AI Studio
              (free, 500MB)        (no GCS mount)      free tier)
              112K products
```

## Database — Supabase (free tier)

Supabase gives a free Postgres with `pgvector` (≥ 0.7, so `halfvec` works). Migration:

1. Create a free Supabase project; grab the **transaction connection-pooler** URL (port 6543;
   URL-encode special chars in the password).
2. Enable the extension and apply the schema:
   `create extension if not exists vector;` then run [`../backend/sql/init.sql`](../backend/sql/init.sql).
3. Load products (`python -m ml.pipeline.run … --skip-training-pairs`) then embeddings on Colab
   GPU ([`../ml/notebooks/embed_products_supabase.ipynb`](../ml/notebooks/embed_products_supabase.ipynb)).

> **No in-DB vector index.** The `halfvec(1024)` column alone is ~230 MB, and a pgvector index
> would add ~230 MB more — over the **500 MB free cap** (and index-free exact scan is ~40 s/query).
> So vector search runs **in-memory on the backend** via a FAISS HNSW index hosted on the HF Hub
> (see *Vector search* below). Postgres stores **metadata + reranking features only**; the
> `embedding` column exists just to build the FAISS index and is unused at serve time. The DB sits
> at ~472 MB — read-only in production, so it won't grow into the cap. Full runbook:
> [`../deploy/hf-space/DEPLOY.md`](../deploy/hf-space/DEPLOY.md).

## Vector search — in-memory FAISS (not pgvector)

A FAISS HNSW index is built offline from the embeddings and uploaded to the HF Hub model repo
(`retrieval/ann/`). The Space loads it into RAM at startup and
[`../backend/app/services/retrieval.py`](../backend/app/services/retrieval.py) runs ANN in-process
(~5 ms) → fetches the Top-100 rows' metadata with `WHERE parent_asin = ANY(...)`. Rebuild + re-upload
the index whenever embeddings change.

## Backend — local or free host

The container (`backend/Dockerfile`, `python:3.12-slim`) is unchanged; what changed is **where
the models come from and where it runs**.

- **Models:** load from the local `ml/model/` directory (mounted by docker-compose at
  `/app/ml/model-gcs/`), **not** a GCS volume. The fine-tuned BGE-M3
  (`retrieval/bge-m3-finetuned-20260202-120852/`) and the LightGBM
  `reranking/lgbm_reranker_current_features_v1.pkl` already live there.
- **Local (primary, simplest $0):** `cd backend && docker-compose up --build -d` → backend on
  `:8080`. Models load from `ml/model/`; point `DATABASE_URL` at Supabase. For a one-off public
  demo, expose it with a tunnel (`cloudflared` / `ngrok`).
- **Free always-online host → Hugging Face Docker Space (recommended).** RAM is the deciding
  factor: BGE-M3 needs ~3–4 GB loaded, so **Render (512 MB) and Fly.io (256 MB) free tiers will
  OOM on startup.** A free HF CPU Space has **16 GB RAM** and is purpose-built for models. Host
  the FastAPI app as a Docker Space, with the fine-tuned BGE-M3 + LightGBM on the **HF Hub** (free
  model repo, downloaded at startup). Ready-to-use Space files + full runbook (incl. Supabase
  load): [`../deploy/hf-space/`](../deploy/hf-space/) and
  [`../deploy/hf-space/DEPLOY.md`](../deploy/hf-space/DEPLOY.md). Free Spaces sleep after ~48h
  idle and re-download the model on cold start (a few minutes; `hf_transfer` speeds it).

## Explanation — Gemini (AI Studio free tier)

Use a **Google AI Studio** API key (free tier for Gemini 2.5 Flash), **not** a paid Vertex AI
endpoint — that is the one usage-based charge a cloud teardown can't remove. Set it as
`GEMINI_API_KEY`.

## Environment variables

| Variable | Used by | Free-tier value |
|---|---|---|
| `GEMINI_API_KEY` | explanation | **AI Studio** free-tier key; service raises at import if unset |
| `DATABASE_URL` | retrieval/reranking | Supabase pooler URL (local default: `postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty`) |
| `BGE_MODEL_PATH` | retrieval | local path under `ml/model/retrieval/...` (docker-compose mount) |
| `RERANK_MODEL_PATH` | reranking | local path under `ml/model/reranking/...` |
| `ALLOWED_ORIGINS` | CORS | comma-separated; the Vercel URL (and `http://localhost:5173` for dev) |
| `DEBUG` | FastAPI | `true`/`false` |
| `DB_PASSWORD` | docker-compose only | local Postgres password |
| `VITE_API_URL` | frontend | backend URL (Vercel env) — local or free-host URL |

Templates: `backend/.env.example`, `frontend/.env.example`. Secrets live in `backend/.env`
(git-ignored) locally, or the host's env-var settings (Vercel / Render / Fly secrets).

## Frontend — Vercel (free)

Static Vite build, framework defaults (no `vercel.json`). Set `VITE_API_URL` to the backend URL
and ensure that origin is in the backend's `ALLOWED_ORIGINS`. Vercel auto-deploys on push — the
only piece of the pipeline that was always free and stays unchanged.

## Operations & observability

- **Logging:** `LatencyMiddleware` logs each request (`<METHOD> <path> completed in <ms>`) and
  the route logs the per-stage `latency` breakdown; responses carry an `X-Total-Latency-Ms`
  header. On a free host these go to that host's log viewer (or stdout locally).
- **Metrics/alerting:** no managed dashboards on the free tier; rely on the logs + header.

---

<details>
<summary>Historical reference — the retired paid GCP deployment (pre-2026-06-02)</summary>

> Kept for reference only. These resources were **deleted on 2026-06-02**; do not recreate them
> without an explicit decision to spend money. Reference script: `deploy/setup-gcp.sh`.

**Topology:** Vercel SPA → Cloud Run (FastAPI) → Cloud SQL (Postgres+pgvector), GCS volume
(BGE-M3 + LightGBM, mounted read-only at `/app/ml/model-gcs/`), Gemini 2.5 Flash.

**Cloud Run (`deploy/setup-gcp.sh`):** enable APIs → create Artifact Registry (`asia-northeast3`)
→ build via Cloud Build (`gcloud builds submit`, no local Docker) → `gcloud run deploy`. Key
flags: `--memory 4Gi --cpu 2`, `--min-instances 1 --max-instances 2` (min 1 prevented cold
starts — and was a constant always-on cost), `--timeout 300`, `--allow-unauthenticated`,
`--add-cloudsql-instances` (private socket `/cloudsql/...`), `--execution-environment gen2`,
`--add-volume`/`--add-volume-mount` for the `chatbeauty-models` GCS bucket → `/app/ml/model-gcs`.

**Post-deploy:** upload models (`gcloud storage cp -r ml/model/* gs://chatbeauty-models/`),
populate the DB via `cloud-sql-proxy`, then `curl $SERVICE_URL/health`.

**Secrets:** reached Cloud Run as plain env vars via `--set-env-vars` (`DATABASE_URL`,
`GEMINI_API_KEY`). GCP Secret Manager was noted as the future hardening step.

**Cost lesson:** the biggest charges were Cloud SQL (`db-custom-2-8192` billed 24/7) and Cloud
Run `--min-instances 1` (always warm = always billed). Artifact Registry also accumulated ~8 GB
of old images. If re-platforming on paid infra, set `--min-instances 0`, use the smallest Cloud
SQL tier (or stay on Supabase), prune the image repo, and set a billing budget alert.

</details>
