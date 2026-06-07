# deploy/

Free-tier deployment for the backend.

- **`hf-space/`** — the **live** deployment: a Hugging Face Docker Space (`Dockerfile`, `start.sh`,
  `README.md` Space card) + **`DEPLOY.md`** (the end-to-end runbook).
- **`setup-gcp.sh`** — the **retired** paid-GCP script (Cloud Run / Cloud SQL / GCS), kept for
  historical reference only. Paid GCP was torn down 2026-06-02 — do **not** run it without an
  explicit decision to spend money.

## Live stack (free, $0)
Vercel (frontend) → **Hugging Face Docker Space** (FastAPI) → **Supabase** (Postgres) + Gemini.
- **Backend:** Space `kevin7548/chatbeauty-backend`. The `Dockerfile` clones GitHub `main`;
  `start.sh` downloads the models + FAISS index from the HF Hub and serves on `:7860`.
- **DB:** Supabase — **metadata + reranking features only**. There is **no pgvector index**
  (it wouldn't fit the 500 MB free cap); serve-time vector search is an **in-memory FAISS** index.
- **Models + FAISS index:** HF Hub `kevin7548/chatbeauty-models`.
- **Space secrets/vars:** `DATABASE_URL`, `GEMINI_API_KEY` (secrets); `MODEL_REPO_ID`,
  `ALLOWED_ORIGINS` (vars). `ANN_INDEX_PATH`/`ANN_ASINS_PATH` are set by `start.sh`.
- **Redeploy:** bump `CACHEBUST` in the Space's `Dockerfile` and re-upload → forces a fresh
  `git clone` of `main`.

## Scope
`deploy/` + `backend/Dockerfile` + `backend/docker-compose.yml`. Frontend → Vercel (set `VITE_API_URL`).

## Docs
Full runbook → `deploy/hf-space/DEPLOY.md` · topology / env-vars → `docs/deployment.md`
