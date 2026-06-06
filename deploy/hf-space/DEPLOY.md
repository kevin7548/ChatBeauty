# Free-tier deployment runbook — Supabase + Hugging Face Space

End-to-end steps to bring the ChatBeauty backend back online at **$0**: database on Supabase,
backend on a free Hugging Face Docker Space, models on the HF Hub, frontend on Vercel, Gemini on
the AI Studio free tier. Replaces the retired paid GCP setup (see [`../../docs/deployment.md`](../../docs/deployment.md)).

Legend: 🧑 = you (accounts / credentials / compute) · 🤖 = artifacts already in this repo.

---

## Phase A — Database on Supabase 🧑

The deployed backend returns nothing without a loaded DB, so do this first.

1. Create a free project at [supabase.com](https://supabase.com). Note the **connection pooler**
   URL (Project → Settings → Database → *Connection pooling*, "Transaction" mode). This is your
   `DATABASE_URL`.
2. In the Supabase **SQL Editor**: `create extension if not exists vector;` then paste & run
   [`../../backend/sql/init.sql`](../../backend/sql/init.sql) (table + B-tree indexes).
3. **Load data + embeddings** (the long pole — needs the raw data files and GPU for embeddings):
   ```bash
   # from repo root, with DATABASE_URL set to the Supabase pooler URL
   python -m ml.pipeline.run \
     --input-reviews=ml/data/raw/All_Beauty.jsonl \
     --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
     --input-keywords=ml/data/processed/keywords_train.jsonl \
     --output-dir=ml/data/processed/beam_output \
     --database-url="$DATABASE_URL"
   python -m ml.scripts.embed_products --database-url="$DATABASE_URL"
   ```
   Embedding ~112k products is slow on CPU — prefer the GPU notebook
   `ml/notebooks/embed_products_colab.ipynb` pointed at the Supabase URL.
4. Build the HNSW index once embeddings exist (SQL Editor):
   ```sql
   CREATE INDEX idx_products_embedding ON products
     USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
   ```
5. Sanity check: `SELECT count(*) FROM products WHERE embedding IS NOT NULL;` (expect ~112k) and
   `SELECT pg_size_pretty(pg_database_size(current_database()));` (stay under the 500 MB free cap).

## Phase B — Push models to the HF Hub 🧑 (script 🤖)

Models aren't in git; host them on a free HF model repo the Space downloads at startup.

```bash
huggingface-cli login                     # paste a WRITE token (hf.co/settings/tokens)
deploy/hf-space/upload-models.sh <hf-username>/chatbeauty-models
```
This creates the repo and uploads `retrieval/bge-m3-...` (~2.1 GB) + the LightGBM `.pkl`.

## Phase C — Create the Space 🧑 (Dockerfile/README/start.sh 🤖)

1. Create a **Docker** Space (public) at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Push the three files in this directory to the Space repo root:
   ```bash
   git clone https://huggingface.co/spaces/<hf-username>/<space-name> hf-space && cd hf-space
   cp ../deploy/hf-space/{Dockerfile,README.md,start.sh} .
   git add . && git commit -m "ChatBeauty backend Space" && git push
   ```
3. In the Space **Settings → Variables and secrets**, add:
   | Key | Type | Value |
   |---|---|---|
   | `MODEL_REPO_ID` | variable | `<hf-username>/chatbeauty-models` |
   | `DATABASE_URL` | secret | Supabase pooler URL (Phase A) |
   | `GEMINI_API_KEY` | secret | Google AI Studio free-tier key |
   | `ALLOWED_ORIGINS` | variable | your Vercel URL (e.g. `https://chatbeauty.vercel.app`) |
4. The Space builds and starts; first boot downloads the model (a few minutes).

## Phase D — Point the frontend at the Space 🧑

- In Vercel, set `VITE_API_URL` to the Space URL: `https://<hf-username>-<space-name>.hf.space`.
- Make sure that exact Vercel origin is in the Space's `ALLOWED_ORIGINS`.

## Phase E — Verify, then merge 🧑

```bash
curl https://<hf-username>-<space-name>.hf.space/health        # {"status":"ok"}
curl -X POST https://<hf-username>-<space-name>.hf.space/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_input":"gentle moisturizer for dry skin"}'        # Top-5 + latency
```
When `/recommend` returns sensible results end-to-end (Vercel → Space → Supabase + Gemini),
**merge PR #4**.

---

## Notes & gotchas

- **Cold starts:** free Spaces have ephemeral storage and sleep after ~48h idle, so the model
  re-downloads on cold start (minutes). `hf_transfer` is enabled in the Dockerfile to speed it.
- **No auth:** the Space endpoint is public (known limitation) — fine for a portfolio demo; add
  rate limiting later (`TODO.md` backlog).
- **Updating the backend:** the Dockerfile clones `main` at build, so trigger a Space rebuild
  (Settings → Factory rebuild) after merging backend changes. A later CD job can automate this.
- **Token roles:** Space creation + model upload need a **write** token, not a read token.
