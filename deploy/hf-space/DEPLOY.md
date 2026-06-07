# Free-tier deployment runbook — Supabase + Hugging Face Space

End-to-end steps to run ChatBeauty at **$0**: database on Supabase, backend on a free Hugging
Face Docker Space, models **and the vector index** on the HF Hub, frontend on Vercel, Gemini on
the AI Studio free tier. Replaces the retired paid GCP setup (see [`../../docs/deployment.md`](../../docs/deployment.md)).

> **Live as of 2026-06-07:** `https://chatbeauty-mu.vercel.app` → Space
> `https://kevin7548-chatbeauty-backend.hf.space` → Supabase + Gemini. `/recommend` ≈ 4.8 s.

## Key architecture decision: vector search is IN-MEMORY (FAISS), not in Postgres

The embeddings (~230 MB) **plus** a pgvector index (~230 MB — pgvector stores a full copy of the
vectors in the index) exceed Supabase's **500 MB free cap**, and an index-free exact scan measured
**~40 s/query** on the shared free CPU. So:

- A **FAISS HNSW** index is built offline from the embeddings and hosted on the HF Hub.
- The Space loads it into RAM at startup; `backend/app/services/retrieval.py` runs ANN in-process
  (~5 ms), then fetches only the Top-100 rows' metadata via `WHERE parent_asin = ANY(...)`.
- **Postgres stores metadata + reranking features only.** The `embedding` column is used solely to
  build the FAISS index; it's unused at serve time (left in place — reclaiming it needs a risky
  `VACUUM FULL` near the cap, and the DB is read-only now).

Legend: 🧑 = you (accounts / credentials / compute) · 🤖 = scripts/automation in this repo.

---

## Phase A — Database on Supabase 🧑

1. Create a free project at [supabase.com](https://supabase.com); copy the **transaction
   connection-pooler** URL (port 6543) — this is `DATABASE_URL`. **URL-encode special characters
   in the password** (`@`→`%40`, `!`→`%21`, …) or reset it to alphanumeric. Put it in `backend/.env`.
2. SQL Editor: `create extension if not exists vector;` then run
   [`../../backend/sql/init.sql`](../../backend/sql/init.sql) (RLS on, no policies — the app
   connects as owner and bypasses RLS).
3. **Load products** (memory-light — skips the ~1M training-pairs branch that OOMs 16 GB):
   ```bash
   python -m ml.pipeline.run \
     --input-reviews=ml/data/raw/All_Beauty.jsonl \
     --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
     --input-keywords=ml/data/processed/keywords_train.jsonl \
     --database-url="$DATABASE_URL" --skip-training-pairs
   ```
4. **Compute embeddings on Colab GPU** with
   [`../../ml/notebooks/embed_products_supabase.ipynb`](../../ml/notebooks/embed_products_supabase.ipynb)
   (loads BGE-M3 from the Hub, writes `halfvec` back to Supabase). ~20–45 min on a T4.
   - **No in-DB index is built** (it wouldn't fit). The notebook only writes embeddings.
5. Sanity-check: `SELECT count(*) FROM products WHERE embedding IS NOT NULL;` ≈ 112,578;
   `SELECT pg_size_pretty(pg_database_size(current_database()));` (~472 MB — under the 500 MB cap).

## Phase B — Models + FAISS index → HF Hub 🧑 login, 🤖 scripts

```bash
huggingface-cli login        # WRITE token; or set HF_TOKEN in backend/.env
```
1. **Upload the models** to a free model repo (`<user>/chatbeauty-models`): BGE-M3 under
   `retrieval/bge-m3-finetuned-20260202-120852/`, LightGBM under `reranking/` (use the
   `huggingface_hub` Python API — the old `huggingface-cli` syntax changed in hub ≥ 1.0).
2. **Build the FAISS HNSW index** from the Supabase embeddings and upload it to the same repo
   under `retrieval/ann/` (`faiss_hnsw_cosine.index` + `asins.json`):
   - stream `(parent_asin, embedding)` from Supabase → `numpy` float32,
   - `faiss.normalize_L2` then `faiss.IndexHNSWFlat(1024, 16)` (efConstruction 200, efSearch 100),
   - `faiss.write_index(...)` → upload. (~477 MB index for 112k vectors.)

## Phase C — Create the Space 🧑/🤖

A Docker Space can be created + configured entirely via the `huggingface_hub` API:
1. `create_repo(repo_type="space", space_sdk="docker")`, upload the three files in this directory
   (`Dockerfile`, `README.md`, `start.sh`) to the Space root.
2. Set **secrets**: `DATABASE_URL`, `GEMINI_API_KEY`. Set **variables**: `MODEL_REPO_ID`
   (`<user>/chatbeauty-models`), `ALLOWED_ORIGINS` (your Vercel origin, no trailing slash).
   - `ANN_INDEX_PATH` / `ANN_ASINS_PATH` are **auto-set in `start.sh`** (the index ships in the
     model repo, downloaded at boot) — no extra config.
3. The Space builds: the `Dockerfile` clones GitHub `main`, installs deps (incl. `faiss-cpu`,
   `google-genai`), and `start.sh` downloads the model+index from the Hub (~2.6 GB) then serves
   on port 7860.
   - **To redeploy after merging backend changes:** bump `CACHEBUST` in the Space's `Dockerfile`
     and re-upload — that busts the `git clone` layer so it pulls fresh `main`.

## Phase D — Point the frontend at the Space 🧑
- Vercel → `VITE_API_URL = https://<user>-<space>.hf.space` → redeploy.
- Ensure that Vercel origin is in the Space's `ALLOWED_ORIGINS`.

## Phase E — Verify 🧑
```bash
curl https://<user>-<space>.hf.space/health            # {"status":"ok"}
curl -X POST https://<user>-<space>.hf.space/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_input":"gentle moisturizer for dry sensitive skin"}'   # Top-5 + Korean explanations
```

---

## Notes & gotchas
- **Gemini:** use a **free-tier** AI Studio key (a key on a pay-as-you-go project 429s with
  "prepayment credits depleted"). `explanation.py` uses `google-genai` with **thinking disabled**
  (`thinking_budget=0`) — ~12 s → ~2.4 s, with no quality loss for this grounded task.
- **Cold starts:** free Spaces have ephemeral storage and sleep after ~48 h idle, so the model +
  index (~2.6 GB) re-download on cold start (a few minutes). The frontend `warmUp()` masks short gaps.
- **No auth / rate limiting:** the endpoint is public (known limitation).
- **Updating embeddings:** if you re-embed, rebuild + re-upload the FAISS index (Phase B step 2),
  then redeploy the Space (bump `CACHEBUST`).
