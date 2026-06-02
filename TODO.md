# ChatBeauty — Project TODO

Single source of truth for remaining work. The frontend runs on Vercel (free) and the models are
trained (saved locally in `ml/model/`). **The paid GCP backend infra — Cloud Run, Cloud SQL,
Artifact Registry, GCS — was torn down on 2026-06-02 to avoid cost.** The project now targets a
**free-tier stack only**: Supabase Postgres, a local or free-host backend, Vercel frontend, and a
Google AI Studio (free-tier) Gemini key. See [`docs/deployment.md`](docs/deployment.md).

The **Current focus** below is the active roadmap, ordered as an execution queue and tagged per
agent/window so it doubles as a dispatch board; the **Backlog** holds documented limitations and
longer-horizon ideas.

Detailed rationale for the documented limitations lives in
[`docs/architecture.md#known-limitations--future-work`](docs/architecture.md#known-limitations--future-work).

**Domain tags:** `[backend]` `[ml]` `[db]` `[deploy]` `[frontend]` `[test]` — map to the
domain agents in `.claude/agents/`.

---

## 🎯 Current focus

> **Dispatch board.** Items are an ordered queue. Each carries a `→ window` owner tag:
> - `→ backend window` / `→ deploy window` / `→ ml window` — an **independent single-domain**
>   task. Run it in its own terminal, tagged for notifications: `CB_AGENT=<domain> claude`,
>   then drive it with *"Do step N from TODO.md."* These can run in parallel (disjoint surfaces).
> - `→ main (cross-cutting)` — context must flow across db/backend/ml. Run it in **one**
>   conversation (subagents, not parallel windows); never split across windows or they collide
>   on `retrieval.py` / `reranking.py`.
>
> **Check-off rule:** when a window finishes its item, it edits **only that item's checkbox** —
> avoids parallel windows clobbering each other's edits to this file.
>
> Steps 3–5 are the cross-cutting tier; within the C-track, **step 4 must precede step 5**
> (shared ONNX/int8 toolchain). Steps 1→2 and 4→5 are hard-ordered; everything else is parallelizable.

**Recently landed** (in code, commit `d1c6456`; boxes intentionally left unchecked):

- [ ] **[db][backend] Migrate embeddings to `halfvec`.** Column is `halfvec(1024)`
      ([`backend/sql/init.sql`](backend/sql/init.sql) L7), `::halfvec` casts in
      [`backend/app/services/retrieval.py`](backend/app/services/retrieval.py) (L27/L30) and the
      write in [`ml/scripts/embed_products.py`](ml/scripts/embed_products.py) (L66). ~50% less
      storage/RAM for the Supabase free tier; requires pgvector ≥ 0.7.
- [ ] **[db] Switch the vector index from IVFFlat to HNSW.** Index recipe in
      [`backend/sql/init.sql`](backend/sql/init.sql) (L34-37) is now
      `USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)`; set
      `ef_search` at query time. `docs/db-schema.md` updated.

### Active queue

- [ ] **Step 0 — `→ deploy window` — [deploy][db] Re-platform on the free-tier stack.**
      Replaces the retired paid GCP infra (torn down 2026-06-02). Bring the app back at $0:
      **DB** → Supabase free tier (Postgres + pgvector + `halfvec`, 500MB); apply
      [`backend/sql/init.sql`](backend/sql/init.sql), re-load via the Beam pipeline +
      [`ml/scripts/embed_products.py`](ml/scripts/embed_products.py), build the HNSW index.
      **Backend** → run locally via [`backend/docker-compose.yml`](backend/docker-compose.yml)
      (primary, simplest $0) or, for an always-online free demo, a **Hugging Face Docker Space**
      (16 GB RAM free — the only free tier big enough for the ~2.1 GB BGE-M3; Render 512 MB /
      Fly 256 MB will OOM). Models load from local `ml/model/` locally, or BGE-M3 from the **HF
      Hub** + the LightGBM `.pkl` committed (256 KB) on a Space. **Gemini** → Google AI Studio
      free-tier key. **Frontend** → Vercel (already free).
      Update env (`DATABASE_URL` → Supabase pooler URL, `BGE_MODEL_PATH` / `RERANK_MODEL_PATH` →
      local paths). **Space files + full runbook ready:** [`deploy/hf-space/`](deploy/hf-space/)
      and [`deploy/hf-space/DEPLOY.md`](deploy/hf-space/DEPLOY.md). Details:
      [`docs/deployment.md`](docs/deployment.md).

- [x] **Step 1 — `→ backend window` — [backend][test] pytest integration test for `POST /recommend`.** ✅ Done
      Landed in `backend/tests/` (`test_recommend.py`, `conftest.py`); exercises the full
      retrieve→rerank→explain path (mock Gemini). Unblocks Step 2.

- [x] **Step 2 — `→ deploy window` — [deploy][test] Add GitHub Actions CI (lint + tests).** ✅ Done
      `.github/workflows/ci.yml` runs ruff (advisory) + `pytest` on every PR / push to main.
      Lean install (`fastapi` + `pytest` + `httpx`) since the test stubs all services offline —
      ~40s, verified locally (3 passed). Goes green on the next push. **CD is deferred / re-scoped:**
      the old CD target (Cloud Run) was retired 2026-06-02; backend auto-deploy waits on a free
      host (Step 0). Add a CD job for that host when chosen. Vercel still auto-deploys the frontend.

- [ ] **Step 3 — `→ main (cross-cutting)` — [ml][db][backend] Add hybrid retrieval (dense + lexical/BM25) with RRF fusion.**
      BGE-M3's dense head alone under-retrieves lexical matches that matter in beauty —
      brands, product lines, ingredients ("niacinamide", "salicylic acid"). Add a Postgres
      `tsvector` GIN index over product text (title + features + keywords) and fuse the dense
      `<=>` ranking with lexical ranking via reciprocal-rank fusion in
      [`backend/app/services/retrieval.py`](backend/app/services/retrieval.py); alternatively
      use BGE-M3's native sparse output. **CPU-only — no GPU needed.** Expected the single
      largest Recall@100 gain. Add a BM25-only baseline to the eval to quantify it.
      **Storage trade-off (Supabase free tier, 500MB):** a GIN index adds ~20–60MB for ~112k
      products. Minimize it by indexing the *expression* (no separate stored `tsvector`
      column) over only title + keywords + features (skip `description`):
      `CREATE INDEX ... USING gin (to_tsvector('english', title || ' ' || coalesce(features,'') || ' ' || coalesce(embedding_text,'')))`.
      Check headroom first (`pg_database_size`); the HNSW index is the largest object, so if
      space is tight the lever is shrinking the vector side (lower `m` / dim), not skipping hybrid.

- [ ] **Step 4 — `→ main (cross-cutting)` — [ml][backend] Export the BGE-M3 query encoder to ONNX + quantize (CPU retrieval latency).**
      Query encoding dominates the ~1.1s retrieval stage (of ~1.4s total) and there is no GPU
      to fall back on. Export the fine-tuned encoder to ONNX Runtime with dynamic int8
      quantization (typically 2–4× on CPU) and load it in
      [`backend/app/services/retrieval.py`](backend/app/services/retrieval.py); regenerate the
      artifact path/convention in [`ml/scripts/embed_products.py`](ml/scripts/embed_products.py)
      if the same encoder is reused offline. **Establishes the ONNX/int8 toolchain reused by
      Step 5. Precedes Step 5.**

- [ ] **Step 5 — `→ main (cross-cutting)` — [ml][backend] Add a query-aware cross-encoder rerank stage.**
      The current LightGBM reranker is ~query-blind: 9 of its 10 features are static item
      priors (review_cnt, vp_ratio, avg_rating, …); only `retrieval_score` depends on the
      query, so it mostly re-sorts by popularity. Add a cross-encoder over the top-N candidates
      in [`backend/app/services/reranking.py`](backend/app/services/reranking.py) — either
      scoring directly or feeding its score in as an 11th LightGBM feature.
      **No GPU:** use a distilled reranker (e.g. `bge-reranker-base` or ms-marco MiniLM)
      exported to ONNX + int8 (reuse the Step 4 toolchain) and rerank only the top ~30–50 to
      stay in budget — a CPU cross-encoder is hundreds of ms vs. today's ~19ms, so cap N and
      measure. **Reuses Step 4.**

- [ ] **Step 6 — `→ ml window` — [ml] Evaluate whether Apache Beam is still warranted.**
      Keyword *extraction* happens upstream — the pipeline only *reads* a pre-made
      `keywords_train.jsonl` ([`ml/pipeline/transforms/parse.py`](ml/pipeline/transforms/parse.py)
      `ParseKeywordFn`). On `DirectRunner` over ~112k products, Beam's real work is just
      feature aggregation + training-pair generation, both doable with a simpler pandas/SQL
      batch script. Decide: keep Beam, or replace
      [`ml/pipeline/`](ml/pipeline/run.py) with a lean script. Independent — parallelizable.

- [ ] **Step 7 — `→ ml window` (spec) then `→ main` (build) — [ml] Derive purchase signals from reviews to build a user-tower (two-tower retrieval).**
      Reviews carry `verified_purchase` (already used in
      [`ml/pipeline/transforms/aggregate.py`](ml/pipeline/transforms/aggregate.py) L93) plus
      `user_id` — use these as implicit purchase interactions to train a user tower alongside
      the item embeddings, enabling personalized retrieval. Larger ML effort; spec the data
      derivation + training in [`docs/ml-pipeline.md`](docs/ml-pipeline.md) first (ml window),
      then the serving-path wiring is cross-cutting (main).

---

## Backlog

### 🟡 Improvements

- [ ] **[deploy]** Move secrets from plain env vars to GCP Secret Manager
      ([`docs/deployment.md`](docs/deployment.md) L61-66).
- [ ] **[backend]** Harden Gemini response parsing in
      [`backend/app/services/explanation.py`](backend/app/services/explanation.py) (L76-81):
      replace the silent `except: pass` with a logged, typed fallback.
- [ ] **[backend]** Honor `top_k` in
      [`backend/app/api/routes/recommend.py`](backend/app/api/routes/recommend.py) instead of
      hardcoding Top-5 ([`docs/api-spec.md`](docs/api-spec.md) L29).
- [ ] **[deploy]** Configure Cloud Run alerting (latency / error-rate); consider a
      `docs/observability.md` ([`docs/deployment.md`](docs/deployment.md) L74).
- [ ] **[backend][deploy]** Add API rate limiting to the public `/recommend` endpoint.
- [ ] **[ml]** Decide the fate of the XGBoost code
      ([`ml/item_ranker/modeling/train/train_xgb.py`](ml/item_ranker/modeling/train/train_xgb.py),
      [`predict/xgb.py`](ml/item_ranker/modeling/predict/xgb.py)): promote to a documented
      experiment or remove as dead code (LightGBM is adopted).

> The pytest `POST /recommend` integration test moved up to **Current focus → Step 1** (CI gate).

### 🟢 Longer-horizon

- [ ] **[db]** Adopt a DB migration tool (Alembic) to replace re-applying
      [`backend/sql/init.sql`](backend/sql/init.sql) by hand.
- [ ] **[ml]** Add MLOps tooling — experiment tracking (MLflow/W&B) + production model/data
      drift monitoring (model versioning is by GCS path convention only).
- [ ] **[backend]** Add an authentication layer if the endpoint moves from public to
      controlled access.
