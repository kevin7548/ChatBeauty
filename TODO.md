# ChatBeauty — Project TODO

Single source of truth for remaining work. ChatBeauty is production-ready (frontend on
Vercel, backend on Cloud Run, models trained and deployed). The **Current focus** below is
the active roadmap; the **Backlog** holds documented limitations and longer-horizon ideas.

Detailed rationale for the documented limitations lives in
[`docs/architecture.md#known-limitations--future-work`](docs/architecture.md#known-limitations--future-work).

**Domain tags:** `[backend]` `[ml]` `[db]` `[deploy]` `[frontend]` `[test]` — map to the
domain agents in `.claude/agents/`.

---

## 🎯 Current focus

- [ ] **[db][backend] Migrate embeddings to `halfvec` to cut memory (Supabase free tier).**
      Switch the column from `vector(1024)` to `halfvec(1024)` — ~50% less storage/RAM (2 vs
      4 bytes/dim). Update in lockstep: the column in
      [`backend/sql/init.sql`](backend/sql/init.sql) (L7), the index opclass
      (`halfvec_cosine_ops`), the `::vector` casts in
      [`backend/app/services/retrieval.py`](backend/app/services/retrieval.py) (L27, L30 →
      `::halfvec`), and the write in
      [`ml/scripts/embed_products.py`](ml/scripts/embed_products.py). Requires pgvector ≥ 0.7;
      recall impact is negligible.

- [ ] **[db] Switch the vector index from IVFFlat to HNSW.**
      Replace the IVFFlat index (commented in [`backend/sql/init.sql`](backend/sql/init.sql)
      L34-36) with HNSW (`USING hnsw (embedding halfvec_cosine_ops)`, tune `m` /
      `ef_construction`, set `ef_search` at query time). Better recall/latency on the ~1.1s
      retrieval stage; pairs naturally with the `halfvec` migration above. Update
      [`docs/db-schema.md`](docs/db-schema.md).

- [ ] **[ml] Derive purchase signals from reviews to build a user-tower (two-tower retrieval).**
      Reviews carry `verified_purchase` (already used in
      [`ml/pipeline/transforms/aggregate.py`](ml/pipeline/transforms/aggregate.py) L93) plus
      `user_id` — use these as implicit purchase interactions to train a user tower alongside
      the item embeddings, enabling personalized retrieval. Larger ML effort; spec the data
      derivation + training in [`docs/ml-pipeline.md`](docs/ml-pipeline.md) first.

- [ ] **[deploy][test] Add GitHub Actions CI/CD.**
      No `.github/workflows/` exists yet. CI: lint + the `POST /recommend` test (see backlog).
      CD: build/deploy the backend to Cloud Run; Vercel already auto-deploys the frontend.
      Source [`deploy/setup-gcp.sh`](deploy/setup-gcp.sh) for the deploy steps.

- [ ] **[ml] Evaluate whether Apache Beam is still warranted.**
      Keyword *extraction* happens upstream — the pipeline only *reads* a pre-made
      `keywords_train.jsonl` ([`ml/pipeline/transforms/parse.py`](ml/pipeline/transforms/parse.py)
      `ParseKeywordFn`). On `DirectRunner` over ~112k products, Beam's real work is just
      feature aggregation + training-pair generation, both doable with a simpler pandas/SQL
      batch script. Decide: keep Beam, or replace
      [`ml/pipeline/`](ml/pipeline/run.py) with a lean script.

---

## Backlog

### 🟡 Improvements

- [ ] **[backend][test]** Add a pytest integration test for `POST /recommend` (docs mark this
      *planned*) — also the gate for CI above.
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

### 🟢 Longer-horizon

- [ ] **[db]** Adopt a DB migration tool (Alembic) to replace re-applying
      [`backend/sql/init.sql`](backend/sql/init.sql) by hand.
- [ ] **[ml]** Add MLOps tooling — experiment tracking (MLflow/W&B) + production model/data
      drift monitoring (model versioning is by GCS path convention only).
- [ ] **[backend]** Add an authentication layer if the endpoint moves from public to
      controlled access.
