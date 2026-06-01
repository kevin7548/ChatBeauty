# Development

End-to-end local setup for the whole stack. Layers: [backend](backend-architecture.md),
[frontend](frontend-architecture.md), [db](db-schema.md), [ml](ml-pipeline.md),
[deploy](deployment.md).

## Prerequisites
- Docker + docker-compose, Python 3.12, Node 18+.
- A `GEMINI_API_KEY` (explanations) and the trained models under `ml/model/`
  (`retrieval/bge-m3-finetuned-20260202-120852/`, `reranking/lgbm_reranker_current_features_v1.pkl`).

## 1. Backend + database (Docker)

```bash
cd backend
cp .env.example .env          # then set GEMINI_API_KEY
docker-compose up --build -d  # Postgres (pgvector) + FastAPI on :8080
# init.sql auto-runs on first DB start; to (re)apply manually:
docker-compose exec -T db psql -U chatbeauty -d chatbeauty < sql/init.sql
```

docker-compose mounts `ml/model/...` read-only into the backend at `/app/ml/model-gcs/`,
so the models must exist locally.

## 2. Load data (Apache Beam)

From the repo root, populate `products` + review-stat features and emit training pairs:

```bash
python -m ml.pipeline.run \
  --input-reviews=ml/data/raw/All_Beauty.jsonl \
  --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
  --input-keywords=ml/data/processed/keywords_train.jsonl \
  --output-dir=ml/data/processed/beam_output \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty
```

## 3. Embeddings + vector index

```bash
python -m ml.scripts.embed_products \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty
```

(Or use `ml/notebooks/embed_products_colab.ipynb` on a GPU.) Then build the HNSW index
once embeddings exist:

```sql
CREATE INDEX idx_products_embedding ON products
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
```

## 4. Frontend

```bash
cd frontend
cp .env.example .env          # VITE_API_URL=http://localhost:8080
npm install
npm run dev                   # Vite dev server on :5173
```

Ensure `http://localhost:5173` is in the backend's `ALLOWED_ORIGINS` (the default).

## 5. Verify

```bash
curl http://localhost:8080/health                         # {"status":"ok"}
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_input":"gentle moisturizer for dry skin"}'   # Top-5 + latency
```

> There is no automated test suite yet
> ([Known limitations](architecture.md#known-limitations--future-work)); verification is
> manual via `/health` + a sample `/recommend`.

## Keeping docs current
`docs/` is the source of truth. When you change a contract, update the matching doc in the
same PR:
- API change → [api-spec.md](api-spec.md)
- DB schema change → [db-schema.md](db-schema.md)
- Architecture / new component → [architecture.md](architecture.md)
- ML pipeline / features / models → [ml-pipeline.md](ml-pipeline.md)

The PR template carries a checklist item for this.
