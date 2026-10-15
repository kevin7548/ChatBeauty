# Development

End-to-end local setup for the whole stack. Layers: [backend](backend-architecture.md),
[frontend](frontend-architecture.md), [db](db-schema.md), [ml](ml-pipeline.md),
[deploy](deployment.md).

## Prerequisites
- Docker + docker-compose, Python 3.12, Node 18+.
- A `GEMINI_API_KEY` (explanations) and the model artifacts under `ml/model/`:
  `retrieval/bge-m3-finetuned-20260202-120852/`, `retrieval/ann/{faiss_hnsw_cosine.index,asins.json}`,
  `reranking/lgbm_reranker_current_features_v1.pkl`. They live in the HF Hub model repo
  (`kevin7548/chatbeauty-models`, same layout); pull whatever is missing locally:

  ```bash
  huggingface-cli download kevin7548/chatbeauty-models --local-dir ml/model
  ```

## 1. Backend + database (Docker)

```bash
cd backend
cp .env.example .env          # then set GEMINI_API_KEY
docker-compose up --build -d  # Postgres (pgvector) + FastAPI on :8080
# init.sql auto-runs on first DB start; to (re)apply manually:
docker-compose exec -T db psql -U chatbeauty -d chatbeauty < sql/init.sql
```

docker-compose mounts `ml/model/` read-only into the backend at `/app/ml/model/` (the
backend's default model paths), so the artifacts above — including the FAISS `retrieval/ann/`
index — must exist locally or the backend fails at import time.

> **Hosted free DB option:** instead of the local docker-compose Postgres, point `DATABASE_URL`
> at a **Supabase** free-tier project (Postgres + pgvector + `halfvec`). This is the project's
> deployment direction since the paid GCP infra was retired — see
> [deployment.md](deployment.md). Local docker-compose remains the simplest dev loop.

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

(Or use `ml/notebooks/embed_products_supabase.ipynb` on a Colab GPU.) Serving does **not**
use a pgvector index: rebuild the in-memory FAISS HNSW index from the new embeddings and
place it under `ml/model/retrieval/ann/` (procedure: [DEPLOY.md](../deploy/hf-space/DEPLOY.md),
Phase B). If you only need to run the stack, skip this step and use the index from the HF Hub.

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

Offline tests (no models/DB/Gemini needed — services are stubbed):

```bash
cd backend && python -m pytest
```

## Keeping docs current
`docs/` is the source of truth. When you change a contract, update the matching doc in the
same PR:
- API change → [api-spec.md](api-spec.md)
- DB schema change → [db-schema.md](db-schema.md)
- Architecture / new component → [architecture.md](architecture.md)
- ML pipeline / features / models → [ml-pipeline.md](ml-pipeline.md)

The PR template carries a checklist item for this.
