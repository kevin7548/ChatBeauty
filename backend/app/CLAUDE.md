# backend/app/

FastAPI application. Orchestrates a 3-stage recommendation pipeline.

## Pipeline (per request, `api/routes/recommend.py`)
1. **Retrieval** (`services/retrieval.py`, ~1,600ms) — encode the query with BGE-M3 (1024-dim),
   search an **in-memory FAISS HNSW index** → Top-100 `parent_asin`, then fetch their metadata
   from Postgres (`WHERE parent_asin = ANY(...)`). No pgvector index — it wouldn't fit the free cap.
2. **Reranking** (`services/reranking.py`, ~850ms) — LightGBM over 10 features (1 batched DB
   feature lookup) → Top-5. `top_k` is **hardcoded to 5** in the route.
3. **Explanation** (`services/explanation.py`, ~2,400ms) — Gemini 2.5 Flash via `google-genai`,
   thinking disabled, **all 5 in one call**.
(Latencies on the free-tier HF Space CPU.)

## Layout
```
app/
├── main.py                       # app, CORS (ALLOWED_ORIGINS), LatencyMiddleware
├── api/routes/recommend.py       # POST /recommend; GET / and /health in main.py
├── services/
│   ├── retrieval.py / reranking.py / explanation.py
│   └── retrieval_resources.py    # singletons: BGE-M3 model + psycopg2 pool (keepalives)
├── models/schemas.py             # RecommendRequest(user_input) / RecommendResponse / ItemScore
└── middleware/latency.py         # X-Total-Latency-Ms header + per-request log
```

## Constraints
- Keep all 5 explanations in **one** Gemini call (latency).
- Embedding dim must stay **1024** (matches the `halfvec(1024)` column).
- BGE-M3 model + LightGBM pickle load at startup from the local **`ml/model/` mount**
  (docker-compose `/app/ml/model-gcs/`), not the image.
- `RecommendRequest` has only `user_input`; don't assume `top_k` is honored.

## Docs
Contract → `docs/api-spec.md` · Design → `docs/backend-architecture.md`
