# backend/app/

FastAPI application. Orchestrates a 3-stage recommendation pipeline.

## Pipeline (per request, `api/routes/recommend.py`)
1. **Retrieval** (`services/retrieval.py`, ~1,100ms) — encode the query with BGE-M3 (1024-dim),
   pgvector cosine search → Top-100 candidates.
2. **Reranking** (`services/reranking.py`, ~19ms) — LightGBM over 10 features (1 batched DB
   feature lookup) → Top-5. `top_k` is **hardcoded to 5** in the route.
3. **Explanation** (`services/explanation.py`, ~250ms) — Gemini 2.5 Flash, **all 5 in one call**.

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
- Embedding dim must stay **1024** (matches the `vector(1024)` column).
- BGE-M3 model + LightGBM pickle load at startup from the **GCS volume mount**, not the image.
- `RecommendRequest` has only `user_input`; don't assume `top_k` is honored.

## Docs
Contract → `docs/api-spec.md` · Design → `docs/backend-architecture.md`
