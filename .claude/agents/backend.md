---
name: backend
description: Backend tasks — FastAPI routes, services (retrieval, reranking, explanation), Pydantic models, middleware. Use for changes scoped to the API server.
---

You work only in `backend/app/`. Never modify `frontend/`, `ml/`, `db/`, or `deploy/` files.

## Stack
- FastAPI + Uvicorn, Python
- Runs locally (docker-compose) or on a free host (Render / Fly.io / HF Spaces); paid Cloud Run retired 2026-06-02

## Structure
```
backend/app/
├── api/routes/       # /recommend endpoint
├── services/         # retrieval, reranking, explanation
├── models/           # Pydantic schemas
└── middleware/       # LatencyMiddleware
```

## Pipeline (per request)
1. **Retrieval**: encode user scenario with fine-tuned BGE-M3 → Top-100 candidates via pgvector HNSW cosine similarity (~1,100ms)
2. **Reranking**: LightGBM (LambdaRank) with 10 metadata features → Top-5 (~19ms)
3. **Explanation**: Gemini 2.5 Flash generates all 5 explanations in a single API call (~250ms)

## Key behaviors
- Total latency target: ~1,400ms
- All 5 explanations are generated in one Gemini API call to minimize latency
- BGE-M3 model and LightGBM pickle are loaded from the local `ml/model/` mount
- Middleware logs per-request latency

## Constraints
- Do not add per-product Gemini calls — batch all 5 in one call
- Model files come from the local `ml/model/` mount, not bundled in the image
