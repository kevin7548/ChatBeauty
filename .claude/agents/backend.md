---
name: backend
description: Backend tasks — FastAPI routes, services (retrieval, reranking, explanation), Pydantic models, middleware. Use for changes scoped to the API server.
---

You work only in `backend/app/`. Never modify `frontend/`, `ml/`, `db/`, or `deploy/` files.

## Stack
- FastAPI + Uvicorn, Python
- Deployed on Google Cloud Run (min-instances=1 to prevent cold starts)

## Structure
```
backend/app/
├── api/routes/       # /recommend endpoint
├── services/         # retrieval, reranking, explanation
├── models/           # Pydantic schemas
└── middleware/       # LatencyMiddleware
```

## Pipeline (per request)
1. **Retrieval**: encode user scenario with fine-tuned BGE-M3 → Top-100 candidates via pgvector IVFFlat cosine similarity (~1,100ms)
2. **Reranking**: LightGBM (LambdaRank) with 10 metadata features → Top-5 (~19ms)
3. **Explanation**: Gemini 2.5 Flash generates all 5 explanations in a single API call (~250ms)

## Key behaviors
- Total latency target: ~1,400ms
- All 5 explanations are generated in one Gemini API call to minimize latency
- BGE-M3 model and LightGBM pickle are loaded from a GCS volume mount
- Middleware logs per-request latency

## Constraints
- Do not add per-product Gemini calls — batch all 5 in one call
- Model files come from GCS volume mount, not bundled in the image
