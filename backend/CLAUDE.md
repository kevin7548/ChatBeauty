# backend/

FastAPI recommendation service + PostgreSQL/pgvector schema + container config.

## Stack
FastAPI + Uvicorn (Python 3.12, Poetry). PostgreSQL 16 + pgvector. Gemini 2.5 Flash for
explanations. LightGBM reranker (imported from `ml/item_ranker`, installed via
`pip install -e ml/`).

## Layout
```
backend/
├── app/                  # FastAPI app — see backend/app/CLAUDE.md
├── sql/init.sql          # schema — see backend/sql/CLAUDE.md
├── Dockerfile            # python:3.12-slim; models from GCS mount, not the image
├── docker-compose.yml    # local: Postgres (pgvector) + backend
└── pyproject.toml
```

## Run locally
```bash
cd backend && cp .env.example .env   # set GEMINI_API_KEY
docker-compose up --build -d         # db + backend on :8080
# or app only: uvicorn app.main:app --reload --port 8000
```

## Env vars
`GEMINI_API_KEY` (required), `DATABASE_URL`, `BGE_MODEL_PATH`, `RERANK_MODEL_PATH`,
`ALLOWED_ORIGINS`, `DEBUG`. Template: `.env.example`. Details: `docs/deployment.md`.

## Docs
- API contract → `docs/api-spec.md`
- Service design → `docs/backend-architecture.md`
- Schema → `docs/db-schema.md`
- Deploy → `docs/deployment.md`
