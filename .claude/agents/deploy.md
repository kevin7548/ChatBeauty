---
name: deploy
description: Deployment and infrastructure tasks — Cloud Run, Cloud SQL, GCS, Vercel, Dockerfile, docker-compose. Use for changes scoped to deployment config.
---

You work only in `deploy/` and deployment-related config files: `backend/Dockerfile`, `backend/docker-compose.yml`. Never modify application code in `frontend/src/`, `backend/app/`, `ml/`, or `backend/sql/`.

## Architecture
```
Vercel (Frontend)              Google Cloud
─────────────────              ─────────────
React + TypeScript  ──API──→  Cloud Run (FastAPI)
                                   │
                      ┌────────────┼────────────┐
                      ▼            ▼            ▼
                 Cloud SQL     GCS Volume    Gemini API
                (pgvector)    (BGE-M3 model)  (2.5 Flash)
                112K products  LightGBM pkl
```

## Key config
- Cloud Run: min-instances=1 (prevents cold starts), build via Cloud Build (not local Docker)
- Cloud SQL: PostgreSQL 16 + pgvector, private IP
- GCS: model volume mounted into Cloud Run at runtime (BGE-M3 + LightGBM pkl)
- Dockerfile build context is project root (`..`) — COPY paths reference `backend/` and `ml/`
- Vercel: static hosting, env var for backend URL

## Deployment commands
```bash
# Build (Cloud Build)
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/chatbeauty/backend --timeout=1800

# Deploy
gcloud run deploy chatbeauty-backend --image=IMAGE_URL --region=REGION ...
```

## Constraints
- Never build the Docker image locally for production — use Cloud Build (avoids local disk limits)
- Model files must come from GCS volume mount, not bundled in the image
- min-instances=1 is intentional — do not set to 0 (cold start is ~10s+)
