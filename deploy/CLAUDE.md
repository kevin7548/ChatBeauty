# deploy/

Infra/deployment for the backend. One file: `setup-gcp.sh` (run from your local machine).

## What `setup-gcp.sh` does
1. Enables GCP APIs (Run, SQL Admin, Artifact Registry, Cloud Build, Storage).
2. Creates the Artifact Registry repo (`asia-northeast3`).
3. Builds the image via **Cloud Build** (`gcloud builds submit`, no local Docker).
4. `gcloud run deploy chatbeauty-backend`: 4Gi/2CPU, min/max instances 1/2, timeout 300s,
   `--allow-unauthenticated`, Cloud SQL socket, gen2, and a **GCS volume** (`chatbeauty-models`)
   mounted at `/app/ml/model-gcs` for the BGE-M3 + LightGBM models.

Secrets (`DATABASE_URL`, `GEMINI_API_KEY`) are passed via `--set-env-vars`; `GEMINI_API_KEY`
must be exported in your shell first. Edit `PROJECT_ID` / `DB_PASSWORD` before running.

## Scope note
This script + `backend/Dockerfile` + `backend/docker-compose.yml` are the deploy surface.
Frontend deploys to Vercel (no config file; set `VITE_API_URL`).

## Docs
Topology, Cloud Run flags, env-var table, secrets, observability → `docs/deployment.md`
