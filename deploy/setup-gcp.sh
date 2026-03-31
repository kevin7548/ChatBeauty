#!/usr/bin/env bash
# =============================================================
# ChatBeauty — Cloud Run Deployment Script
# Run this from your LOCAL machine inside backend/
# =============================================================
set -euo pipefail

PROJECT_ID="your-gcp-project-id"    # <-- CHANGE THIS
REGION="asia-northeast3"
DB_INSTANCE="chatbeauty-db"
DB_PASSWORD="your-db-password"       # <-- CHANGE THIS
BUCKET="chatbeauty-models"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/chatbeauty/backend"

echo "=== Step 1: Enable APIs ==="
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  --project="$PROJECT_ID"

echo "=== Step 2: Create Artifact Registry ==="
gcloud artifacts repositories create chatbeauty \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID" 2>/dev/null || echo "Repository already exists"

echo "=== Step 3: Build & push Docker image ==="
cd "$(dirname "$0")/../backend"
gcloud builds submit --tag "$IMAGE" --timeout=1800 --project="$PROJECT_ID"

echo "=== Step 4: Deploy to Cloud Run ==="
gcloud run deploy chatbeauty-backend \
  --image="$IMAGE" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --memory=4Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=2 \
  --timeout=300 \
  --allow-unauthenticated \
  --add-cloudsql-instances="$PROJECT_ID:$REGION:$DB_INSTANCE" \
  --set-env-vars="DATABASE_URL=postgresql://postgres:$DB_PASSWORD@/chatbeauty?host=/cloudsql/$PROJECT_ID:$REGION:$DB_INSTANCE" \
  --set-env-vars="GEMINI_API_KEY=$GEMINI_API_KEY" \
  --set-env-vars="BGE_MODEL_PATH=/app/ml/model-gcs/retrieval/bge-m3-finetuned-20260202-120852" \
  --set-env-vars="RERANK_MODEL_PATH=/app/ml/model-gcs/reranking/lgbm_reranker_current_features_v1.pkl" \
  --execution-environment=gen2 \
  --add-volume=name=models,type=cloud-storage,bucket="$BUCKET" \
  --add-volume-mount=volume=models,mount-path=/app/ml/model-gcs

SERVICE_URL=$(gcloud run services describe chatbeauty-backend \
  --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')

echo ""
echo "=== Done! ==="
echo "Service URL: $SERVICE_URL"
echo ""
echo "Next steps:"
echo "  1. Upload models to GCS:  gcloud storage cp -r ml/model/* gs://$BUCKET/"
echo "  2. Populate database:     cloud-sql-proxy $PROJECT_ID:$REGION:$DB_INSTANCE"
echo "  3. Verify:                curl $SERVICE_URL/health"
