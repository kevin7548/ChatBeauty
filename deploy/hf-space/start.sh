#!/usr/bin/env bash
# Space entrypoint: pull the fine-tuned models from the HF Hub, then launch the API.
set -euo pipefail

: "${MODEL_REPO_ID:?Set MODEL_REPO_ID (e.g. <hf-username>/chatbeauty-models) in Space variables}"
: "${DATABASE_URL:?Set DATABASE_URL (Supabase pooler URL) in Space secrets}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY (Google AI Studio free-tier key) in Space secrets}"

MODEL_DIR="${MODEL_DIR:-/app/models}"

echo "Downloading models from HF Hub: ${MODEL_REPO_ID} -> ${MODEL_DIR}"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_REPO_ID"],
    repo_type="model",
    local_dir=os.environ.get("MODEL_DIR", "/app/models"),
)
PY

export BGE_MODEL_PATH="${MODEL_DIR}/retrieval/bge-m3-finetuned-20260202-120852"
export RERANK_MODEL_PATH="${MODEL_DIR}/reranking/lgbm_reranker_current_features_v1.pkl"
# In-memory FAISS ANN index (vector search runs in-process; see retrieval.py).
export ANN_INDEX_PATH="${MODEL_DIR}/retrieval/ann/faiss_hnsw_cosine.index"
export ANN_ASINS_PATH="${MODEL_DIR}/retrieval/ann/asins.json"

cd /app/src/backend
exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
