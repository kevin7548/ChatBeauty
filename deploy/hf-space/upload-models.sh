#!/usr/bin/env bash
# Push the local fine-tuned models to a free HF Hub model repo so the Space can download
# them at startup. Run from the repo ROOT after `huggingface-cli login` (WRITE token).
#
#   deploy/hf-space/upload-models.sh <hf-username>/chatbeauty-models
#
# Mirrors the local ml/model/ layout into the repo (retrieval/... + reranking/...).
set -euo pipefail

REPO_ID="${1:?Usage: $0 <hf-username>/chatbeauty-models}"
BGE_DIR="ml/model/retrieval/bge-m3-finetuned-20260202-120852"
LGBM_PKL="ml/model/reranking/lgbm_reranker_current_features_v1.pkl"

[ -f "$BGE_DIR/model.safetensors" ] || { echo "Missing $BGE_DIR/model.safetensors — run from repo root."; exit 1; }
[ -f "$LGBM_PKL" ] || { echo "Missing $LGBM_PKL — run from repo root."; exit 1; }

huggingface-cli repo create "$REPO_ID" --type model -y || true

echo "Uploading BGE-M3 (~2.1 GB)…"
huggingface-cli upload "$REPO_ID" "$BGE_DIR" "retrieval/bge-m3-finetuned-20260202-120852" --repo-type model

echo "Uploading LightGBM reranker…"
huggingface-cli upload "$REPO_ID" "$LGBM_PKL" "reranking/lgbm_reranker_current_features_v1.pkl" --repo-type model

echo "Done. Set MODEL_REPO_ID=$REPO_ID as a Space variable."
