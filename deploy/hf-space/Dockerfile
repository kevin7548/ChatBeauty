# Hugging Face Docker Space for the ChatBeauty backend (free CPU Space, 16 GB RAM).
#
# Self-contained: clones the public repo at build (models are NOT in git), installs the
# backend exactly like backend/Dockerfile, and downloads the fine-tuned models from the
# HF Hub at startup (see start.sh). HF Spaces serve on port 7860.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Application code from GitHub. To deploy a fresh main, bump CACHEBUST in the Space's
# Dockerfile and re-upload — that busts this layer's cache so the clone re-runs.
ARG REPO_URL=https://github.com/kevin7548/ChatBeauty
ARG REPO_REF=main
ARG CACHEBUST=1
RUN echo "cachebust=${CACHEBUST}" && git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} src

# Backend deps (mirrors backend/Dockerfile) + the local item_ranker package + hub tooling.
RUN pip install --no-cache-dir poetry huggingface_hub hf_transfer && \
    cd src/backend && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main --no-root && \
    pip install -e /app/src/ml

# HF Spaces run as a non-root user; model-download + cache dirs must be writable at runtime.
ENV HF_HOME=/app/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_DIR=/app/models \
    PYTHONPATH=/app/src/backend
RUN mkdir -p /app/models /app/.cache && chmod -R 777 /app/models /app/.cache

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860
CMD ["/app/start.sh"]
