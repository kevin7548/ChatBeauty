FROM python:3.12-slim

WORKDIR /app

# System dependencies for lightgbm (libgomp) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies via Poetry
COPY backend/pyproject.toml ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main --no-root

# Install item_ranker local package
COPY ml/setup.py ml/setup.py
COPY ml/item_ranker/ ml/item_ranker/
RUN pip install -e ml/

# Copy application code
COPY backend/app/ app/

# Models are not baked into the image. docker-compose mounts the local ml/model/
# read-only at /app/ml/model/ (the default paths in app/services/). Required:
#   /app/ml/model/retrieval/bge-m3-finetuned-20260202-120852/
#   /app/ml/model/retrieval/ann/faiss_hnsw_cosine.index + asins.json
#   /app/ml/model/reranking/lgbm_reranker_current_features_v1.pkl

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
