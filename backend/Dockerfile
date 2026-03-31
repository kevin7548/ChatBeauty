FROM python:3.12-slim

WORKDIR /app

# System dependencies for lightgbm (libgomp) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies via Poetry
COPY pyproject.toml ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main --no-root

# Install item_ranker local package
COPY ml/setup.py ml/setup.py
COPY ml/item_ranker/ ml/item_ranker/
RUN pip install -e ml/

# Copy application code
COPY app/ app/

# Models are loaded from GCS volume mount at /app/ml/model-gcs/
# Required:
#   /app/ml/model-gcs/retrieval/bge-m3-finetuned-20260202-120852/
#   /app/ml/model-gcs/reranking/lgbm_reranker_current_features_v1.pkl

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
