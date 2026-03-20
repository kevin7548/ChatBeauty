FROM python:3.12-slim

WORKDIR /app

# System dependencies for lightgbm (libgomp) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies via Poetry
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main

# Install item_ranker local package
COPY ml/setup.py ml/setup.py
COPY ml/item_ranker/ ml/item_ranker/
RUN pip install -e ml/

# Copy application code
COPY app/ app/

# Models and data are mounted as volumes, NOT baked into the image.
# Expected mount points:
#   /app/ml/model/retrieval/bge-m3-finetuned-20260202-120852/
#   /app/ml/data/chromadb/
#   /app/ml/data/cleaned/item_v1/item_features_v1.csv
#   /app/ml/model/reranking/lgbm_reranker_current_features_v1.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
