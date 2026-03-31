# Backend

LLM & RAG 기반 뷰티 제품 추천 시스템의 백엔드입니다.

## Architecture Overview

```
backend/
├── app/                          # FastAPI 애플리케이션 (API 서버)
│   ├── api/routes/               # API 엔드포인트
│   ├── services/                 # 비즈니스 로직 (retrieval, reranking, explanation)
│   ├── models/                   # Pydantic 스키마
│   ├── middleware/               # 레이턴시 미들웨어
│   └── main.py                   # FastAPI 앱 진입점
│
├── ml/                           # ML 파이프라인
│   ├── item_ranker/              # Re-ranking 모델 라이브러리 (LightGBM / XGBoost)
│   │   ├── dataset/              #   데이터셋 파싱
│   │   ├── features/             #   피처 빌더 (TreeFeatureBuilder)
│   │   └── modeling/             #   학습 및 예측
│   ├── pipeline/                 # Apache Beam 데이터 파이프라인
│   │   └── transforms/           #   Parse, Validate, Aggregate, Join, Sink
│   ├── model/                    # 학습된 모델 저장소 (git 미포함)
│   │   ├── retrieval/            #   Fine-tuned BGE-M3
│   │   └── reranking/            #   LightGBM reranker (.pkl)
│   └── data/                     # 데이터셋 (git 미포함)
│
├── sql/init.sql                  # PostgreSQL 스키마 (pgvector)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## System Flow

```
[User Scenario]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (app/)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Retrieval  │→ │  Reranking  │→ │   Explanation   │ │
│  │  (Top-100)  │  │   (Top-5)   │  │  (LLM 생성)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  PostgreSQL          LightGBM             Gemini 2.0
  + pgvector          (ml/model/)          Flash API
```

## Quick Start (Local with Docker)

```bash
cd backend

# 1. Create .env
cp .env.example .env
# Fill in GEMINI_API_KEY

# 2. Start PostgreSQL + Backend
docker-compose up --build -d

# 3. Apply schema (first time only)
docker-compose exec -T db psql -U chatbeauty -d chatbeauty < sql/init.sql

# 4. Run Beam pipeline to populate database
pip install apache-beam psycopg2-binary
python -m ml.pipeline.run \
  --input-reviews=ml/data/raw/All_Beauty.jsonl \
  --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
  --input-keywords=ml/data/processed/keywords_train.jsonl \
  --output-dir=ml/data/processed/beam_output \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty

# 5. Verify
curl http://localhost:8000/health
```

API docs: http://localhost:8000/docs

## Cloud Run Deployment

See [deploy/setup-gcp.sh](../deploy/setup-gcp.sh) for the full deployment script.

## Tech Stack

- **API Framework**: FastAPI, Uvicorn
- **Embedding Model**: BAAI/bge-m3 (fine-tuned)
- **Database**: PostgreSQL + pgvector
- **Explanation LLM**: Google Gemini 2.0 Flash
- **Reranker**: LightGBM (LambdaRank)
- **Data Pipeline**: Apache Beam

## Dataset

- **Amazon Reviews 2023 (All_Beauty)**
- 632k users / 112k items / 701k ratings
- Source: https://amazon-reviews-2023.github.io/
