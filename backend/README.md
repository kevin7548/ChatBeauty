# Backend

LLM & RAG 기반 뷰티 제품 추천 시스템의 백엔드입니다.

## Architecture Overview

```
backend/
├── app/                          # FastAPI 애플리케이션 (API 서버)
│   ├── api/routes/               # /recommend 엔드포인트
│   ├── services/                 # 비즈니스 로직 (retrieval, reranking, explanation)
│   ├── models/                   # Pydantic 스키마
│   ├── middleware/               # 레이턴시 로깅
│   └── main.py                   # FastAPI 앱 진입점
│
├── ml/                           # ML 파이프라인
│   ├── item_ranker/              # Re-ranking 모델 라이브러리
│   │   ├── dataset/              #   데이터셋 파싱
│   │   ├── features/             #   피처 빌더 (TreeFeatureBuilder)
│   │   └── modeling/             #   학습 및 예측
│   ├── pipeline/                 # Apache Beam 데이터 파이프라인
│   │   └── transforms/           #   Parse, Validate, Aggregate, Join, Sink
│   ├── scripts/                  # 유틸리티 스크립트
│   │   └── embed_products.py     #   BGE-M3 임베딩 계산 → DB 저장
│   ├── model/                    # 학습된 모델 저장소 (git 미포함)
│   │   ├── retrieval/            #   Fine-tuned BGE-M3
│   │   └── reranking/            #   LightGBM reranker (.pkl)
│   └── data/                     # 데이터셋 (git 미포함)
│
├── notebooks/                    # Colab 노트북
│   ├── retrieve_and_train_lgbm.py  # LightGBM 학습 (T4 GPU)
│   └── embed_products_colab.ipynb  # 임베딩 계산 (T4 GPU)
│
├── sql/init.sql                  # PostgreSQL 스키마 (pgvector)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## System Flow

```
[User Query: "I have dry skin and want a gentle moisturizer"]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (app/)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Retrieval  │→ │  Reranking  │→ │   Explanation   │ │
│  │  (Top-100)  │  │   (Top-5)   │  │ (Gemini batch)  │ │
│  │  ~1,100ms   │  │   ~19ms     │  │   ~250ms        │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  Cloud SQL           GCS Volume          Gemini 2.5 Flash
  + pgvector         (BGE-M3, LightGBM)      API
  112K products
```

## Setup & Deployment

### 1. 로컬 실행 (Docker)

```bash
cd backend

# .env 파일 생성 (GEMINI_API_KEY 설정)
cp .env.example .env

# PostgreSQL + Backend 실행
docker-compose up --build -d

# 스키마 적용 (최초 1회)
docker-compose exec -T db psql -U chatbeauty -d chatbeauty < sql/init.sql
```

### 2. 데이터 적재 (Apache Beam)

```bash
pip install apache-beam psycopg2-binary

python -m ml.pipeline.run \
  --input-reviews=ml/data/raw/All_Beauty.jsonl \
  --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
  --input-keywords=ml/data/processed/keywords_train.jsonl \
  --output-dir=ml/data/processed/beam_output \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty
```

### 3. 임베딩 계산

로컬 (CPU, ~4시간):
```bash
python -m ml.scripts.embed_products \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty
```

또는 Colab T4 GPU (~15분):
`notebooks/embed_products_colab.ipynb` 사용

### 4. 검색 인덱스 생성

```sql
CREATE INDEX idx_products_embedding ON products
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 5. Cloud Run 배포

```bash
# Cloud Build로 이미지 빌드 (로컬 디스크 불필요)
gcloud builds submit \
  --tag REGION-docker.pkg.dev/PROJECT_ID/chatbeauty/backend \
  --timeout=1800

# Cloud Run 배포
gcloud run deploy chatbeauty-backend \
  --image=IMAGE_URL \
  --region=REGION \
  --add-cloudsql-instances=PROJECT_ID:REGION:chatbeauty-db \
  --add-volume=name=models,type=cloud-storage,bucket=chatbeauty-models \
  --add-volume-mount=volume=models,mount-path=/app/ml/model-gcs
```

전체 배포 스크립트: [deploy/setup-gcp.sh](../deploy/setup-gcp.sh)

### 6. 확인

```bash
# 헬스 체크
curl https://your-service-url/health

# API 문서
open https://your-service-url/docs
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API 키 |
| `DATABASE_URL` | PostgreSQL 연결 URL |
| `BGE_MODEL_PATH` | BGE-M3 모델 경로 (기본: GCS 마운트 경로) |
| `RERANK_MODEL_PATH` | LightGBM 모델 경로 (기본: GCS 마운트 경로) |
| `ALLOWED_ORIGINS` | CORS 허용 도메인 (Vercel URL) |

## Tech Stack

- **API Framework**: FastAPI, Uvicorn
- **Embedding Model**: BAAI/bge-m3 (fine-tuned), sentence-transformers
- **Database**: PostgreSQL 16 + pgvector (IVFFlat index)
- **Explanation LLM**: Google Gemini 2.5 Flash
- **Reranker**: LightGBM (LambdaRank, NDCG@5=0.3655)
- **Data Pipeline**: Apache Beam (DirectRunner)
- **Deployment**: Docker, Google Cloud Run, Cloud SQL, Cloud Storage

## Dataset

- **Amazon Reviews 2023 (All_Beauty)**
- 112,578 products indexed with 1024-dim BGE-M3 embeddings
- Source: https://amazon-reviews-2023.github.io/
