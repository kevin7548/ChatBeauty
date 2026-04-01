# ChatBeauty - LLM & RAG 기반 아마존 뷰티 상품 추천 시스템

> [English version (README_EN.md)](README_EN.md)

ChatBeauty는 사용자의 피부 타입, 피부 고민, 선호 조건을 기반으로
**개인 맞춤형 화장품을 추천하고 추천 이유까지 설명해주는 서비스**입니다.

> "어떤 화장품이 나한테 맞을까?"
> "성분은 괜찮을까?"
> "종류는 많은데 뭘 골라야 할지 모르겠어…"

이런 고민을 해결하기 위해 만들어졌습니다.

---

## Demo

![ChatBeauty Demo](images/demo_video.gif)

[Demo Video (YouTube)](https://youtu.be/g0UO8cHWX9I)

---

## Project Overview

### What is ChatBeauty?

ChatBeauty는 자연어로 입력한 사용자 질의와 피부 정보를 기반으로
가장 적합한 화장품을 추천하는 AI 추천 시스템입니다.

단순 인기 제품 추천이 아니라,

- 왜 이 제품이 나에게 맞는지
- 어떤 성분이 도움이 되는지

를 함께 설명하는 것이 핵심 목표입니다.

### Expected Impact

- 사용자 **피부 타입, 고민, 선호 조건 반영**
- 대규모 화장품 데이터 기반 추천
- 추천 결과에 대한 **설명 제공**
- 선택에 대한 불확실성 감소

> 즉, "많이 팔린 제품"이 아니라
> **"나에게 맞는 제품"을 추천합니다.**

---

## Service Pipeline

사용자 시나리오 입력부터 추천 결과까지 3단계로 구성됩니다.

1. **Retrieval**: Fine-tuned BGE-M3로 사용자 시나리오를 인코딩하고, PostgreSQL + pgvector에서 IVFFlat 인덱스 기반 코사인 유사도 Top-100 후보 추출
2. **Re-ranking**: LightGBM (LambdaRank)으로 가격, 평점, 리뷰 수 등 10개 메타데이터 피처를 활용하여 Top-5 선정
3. **Explanation**: Gemini 2.5 Flash가 사용자 시나리오와 실제 리뷰 데이터를 기반으로 5개 상품에 대한 추천 이유를 한 번의 API 호출로 생성

![Service Pipeline](images/service_pipeline.png)

---

## Data

### Data Source

- **Amazon Reviews 2023 (All_Beauty)**
- 632k users / 112k items / 701k ratings
- Source: https://amazon-reviews-2023.github.io/

### Data Structure

![data structure](images/Amazon_data.png)

### EDA

![eda](images/EDA.png)

### Data Preprocessing

**문제 상황**: 사용자 리뷰 데이터에 대한 신뢰성 확보

**해결 방법**:
- 활동 시간 대비 리뷰 과다: 1시간 이내 리뷰를 10개 이상 작성한 유저
- 평점 분산 기반: 리뷰 수가 5개 이상인 유저 중 모든 평점을 동일하게 작성한 유저

→ 위 조건 중 하나라도 만족할 경우 비정상 의심 유저로 분류 → 약 0.3% 유저 데이터 제거

### Data Pipeline

Apache Beam 파이프라인으로 원본 데이터를 처리합니다:

```
All_Beauty.jsonl + meta_All_Beauty.jsonl
       │
       ▼
  Parse → Validate → Aggregate → Join
       │
       ├──→ PostgreSQL (112,578 products + 메타데이터 + 리뷰 통계)
       └──→ training_pairs.jsonl (BGE-M3 fine-tuning 학습 데이터)
```

별도 스크립트(`embed_products.py`)로 BGE-M3 임베딩을 계산하여 DB에 저장합니다.

### Database Schema

![database schema](images/data_schema.png)

---

## Recommendation Model

### Architecture

ChatBeauty는 Two-Tower 기반 구조를 사용합니다.

- **Query Tower**: 사용자 시나리오 텍스트 → Fine-tuned BGE-M3 → 1024-dim 벡터
- **Item Tower**: 제목 + 리뷰 키워드 + 설명 요약 + 특성 → Fine-tuned BGE-M3 → PostgreSQL + pgvector 저장
- **Fine-tuning**: MultipleNegativesRankingLoss, 두 가지 접근법 (리뷰 텍스트 기반 / LLM 생성 쿼리 기반)

![model architecture](images/model_architecture.png)

### Candidate Generation (Retrieval)

**Item Tower**: 4가지 텍스트를 결합하여 `embedding_text` 구성
- `[Title]` 상품명
- `[Review Keywords]` Llama 3.1로 리뷰에서 추출한 키워드 (WHO/WHEN/WHY)
- `[Description Summary]` Llama 3.1로 요약한 상품 설명
- `[Features]` 상품 특성

**User Tower**: 학습 시 리뷰 텍스트를 query로 사용하여 학습하고, 서비스 시 사용자가 입력한 자연어 시나리오를 query로 인코딩

**Fine-tuning 방법 — Review-based**

Raw review 텍스트를 query로, item의 `embedding_text`를 positive로 사용하여 학습

| 항목 | 값 |
|------|-----|
| Loss | MultipleNegativesRankingLoss |
| Training Pairs | ~1M |
| Epochs | 2 |
| Batch Size | 32 |
| Embedding Dim | 1024 |
| **Valid Recall@100** | 0.2015 → **0.3543** |
| **Test Recall@100** | **0.3728** |

### Re-ranking

1-stage Retrieval만으로는 구매 관점의 정렬이 부족하여, 후보를 정교하게 재정렬하는 2-stage 구조를 적용했습니다.

**모델 선정**: LightGBM (Leaf-wise 방식이 상위 랭크 패턴에 집중 → NDCG@5에 더 적합)

학습 데이터: 36.4M candidate rows (364K queries x 100 candidates)

| Metric | Value |
|--------|-------|
| **NDCG@5** | 0.3655 |
| **NDCG@10** | 0.4015 |

**Features** (10개):

| Feature | 설명 |
|---------|------|
| `retrieval_score` | BGE-M3 코사인 유사도 |
| `review_cnt` | 전체 리뷰 수 |
| `vp_review_cnt` | 구매 인증 리뷰 수 |
| `vp_ratio` | 구매 인증 리뷰 비율 |
| `recent_review_cnt` | 최근 리뷰 수 |
| `avg_rating` | 평균 평점 |
| `rating_std` | 평점 표준 편차 |
| `avg_review_len` | 평균 리뷰 길이 |
| `log_median_price` | 로그 변환 중간 가격 |
| `price_cnt` | 가격 샘플 수 |

### 추천 설명문 생성

Top-5 추천 상품에 대해 **Gemini 2.5 Flash**가 사용자 입력 시나리오와 상품 메타데이터(features, details, top_reviews)를 기반으로 개인화된 추천 이유를 생성합니다. 5개 상품의 설명을 단일 API 호출로 처리하여 지연시간을 최소화합니다.

---

## Production Latency

| 단계 | 시간 |
|------|------|
| Retrieval (pgvector IVFFlat) | ~1,100ms |
| Reranking (LightGBM) | ~19ms |
| Explanation (Gemini 2.5 Flash) | ~250ms |
| **Total** | **~1,400ms** |

---

## Deployment Architecture

```
Vercel (Frontend)                    Google Cloud
──────────────────                  ─────────────
React + TypeScript   ──API call──→  Cloud Run (FastAPI)
                                        │
                           ┌────────────┼────────────┐
                           ▼            ▼            ▼
                      Cloud SQL    GCS Volume    Gemini API
                     (pgvector)   (BGE-M3 model)  (2.5 Flash)
                     112K products  LightGBM pkl
```

- **Frontend**: Vercel (정적 호스팅, CDN)
- **Backend**: Cloud Run (서버리스, min-instances=1로 cold start 방지)
- **Database**: Cloud SQL PostgreSQL 16 + pgvector (IVFFlat 인덱스)
- **Model Storage**: GCS 버킷 → Cloud Run 볼륨 마운트

---

## Quick Start

### 로컬 실행 (Docker)

```bash
cd backend
docker-compose up --build -d
docker-compose exec -T db psql -U chatbeauty -d chatbeauty < sql/init.sql

python -m ml.pipeline.run \
  --input-reviews=ml/data/raw/All_Beauty.jsonl \
  --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
  --input-keywords=ml/data/processed/keywords_train.jsonl \
  --output-dir=ml/data/processed/beam_output \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty
```

### Cloud Run 배포

```bash
# 이미지 빌드 (Cloud Build)
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/chatbeauty/backend --timeout=1800

# 배포
gcloud run deploy chatbeauty-backend --image=IMAGE_URL --region=REGION ...
```

자세한 배포 방법은 [deploy/setup-gcp.sh](deploy/setup-gcp.sh)를 참고하세요.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)

| 분류 | 기술 |
|------|------|
| **Frontend** | React, TypeScript, Vite, Vercel |
| **Backend** | FastAPI, Uvicorn, Docker |
| **Embedding** | BAAI/bge-m3 (fine-tuned), sentence-transformers |
| **Database** | PostgreSQL 16 + pgvector (IVFFlat) |
| **LLM** | Llama 3.1:8B (키워드 추출), Gemini 2.5 Flash (추천 설명) |
| **Re-ranking** | LightGBM (LambdaRank) |
| **Data Pipeline** | Apache Beam (DirectRunner) |
| **Deployment** | Google Cloud Run, Cloud SQL, Cloud Storage, Vercel |

---

## Repository Structure

```
.
├── backend/
│   ├── app/                     # FastAPI API 서버
│   │   ├── api/routes/          #   /recommend 엔드포인트
│   │   ├── services/            #   retrieval, reranking, explanation
│   │   ├── models/              #   Pydantic 스키마
│   │   └── middleware/          #   레이턴시 로깅
│   ├── ml/
│   │   ├── item_ranker/         #   LightGBM re-ranking 라이브러리
│   │   ├── pipeline/            #   Apache Beam 데이터 파이프라인
│   │   ├── scripts/             #   embed_products.py (임베딩 계산)
│   │   └── model/               #   학습된 모델 (git 미포함)
│   ├── notebooks/               #   Colab 노트북 (LightGBM 학습, 임베딩)
│   ├── sql/init.sql             #   PostgreSQL 스키마 + pgvector
│   ├── Dockerfile
│   └── docker-compose.yml
├── frontend/                    # React 프론트엔드
├── deploy/setup-gcp.sh          # Cloud Run 배포 스크립트
├── images/                      # README 이미지
└── README.md
```

---

## 자체 평가

### 기술적 성과
- **2-stage 추천 파이프라인 구현**: Bi-encoder 기반 Retrieval과 LightGBM Reranking 단계를 분리하여, 112K 아이템 환경에서 ~1.4초 이내 추천 응답
- **PostgreSQL + pgvector 기반 벡터 검색**: 관계형 DB에서 메타데이터 필터링과 IVFFlat 인덱스 벡터 유사도 검색을 통합
- **LLM 기반 설명 가능한 추천 제공**: Gemini 2.5 Flash로 5개 상품 설명을 단일 API 호출로 생성하여 지연시간 최소화
- **서버리스 프로덕션 배포**: Cloud Run + Cloud SQL + GCS 볼륨 마운트 + Vercel로 확장 가능한 아키텍처 구현
- **Apache Beam 데이터 파이프라인**: 리뷰/메타데이터 파싱, 검증, 집계, 조인을 자동화하여 112K 상품을 DB에 적재

### 한계점
- 실제 서비스 로그가 없어, 오프라인 지표(Recall@100, NDCG@5) 중심의 제한적인 성능 평가에 머무름
- LLM이 생성한 추천 이유에 대해 체계적인 평가 기준을 충분히 마련하지 못함
- Retrieval 단계 지연시간(~1.1초)이 전체 응답시간의 대부분을 차지

### 향후 발전 계획
- **사용자 행동 데이터 기반 추천 고도화**: 클릭/선택 로그를 활용한 온라인 학습 및 추천 성능 개선
- **멀티모달 확장**: 텍스트뿐만 아니라 사용자의 피부 사진을 분석하여 추천에 반영하는 멀티모달 추천 시스템으로 고도화
- **Retrieval 최적화**: HNSW 인덱스 도입, 프로브 수 튜닝 등으로 검색 지연시간 개선

---

## Team

ChatBeauty Project Team - RecSys-07

![team members](images/team_members.png)
