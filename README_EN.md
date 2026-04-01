# ChatBeauty - LLM & RAG-based Amazon Beauty Product Recommendation System

ChatBeauty is a service that recommends **personalized cosmetics based on the user's skin type, concerns, and preferences, along with explanations for each recommendation.**

> "Which cosmetics are right for me?"
> "Are the ingredients safe?"
> "There are so many options — how do I choose?"

Built to solve these common dilemmas.

---

## Demo

![ChatBeauty Demo](images/demo_video.gif)

[Demo Video (YouTube)](https://youtu.be/g0UO8cHWX9I)

---

## Project Overview

### What is ChatBeauty?

ChatBeauty is an AI recommendation system that suggests the most suitable cosmetics based on natural language queries and skin information.

Rather than simply recommending popular products, the core goal is to explain:

- Why this product suits you
- Which ingredients are beneficial

### Expected Impact

- Reflects user **skin type, concerns, and preferences**
- Recommendations based on large-scale cosmetics data
- **Explainable recommendations**
- Reduces uncertainty in product selection

> Not "best-selling products,"
> but **"the right products for you."**

---

## Service Pipeline

The pipeline consists of 3 stages from user scenario input to recommendation results.

1. **Retrieval**: Encode user scenario with fine-tuned BGE-M3, extract Top-100 candidates from PostgreSQL + pgvector via IVFFlat index cosine similarity
2. **Re-ranking**: Use LightGBM (LambdaRank) with 10 metadata features (price, rating, review count, etc.) to select Top-5
3. **Explanation**: Gemini 2.5 Flash generates personalized recommendation reasons for all 5 products in a single API call based on the user's scenario and actual review data

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

**Problem**: Ensuring reliability of user review data

**Solution**:
- Excessive reviews relative to activity time: users who wrote 10+ reviews within 1 hour
- Rating variance-based: users with 5+ reviews who gave identical ratings for all

→ Users meeting any of the above criteria were classified as suspected abnormal users → ~0.3% of user data removed

### Data Pipeline

Apache Beam pipeline processes raw data:

```
All_Beauty.jsonl + meta_All_Beauty.jsonl
       │
       ▼
  Parse → Validate → Aggregate → Join
       │
       ├──→ PostgreSQL (112,578 products + metadata + review stats)
       └──→ training_pairs.jsonl (BGE-M3 fine-tuning data)
```

A separate script (`embed_products.py`) computes BGE-M3 embeddings and stores them in the database.

### Database Schema

![database schema](images/data_schema.png)

---

## Recommendation Model

### Architecture

ChatBeauty uses a Two-Tower architecture.

- **Query Tower**: User scenario text → Fine-tuned BGE-M3 → 1024-dim vector
- **Item Tower**: Title + review keywords + description summary + features → Fine-tuned BGE-M3 → stored in PostgreSQL + pgvector
- **Fine-tuning**: MultipleNegativesRankingLoss, two approaches (review text-based / LLM-generated query-based)

![model architecture](images/model_architecture.png)

### Candidate Generation (Retrieval)

**Item Tower**: Combines 4 text fields into `embedding_text`
- `[Title]` Product name
- `[Review Keywords]` Keywords extracted from reviews via Llama 3.1 (WHO/WHEN/WHY)
- `[Description Summary]` Product description summarized by Llama 3.1
- `[Features]` Product features

**User Tower**: Trained using review text as queries; at inference, encodes the user's natural language scenario as the query

**Fine-tuning Approach A — Review-based (adopted)**

Raw review text as query, item's `embedding_text` as positive

| Item | Value |
|------|-------|
| Loss | MultipleNegativesRankingLoss |
| Training Pairs | ~1M |
| Epochs | 2 |
| Batch Size | 32 |
| Embedding Dim | 1024 |
| **Valid Recall@100** | 0.2015 → **0.3543** |
| **Test Recall@100** | **0.3728** |

**Fine-tuning Approach B — Generated Query (experimental)**

Natural language queries generated from reviews via Llama 3.1 (excluded: negative reviews + rating < 4.0 + rating_number < 20)

| Item | Value |
|------|-------|
| Training Pairs | ~100K |
| Batch Size | 16 |
| **Valid Recall@100** | 0.0543 → **0.1092** |
| **Test Recall@100** | **0.1587** |

### Re-ranking

1-stage retrieval alone lacks purchase-oriented ranking, so a 2-stage structure was applied for refined re-ranking of candidates.

**Model**: LightGBM (leaf-wise approach focuses on top-rank patterns → better suited for NDCG@5)

Training data: 36.4M candidate rows (364K queries x 100 candidates)

| Metric | Value |
|--------|-------|
| **NDCG@5** | 0.3655 |
| **NDCG@10** | 0.4015 |

**Features** (10):

| Feature | Description |
|---------|-------------|
| `retrieval_score` | BGE-M3 cosine similarity |
| `review_cnt` | Total review count |
| `vp_review_cnt` | Verified purchase review count |
| `vp_ratio` | Verified purchase ratio |
| `recent_review_cnt` | Recent review count |
| `avg_rating` | Average rating |
| `rating_std` | Rating standard deviation |
| `avg_review_len` | Average review length |
| `log_median_price` | Log-transformed median price |
| `price_cnt` | Number of price samples |

### Recommendation Explanation

For the final Top-5 products, **Gemini 2.5 Flash** generates personalized recommendation reasons based on the user's input scenario and item metadata (features, details, top_reviews). All 5 explanations are generated in a single API call to minimize latency.

---

## Production Latency

| Stage | Time |
|-------|------|
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

- **Frontend**: Vercel (static hosting, CDN)
- **Backend**: Cloud Run (serverless, min-instances=1 to prevent cold starts)
- **Database**: Cloud SQL PostgreSQL 16 + pgvector (IVFFlat index)
- **Model Storage**: GCS bucket → Cloud Run volume mount

---

## Quick Start

### Local (Docker)

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

### Cloud Run Deployment

```bash
# Build image (Cloud Build — no local disk needed)
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/chatbeauty/backend --timeout=1800

# Deploy
gcloud run deploy chatbeauty-backend --image=IMAGE_URL --region=REGION ...
```

See [deploy/setup-gcp.sh](deploy/setup-gcp.sh) for the full deployment script.

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

| Category | Technologies |
|----------|-------------|
| **Frontend** | React, TypeScript, Vite, Vercel |
| **Backend** | FastAPI, Uvicorn, Docker |
| **Embedding** | BAAI/bge-m3 (fine-tuned), sentence-transformers |
| **Database** | PostgreSQL 16 + pgvector (IVFFlat) |
| **LLM** | Llama 3.1:8B (keyword extraction), Gemini 2.5 Flash (explanations) |
| **Re-ranking** | LightGBM (LambdaRank) |
| **Data Pipeline** | Apache Beam (DirectRunner) |
| **Deployment** | Google Cloud Run, Cloud SQL, Cloud Storage, Vercel |

---

## Repository Structure

```
.
├── backend/
│   ├── app/                     # FastAPI API server
│   │   ├── api/routes/          #   /recommend endpoint
│   │   ├── services/            #   retrieval, reranking, explanation
│   │   ├── models/              #   Pydantic schemas
│   │   └── middleware/          #   latency logging
│   ├── ml/
│   │   ├── item_ranker/         #   LightGBM re-ranking library
│   │   ├── pipeline/            #   Apache Beam data pipeline
│   │   ├── scripts/             #   embed_products.py (embedding computation)
│   │   └── model/               #   trained models (not in git)
│   ├── notebooks/               #   Colab notebooks (LightGBM training, embedding)
│   ├── sql/init.sql             #   PostgreSQL schema + pgvector
│   ├── Dockerfile
│   └── docker-compose.yml
├── frontend/                    # React frontend
├── deploy/setup-gcp.sh          # Cloud Run deployment script
├── images/                      # README images
└── README.md
```

---

## Evaluation

### Technical Achievements
- **2-stage recommendation pipeline**: Bi-encoder retrieval + LightGBM reranking delivering recommendations within ~1.4s across 112K items
- **PostgreSQL + pgvector vector search**: Integrated metadata filtering and IVFFlat index vector similarity in a relational DB
- **LLM-based explainable recommendations**: Gemini 2.5 Flash generates all 5 product explanations in a single API call for minimal latency
- **Serverless production deployment**: Cloud Run + Cloud SQL + GCS volume mounts + Vercel for a scalable architecture
- **Apache Beam data pipeline**: Automated parsing, validation, aggregation, and joining of 112K products into the database

### Limitations
- Limited to offline metric-based evaluation (Recall@100, NDCG@5) due to the absence of real service logs
- Insufficient systematic evaluation criteria for LLM-generated recommendation explanations
- Retrieval latency (~1.1s) dominates total response time

### Future Plans
- **User behavior data-driven improvement**: Online learning and recommendation refinement using click/selection logs
- **Multimodal extension**: Evolve into a multimodal recommendation system that analyzes user skin photos in addition to text
- **Retrieval optimization**: Introduce HNSW index, tune probe count to reduce search latency

---

## Team

ChatBeauty Project Team - RecSys-07

![team members](images/team_members.png)
