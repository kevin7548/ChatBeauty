---
name: ml
description: ML tasks — BGE-M3 fine-tuning, LightGBM reranker, Apache Beam data pipeline, embedding scripts. Use for changes scoped to the ML layer.
---

You work only in the `ml/` directory. Never modify `frontend/`, `backend/`, `db/`, or `deploy/` files.

## Stack
- Embedding model: BAAI/bge-m3 (fine-tuned), sentence-transformers, PyTorch
- Reranker: LightGBM (LambdaRank)
- Data pipeline: Apache Beam (DirectRunner)
- Keyword extraction: Llama 3.1:8B

## Structure
```
ml/
├── item_ranker/      # LightGBM re-ranking library (used by backend at serve time)
├── pipeline/         # Apache Beam data pipeline (offline, training/processing)
├── scripts/          # embed_products.py (compute BGE-M3 embeddings → write to DB)
├── notebooks/        # Colab notebooks for embedding computation
└── model/            # trained models (not in git — kept locally; loaded by the backend mount)
```

## Pipeline
```
All_Beauty.jsonl + meta_All_Beauty.jsonl
  → Parse → Validate → Aggregate → Join
  → PostgreSQL (112K products + metadata + review stats)
  → training_pairs.jsonl (BGE-M3 fine-tuning data)
```

## Model details
- BGE-M3 fine-tuned with MultipleNegativesRankingLoss, ~1M training pairs, 2 epochs
- Embedding text = [Title] + [Review Keywords] + [Description Summary] + [Features]
- Valid Recall@100: 0.3543 (up from 0.2015 baseline)
- LightGBM trained on 36.4M candidate rows (364K queries × 100 candidates), NDCG@5: 0.3655

## Constraints
- `item_ranker/` is imported by `backend/app/` at serve time — keep its public API stable
- Embedding dimension must stay at 1024 to match the pgvector column
- Model artifacts stay in local `ml/model/` (loaded via the backend mount), not in the Docker image
