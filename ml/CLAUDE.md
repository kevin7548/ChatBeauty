# ml/

Offline ML layer: Apache Beam data pipeline, BGE-M3 fine-tuning (retrieval), LightGBM reranker.

## Three components
- **Retrieval** — `BAAI/bge-m3` fine-tuned (MultipleNegativesRankingLoss, ~1M pairs, dim 1024).
  Embeddings computed by `scripts/embed_products.py` → PostgreSQL. Valid Recall@100 0.3543.
- **Reranking** — `item_ranker/` LightGBM LambdaRank over 10 features. NDCG@5 0.3655.
- **Pipeline** — `pipeline/` Apache Beam (DirectRunner), offline.

## Layout
```
ml/
├── pipeline/        # Beam DAG — see ml/pipeline/CLAUDE.md
├── item_ranker/     # LightGBM lib (imported by backend!) — see ml/item_ranker/CLAUDE.md
├── scripts/         # embed_products.py (BGE-M3 → DB)
├── notebooks/       # Colab (GPU embedding/training; embed_products_supabase.ipynb)
└── model/           # trained artifacts (not in git; kept locally + on the HF Hub)
```

## Constraints
- `item_ranker/` is installed (`pip install -e ml/`) and **imported by `backend/app/` at
  serve time** — keep its public API stable.
- Embedding dimension must stay **1024** (matches the `halfvec(1024)` column + the FAISS index).
- Model artifacts live in local `ml/model/` and on the **HF Hub**, not in the Docker image.
- Serve-time vector search is an **in-memory FAISS** index built from the embeddings (the DB has
  no pgvector index — it wouldn't fit the Supabase free cap).

## Docs
Beam DAG, fine-tuning, LightGBM feature table, versioning → `docs/ml-pipeline.md`
