# ML

Offline ML layer for ChatBeauty: the **Apache Beam** data pipeline, **BGE-M3** fine-tuning
(retrieval), and the **LightGBM** reranker. Serve-time vector search runs as an **in-memory FAISS**
index on the backend (the DB has no pgvector index — see below).

> Full engineering reference (Beam DAG, fine-tuning, the 10-feature table, metrics, versioning) →
> [docs/ml-pipeline.md](../docs/ml-pipeline.md). Per-directory context lives in each `CLAUDE.md`.

## Structure

```
ml/
├── pipeline/        # Apache Beam (DirectRunner): Parse → Validate → Aggregate → Join
│                    #   → PostgreSQL (products) + training_pairs.jsonl   (see pipeline/CLAUDE.md)
├── item_ranker/     # LightGBM LambdaRank reranking library — installed via `pip install -e ml/`
│                    #   and imported by backend/app at serve time       (see item_ranker/CLAUDE.md)
├── scripts/         # embed_products.py — compute BGE-M3 embeddings → PostgreSQL
├── notebooks/       # Colab (GPU): embed_products_supabase.ipynb (Supabase + model from HF Hub)
├── model/           # trained artifacts (not in git; kept locally + on the HF Hub)
│                    #   retrieval/ (fine-tuned BGE-M3), reranking/ (LightGBM .pkl)
├── data/            # raw / processed / evaluation datasets (not in git)
└── setup.py         # packaging for item_ranker
```

## Offline workflow

1. **Load + features (Beam):**
   ```bash
   python -m ml.pipeline.run \
     --input-reviews=ml/data/raw/All_Beauty.jsonl \
     --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
     --input-keywords=ml/data/processed/keywords_train.jsonl \
     --database-url="$DATABASE_URL" [--skip-training-pairs]
   ```
   Populates the `products` table (112,578 rows + the 9 stored reranking features) and — unless
   `--skip-training-pairs` — emits `training_pairs.jsonl` (~1M pairs). Use `--skip-training-pairs`
   for a memory-light products-only load (the pairs branch OOMs ~16 GB).
2. **Fine-tune BGE-M3** on the pairs (MultipleNegativesRankingLoss, ~1M pairs, 2 epochs, dim 1024).
   Artifact → `model/retrieval/bge-m3-finetuned-<timestamp>/`. (Done offline; details in the docs.)
3. **Embeddings:** run `notebooks/embed_products_supabase.ipynb` on a Colab GPU — loads the
   fine-tuned model from the HF Hub and writes `halfvec` embeddings to Postgres (~15–30 min).
4. **Reranker:** train LightGBM (`item_ranker/modeling/train/`) on retrieval candidates (10 features).
5. **Serve-time index:** a **FAISS HNSW** index is built from the embeddings, uploaded to the HF
   Hub, and loaded into the backend's RAM (the DB embedding column exists only to build it).

## Models & metrics

- **Retrieval** — `BAAI/bge-m3` fine-tuned: **Valid Recall@100 0.2015 → 0.3543**, Test 0.3728.
- **Reranking** — LightGBM LambdaRank over 10 features (36.4M candidate rows):
  **NDCG@5 0.3655**, NDCG@10 0.4015.

## Constraints

- `item_ranker/` public API is imported by `backend/app/` at serve time — keep it stable.
- Embedding dimension must stay **1024** (matches the `halfvec(1024)` column + the FAISS index).
- Model artifacts live in local `ml/model/` and on the **HF Hub**, not in the Docker image.

## Data source

Amazon Reviews 2023 (All_Beauty) — 112k products / 701k reviews.
https://amazon-reviews-2023.github.io/
