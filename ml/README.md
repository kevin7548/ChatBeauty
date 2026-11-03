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
├── preprocess/      # extract_keywords.py (Llama 3.1 8B via vLLM) → split_keywords.py (timestamp split)
├── retrieval/       # create_training_pairs.py → finetune_bge_m3.py (the served model's recipe)
├── evaluation/      # retrieve_candidates.py (FAISS) → evaluate_recall.py (Recall@K, MRR)
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
   `--skip-training-pairs` — emits `training_pairs.jsonl` (~518K keyword-query pairs). Use `--skip-training-pairs`
   for a memory-light products-only load (the pairs branch OOMs ~16 GB).
2. **Fine-tune BGE-M3** (MultipleNegativesRankingLoss, 2 epochs, batch 32, dim 1024). The served
   model was trained on `retrieval/create_training_pairs.py QUERY_TYPE=both` (keyword + raw-review
   queries, 1,006,278 pairs; `finetune_bge_m3.py` holds out 10% → 905,650), not on the Beam pairs
   (keyword queries only, ~518K):
   ```bash
   python ml/preprocess/extract_keywords.py            # GPU; → keywords_output.jsonl
   python ml/preprocess/split_keywords.py              # → keywords_{train,valid,test}.jsonl
   python ml/retrieval/create_training_pairs.py QUERY_TYPE=both
   python ml/retrieval/finetune_bge_m3.py EPOCHS=2 BATCH_SIZE=32
   ```
   Artifact → `model/retrieval/bge-m3-finetuned-<timestamp>/`.
   **Evaluate** (leak-free baseline: Valid Recall@100 **0.334** with `INDEX=flat` and the fine-tuned model, see
   `experiments/B/RESULTS.md`. The old `data/evaluation/retrieval_candidates_valid.jsonl` scores 0.3543 but comes
   from a ChromaDB-era index that likely leaked valid/test review keywords; don't use it as the baseline):
   ```bash
   python ml/evaluation/retrieve_candidates.py SPLIT=valid                               # serving HNSW index
   python ml/evaluation/retrieve_candidates.py MODEL_PATH=BAAI/bge-m3 INDEX=flat SPLIT=valid  # any other model
   python ml/evaluation/evaluate_recall.py SPLIT=valid TAG=<tag printed above>
   ```
3. **Embeddings:** run `notebooks/embed_products_supabase.ipynb` on a Colab GPU — loads the
   fine-tuned model from the HF Hub and writes `halfvec` embeddings to Postgres (~15–30 min).
4. **Reranker:** train LightGBM (`item_ranker/modeling/train/`) on retrieval candidates (10 features).
5. **Serve-time index:** a **FAISS HNSW** index is built from the embeddings, uploaded to the HF
   Hub, and loaded into the backend's RAM (the DB embedding column exists only to build it).

## Models & metrics

- **Retrieval** — `BAAI/bge-m3` fine-tuned: **Valid Recall@100 0.176 (zero-shot) → 0.334**, Test 0.176 → 0.341 (exact search, current corpus).
  Older 0.2015 → 0.3543 / Test 0.3728 are leaky ChromaDB-era values.
- **Reranking** — LightGBM LambdaRank over 10 features (36.4M candidate rows):
  **NDCG@5 0.3655**, NDCG@10 0.4015.

## Constraints

- `item_ranker/` public API is imported by `backend/app/` at serve time — keep it stable.
- Embedding dimension must stay **1024** (matches the `halfvec(1024)` column + the FAISS index).
- Model artifacts live in local `ml/model/` and on the **HF Hub**, not in the Docker image.

## Data source

Amazon Reviews 2023 (All_Beauty) — 112k products / 701k reviews.
https://amazon-reviews-2023.github.io/
