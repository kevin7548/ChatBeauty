# ML Pipeline

Offline ML layer in `ml/`: the Apache Beam data pipeline, BGE-M3 fine-tuning (retrieval),
and the LightGBM reranker. The fuller narrative (EDA, two fine-tuning approaches) is in the
root README; this doc is the engineering reference. See also
[architecture.md](architecture.md) for how it feeds serving.

## Layout

```
ml/
├── pipeline/              # Apache Beam (DirectRunner) — offline data processing
│   ├── run.py             #   DAG wiring (entry point)
│   ├── options.py         #   ChatBeautyPipelineOptions (CLI args)
│   ├── schemas.py         #   Pydantic record models
│   └── transforms/        #   parse, validate, aggregate, join, pairs, io, sink
├── item_ranker/           # LightGBM reranking library (imported by the backend!)
│   ├── dataset/base.py    #   RerankSample, iter_samples
│   ├── features/tree.py   #   TreeFeatureBuilder (the 10 features)
│   └── modeling/{train,predict}/   # train_lgbm.py / train_xgb.py, lgbm.py / xgb.py
├── scripts/embed_products.py       # compute BGE-M3 embeddings → PostgreSQL
├── notebooks/             # Colab (GPU embedding / training)
└── model/                 # trained artifacts (not in git; deployed via GCS)
    ├── retrieval/bge-m3-finetuned-<timestamp>/
    └── reranking/lgbm_reranker_current_features_v1.pkl
```

> `item_ranker/` is installed (`pip install -e ml/`) and imported by `backend/app/` at
> serve time. Keep its public API stable; keep the embedding dimension at 1024 to match the
> `vector(1024)` column.

## Beam pipeline (`ml/pipeline/run.py`)

DAG (3 parallel sources → aggregate → join → sinks):

```
ReadReviews   → ParseReviewFn   → ValidateReviewFn   ┐
ReadMetadata  → ParseMetadataFn → ValidateMetadataFn ┤
ReadKeywords  → ParseKeywordFn  → ValidateKeywordFn  ┘
   keywords ─ AggregateKeywordsPerProduct ─┐
                                            ├─ JoinMetadataAndKeywords → items ─┐
   reviews + metadata ─ ComputeRerankingFeatures ─ features ──────────────────┤
                                                       MergeItemAndFeatures → products
                                                                                │
   products ─ WriteToPostgreSQL  (if --database-url)                           │
   keywords + items ─ CreateTrainingPairs → WriteJsonl(training_pairs)
```

Two sinks: the `products` table (112,578 rows, metadata + the 9 stored reranking features)
and `training_pairs.jsonl` (~1M `(query, positive)` pairs for fine-tuning).

Run locally (DirectRunner):

```bash
python -m ml.pipeline.run \
  --input-reviews=ml/data/raw/All_Beauty.jsonl \
  --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
  --input-keywords=ml/data/processed/keywords_train.jsonl \
  --output-dir=ml/data/processed/beam_output \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty
```

## Retrieval — BGE-M3 fine-tuning
- Base `BAAI/bge-m3`, fine-tuned with `MultipleNegativesRankingLoss` (sentence-transformers).
- Embedding text = `[Title]` + `[Review Keywords]` (Llama 3.1, WHO/WHEN/WHY) +
  `[Description Summary]` (Llama 3.1) + `[Features]`.
- Adopted (review-based): ~1M pairs, 2 epochs, batch 32, dim 1024.
  **Valid Recall@100 0.2015 → 0.3543**, Test Recall@100 0.3728.
- Embeddings are computed by `ml/scripts/embed_products.py` and written to the `embedding`
  column; build the IVFFlat index afterwards ([db-schema.md](db-schema.md)).

## Reranking — LightGBM LambdaRank
- `item_ranker/modeling/train/train_lgbm.py`; training data 36.4M candidate rows
  (364K queries × 100 candidates). **NDCG@5 0.3655**, NDCG@10 0.4015.
- The 10 features (`TreeFeatureBuilder.FEATURE_NAMES`, mirrored by `reranking.py`):

  | Feature | Description |
  |---|---|
  | `retrieval_score` | BGE-M3 cosine similarity (query-time) |
  | `review_cnt` | total review count |
  | `vp_review_cnt` | verified-purchase review count |
  | `vp_ratio` | verified-purchase ratio |
  | `recent_review_cnt` | recent review count |
  | `avg_rating` | average rating |
  | `rating_std` | rating standard deviation |
  | `avg_review_len` | average review length |
  | `log_median_price` | log-transformed median price |
  | `price_cnt` | number of price samples |

- Trained artifact: `ml/model/reranking/lgbm_reranker_current_features_v1.pkl`. At training
  time features come from a CSV (`TreeFeatureBuilder`); at serve time the same features are
  read from PostgreSQL (`backend/app/services/reranking.py`) — keep the two in sync.

## Model artifacts & versioning
- Versioning is by **GCS path convention** (e.g. the timestamped
  `bge-m3-finetuned-20260202-120852/` dir, `..._v1.pkl`). No experiment tracking or
  drift monitoring yet — see
  [Known limitations](architecture.md#known-limitations--future-work).
- Artifacts live in GCS and are volume-mounted into Cloud Run, not baked into the image
  ([deployment.md](deployment.md)).
