# ml/item_ranker/

LightGBM LambdaRank reranking library. **Installed (`pip install -e ml/`) and imported by
`backend/app/services/reranking.py` at serve time** — keep the public API stable.

## Layout
```
item_ranker/
├── dataset/base.py            # RerankSample, iter_samples
├── features/tree.py           # TreeFeatureBuilder — the 10 features (FEATURE_NAMES)
└── modeling/
    ├── train/train_lgbm.py    # LambdaRank trainer (NDCG@5 0.3655); train_xgb.py experimental
    └── predict/lgbm.py        # inference; xgb.py, base*.py
```

## 10 features (`TreeFeatureBuilder.FEATURE_NAMES`)
`retrieval_score`, `review_cnt`, `vp_review_cnt`, `vp_ratio`, `recent_review_cnt`,
`avg_rating`, `rating_std`, `avg_review_len`, `log_median_price`, `price_cnt`.

`retrieval_score` is query-time; the other 9 are stored in `products`. Training reads them
from CSV (`TreeFeatureBuilder`); serving reads the same 9 from PostgreSQL — **keep the two
in sync** (same names/order). Trained artifact:
`ml/model/reranking/lgbm_reranker_current_features_v1.pkl`.

## Docs
Feature table, training data, metrics → `docs/ml-pipeline.md`
