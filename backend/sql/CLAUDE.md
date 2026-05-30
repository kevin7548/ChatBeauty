# backend/sql/

PostgreSQL schema for the recommender. One file: `init.sql`.

## Model
A single denormalized `products` table (PK `parent_asin`) holds everything serving needs:
`embedding vector(1024)`, explanation text (`description`, `features`, `top_reviews`,
`details`, `image`, `store`), the **9 stored reranking features**
(`review_cnt`, `vp_review_cnt`, `vp_ratio`, `recent_review_cnt`, `avg_rating`, `rating_std`,
`avg_review_len`, `log_median_price`, `price_cnt`), and filter fields (`price`,
`average_rating`, `rating_number`). No joins on the serving path.

## Key rules
- Reranking-feature columns are **pre-aggregated by the Beam pipeline** (`ml/`), not at query time.
- `init.sql` creates the `vector` extension, the table, and 3 B-tree indexes (price, rating,
  store). It does **not** create the vector index.
- Build the IVFFlat index **after** embeddings are loaded:
  `CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);`
- **No migration tool** (no Alembic) — change `init.sql` and re-apply. No triggers, no RLS.

## Docs
Full column/index reference + load workflow → `docs/db-schema.md`
