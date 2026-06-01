# backend/sql/

PostgreSQL schema for the recommender. One file: `init.sql`.

## Model
A single denormalized `products` table (PK `parent_asin`) holds everything serving needs:
`embedding halfvec(1024)`, explanation text (`description`, `features`, `top_reviews`,
`details`, `image`, `store`), the **9 stored reranking features**
(`review_cnt`, `vp_review_cnt`, `vp_ratio`, `recent_review_cnt`, `avg_rating`, `rating_std`,
`avg_review_len`, `log_median_price`, `price_cnt`), and filter fields (`price`,
`average_rating`, `rating_number`). No joins on the serving path.

## Key rules
- Reranking-feature columns are **pre-aggregated by the Beam pipeline** (`ml/`), not at query time.
- `init.sql` creates the `vector` extension, the table, and 3 B-tree indexes (price, rating,
  store). It does **not** create the vector index.
- `embedding` is `halfvec(1024)` (2 bytes/dim, ~50% RAM vs. `vector`, for the Supabase free
  tier); requires pgvector ≥ 0.7.
- Build the HNSW index **after** embeddings are loaded:
  `CREATE INDEX ... USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);`
- **No migration tool** (no Alembic) — change `init.sql` and re-apply. No triggers, no RLS.

## Docs
Full column/index reference + load workflow → `docs/db-schema.md`
