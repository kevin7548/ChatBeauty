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
- **No in-DB vector index.** The `halfvec` column (~230 MB) + a pgvector index (~230 MB) exceed
  Supabase's 500 MB free cap, and exact scan is ~40 s/query. Serve-time vector search runs as an
  **in-memory FAISS HNSW index** on the backend (built offline from the embeddings; the `embedding`
  column exists only to build it). The commented HNSW recipe in `init.sql` is for non-free hosting.
- **No migration tool** (no Alembic) — change `init.sql` and re-apply. No triggers. On Supabase,
  **RLS is enabled with no policies** (the app connects as the owner role and bypasses RLS, so the
  public PostgREST API is closed while the backend is unaffected).

## Docs
Full column/index reference + load workflow → `docs/db-schema.md`
