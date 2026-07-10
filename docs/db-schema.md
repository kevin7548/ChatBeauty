# Database Schema

PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector). Source of truth:
`backend/sql/init.sql`. A single denormalized `products` table holds everything the
runtime needs — vector, explanation text, and pre-aggregated reranking features — so the
serving path needs no joins.

> **No migration tool.** Schema changes are made by editing `init.sql` and re-applying it;
> there is no Alembic/migration history. Adopting one is future work — see
> [Known limitations](architecture.md#known-limitations--future-work).

## `products`

| Column | Type | Purpose |
|---|---|---|
| `parent_asin` | `VARCHAR(20)` PK | product id |
| `title` | `TEXT NOT NULL` | product name (→ `item_name`) |
| `embedding_text` | `TEXT` | text that was embedded (Title + keywords + summary + features); only the offline ML pipeline reads/writes it — the API never selects it, and the live Supabase DB keeps it NULL to stay under the 500 MB free cap |
| `embedding` | `halfvec(1024)` | fine-tuned BGE-M3 embedding (cosine); `halfvec` (2 bytes/dim) halves storage/index RAM to fit the Supabase free tier — requires pgvector ≥ 0.7 |
| **Explanation metadata** | | passed to Gemini |
| `description` | `TEXT` | |
| `features` | `TEXT` | |
| `top_reviews` | `TEXT` | |
| `details` | `TEXT` | |
| `image` | `TEXT` | image URL |
| `store` | `VARCHAR(255)` | store / brand |
| **Reranking features (9 stored)** | | the 10th, `retrieval_score`, is computed at query time |
| `review_cnt` | `INTEGER` | total reviews |
| `vp_review_cnt` | `INTEGER` | verified-purchase reviews |
| `vp_ratio` | `FLOAT` | verified-purchase ratio |
| `recent_review_cnt` | `INTEGER` | recent reviews |
| `avg_rating` | `FLOAT` | mean rating (from reviews) |
| `rating_std` | `FLOAT` | rating std dev |
| `avg_review_len` | `FLOAT` | mean review length |
| `log_median_price` | `FLOAT` | log of median price |
| `price_cnt` | `INTEGER` | number of price samples |
| **Filter / display** | | |
| `price` | `NUMERIC` | |
| `average_rating` | `FLOAT` | |
| `rating_number` | `INTEGER` | number of ratings |

The reranking-feature columns are **pre-aggregated by the Beam pipeline**
(`ComputeRerankingFeatures`), not computed at request time. See
[ml-pipeline.md](ml-pipeline.md).

## Indexes

`init.sql` creates three B-tree indexes immediately:

```sql
idx_products_price   ON products (price)
idx_products_rating  ON products (average_rating)
idx_products_store   ON products (store)
```

The vector index is **not** created by `init.sql`. HNSW does not strictly require data to
exist first, but building after embeddings are loaded is faster:

```sql
CREATE INDEX idx_products_embedding ON products
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
```

Tune recall at query time with `SET hnsw.ef_search = 100;` (vs. the Top-100 `LIMIT`).

## Load workflow

1. `CREATE EXTENSION vector;` + table + B-tree indexes → apply `backend/sql/init.sql`.
2. Populate rows + review-stat features → Beam pipeline (`ml.pipeline.run`).
3. Compute & write `embedding` → `ml/scripts/embed_products.py` (or the Colab notebook).
4. Build the HNSW index (SQL above).

Full runbook: [development.md](development.md).

## Notes
- No triggers, no row-level security.
- Query-time retrieval uses the cosine operator `<=>` (`vector_cosine_ops`); the stored
  `score` is `1 - (embedding <=> query)`.
