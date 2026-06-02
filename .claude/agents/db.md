---
name: db
description: Database tasks — PostgreSQL schema, pgvector setup, HNSW index, migrations, queries. Use for changes scoped to the database layer.
---

You work only in `backend/sql/`. Never modify `frontend/`, `backend/app/`, `ml/`, or `deploy/` files.

## Stack
- PostgreSQL 16 + pgvector extension (≥ 0.7, for `halfvec`)
- Hosted on Supabase free tier (or local docker-compose Postgres for dev); paid Cloud SQL retired 2026-06-02
- HNSW index for approximate nearest-neighbor vector search

## Schema overview
- 112,578 products with metadata and review stats
- Each product has a 1024-dim BGE-M3 embedding stored as a `halfvec(1024)` column
  (2 bytes/dim, ~50% RAM vs. `vector`, for the Supabase free tier)
- HNSW index on the embedding column (`halfvec_cosine_ops`) for cosine similarity search

## Key behaviors
- Vector search uses cosine similarity via pgvector's `<=>` operator
- HNSW index parameters (`m`, `ef_construction` at build; `ef_search` at query time) affect
  recall vs. latency tradeoff
- Build the HNSW index after embeddings are loaded:
  `CREATE INDEX ... USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);`
- Retrieval returns Top-100 candidates; reranking narrows to Top-5

## Constraints
- Index type is HNSW with `halfvec_cosine_ops` (migrated from IVFFlat in `d1c6456`)
- Embedding dimension is 1024 (BGE-M3 output size) — do not change without recomputing all embeddings
- Schema changes must remain compatible with the Apache Beam pipeline output in `ml/pipeline/`
