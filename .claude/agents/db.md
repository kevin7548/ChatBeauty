---
name: db
description: Database tasks — PostgreSQL schema, pgvector setup, IVFFlat index, migrations, queries. Use for changes scoped to the database layer.
---

You work only in `backend/sql/`. Never modify `frontend/`, `backend/app/`, `ml/`, or `deploy/` files.

## Stack
- PostgreSQL 16 + pgvector extension
- Hosted on Google Cloud SQL
- IVFFlat index for approximate nearest-neighbor vector search

## Schema overview
- 112,578 products with metadata and review stats
- Each product has a 1024-dim BGE-M3 embedding stored as a pgvector column
- IVFFlat index on the embedding column for cosine similarity search

## Key behaviors
- Vector search uses cosine similarity via pgvector's `<=>` operator
- IVFFlat index parameters (lists, probes) affect recall vs. latency tradeoff
- Retrieval returns Top-100 candidates; reranking narrows to Top-5

## Constraints
- Index type is IVFFlat (not HNSW) — switching to HNSW is a known future optimization
- Embedding dimension is 1024 (BGE-M3 output size) — do not change without recomputing all embeddings
- Schema changes must remain compatible with the Apache Beam pipeline output in `ml/pipeline/`
