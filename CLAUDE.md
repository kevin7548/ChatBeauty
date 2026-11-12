# CLAUDE.md

## Workflow

Work one task at a time, sized to a single item in `TODO.md`: one conversation per task, scoped
to the files that task needs. Domain tags in `TODO.md` (`[backend]`, `[ml]`, ...) only label
which part of the codebase an item touches; they are not separate agents. Cross-cutting changes
(db + backend + ml) stay in one conversation so context flows across layers.

Scope is controlled by fixed rules and automated checks rather than by splitting work across
agents:

- **Fixed invariants:** the explanation step is one Gemini call for all 5 products; embeddings
  stay 1024-dimensional (BGE-M3). Do not change these without an explicit decision.
- **PR only:** every change lands through a pull request, never a direct push to `main`.
- **CI gate:** the `POST /recommend` integration test in `.github/workflows/ci.yml` must pass
  before merge.
- **Evaluation scripts:** before trusting a new or changed evaluation script, reproduce a known
  baseline number with it first.

## Architecture

ChatBeauty is an LLM/RAG beauty-product recommender. Per request, the FastAPI backend runs
**Retrieval** (BGE-M3 + in-memory FAISS HNSW → Top-100) → **Reranking** (LightGBM → Top-5) →
**Explanation** (Gemini 2.5 Flash, one batched call). The frontend is a React/Vite SPA; the
`ml/` layer (Apache Beam pipeline, BGE-M3 fine-tuning, LightGBM) is offline. There is no
authentication. Full picture: `docs/architecture.md`.

Run locally: see `docs/development.md` (docker-compose for backend+db, Beam load, embed,
`npm run dev` for frontend).

## Reference Docs

Detailed specs in `docs/` - Claude reads these on-demand when relevant:
- `docs/architecture.md` - System architecture + Mermaid diagrams; known limitations
- `docs/backend-architecture.md` - Service layer, request orchestration, singletons
- `docs/api-spec.md` - Endpoints, request/response schemas
- `docs/db-schema.md` - `products` table, indexes, HNSW, load workflow
- `docs/frontend-architecture.md` - SPA state, components, hooks, styling
- `docs/ml-pipeline.md` - Beam DAG, BGE-M3 fine-tuning, LightGBM features
- `docs/deployment.md` - Free-tier deploy: Supabase, local/free host, Vercel, env vars (paid GCP retired 2026-06-02)
- `docs/development.md` - End-to-end local setup runbook