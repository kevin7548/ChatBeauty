# CLAUDE.md

## Agent Workflow

Use a task-type router to decide how to handle each request:

| Task type | Example | Mode |
|---|---|---|
| Single-domain | "Fix broken button styling" | Spawn the relevant domain agent |
| Cross-cutting feature | "Add new recommendation filter" | Single main-agent conversation — context must flow across layers |
| Parallel independent | "Update embeddings + fix UI bug" | Two domain agents in parallel |
| Architectural change | "Switch retrieval index to HNSW" | Main conversation only |

**Domain agents** (defined in `.claude/agents/`):

| Agent | Directory | Scope |
|---|---|---|
| `frontend` | `frontend/` | React, TypeScript, Vite, Vercel |
| `backend` | `backend/app/` | FastAPI routes, retrieval/reranking/explanation services |
| `db` | `backend/sql/` | PostgreSQL, pgvector, HNSW index, schema |
| `ml` | `ml/` | BGE-M3 fine-tuning, LightGBM, Apache Beam pipeline |
| `deploy` | `deploy/`, `backend/Dockerfile`, `backend/docker-compose.yml` | Free-tier: Supabase, local/free-host backend, Vercel (paid GCP retired 2026-06-02) |

## Architecture

ChatBeauty is an LLM/RAG beauty-product recommender. Per request, the FastAPI backend runs
**Retrieval** (BGE-M3 + pgvector → Top-100) → **Reranking** (LightGBM → Top-5) →
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