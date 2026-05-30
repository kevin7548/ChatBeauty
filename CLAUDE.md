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
| `db` | `backend/sql/` | PostgreSQL, pgvector, IVFFlat index, schema |
| `ml` | `ml/` | BGE-M3 fine-tuning, LightGBM, Apache Beam pipeline |
| `deploy` | `deploy/`, `backend/Dockerfile`, `backend/docker-compose.yml` | Cloud Run, Cloud SQL, GCS, Vercel config |

## Architecture
### Authentication Flow

### File Processing Pipeline

## Reference Docs

Detailed specs in `docs/` - Claude reads these on-demand when relevant:
- `docs/api-spec.md` - API endpoints, schemas, enums
- `docs/db-schema.md` - Tables, triggers, RLS, migrations
- `docs/frontend-architecture.md` - Routes, components, hooks, styling
- `docs/architecture.md` - System architecture diagrams (Mermaid)