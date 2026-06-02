---
title: ChatBeauty Backend
emoji: 💄
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ChatBeauty Backend (FastAPI)

Free Hugging Face Docker Space hosting the ChatBeauty recommendation API
(retrieval → reranking → explanation). Endpoints: `GET /health`, `POST /recommend`.

> **This file is the Space card.** The two files in this directory — this `README.md` and the
> `Dockerfile` (plus `start.sh`) — are what you push to the Space git repo. Full setup runbook:
> [`DEPLOY.md`](DEPLOY.md).

## Required Space settings

| Key | Type | Value |
|---|---|---|
| `MODEL_REPO_ID` | variable | `<hf-username>/chatbeauty-models` (the model repo from `upload-models.sh`) |
| `DATABASE_URL` | secret | Supabase connection-pooler URL |
| `GEMINI_API_KEY` | secret | Google AI Studio free-tier key |
| `ALLOWED_ORIGINS` | variable | your Vercel frontend URL (comma-separated) |

Cold starts re-download the ~2.1 GB model (free Spaces have ephemeral storage), so the first
request after a sleep takes a few minutes; the frontend's `warmUp()` ping helps mask shorter gaps.
