# Backend Architecture

FastAPI service that serves the recommendation pipeline. Code lives in `backend/app/`.
For the request/response contract see [api-spec.md](api-spec.md); for the data model see
[db-schema.md](db-schema.md); for the system picture see [architecture.md](architecture.md).

> No automated tests exist yet. See
> [Known limitations](architecture.md#known-limitations--future-work).

## Layout

```
backend/app/
├── main.py                       # FastAPI app, CORS, middleware, router include
├── api/routes/recommend.py       # POST /recommend orchestration
├── services/
│   ├── retrieval.py              # pgvector cosine search → Top-100 candidates
│   ├── reranking.py              # LightGBM rerank → Top-5
│   ├── explanation.py            # Gemini 2.5 Flash batch explanations
│   └── retrieval_resources.py    # shared singletons: BGE-M3 model + DB pool
├── models/schemas.py             # Pydantic request/response models
└── middleware/latency.py         # LatencyMiddleware (per-request timing)
```

## Request orchestration

`recommend()` in `api/routes/recommend.py` runs the three stages in sequence and times
each with `time.perf_counter()`:

1. `retrieve_candidates(user_input)` → up to 100 candidate dicts (default `n=100`).
2. `rerank_items(query, candidates, top_k=5)` → Top-5 (note: **`top_k` is hardcoded to 5**
   in the route, not read from the request).
3. `build_explanation_input(...)` then `generate_explanation(...)` → `{item_id:
   explanation}` map, joined back onto the ranked items.

The response carries `recommendations` plus a `latency` dict
(`retrieval_ms`, `reranking_ms`, `explanation_ms`, `total_ms`), which is also logged.

## Service layer

### retrieval.py
- Encodes the query with the shared `model` (BGE-M3) → 1024-dim vector.
- Runs `RETRIEVE_SQL`: `ORDER BY embedding <=> %s::vector LIMIT n` with
  `1 - (embedding <=> v)` as the cosine `score`; filters `WHERE embedding IS NOT NULL`.
- Returns candidate dicts including the metadata needed downstream (`title`, `price`,
  `average_rating`, `rating_number`, `store`, `features`, `description`, `top_reviews`,
  `details`, `image`, `embedding_text`, `score`).
- Empty/blank query short-circuits to `[]`.

### reranking.py
- Loads the LightGBM model once at import time from `RERANK_MODEL_PATH`
  (`pickle.load`, an `lgb.LGBMRanker`).
- Fetches the 9 stored reranking features for the candidate ASINs in **one** batched query
  (`WHERE parent_asin = ANY(%s)`), combines them with the per-candidate `retrieval_score`,
  builds a `pandas.DataFrame` over `FEATURE_NAMES` (10 features), predicts, sorts
  descending, returns the Top-`top_k`. Feature list: [db-schema.md](db-schema.md).

### explanation.py
- Single Gemini `gemini-2.5-flash` model configured with a Korean system prompt,
  `temperature=0.2`, `top_p=0.8`, `max_output_tokens=8192`,
  `response_mime_type="application/json"`.
- `generate_explanation(input)` sends **all 5 items in one call** and parses the JSON
  `{"explanations": [{"item_id, explanation}]}`. On JSON parse failure it logs
  `finish_reason`/token counts and returns a fallback; on API error it returns
  `{"explanations": []}`. Requires `GEMINI_API_KEY` (raises at import if missing).

### retrieval_resources.py (shared singletons)
- Loads the BGE-M3 `SentenceTransformer` once from `BGE_MODEL_PATH`.
- Holds a `psycopg2` `ThreadedConnectionPool` (min 1, max 5) with TCP keepalives
  (`keepalives_idle=30`, `interval=10`, `count=3`) — Cloud Run pauses idle instances and
  can silently kill pooled sockets.
- `get_db_connection()` runs a `SELECT 1` liveness check and transparently replaces a dead
  connection; `release_db_connection()` returns it to the pool.

## Cross-cutting

- **Startup singletons:** the BGE-M3 model, the LightGBM pickle, the DB pool, and the
  Gemini model are all created at import time, so the first import pays the load cost (this
  is why the frontend warms the instance — see
  [frontend-architecture.md](frontend-architecture.md)).
- **CORS:** `main.py` reads `ALLOWED_ORIGINS` (comma-separated; default
  `http://localhost:5173`); methods limited to `GET`/`POST`.
- **Latency middleware:** `LatencyMiddleware` adds an `X-Total-Latency-Ms` response header
  and logs `"<METHOD> <path> completed in <ms>ms"` for every request.
- **Constraints to preserve:** keep all 5 explanations in one Gemini call; the embedding
  dimension must stay 1024 to match the `vector(1024)` column; model files come from the
  GCS volume mount, not the image.
