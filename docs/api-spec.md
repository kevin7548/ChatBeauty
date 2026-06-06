# API Spec

Backend HTTP API. Source of truth: `backend/app/api/routes/recommend.py` and
`backend/app/models/schemas.py`. Interactive docs at `/docs` (Swagger) and `/redoc`.

Base URL: the deployed backend URL (local docker-compose or a free host — the paid Cloud Run
deployment was retired 2026-06-02); local dev: `http://localhost:8000` (uvicorn) or `:8080`
(docker-compose).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/recommend` | Recommend Top-5 products for a user scenario |
| `GET` | `/` | Liveness — returns `{"message": "hi"}` |
| `GET` | `/health` | Health/warmup — returns `{"status": "ok"}` |

There is no authentication. CORS allows `GET`/`POST` from `ALLOWED_ORIGINS`.

## POST /recommend

### Request — `RecommendRequest`

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_input` | string | yes | The user's situation / need, free text |

> **`top_k` is not a request field.** The Pydantic model only has `user_input`; the route
> hardcodes Top-5. Any `top_k` a client sends (the frontend currently does) is ignored.
> TODO: honor `top_k` in `recommend.py` if variable result counts are needed. See
> [Known limitations](architecture.md#known-limitations--future-work).

```json
{ "user_input": "I have thin hair and want a volumizing shampoo" }
```

### Response — `RecommendResponse`

| Field | Type | Notes |
|---|---|---|
| `recommendations` | `ItemScore[]` | Ranked Top-5 |
| `latency` | object \| null | `{retrieval_ms, reranking_ms, explanation_ms, total_ms}` |

#### `ItemScore`

| Field | Type | Notes |
|---|---|---|
| `item_id` | string | product ASIN (`parent_asin`) |
| `item_name` | string | product title |
| `score` | float | LightGBM rerank score (not 0–1 normalized) |
| `explanation` | string \| null | Korean, Gemini-generated, review-grounded |
| `image` | string \| null | image URL |
| `price` | float \| null | |
| `average_rating` | float \| null | |
| `rating_number` | int \| null | number of ratings |
| `store` | string \| null | store / brand |

```json
{
  "recommendations": [
    {
      "item_id": "B0XXXXXXX",
      "item_name": "...",
      "score": 1.83,
      "explanation": "검색하신 '볼륨'과 ... 실제 구매자 리뷰에서 ...",
      "image": "https://...",
      "price": 12.99,
      "average_rating": 4.4,
      "rating_number": 1203,
      "store": "..."
    }
  ],
  "latency": { "retrieval_ms": 1102.3, "reranking_ms": 18.7, "explanation_ms": 248.1, "total_ms": 1369.1 }
}
```

Every response also carries an `X-Total-Latency-Ms` header (added by `LatencyMiddleware`).

### Behavior & errors
- A blank `user_input` yields an empty candidate set → empty `recommendations`.
- If Gemini fails or returns non-JSON, explanations degrade gracefully (items still
  returned; `explanation` may be `null` or a fallback string) — the request does not 500.
