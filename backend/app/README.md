# App — FastAPI Application

뷰티 제품 추천 API 서버입니다. 한 번의 요청에서 **Retrieval → Reranking → Explanation**
3단계 파이프라인을 실행합니다.

> 상세 스펙은 `docs/`를 참고하세요 (단일 출처):
> API 계약은 [docs/api-spec.md](../../docs/api-spec.md),
> 서비스 설계는 [docs/backend-architecture.md](../../docs/backend-architecture.md).

## Directory Structure

```
app/
├── main.py                       # FastAPI 앱, CORS, LatencyMiddleware
├── api/routes/recommend.py       # POST /recommend 오케스트레이션
├── services/
│   ├── retrieval.py              # 인메모리 FAISS 검색 → Top-100 (메타데이터는 Postgres)
│   ├── reranking.py              # LightGBM 재정렬 → Top-5
│   ├── explanation.py            # Gemini 2.5 Flash 설명 (5개 한 번에)
│   └── retrieval_resources.py    # BGE-M3 모델 + DB 커넥션 풀 (싱글톤)
├── models/schemas.py             # Pydantic 스키마
└── middleware/latency.py         # 요청별 레이턴시 로깅
```

## Quick Start

```bash
cd backend
uvicorn app.main:app --reload --port 8000
# Swagger: http://localhost:8000/docs  ·  ReDoc: /redoc
```

## Endpoints

- `POST /recommend` — 사용자 시나리오(`user_input`)로 Top-5 추천 + 설명을 반환.
  요청 본문은 `{"user_input": "..."}`. (`top_k`는 받지 않으며 항상 5개를 반환합니다.)
- `GET /` — `{"message": "hi"}`
- `GET /health` — `{"status": "ok"}` (프론트엔드 warmup에 사용)

요청/응답 스키마와 예시는 [docs/api-spec.md](../../docs/api-spec.md)를 참고하세요.

## Service Layer

1. **Retrieval** (`services/retrieval.py`) — Fine-tuned BGE-M3로 쿼리를 인코딩하고
   **인메모리 FAISS HNSW 인덱스**로 코사인 Top-100 ANN 검색 후, 해당 후보들의 메타데이터를
   Postgres에서 일괄 조회 (pgvector 인덱스가 무료 한도를 넘어 벡터 검색은 인메모리로 수행).
2. **Reranking** (`services/reranking.py`) — 10개 피처에 대한 LightGBM LambdaRank로
   재정렬하여 Top-5 선정. 피처는 DB에서 단일 배치 쿼리로 조회.
3. **Explanation** (`services/explanation.py`) — Gemini 2.5 Flash가 5개 상품 설명을
   단일 API 호출로 생성 (한국어).

## Configuration

CORS는 `ALLOWED_ORIGINS`(쉼표 구분, 기본 `http://localhost:5173`)로 설정합니다.
환경 변수 전체 목록과 배포 설정은 [docs/deployment.md](../../docs/deployment.md) 참고.

## Dependencies

`backend/pyproject.toml` 참조 — fastapi, uvicorn, lightgbm, faiss-cpu, sentence-transformers,
psycopg2-binary, pgvector, google-genai 등.
