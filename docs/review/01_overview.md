# 01. ChatBeauty 프로젝트 구조 개요

> 기준: `main` @ `e7cb606` (2026-10-12). 코드를 직접 읽어서 정리한 문서이며, 문서(README/docs)와
> 코드가 서로 다른 부분은 **⚠️** 로 표시했습니다. 이 리뷰에서는 코드를 수정하지 않았습니다.

## 0. 한눈에 보기

ChatBeauty는 "사용자 상황 텍스트 → 뷰티 상품 Top-5 + 추천 이유"를 돌려주는 RAG 추천기입니다.

```
[Vercel] React SPA ──POST /recommend──▶ [HF Space] FastAPI (uvicorn, 1 worker)
                                           │ 1. Retrieval : BGE-M3(파인튜닝) 쿼리 인코딩 → 인메모리 FAISS HNSW → Top-100 ASIN
                                           │              → Postgres에서 메타데이터 조회 (WHERE parent_asin = ANY)
                                           │ 2. Reranking : Postgres에서 피처 9개 조회 + retrieval_score → LightGBM LambdaRank → Top-5
                                           │ 3. Explanation: Gemini 2.5 Flash 1회 배치 호출 → 상품별 한국어 설명(JSON)
                                           ▼
                                    [Supabase] PostgreSQL 16 (products 테이블, 112,578행)
오프라인: Apache Beam(DirectRunner) → products 적재 + training_pairs.jsonl
        → BGE-M3 파인튜닝 → 상품 임베딩(Colab GPU) → FAISS 인덱스 빌드 → HF Hub 업로드
        → LightGBM 학습
```

레이어는 4개 + 배포 1개입니다: `frontend/`(서빙 UI), `backend/app/`(온라인 추론),
`backend/sql/`(스키마), `ml/`(오프라인 데이터·학습), `deploy/`(무료 티어 배포).

---

## 1. 디렉토리 트리와 역할

`node_modules/`, `.git/`, 데이터/모델 실물(gitignore 대상)은 제외했습니다. git 추적 파일 145개.

```
ChatBeauty/
├── CLAUDE.md                 # Claude Code용 프로젝트 지침 (도메인 에이전트 라우팅 표, 아키텍처 요약)
├── README.md / README.ko.md  # 프로젝트 소개(영/한): 파이프라인, 모델 지표, 기술 스택, 데모
├── TODO.md                   # 남은 작업의 단일 출처 (라이브 URL, 로드맵)
├── .gitignore                # 루트 규칙: .env*, ml/data/**, ml/model/**, *.pkl/*.safetensors/*.jsonl 등 제외
│
├── .github/
│   ├── workflows/ci.yml          # PR/main push 시 backend pytest(+ruff 권고 모드) 실행
│   ├── workflows/keepalive.yml   # 월·목 cron: HF Space /health 확인→필요 시 재시작, 실제 /recommend 호출로 Supabase 깨우기
│   ├── ISSUE_TEMPLATE/           # 버그/기능 이슈 템플릿
│   └── pull_request_template.md  # PR 템플릿
│
├── .claude/
│   ├── agents/{frontend,backend,db,ml,deploy}.md  # 도메인 서브에이전트 정의
│   ├── hooks/                    # (미추적) 작업 완료 알림 스크립트
│   ├── settings.local.json       # 로컬 권한 설정
│   └── worktrees/kind-turing-6714e5/  # ⚠️ 오래된 git worktree (구 코드 사본, backend/ml 등 중복) — 검색 시 노이즈
│
├── backend/                  # FastAPI 서빙 서버 (온라인 추론)
│   ├── pyproject.toml            # Poetry 의존성 (Python ≥3.12), pytest 설정
│   ├── Dockerfile                # 로컬/구 Cloud Run용 이미지: poetry install + `pip install -e ml/` + uvicorn :8080
│   ├── docker-compose.yml        # 로컬: pgvector/pgvector:pg16 DB + backend, 모델 디렉토리 read-only 마운트
│   ├── .env.example              # 백엔드 환경변수 템플릿
│   ├── .dockerignore / .gcloudignore / .gitignore
│   ├── README.md, CLAUDE.md
│   ├── sql/
│   │   ├── init.sql              # products 테이블 DDL (halfvec(1024), 피처 9개, 필터 인덱스). compose 최초 기동 시 실행
│   │   └── CLAUDE.md
│   ├── app/
│   │   ├── main.py               # ★ FastAPI 앱 생성, CORS·Latency 미들웨어, `/`·`/health`, recommend 라우터 등록
│   │   ├── api/routes/recommend.py   # POST /recommend: retrieve→rerank(top_k=5 하드코딩)→explain 오케스트레이션, 단계별 latency
│   │   ├── models/schemas.py     # Pydantic: RecommendRequest{user_input}, ItemScore, RecommendResponse{recommendations, latency}
│   │   ├── middleware/latency.py # 모든 응답에 X-Total-Latency-Ms 헤더 + 로그
│   │   └── services/             # (__init__.py 없음 — namespace package)
│   │       ├── retrieval_resources.py  # 임포트 시 싱글톤 로드: SentenceTransformer, FAISS 인덱스+asins.json, psycopg2 커넥션 풀(1~5)
│   │       ├── retrieval.py      # 쿼리 인코딩→L2 정규화→FAISS search(n=100)→cosine 환산→DB 메타데이터 배치 조회
│   │       ├── reranking.py      # 임포트 시 LightGBM .pkl 로드, DB에서 피처 9개 조회 + retrieval_score → predict → 정렬
│   │       └── explanation.py    # 임포트 시 load_dotenv + Gemini 클라이언트 생성(키 없으면 RuntimeError), JSON 응답 파싱
│   └── tests/
│       ├── conftest.py           # 서비스 모듈 4개를 sys.modules 스텁으로 교체 → 모델/DB/Gemini 없이 오프라인 테스트
│       └── test_recommend.py     # /recommend 통합 테스트 (Top-5, 설명 매핑, latency 키, 헤더, 빈 입력)
│
├── frontend/                 # React 19 + Vite 7 + TS SPA (Vercel 배포)
│   ├── package.json / package-lock.json  # 스크립트: dev / build(tsc -b && vite build) / lint / preview
│   ├── vite.config.ts            # react 플러그인만 사용 (프록시 등 없음)
│   ├── tsconfig*.json, eslint.config.js, index.html
│   ├── .env.example              # VITE_API_URL
│   └── src/
│       ├── main.tsx              # ★ React 엔트리 (App 마운트)
│       ├── App.tsx               # 단일 페이지 상태 관리: 쿼리, 로딩/에러/결과, 마운트 시 warmUp()
│       ├── api/recommend.ts      # fetchRecommend(POST /recommend), warmUp(GET /health)
│       ├── types/recommend.ts    # 요청/응답 TS 타입
│       ├── hooks/useLoadingMessage.ts  # 경과 시간별 로딩 문구 (콜드스타트 안내)
│       └── components/           # Header, Hero(예시 칩), ProductCard, SkeletonCard, SearchFooter(입력창), WarmupBanner (+ 각 .css)
│
├── ml/                       # 오프라인 데이터 파이프라인·학습
│   ├── setup.py                  # `item_ranker` 패키지 정의 (install_requires 없음)
│   ├── README.md, CLAUDE.md
│   ├── pipeline/                 # Apache Beam DAG
│   │   ├── run.py                # ★ 파이프라인 엔트리: Read→Parse→Validate→Aggregate→Join→Sink(DB, training_pairs)
│   │   ├── options.py            # 커스텀 CLI 옵션 (--input-*, --database-url, --output-dir, --skip-training-pairs, --max-keywords)
│   │   ├── schemas.py            # NamedTuple 레코드 (Review, Metadata, KeywordRecord, ProductRecord, TrainingPair …)
│   │   └── transforms/
│   │       ├── io.py             # ReadJsonl / WriteJsonl
│   │       ├── parse.py          # raw dict → NamedTuple
│   │       ├── validate.py       # 유효성 필터 + Beam Metrics
│   │       ├── aggregate.py      # 상품별 키워드 병합, 리뷰 통계 → 리랭킹 피처 (CombineFn)
│   │       ├── join.py           # CoGroupByKey: 메타+키워드 → 아이템, 아이템+피처 → ProductRecord(embedding_text 포함)
│   │       ├── pairs.py          # (query, positive) 학습 페어 생성 (BGE-M3 파인튜닝용)
│   │       └── sink.py           # psycopg2 배치 upsert (ON CONFLICT) → products
│   ├── item_ranker/              # 리랭커 학습/오프라인 추론 라이브러리
│   │   ├── dataset/base.py       # Candidate/RerankSample, iter_samples(JSONL; 정답 ASIN=1, 나머지=0)
│   │   ├── features/tree.py      # TreeFeatureBuilder: CSV에서 아이템 피처 로드 → 10개 피처 DataFrame
│   │   └── modeling/
│   │       ├── train/train_lgbm.py   # train_reranker(): LGBMRanker(lambdarank, early stopping) → pickle
│   │       ├── train/train_xgb.py    # XGBoost 대안 학습 (실험)
│   │       └── predict/{base,base_tree,lgbm,xgb}.py  # 오프라인 스코어링 래퍼
│   ├── scripts/embed_products.py # ★ 임베딩 엔트리(CLI): embedding IS NULL 행을 인코딩 → halfvec UPDATE
│   ├── notebooks/
│   │   ├── embed_products_supabase.ipynb  # 현행: Colab T4에서 HF Hub 모델로 Supabase에 halfvec 임베딩 기록
│   │   └── embed_products_colab.ipynb     # 레거시: Cloud SQL + IVFFlat (GCP 철수 후 사용 안 함)
│   ├── data/{raw,cleaned,processed,evaluation}/  # gitignore (.gitkeep만 추적). 로컬엔 All_Beauty.jsonl, keywords_*.jsonl, training_pairs.jsonl 등 존재
│   └── model/{retrieval,reranking,explanation}/  # gitignore. 로컬: bge-m3-finetuned-20260202-120852/, lgbm_reranker_current_features_v1.pkl
│
├── deploy/
│   ├── CLAUDE.md
│   ├── hf-space/
│   │   ├── Dockerfile            # HF Docker Space: GitHub main을 clone(CACHEBUST로 갱신) → poetry install → :7860
│   │   ├── start.sh              # ★ 프로덕션 서빙 엔트리: HF Hub에서 모델+FAISS 인덱스 다운로드 → 경로 env export → uvicorn
│   │   ├── upload-models.sh      # 로컬 ml/model → HF Hub 모델 리포 업로드
│   │   ├── DEPLOY.md             # 무료 티어 배포 런북 (Supabase / HF Hub / Space / Vercel, FAISS 빌드 절차)
│   │   └── README.md             # HF Space 메타데이터 카드
│   └── setup-gcp.sh              # 레거시: Cloud Run + Cloud SQL 배포 스크립트 (2026-06-02 GCP 철수로 사용 안 함)
│
├── docs/
│   ├── architecture.md, backend-architecture.md, api-spec.md, db-schema.md,
│   │   frontend-architecture.md, ml-pipeline.md, deployment.md, development.md  # 레이어별 상세 스펙
│   ├── refactoring.md            # 과거 리팩토링(ML 파이프라인 분리 등) 기록
│   ├── testing-cicd-primer.md    # 테스트/CI 학습 노트
│   ├── plans/                    # (미추적) 임시 설계 메모: halfvec·HNSW 마이그레이션, 차원 축소 recall 평가
│   └── review/01_overview.md     # (이 문서)
│
└── images/                   # README용 다이어그램/EDA/데모 GIF
```

---

## 2. 엔트리포인트

### 2.1 데이터 적재 · 피처 생성 (오프라인)

| 단계 | 엔트리포인트 | 입력 → 출력 |
|---|---|---|
| Beam 파이프라인 | `python -m ml.pipeline.run …` ([run.py](../../ml/pipeline/run.py)) | `All_Beauty.jsonl`, `meta_All_Beauty.jsonl`, `keywords_train.jsonl` → `products` 테이블 + `training_pairs.jsonl` |

```bash
python -m ml.pipeline.run \
  --input-reviews=ml/data/raw/All_Beauty.jsonl \
  --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
  --input-keywords=ml/data/processed/keywords_train.jsonl \
  --output-dir=ml/data/processed/beam_output \
  --database-url="$DATABASE_URL" [--skip-training-pairs]
```

### 2.2 학습

| 모델 | 엔트리포인트 | 비고 |
|---|---|---|
| BGE-M3 파인튜닝 | ⚠️ **레포에 코드 없음** | README/docs에는 MultipleNegativesRankingLoss, ~1M 페어, 2 epoch로 기술. 산출물 `ml/model/retrieval/bge-m3-finetuned-20260202-120852/`만 로컬·HF Hub에 존재 |
| LightGBM 리랭커 | `train_reranker()` 함수 ([train_lgbm.py](../../ml/item_ranker/modeling/train/train_lgbm.py)) | ⚠️ `__main__`/CLI 없음 — 노트북 등에서 import해 호출하는 라이브러리 함수. 입력: `retrieval_candidates_{train,valid}.jsonl` + 아이템 피처 CSV(`TreeFeatureBuilder`) |
| XGBoost (실험) | [train_xgb.py](../../ml/item_ranker/modeling/train/train_xgb.py) | 동일하게 함수 형태 |

학습 입력 중 아래 산출물을 만드는 코드도 레포에 없습니다: `keywords_*.jsonl`(Llama 3.1 8B 키워드
추출), `retrieval_candidates_*.jsonl`(리랭커 학습용 Top-100 후보), `TreeFeatureBuilder`가 읽는 아이템
피처 CSV.

### 2.3 인덱싱 (임베딩 → 벡터 인덱스)

| 단계 | 엔트리포인트 | 비고 |
|---|---|---|
| 상품 임베딩 | [ml/notebooks/embed_products_supabase.ipynb](../../ml/notebooks/embed_products_supabase.ipynb) (현행, Colab GPU) 또는 `python -m ml.scripts.embed_products --database-url=… --model-path=…` ([embed_products.py](../../ml/scripts/embed_products.py)) | `embedding_text` → `products.embedding halfvec(1024)` |
| FAISS HNSW 빌드 | ⚠️ **레포에 코드 없음** | 절차만 [DEPLOY.md](../../deploy/hf-space/DEPLOY.md) Phase B에 기술: Supabase에서 임베딩 스트리밍 → `normalize_L2` → `IndexHNSWFlat(1024, 16)`, efConstruction 200 → `faiss_hnsw_cosine.index` + `asins.json` → HF Hub `retrieval/ann/` |
| pgvector HNSW 인덱스 | `init.sql` 주석 / `embed_products.py` 로그 안내 | 현재 서빙에서는 **사용 안 함** (Supabase 500MB 무료 한도 때문에 FAISS로 전환) |

### 2.4 서빙 / 추론 (온라인)

| 환경 | 엔트리포인트 | 포트 |
|---|---|---|
| 프로덕션 (HF Space) | [deploy/hf-space/start.sh](../../deploy/hf-space/start.sh) → `uvicorn app.main:app` | 7860 |
| 로컬 Docker | [backend/Dockerfile](../../backend/Dockerfile) CMD (`docker compose up`) | 8080 |
| 로컬 bare | `cd backend && uvicorn app.main:app --reload` | 8000 |
| 앱 객체 | [backend/app/main.py](../../backend/app/main.py) → 라우트 [recommend.py](../../backend/app/api/routes/recommend.py) | — |
| 프론트엔드 | `npm run dev` / [frontend/src/main.tsx](../../frontend/src/main.tsx) | 5173 |
| 테스트 | `cd backend && python -m pytest` | — |

모델/DB/Gemini 클라이언트는 **요청 시점이 아니라 모듈 임포트 시점**에 로드됩니다
(`retrieval_resources.py`, `reranking.py`, `explanation.py`). 따라서 `app.main` 임포트 = 전체 모델
로드이며, 하나라도 실패하면 서버가 기동되지 않습니다(테스트가 `sys.modules` 스텁을 쓰는 이유).

---

## 3. 주요 라이브러리와 버전

### 3.1 Backend — [backend/pyproject.toml](../../backend/pyproject.toml) (Python `>=3.12`, Docker는 `python:3.12-slim`)

| 패키지 | 선언 범위 | 용도 |
|---|---|---|
| fastapi | `>=0.128.0,<0.129.0` | API 서버 |
| uvicorn | `>=0.40.0,<0.41.0` | ASGI 서버 |
| sentence-transformers | `>=3.0.0` | BGE-M3 쿼리 인코딩 (PyTorch는 전이 의존성으로 설치) |
| faiss-cpu | `>=1.8.0` | 인메모리 HNSW ANN |
| lightgbm | `>=4.6.0,<5.0.0` | 리랭커 추론 (pickle 로드) |
| google-genai | `>=1.0.0` | Gemini 2.5 Flash 호출 |
| psycopg2-binary | `>=2.9.0` | Postgres 커넥션 풀 |
| pgvector | `>=0.3.0` | ⚠️ 선언만 되어 있고 `app/`에서 import하지 않음 |
| pandas / numpy | `>=2.2.0` / `>=1.26.0` | 피처 DataFrame |
| requests | `>=2.31.0` | ⚠️ `app/`에서 사용처 없음 |
| python-dotenv | `>=1.0.0` | `.env` 로드 |
| (test 그룹) pytest / httpx | `>=8.0.0,<10.0.0` / `>=0.27.0` | 테스트 |

- ⚠️ **lock 파일(`poetry.lock`) 미커밋** → fastapi/uvicorn/lightgbm 외에는 하한만 있어서 빌드 시점마다
  sentence-transformers·torch·faiss·google-genai 버전이 달라질 수 있습니다. HF Space는 빌드할 때마다
  최신을 받습니다.
- CI는 pyproject 전체가 아니라 `fastapi>=0.128,<0.129 pytest httpx ruff`만 설치합니다(무거운 ML 의존성 회피).

### 3.2 ML — ⚠️ 의존성 선언 파일 없음

`ml/setup.py`에 `install_requires`가 없고 `requirements.txt`도 없습니다. 코드에서 import하는 것 기준:

| 패키지 | 사용 위치 | 버전 |
|---|---|---|
| apache-beam | `ml/pipeline/**` | 미고정 |
| psycopg2 | `pipeline/transforms/sink.py`, `scripts/embed_products.py` | 미고정 |
| sentence-transformers, tqdm, numpy | `scripts/embed_products.py` | 미고정 |
| lightgbm, pandas, scikit-learn(`ndcg_score`) | `item_ranker/modeling/train/train_lgbm.py` | 미고정 |
| xgboost | `item_ranker/modeling/*/xgb.py` | 미고정 |
| huggingface_hub | 노트북, `deploy/hf-space` | 미고정 |

모델: `BAAI/bge-m3`(파인튜닝, 1024차원), LightGBM `LGBMRanker(objective="lambdarank")`,
Gemini `gemini-2.5-flash`(thinking_budget=0, temperature 0.2, JSON 응답 모드).

### 3.3 Frontend — [frontend/package.json](../../frontend/package.json) (선언 범위 → lock 기준 실제 버전)

| 패키지 | 선언 | lock |
|---|---|---|
| react / react-dom | `^19.2.0` | 19.2.4 |
| vite | `^7.2.4` | 7.3.1 |
| @vitejs/plugin-react | `^5.1.1` | 5.1.3 |
| typescript | `~5.9.3` | 5.9.3 |
| eslint (+ typescript-eslint `^8.46.4`, react-hooks `^7.0.1`) | `^9.39.1` | 9.39.2 |

런타임 의존성은 react/react-dom 두 개뿐입니다(라우터·상태관리·UI 라이브러리 없음, fetch 직접 사용).

### 3.4 인프라

| 구성 | 버전/이미지 |
|---|---|
| 로컬 DB | `pgvector/pgvector:pg16` (halfvec에는 pgvector ≥0.7 필요) |
| 프로덕션 DB | Supabase PostgreSQL (트랜잭션 풀러 :6543) |
| 서빙 | Hugging Face Docker Space (무료 CPU, 16GB RAM) |
| 모델 저장소 | HF Hub `kevin7548/chatbeauty-models` (BGE-M3 + LightGBM + FAISS 인덱스, ~2.6GB) |
| 프론트 | Vercel |
| CI | GitHub Actions (`actions/checkout@v4`, `actions/setup-python@v5`, Python 3.12) |

---

## 4. 설정 파일과 환경변수

별도의 config 모듈/YAML은 없고, **모든 설정이 환경변수**이며 각 모듈이 임포트 시점에
`os.environ.get(KEY, 기본값)`으로 직접 읽습니다.

### 4.1 Backend 환경변수

| 변수 | 읽는 곳 | 기본값 | 프로덕션에서 주입 |
|---|---|---|---|
| `GEMINI_API_KEY` | `services/explanation.py` | 없음 — **미설정 시 임포트 단계에서 RuntimeError** | Space secret |
| `DATABASE_URL` | `services/retrieval_resources.py` | `postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty` | Space secret (Supabase pooler) |
| `ALLOWED_ORIGINS` | `main.py` | `http://localhost:5173` (콤마 구분) | Space variable (Vercel origin) |
| `DEBUG` | `main.py` | `false` | — |
| `BGE_MODEL_PATH` | `retrieval_resources.py` | `<backend>/ml/model-gcs/retrieval/bge-m3-finetuned-20260202-120852` | `start.sh`가 export |
| `ANN_INDEX_PATH` | `retrieval_resources.py` | `<backend>/ml/model-gcs/retrieval/ann/faiss_hnsw_cosine.index` | `start.sh`가 export |
| `ANN_ASINS_PATH` | `retrieval_resources.py` | `<backend>/ml/model-gcs/retrieval/ann/asins.json` | `start.sh`가 export |
| `ANN_EF_SEARCH` | `retrieval_resources.py` | `100` | — |
| `RERANK_MODEL_PATH` | `services/reranking.py` | `<backend>/ml/model-gcs/reranking/lgbm_reranker_current_features_v1.pkl` | `start.sh`가 export |
| `PORT` | `backend/Dockerfile` CMD | `8080` | (Space는 7860 고정) |
| `MODEL_REPO_ID`, `MODEL_DIR`, `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER` | `deploy/hf-space/start.sh`, Dockerfile | `MODEL_DIR=/app/models` | Space variable |
| `DB_PASSWORD` | `docker-compose.yml` | `chatbeauty` | 로컬 전용 |

`<backend>`는 `Path(__file__).parents[2]`, 즉 `backend/`입니다. 기본값의 `model-gcs`는 구 Cloud Run
시절 GCS 볼륨 마운트 경로의 흔적입니다.

**설정이 주입되는 경로 (환경별)**

1. **HF Space(프로덕션):** Space secrets/variables → `start.sh`가 HF Hub에서 모델을 받은 뒤 모델 경로
   4개를 export → uvicorn. `.env` 파일은 쓰지 않습니다.
2. **docker-compose(로컬):** `env_file: .env`로 전체 주입, `DATABASE_URL`은 compose가 `db` 서비스로 덮어씀.
   모델 디렉토리는 기본 경로(`/app/ml/model-gcs/...`)에 read-only 마운트.
3. **bare uvicorn(로컬):** 셸 환경변수 + `backend/.env`. 단 `.env` 로딩(`load_dotenv()`)은
   `explanation.py` 안에서만 일어납니다.

### 4.2 Frontend

| 변수 | 읽는 곳 | 기본값 |
|---|---|---|
| `VITE_API_URL` | `src/api/recommend.ts` | `http://localhost:8000` |

Vite가 빌드 시점에 `VITE_` 접두사 변수만 번들에 넣습니다. 프로덕션 값은 Vercel 환경변수(HF Space URL)이고,
값을 바꾸면 재배포가 필요합니다.

### 4.3 ML / 기타

- Beam: 설정 파일 없이 CLI 인자(`options.py`) — `--database-url` 생략 시 DB 쓰기 스킵.
- `embed_products.py`: `--database-url`, `--model-path`, `--batch-size` CLI 인자.
- 노트북: 셀 안에 `DATABASE_URL`, `MODEL_REPO_ID`를 직접 붙여넣는 방식.
- GitHub Actions secret: `HF_TOKEN`(keepalive에서 Space 재시작용).
- 비밀값은 `.gitignore`가 `.env`, `.env.*`를 막고 `*.example`만 추적합니다.

---

## 5. 문서·코드 불일치 및 리뷰 메모

구조를 훑으면서 발견한 사항입니다(수정하지 않음, 이후 리뷰 단계 입력용).

1. **docker-compose 로컬 기동이 실패할 가능성이 높음.** `retrieval_resources.py`는 임포트 시
   `faiss.read_index(ANN_INDEX_PATH)`를 호출하는데, compose는 BGE-M3와 reranking 디렉토리만 마운트하고
   `retrieval/ann/`은 마운트하지 않습니다. 로컬 `ml/model/retrieval/`에도 `ann/` 디렉토리가 없습니다.
   `.env.example`에도 `ANN_*` 변수가 없습니다.
2. **bare uvicorn에서 `.env`의 일부 값이 무시됨.** 임포트 순서가
   `main → routes.recommend → retrieval(→ retrieval_resources) → reranking → explanation(load_dotenv)`이라,
   `DATABASE_URL`·`BGE_MODEL_PATH`·`ANN_*`·`RERANK_MODEL_PATH`는 `.env`가 로드되기 **전에** 읽힙니다.
   `.env`에서 실제로 적용되는 건 `GEMINI_API_KEY`, `DEBUG`, `ALLOWED_ORIGINS` 정도입니다(셸에서 export하거나
   compose를 쓰면 문제없음).
3. **재현성 공백:** BGE-M3 파인튜닝, FAISS 인덱스 빌드, LLM 키워드 추출, 리랭커 학습 후보 생성 코드가
   레포에 없고, LightGBM 학습도 CLI가 없습니다. ML 의존성 선언 파일과 backend lock 파일도 없습니다.
4. **`item_ranker` 서빙 의존성 설명과 실제 코드가 다름.** `ml/README.md`는 "backend/app이 serve time에 import"한다고
   쓰지만, `backend/app`에는 `item_ranker` import가 없습니다. `reranking.py`가 피처 목록을 **복제**해 DB에서
   읽습니다(학습은 CSV, 서빙은 DB — 피처 정합성은 수동으로 맞춰야 함). 그런데도 두 Dockerfile 모두
   `pip install -e ml/`을 실행합니다.
5. **API 계약 불일치:** 프론트는 `top_k: 5`를 보내지만 `RecommendRequest`에는 `top_k`가 없어 무시되고,
   라우트에서 `top_k=5`로 하드코딩합니다. 프론트 `RecommendResponse` 타입에는 `latency` 필드가 없습니다.
6. **오래된 서술:** `frontend/src/api/recommend.ts`의 warmUp 주석(Cloud Run min-instances=0),
   `retrieval_resources.py` 독스트링("pgvector similarity search"), `.claude/agents/backend.md`
   ("pgvector HNSW"), `.env.example`("GCS volume mount"), `embed_products.py` 종료 로그("Cloud SQL Studio"),
   `deploy/setup-gcp.sh`, `embed_products_colab.ipynb`가 아직 GCP·pgvector 시절 기준입니다.
7. **미사용 의존성:** `pgvector`, `requests`(backend `app/` 기준).
8. **`.claude/worktrees/kind-turing-6714e5/`** 에 구 코드 사본이 있어 전역 grep 결과를 오염시킵니다.
