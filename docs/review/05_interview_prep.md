# 05. 면접 대비: 설계 결정, 약점, 예상 질문

> 기준: `main` @ `e7cb606` (2026-10-12). 코드는 수정하지 않았습니다.
> 01~04 리뷰 문서에서 **코드, git 기록, 모델 파일, 로컬 데이터로 확인한 것만** 근거로 씁니다.
> 근거는 `파일:줄` 또는 `[0N 문서 §]`로 표시했습니다.
> 확인되지 않은 것은 "확인 불가"로 남겼습니다.
>
> **먼저 짚을 점:** README의 리랭커 수치(NDCG@5 0.3655)는 피처에 미래 리뷰가 섞인 상태에서 나온 값입니다 [04 §2.3].
> 면접에서 이 수치를 내세우기 전에, 이 사실을 **먼저 말하고 어떻게 고칠지 설명하는 쪽**이 훨씬 방어하기 좋습니다(Q4).

---

## 1. 핵심 설계 결정 5가지와 trade-off

### D1. Two-stage: Dense retrieval Top-100 → GBDT rerank Top-5

- **코드**: [routes/recommend.py](../../backend/app/api/routes/recommend.py) — `retrieve_candidates(n=100)` → `rerank_items(top_k=5)` → `generate_explanation`
- **얻는 것**: 112K 상품 전체에 비싼 모델을 돌리지 않습니다. 비싼 연산은 후보 100개에만 씁니다. 두 단계를 따로 학습하고 교체할 수 있습니다.
- **잃는 것**:
  - 최종 품질의 상한이 Stage 1 recall로 정해집니다. Valid Recall@100이 0.3543이라 **쿼리의 64.6%는 리랭커가 무엇을 해도 정답을 찾을 수 없습니다** [04 §3.2].
  - 리랭커는 정답이 Top-100에 있는 쿼리로만 학습하고 평가합니다 ([train_lgbm.py:57](../../ml/item_ranker/modeling/train/train_lgbm.py:57)).
  - 두 단계 사이에 넘어가는 신호는 cosine 점수 하나뿐입니다 ([reranking.py:90](../../backend/app/services/reranking.py:90)).

### D2. 파인튜닝한 BGE-M3 bi-encoder (in-batch negative MNRL)와 LLM 키워드로 보강한 아이템 텍스트

- **코드/근거**:
  - 모델카드: `MultipleNegativesRankingLoss(scale=20)`, batch 32, 2 epoch, CLS pooling + Normalize [03 §1.2-1.4]
  - 아이템 텍스트: `[Title] [Review Keywords] [Description Summary] [Features]` ([join.py](../../ml/pipeline/transforms/join.py) `BuildEmbeddingTextFn`)
- **얻는 것**:
  - 다국어 사전학습 모델이라 한국어 쿼리를 영어 상품 텍스트에 그대로 매칭할 수 있습니다.
  - 리뷰에서 뽑은 상황 키워드(WHO/WHEN/WHY)를 아이템 텍스트에 넣어서, "상황 문장 → 상품" 매칭을 학습할 수 있게 했습니다.
  - Valid Recall@100이 0.2015에서 0.3543으로 올랐습니다 [문서].
- **잃는 것**:
  - 568M 파라미터 인코더를 공유 CPU에서 돌려서 retrieval에 약 1.6초가 걸립니다 [README latency 표].
  - 네거티브가 in-batch뿐이고 hard negative가 없습니다. 같은 상품의 리뷰가 한 배치에 들어가면 false negative가 생깁니다(`NO_DUPLICATES` 미사용) [03 §1.3].
  - 학습 쿼리는 영어 리뷰와 키워드인데 서빙 쿼리는 한국어 문장입니다. 이 차이를 잰 평가는 없습니다.

### D3. pgvector 대신 인메모리 FAISS HNSW, Postgres는 메타데이터 전용

- **코드/근거**:
  - [retrieval_resources.py:27-51](../../backend/app/services/retrieval_resources.py:27): 주석 "embeddings + index don't fit the Supabase free-tier cap"
  - [DEPLOY.md](../../deploy/hf-space/DEPLOY.md): 임베딩 약 230 MB + 인덱스 약 230 MB가 500 MB 무료 한도를 넘음. 인덱스 없이 전체를 스캔하면 쿼리당 약 40초
- **얻는 것**: ANN 검색이 약 5 ms입니다. DB 비용이 들지 않습니다. DB는 `WHERE parent_asin = ANY(...)` 조회 한 번만 합니다.
- **잃는 것**:
  - 인덱스(약 477 MB)와 모델(약 2.1 GB)이 프로세스 메모리에 올라갑니다. 기동할 때 약 2.6 GB를 내려받아야 해서 콜드스타트가 몇 분입니다.
  - 인덱스는 오프라인에서 수동으로 다시 빌드하고 다시 올려야 합니다. 빌드 코드도 레포에 없습니다.
  - `asins.json`과 DB 행이 따로 관리됩니다. 둘이 어긋나면 해당 후보는 조용히 빠집니다 ([retrieval.py](../../backend/app/services/retrieval.py) 재조립 루프).
  - 메타데이터 조건(가격 등)을 걸 수 없습니다. 실시간 증분 업데이트도 안 됩니다.

### D4. 리랭커: LightGBM LambdaRank, 피처 10개 (검색 점수 1개 + 상품 통계 9개)

- **코드/근거**:
  - [train_lgbm.py](../../ml/item_ranker/modeling/train/train_lgbm.py)
  - pkl에서 확인: 트리 70개, num_leaves 31, depth 5. 분기 횟수는 `retrieval_score` 875, `review_cnt` 528, `vp_review_cnt` 452, `price_cnt` 0 [03 §2.2]
- **얻는 것**: CPU에서 100행 예측이 1 ms 수준입니다. 피처 중요도로 동작을 설명할 수 있습니다. 쿼리 안의 순서를 직접 최적화합니다(NDCG 가중 pairwise).
- **잃는 것**:
  - 쿼리와 상품의 상호작용 피처가 없습니다. 쿼리별로 달라지는 건 cosine 하나뿐이라, 사실상 "유사도 + 인기도" 재정렬입니다.
  - 학습 피처는 CSV, 서빙 피처는 DB에서 와서 정의가 다릅니다(`recent_review_cnt`) [02 부록 A].
  - CSV에는 미래 리뷰가 섞여 있습니다 [04 §2.3].

### D5. 설명 생성: Gemini 2.5 Flash를 요청 경로 안에서 동기로 한 번 호출 (thinking off, JSON 모드)

- **코드**: [explanation.py:50-60](../../backend/app/services/explanation.py:50). 주석: thinking을 켜면 약 12초, 끄면 약 2-3초. `temperature 0.2`, `response_mime_type="application/json"`, 상품 5개를 한 번에 요청
- **얻는 것**: 호출 1회로 끝나서 비용과 지연이 예측 가능합니다. thinking을 끄면 지연이 크게 줄고 출력 토큰 낭비도 막습니다. 리뷰 인용으로 근거를 대라고 프롬프트에서 강제합니다.
- **잃는 것**:
  - 전체 약 4.8초 중 약 2.4초가 이 호출입니다. 추천 결과를 먼저 보여주고 설명을 나중에 붙이는 구조(스트리밍/비동기)가 아닙니다.
  - 실패해도 200을 반환하고 설명만 비어 있습니다 (`{"item_id": "all"}` 폴백 → 5개 모두 None, [explanation.py](../../backend/app/services/explanation.py)).
  - 설명 품질(사실성, 환각) 평가가 없습니다.
  - Beam으로 다시 적재하면 근거 필드인 `top_reviews`가 비게 됩니다 [02 §D-1].

> (참고) 무료 티어 운영도 설계 제약입니다: HF Space + Supabase + 주 2회 keepalive cron.
> [keepalive.yml](../../.github/workflows/keepalive.yml) 주석에 "DB ping으로는 Supabase pause를 막지 못했다"는 경험이 남아 있습니다. 비용 0의 대가로 가용성이 낮습니다.

---

## 2. 코드에서 보이는 약점과 개선 방향

### 2.1 평가의 허점 (가장 중요)

| 약점 | 근거 | 개선 |
|---|---|---|
| **리랭커 피처에 미래 정보가 섞임** | CSV의 `review_cnt`가 98.7% 상품에서 valid/test까지 포함한 전체 기간 리뷰 수와 같음. 인기도만으로 정렬해도 NDCG@5가 0.1758(train 기간만)에서 0.2704(CSV)로 바뀜 [04 §2.3, §4.2] | 쿼리 시점 이전 리뷰로만 피처를 다시 계산(point-in-time)하고, 다시 학습해서 NDCG를 다시 보고 |
| 검증셋 재사용, test 없음 | early stopping에 쓴 valid의 `best_score`를 그대로 보고. 리랭커 test 평가 없음 [04 §2.4] | valid는 모델 선택에만, test로 최종 보고 |
| 조건부 지표 | 정답이 Top-100 안에 있는 35.4% 쿼리만으로 NDCG 계산 [04 §1.2] | 전체 쿼리 기준 end-to-end HitRate@5를 함께 보고 |
| 기준선 미보고 | 기준선 스크립트는 있지만 결과가 없음. 이번에 잰 값: 리트리버 순서 0.2165 [04 §4.2] | 리트리버 순서, 인기도 순서와 항상 같이 보고 |
| 평가 쿼리 ≠ 서빙 쿼리 | 평가는 영어 리뷰 원문, 서빙은 한국어 상황 문장. UI 예시 4개 중 3개가 한국어 ([Hero.tsx](../../frontend/src/components/Hero.tsx)) | 한국어 상황 쿼리로 작은 평가셋 구축 |
| 라벨 편향 | 리뷰 1건 = 쿼리 1개라 리뷰 많은 상품이 정답으로 자주 등장 | 상품별로 쿼리 수를 맞춘(층화) 평가 추가 |
| 설명 품질 평가 없음 | 코드에서 확인 불가 [04 §1.3] | 근거 인용 정확도 자동 검사 + 소량 사람 평가 |

### 2.2 재현성

| 약점 | 근거 |
|---|---|
| 평가·파인튜닝·학습 쌍 생성 코드가 `main`에 없고 git 기록에만 있음 | [04 머리말] (`17fd99f`, `1e9f188` 등) |
| 피처 CSV, FAISS 인덱스, 키워드 추출 결과를 만드는 코드 없음 | [01 §2.2-2.3], [04 §2.3] |
| backend lock 파일 없음, ML 의존성 선언 없음 | [01 §3] |
| 로컬 `training_pairs.jsonl`이 다른 파일로 덮어써짐 (518,393줄 ≠ 학습에 쓴 약 1.0M쌍) | [04 §3.2] |
| 실험 기록(MLflow)이 로컬에 없음 | [04 §3.1] |
| docker-compose가 FAISS 인덱스를 마운트하지 않아 로컬 기동 실패 가능 | [01 §5-1] |

→ 개선: `ml/evaluation/`과 학습 스크립트를 되살리고, 데이터와 모델 산출물에 버전 해시를 남기고, lock 파일을 커밋하고, 실험 결과를 레포 안 파일(CSV/JSON)로 남깁니다.

### 2.3 레이턴시

README 기준 약 4.8초 = retrieval 약 1.6초 + rerank 약 0.85초 + 설명 약 2.4초.

| 원인 | 근거 | 개선 |
|---|---|---|
| 세 단계가 순서대로 동기 실행, 설명 대기가 전체의 절반 | [routes/recommend.py](../../backend/app/api/routes/recommend.py) | Top-5를 먼저 응답하고 설명은 스트리밍/별도 요청으로 |
| CPU에서 568M 인코더로 쿼리 인코딩 | [retrieval.py:47](../../backend/app/services/retrieval.py:47) | ONNX + int8 양자화(TODO에도 있음), 쿼리 임베딩 캐시 |
| 리랭크 시간의 대부분이 DB 왕복 | 같은 테이블을 2번 조회하고, 매번 `SELECT 1`로 연결 확인 ([retrieval_resources.py](../../backend/app/services/retrieval_resources.py) `get_db_connection`) | 메타데이터 + 피처를 한 번에 조회, 정적 피처는 메모리에 캐시 (112K행 × 9개는 작음) |
| uvicorn worker 1개, 모델을 import할 때 로드 | [start.sh](../../deploy/hf-space/start.sh) | 무료 티어 제약. 유료라면 replica + readiness probe |

### 2.4 확장성

- 프로세스 하나가 모델과 인덱스(약 2.6 GB)를 모두 들고 있어서, 수평 확장하면 replica마다 이 메모리가 그대로 필요합니다.
- 인덱스 재빌드는 수동이고 오프라인입니다. 신규 상품을 바로 반영할 수 없습니다.
- DB 풀은 최대 5개입니다. 라우트는 동기 `def`라서 FastAPI 스레드풀에서 블로킹 I/O(DB, Gemini)를 기다립니다.
- Beam은 DirectRunner로 돌고, 학습 쌍 생성 단계는 약 16 GB에서 OOM이 납니다 ([ml/README.md](../../ml/README.md)).

### 2.5 견고성과 코드 품질

- **조용히 실패하는 곳**:
  - Beam sink는 예외가 나면 로그만 남깁니다.
  - DB에 없는 ASIN은 검색 결과에서 그냥 빠집니다.
  - Gemini 파싱이 실패해도 200을 반환합니다 [02 §D-7].
- **피처 계약이 복사본**: `FEATURE_NAMES`가 학습([tree.py](../../ml/item_ranker/features/tree.py))과 서빙([reranking.py](../../backend/app/services/reranking.py))에 따로 있습니다.
  테스트([conftest.py](../../backend/tests/conftest.py))는 서비스를 모두 stub으로 바꾸므로 SQL과 피처 순서는 검증하지 않습니다.
- **인증과 rate limit이 없습니다** (공개 엔드포인트, [architecture.md](../architecture.md) 한계 항목).
- **API 계약 불일치**: 요청의 `top_k`는 무시되고, 응답의 `score`는 LGBM 점수(의미 있는 스케일 아님)입니다.

---

## 3. 예상 질문 10개와 답변 초안

> 답변은 "결정 → 이유(코드 근거) → 한계 인정 → 다음 단계" 순서로 썼습니다.

**Q1. 왜 two-stage 구조인가요?**
> 상품 112K개 전체에 리랭커를 돌릴 수 없어서, 싼 dense ANN으로 100개를 먼저 뽑고(`retrieve_candidates(n=100)`), 100개에만 LightGBM을 돌립니다(`rerank_items(top_k=5)`).
> 한계도 분명합니다. Valid Recall@100이 0.354라 쿼리의 약 65%는 정답이 후보에 아예 없습니다. 그래서 다음 개선은 리랭커보다 **retrieval recall**에 투자하는 게 맞다고 봅니다. 예를 들면 BM25를 섞는 hybrid 검색과 hard negative 학습입니다.

**Q2. 왜 BGE-M3를 파인튜닝했고, 학습은 어떻게 했나요?**
> 한국어 쿼리를 영어 상품 데이터에 매칭해야 해서 다국어 모델이 필요했습니다.
> 리뷰를 쿼리로, 상품의 `embedding_text`를 positive로 잡고 MNRL(in-batch negative, batch 32, 2 epoch)로 학습했습니다. 키워드 쿼리와 리뷰 원문 쿼리를 둘 다 썼습니다(`QUERY_TYPE=both`, 약 1.0M쌍 중 90%).
> Valid Recall@100은 0.20에서 0.35로 올랐습니다.
> 아쉬운 점은 hard negative가 없다는 것, 같은 상품이 한 배치에 들어가면 false negative가 생긴다는 것, 그리고 학습 코드가 지금 `main`에 없어서 재현이 어렵다는 것입니다.

**Q3. 왜 리랭커로 LightGBM LambdaRank를 골랐나요? XGBoost나 cross-encoder는요?**
> 피처가 표 형태로 10개뿐이고 서빙이 CPU라서, 지연과 해석 가능성을 보고 GBDT를 골랐습니다. 목적함수는 쿼리 안의 순서를 직접 최적화하는 LambdaRank입니다.
> XGBoost(`rank:ndcg`)도 같은 피처로 학습하는 코드가 있습니다. 다만 결과가 MLflow에만 있었고 지금 남아 있지 않아서 **수치로 비교했다고 말할 수는 없습니다**.
> cross-encoder는 쿼리-상품 상호작용을 볼 수 있어서 다음 단계로 생각하고 있습니다. 지금 피처는 쿼리별로 바뀌는 게 cosine 하나뿐이라, 리랭커가 사실상 "유사도 + 인기도" 재정렬을 하고 있기 때문입니다(분기 횟수: `retrieval_score` 875, `review_cnt` 528).

**Q4. NDCG@5 0.3655는 믿을 만한 수치인가요?** ← 가장 중요
> 솔직히 말하면 부풀려진 수치입니다. 코드를 다시 보면서 찾았습니다.
> 리랭커 피처 CSV의 리뷰 수에 valid/test 기간 리뷰가 포함되어 있습니다. 그래서 평가 쿼리로 쓴 리뷰 자체가 정답 상품의 리뷰 수에 들어가 있습니다.
> 리뷰 수 하나로만 정렬해도 train 기간 통계로는 NDCG@5 0.176, 누수된 통계로는 0.270입니다. 리트리버 순서는 0.217입니다.
> 거기에 early stopping에 쓴 같은 valid로 보고했고, test 평가도 없습니다.
> 고칠 방향은 point-in-time 피처로 다시 계산하고, test로 보고하고, 리트리버와 인기도 기준선을 함께 보고하는 것입니다. 온라인 지표가 없으니 이 수치는 어디까지나 **오프라인 relevance 지표**입니다.

**Q5. 왜 pgvector 대신 FAISS를 메모리에 올렸나요?**
> 무료 티어 제약 때문입니다. Supabase 500 MB 안에 임베딩과 pgvector 인덱스(인덱스가 벡터를 한 벌 더 저장)를 같이 넣을 수 없었고, 인덱스 없이 전체 스캔하면 쿼리당 약 40초가 걸렸습니다.
> 그래서 HNSW(M=16, efSearch=100)를 메모리에 올리고, DB는 Top-100 메타데이터만 조회하게 했습니다. ANN은 약 5 ms입니다.
> 대가는 콜드스타트(약 2.6 GB 다운로드), 수동 재빌드, 인덱스와 DB 정합성을 따로 맞춰야 하는 점, 필터 검색이 안 되는 점입니다.
> 다만 이건 비용 제약 아래의 선택입니다. 일반적인 상황이라면 pgvector든 FAISS든 recall과 지연을 재서 고를 것입니다. 지금 efSearch가 k와 같아서 exact 검색 대비 recall을 재 본 적이 없다는 것도 인정해야 합니다.

**Q6. 응답이 4.8초인데 어떻게 줄이겠어요?**
> 측정 기준으로 설명이 약 2.4초, 검색이 약 1.6초, 리랭크가 약 0.85초입니다.
> ① 설명을 요청 경로에서 빼서 Top-5를 먼저 응답하고 설명은 스트리밍으로 붙입니다.
> ② 쿼리 인코더를 ONNX와 int8로 바꾸고 쿼리 임베딩을 캐시합니다.
> ③ 리랭크 시간은 모델이 아니라 DB 왕복입니다. 같은 테이블을 두 번 조회하고 매번 `SELECT 1`을 보냅니다. 정적 피처를 메모리에 올리면 거의 0이 됩니다.
> thinking을 끈 것만으로 설명 시간이 약 12초에서 2-3초로 줄었던 경험이 코드 주석에 남아 있습니다.

**Q7. 데이터가 100배(상품 1천만 개, 리뷰 수억 건)가 되면 어떻게 바꾸겠어요?**
> - 인덱스: HNSW Flat은 메모리를 너무 많이 씁니다(1천만 × 1024 × 4B ≈ 40 GB). IVF-PQ 같은 압축 인덱스나 샤딩된 벡터 DB로 바꾸고, 차원 축소도 검토합니다(차원 축소 평가 계획이 `docs/plans`에 있습니다).
> - 오프라인 파이프라인: Beam이 이미 Dataflow로 옮길 수 있는 형태입니다(`options.py`가 `gs://` 경로를 받습니다). 다만 학습 쌍 생성 단계의 CoGroupByKey가 OOM이 나는 걸 먼저 고쳐야 합니다.
> - 피처: CSV와 DB를 따로 쓰는 대신 피처 스토어로 학습과 서빙의 피처를 통일하고 point-in-time으로 조회합니다.
> - 검색: dense 하나 대신 dense + lexical + 인기/카테고리 등 여러 경로로 후보를 모으고 섞습니다.
> - 서빙: 인코더와 랭커를 별도 서비스로 분리하고 GPU로 배치 추론합니다.

**Q8. 평가는 어떻게 설계했나요?**
> 리뷰 timestamp 기준 80/10/10 시간 분할입니다. 로컬 데이터로 세 구간이 겹치지 않는 것을 확인했습니다.
> retrieval은 Recall@{1..100}과 MRR(valid R@100 0.354, test 0.373), 리랭커는 NDCG@5/10입니다.
> 아이템 텍스트는 train 기간 키워드로만 만들어서 retrieval 쪽에는 누수가 없습니다(valid/test에만 나오는 키워드를 가진 상품 0.02%).
> 반면 리랭커 피처에는 누수가 있습니다(Q4). 그 밖의 한계는 정답이 하나뿐인 대리 라벨, 영어 리뷰 쿼리와 한국어 서빙 쿼리의 차이, 온라인 지표가 없다는 점입니다.

**Q9. 학습은 영어, 서비스는 한국어인데 괜찮나요?**
> BGE-M3의 다국어 정렬에 기대고 있습니다. 쿼리와 상품이 같은 인코더를 쓰고 prompt도 없습니다.
> 하지만 한국어 쿼리로 검증한 결과는 **없습니다**.
> 이 차이는 리랭커에도 영향을 줍니다. 리랭커가 가장 많이 쓰는 피처가 cosine인데, 학습 때 cosine 범위는 0.19~0.85였습니다. 한국어 쿼리에서는 이 분포가 달라질 수 있습니다.
> 다음 단계는 한국어 상황 쿼리와 정답 쌍을 소량 만들어서 Recall과 cosine 분포를 비교하는 것입니다.

**Q10. 실서비스로 가려면 무엇이 필요하고, 신규 상품은 어떻게 처리하나요?**
> 지금은 노출이나 클릭을 기록하지 않습니다. 남기는 건 latency 로그뿐입니다(`LatencyMiddleware`, 라우트 `logger.info`).
> 먼저 요청, 노출, 클릭 로그를 남기고, 그걸로 클릭 기반 라벨, A/B 테스트, 위치 편향 보정을 해야 합니다.
> 신규 상품은 특별히 처리하지 않습니다. 검증 쿼리의 39.7%가 train 기간 리뷰 0건 상품인데, 이런 상품은 아이템 텍스트에 리뷰 키워드가 없고 인기도 피처도 낮습니다.
> 개선 방향은 메타데이터와 설명 요약만으로도 텍스트를 만들고, 신규 상품을 일정 비율 노출(탐색)하고, 인기도 피처를 시간 감쇠나 정규화로 다루는 것입니다.

---

## 4. 실제 광고/추천 시스템(CTR 예측, ad ranking)과 비교했을 때 부족한 점

| 항목 | 실제 광고/커머스 추천 | ChatBeauty (코드 기준) |
|---|---|---|
| **목적함수** | 노출 대비 클릭/구매 확률(pCTR, pCVR). 광고는 eCPM = pCTR × bid | 리뷰 → 리뷰 대상 상품을 맞히는 대리 relevance. LambdaRank 점수는 쿼리 안 순서용이라 확률이 아님 |
| **라벨 출처** | 노출 로그(클릭/비클릭), implicit feedback | 오프라인 리뷰 데이터뿐. 노출/클릭 로깅 없음 |
| **보정(calibration)** | 입찰가와 곱해야 하므로 logloss, calibration 지표가 핵심 | 보정 없음. 응답의 `score`는 스케일 의미가 없음 |
| **개인화** | user ID와 행동 시퀀스(DIN/SIM 등), user tower | 요청은 `user_input` 문자열 하나뿐(`RecommendRequest`). 사용자 피처 없음 (TODO의 user-tower는 미구현) |
| **피처 규모** | 수억 개 희소 ID 피처, 피처 교차, 임베딩 (DeepFM/DCN 등) | 밀집 피처 10개. 실제로는 2~3개만 사용 |
| **피처 신선도와 point-in-time** | 피처 스토어, 실시간 집계, 시점 일관성 | 한 번 만든 배치 CSV와 DB. 학습/서빙 정의가 다르고 미래 정보가 섞임 |
| **위치/노출 편향** | IPS, position feature, 탐색 트래픽 | 해당 없음 (로그 없음). `lambdarank_position_bias_regularization=0` |
| **후보 생성** | 여러 경로(협업 필터링, 인기, 카테고리, 최근 본 상품 등) | dense ANN 하나뿐, 필터 없음 |
| **비즈니스 제약** | 다양성, 중복 제거, 재고/가격, 예산 pacing, 광고 경매 | 없음. Top-5를 그대로 자름 |
| **학습 주기** | 일 단위 또는 online learning, 드리프트 모니터링 | 모델 1회 학습 후 고정. 재학습 파이프라인, 모니터링 없음 |
| **실험 인프라** | A/B 테스트, interleaving, 가드레일 지표 | 없음 |
| **지연 예산** | 랭킹 수십 ms (p99 기준) | 약 4.8초 (공유 CPU, 무료 티어) |
| **다목적 최적화** | CTR, CVR, GMV, 체류 시간 등을 함께 최적화 | 단일 relevance 목적 |

**면접에서 쓸 수 있는 한 줄 요약**
> "이 프로젝트는 **피드백 루프가 없는 콘텐츠 기반 검색 + 학습 랭킹**입니다. CTR 시스템으로 가려면 ① 노출/클릭 로깅, ② 보정된 확률 모델과 point-in-time 피처, ③ 개인화 신호, ④ A/B 실험 인프라가 필요하고, 지금 코드에는 이 네 가지가 모두 없습니다."
>
> 이렇게 범위를 정확히 말하는 쪽이 신뢰를 얻습니다. 반대로 "추천 시스템을 운영했다"처럼 과장하면, 이 표의 어느 칸을 물어봐도 막힙니다.
