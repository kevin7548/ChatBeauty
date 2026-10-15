# 04. 평가·실험 코드 분석

> 기준: `main` @ `e7cb606` (2026-10-12). 코드는 수정하지 않았습니다.
>
> **현재 `main`에는 평가 코드가 없습니다.** 평가·실험 스크립트는 2026-02 무렵 git 기록에만 남아 있고, 이후 리팩토링 때 삭제되었습니다.
> 이 문서는 그 **git 기록의 코드**를 근거로 씁니다. 각 파일 옆에 `커밋:경로`를 적었고, `git show <커밋>:<경로>`로 열어 볼 수 있습니다.
>
> 표기 규칙:
> - **[기록]**: git 기록에 남은 코드
> - **[현행]**: 현재 `main`의 코드
> - **[문서]**: README 등에 적힌 수치
> - **[측정]**: 이번 리뷰에서 로컬 `ml/data`로 직접 계산한 값. 일회성 스크립트라 레포에는 넣지 않았습니다.
> - **코드에서 확인 불가**: 근거가 되는 코드나 산출물이 없는 항목

## 0. 요약

| 질문 | 답 |
|---|---|
| 지표 | Retrieval: **Recall@{1,5,10,20,50,100}, MRR@100**. Reranking: **NDCG@5, NDCG@10** (LightGBM 내장 지표와 sklearn `ndcg_score`) |
| 분할 | 리뷰 **timestamp 기준 80/10/10 시간 분할**. 로컬 데이터에서 세 구간이 겹치지 않는 것을 확인 |
| leakage | ① Retrieval은 **깨끗함** (아이템 텍스트는 train 기간 리뷰만 사용). ② **리랭커 피처에 미래 리뷰가 섞임**: `item_features_v1.csv`의 리뷰 통계가 valid/test 기간 리뷰를 포함함 → **오프라인 NDCG가 부풀려짐** (§2.3). ③ 리랭커는 early stopping에 쓴 검증셋의 점수를 그대로 최종 성능으로 보고하고, test 평가는 없음 |
| 결과 위치 | README 표, 삭제된 `backend/ml/README.md`(git 기록), 모델카드의 학습 손실 로그, pkl 안의 검증 곡선. MLflow 기록은 로컬에 없음 |
| baseline | Retrieval: 파인튜닝 전 BGE-M3 (Recall@100 0.2015). Reranker: 기준선 **코드**는 있지만 **결과 기록이 없음** → 이번에 직접 계산 (§4) |

---

## 1. 평가 지표와 계산 방식

### 1.1 Retrieval: Recall@K, MRR

**[기록] `17fd99f:backend/ml/evaluation/evaluate_recall.py`**

```python
k_values = [1, 5, 10, 20, 50, 100]
rank = next((i + 1 for i, a in enumerate(candidate_asins) if a == ground_truth), None)
for k in k_values:
    if rank is not None and rank <= k: hits[k] += 1
reciprocal_ranks.append(1.0 / rank if rank else 0.0)
# Recall@K = hits[k] / total,  MRR = mean(reciprocal_ranks)
```

| 항목 | 내용 |
|---|---|
| 쿼리 단위 | 리뷰 1건 (`query_text` = 리뷰 원문) |
| 정답 | 그 리뷰를 쓴 상품 `parent_asin` **1개** |
| Recall@K | 정답이 1개이므로 사실상 **HitRate@K** (Top-K 안에 정답이 있는 쿼리 비율) |
| MRR | 정답 순위의 역수 평균. Top-100 밖이면 0 → 엄밀히는 **MRR@100** |
| 분모 | 해당 split의 모든 쿼리 (정답이 후보에 없는 쿼리도 포함) |
| 입력 | `retrieval_candidates_{split}.jsonl`. 생성 코드: **[기록] `17fd99f:backend/ml/evaluation/retrieve_candidates.py`** (ChromaDB cosine Top-100, `score = 1 − distance`) |

### 1.2 Reranking: NDCG@5, NDCG@10

| 계산 경로 | 위치 | 대상 데이터 |
|---|---|---|
| LightGBM 내장 `ndcg@5/10` (early stopping 기준) | [현행] `ml/item_ranker/modeling/train/train_lgbm.py` (`metric="ndcg"`, `eval_at=[5,10]`) | `retrieval_candidates_valid.jsonl` |
| sklearn `ndcg_score(y_true, y_pred, k)` 쿼리별 평균 | [기록] `719fb09:backend/ml/scripts/evaluate_lgbm.py` | ⚠️ **`retrieval_candidates_train.jsonl`** (지표 이름도 `train_ndcg_5`) |
| 〃 (XGBoost) | [기록] `719fb09:backend/ml/scripts/evaluate_xgb.py` | ⚠️ 마찬가지로 **train** |
| 〃 (리트리버 점수 순서만 사용) | [기록] `650c27e:backend/ml/scripts/evaluate_retrieval_baseline.py` | valid |

- 라벨은 이진값입니다. 후보 100개 중 정답 상품만 1이고 나머지는 0입니다([현행] `ml/item_ranker/dataset/base.py:55`).
- **정답이 Top-100 안에 없는 쿼리는 NDCG 계산에서 뺍니다**([현행] `train_lgbm.py:57`, 기준선 스크립트도 같음).
  그래서 NDCG@5 0.3655는 **전체 검증 쿼리의 35.4%(22,203 / 62,669)만 대상으로 한 조건부 지표**입니다.
  전체 쿼리를 분모로 하면 대략 0.3655 × 0.3543 ≈ 0.13 수준입니다.
- 별도 평가 스크립트(`evaluate_lgbm.py`, `evaluate_xgb.py`)는 **학습 데이터로 평가**합니다. 따라서 일반화 성능을 보는 스크립트가 아닙니다.

### 1.3 기타

- 설명 생성(Gemini) 품질에 대한 평가 코드나 지표는 **코드에서 확인 불가**입니다. README의 한계 항목에도 "평가 기준 부족"이라고 적혀 있습니다.
- 온라인 지표(CTR, CVR, latency SLO 등)는 **코드에서 확인 불가**입니다. 서빙 로그에 latency만 남깁니다.
- 파인튜닝 중 검증: `InformationRetrievalEvaluator`를 만들기는 하지만 `model.fit(... #evaluator=evaluator ...)`로 **주석 처리**되어 있습니다([기록] `1e9f188:backend/ml/retriever/keyword_based/finetune_bge_m3.py:251`). 모델카드의 `eval_strategy: no`와도 일치합니다.

---

## 2. train / valid / test 분할과 leakage

### 2.1 분할 방식

**[기록] `retrieve_candidates.py:load_and_split_reviews`**

```python
# text가 20자 미만이거나 parent_asin이 없는 리뷰는 제외
reviews.sort(key=lambda x: x['timestamp'])
train_end, valid_end = int(total * 0.8), int(total * 0.9)
# train = [:train_end], valid = [train_end:valid_end], test = [valid_end:]
```

**[측정] 로컬 split 파일 확인**

| split | 리뷰 수 | 상품 수 | 기간 (timestamp → 날짜) |
|---|---|---|---|
| `reviews_train.jsonl` | 561,222 | 90,382 | 2000-11 ~ 2021-05-24 |
| `reviews_valid.jsonl` | 70,152 | 25,248 | 2021-05-24 ~ 2022-01-26 |
| `reviews_test.jsonl` | 70,154 | 20,382 | 2022-01-26 ~ 2023-09 |

- `train.max ≤ valid.min`, `valid.max ≤ test.min`이 **모두 참**입니다. 시간 분할이 맞습니다.
- 평가 쿼리 수가 62,669로 리뷰 수와 다른 이유는 `retrieve_candidates.py`가 20자 미만 리뷰를 따로 빼기 때문입니다.
- 사용자 단위 분할은 아닙니다. 같은 사용자가 여러 split에 나올 수 있습니다. 다만 사용자 피처를 쓰지 않으므로 영향은 없습니다.
- `reviews_*.jsonl`, `keywords_*.jsonl` 파일 자체를 만든 코드는 **코드에서 확인 불가**입니다.
  로컬 파일의 경계가 위 로직과 맞으므로 같은 규칙으로 만든 것으로 보입니다.

**파인튜닝 내부 분할**: [기록] `finetune_bge_m3.py:load_training_pairs`가 학습 쌍(전부 train 기간)을 `random.seed(42)`로 섞은 뒤 90/10으로 나눕니다(`val_ratio=0.1`).
10% 검증 쪽은 evaluator가 꺼져 있어서 실제로는 쓰이지 않습니다. 대신 이 분할을 보면 모델카드의 학습 쌍 수가 설명됩니다(§3.2).

### 2.2 Retrieval leakage: 없음 (확인함)

| 점검 | 근거 | 결과 |
|---|---|---|
| 아이템 텍스트(`[Review Keywords]`)가 평가 기간 리뷰를 쓰는가 | [기록] `3f6c62c^:backend/ml/features/merge_metadata.py:457`이 `keywords_train.jsonl`만 읽음. [현행] Beam도 `keywords_train`만 사용 | **[측정]** valid/test 키워드가 있는 아이템 37,220개 중, valid/test에만 나오는 키워드를 가진 아이템은 **7개 (0.02%)** → 사실상 없음 |
| 파인튜닝 학습 쌍이 평가 기간을 포함하는가 | [기록] `create_training_pairs.py:200`이 `keywords_train.jsonl`만 사용 | 없음 |
| valid 쿼리와 train 리뷰가 문장 그대로 겹치는가 | **[측정]** md5 비교 | 0.63%가 다른 상품의 train 리뷰와 같은 문장("Great product" 같은 짧은 리뷰로 추정). **같은 상품에서는 0%** |

참고로 valid 쿼리의 **39.7%는 train 기간에 리뷰가 하나도 없는 상품**이 정답입니다 [측정].
이런 상품의 아이템 텍스트에는 `[Review Keywords]`가 없어서 Recall이 구조적으로 낮을 수 있습니다. 이 구간만 따로 잰 결과는 **코드에서 확인 불가**입니다.

### 2.3 Reranker leakage: 미래 정보가 피처에 섞임 (확인함)

리랭커 피처 9개는 `item_features_v1.csv`에서 옵니다([현행] `ml/item_ranker/features/tree.py`).
이 CSV를 만든 코드는 git 기록 전체를 뒤져도 찾지 못했습니다(**코드에서 확인 불가**). 그래서 값을 직접 대조했습니다.

**[측정]**

| 비교 | `review_cnt` 일치율 |
|---|---|
| CSV vs **train 기간** 리뷰 수 | 65.8% |
| CSV vs **train+valid+test 전체** 리뷰 수 | **98.7%** |
| valid/test 리뷰가 있는 상품 37,872개만 보면: CSV vs 전체 기간 | **98.0%** |

즉 리랭커의 리뷰 통계는 **평가 기간(valid/test)의 리뷰까지 포함해서** 계산되어 있습니다. 결과적으로 다음 두 가지가 일어납니다.

1. 검증 쿼리 자체(리뷰 1건)가 정답 상품의 `review_cnt`, `vp_review_cnt`, `avg_rating`, `avg_review_len`에 **이미 반영되어 있습니다**.
2. train 기간에 리뷰가 없던 상품도 CSV에서는 리뷰 수가 1 이상입니다.
   Top-100 안에 정답이 있는 검증 쿼리 중 **정답 상품의 train 기간 리뷰 수가 0인 경우는 30.9%** 입니다. CSV 기준으로는 **0.1%** 입니다.
   "이번 기간에 리뷰가 달린 상품"이라는 미래 정보가 피처로 새어 들어간 것입니다.

**영향 크기** (§4.2): 리뷰 수 하나로만 정렬해도 NDCG@5가 **0.2704(CSV) vs 0.1758(train 기간만)** 으로 달라집니다.
LightGBM은 이 피처를 두 번째로 많이 씁니다(split 528회, [03 문서](03_models.md) §2.2). 따라서 **보고된 0.3655는 leakage로 부풀려진 값**입니다.
깨끗한 피처로 다시 학습했을 때의 NDCG는 **코드에서 확인 불가**입니다. 이 컴퓨터에는 lightgbm 실행 환경(libomp)이 없어서 이번에 재학습하지 못했습니다.

### 2.4 그 밖의 평가 설계 문제

| 항목 | 내용 | 근거 |
|---|---|---|
| 검증셋 재사용 | early stopping과 최종 보고에 같은 valid를 씀 (`best_score`를 그대로 보고). 낙관적으로 치우침 | [현행] `train_lgbm.py`, pkl `_best_score` 0.36562 ≈ README 0.3655 |
| 리랭커 test 평가 | 없음. `retrieval_candidates_test.jsonl`이 로컬에 없고 test NDCG 기록도 없음 | **코드에서 확인 불가** |
| 조건부 평가 | 정답이 Top-100에 없는 쿼리(64.6%)를 뺀 NDCG | §1.2 |
| 쿼리 분포 | 평가 쿼리는 영어 리뷰 원문이고, 서빙 쿼리는 한국어/영어 상황 문장 → 오프라인 지표가 서비스 품질을 대표하지 못함 | 03 문서 §3.3 |
| 라벨 편향 | 리뷰 1건이 쿼리 1개 → 리뷰가 많은 상품일수록 정답으로 자주 등장 → 인기도 피처에 유리 | 03 문서 §2.4 |
| 상품 메타 시점 | `meta_All_Beauty.jsonl`의 `average_rating`, `rating_number`는 수집 시점(2023) 값. 후보 파일에 들어 있지만 현재 리랭커 피처로는 쓰지 않음 | [기록] `retrieve_candidates.py` |

---

## 3. 실험 결과가 남아 있는 곳

### 3.1 위치별 정리

| 위치 | 내용 | 상태 |
|---|---|---|
| `README.md`, `docs/ml-pipeline.md`, `ml/README.md` | Valid/Test Recall@100, NDCG@5/10, Approach B 수치 | [문서] 현행 |
| **[기록] `f4a469e:backend/ml/README.md`** (2026-02-05, 이후 삭제) | `evaluate_recall.py`의 **출력 원문** (valid/test, Recall@K 전체와 MRR) | git 기록에만 있음 |
| `ml/model/retrieval/bge-m3-finetuned-…/README.md` | 파인튜닝 학습 손실 로그 (500 step마다, 113줄), 하이퍼파라미터 | 로컬 모델 폴더 (gitignore) |
| `ml/model/reranking/lgbm_…_v1.pkl` 안의 `_evals_result` | LightGBM 검증 NDCG@5/10 곡선 (120 iter) | 로컬 (gitignore) |
| MLflow: 실험 `Reranker_LGBM_Current_Features`, `Reranker_XGB_Current_Features`, `Reranker_Feature_Expansion` | 리랭커 실험 지표, feature importance CSV/PNG 아티팩트 | `mlruns/`가 gitignore 대상이고 로컬에도 없음 → **코드에서 확인 불가** |
| `ml/notebooks/*.ipynb` | 임베딩 적재만 함. 평가 결과 없음 | — |
| `docs/plans/dim-recall-eval.md` | 차원 축소 recall 평가 **계획서**. 결과 표가 비어 있음(`…`) | 미실행 |

### 3.2 Retrieval 수치

**[기록] `evaluate_recall.py` 출력** (`f4a469e:backend/ml/README.md`)

| split | 쿼리 | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR |
|---|---|---|---|---|---|---|---|---|
| valid | 62,669 | 0.0462 | 0.1058 | 0.1467 | 0.1950 | 0.2795 | **0.3543** | 0.0795 |
| test | 62,669 | 0.0599 | 0.1307 | 0.1715 | 0.2216 | 0.3015 | **0.3728** | 0.0971 |

- **[측정]** 로컬 `retrieval_candidates_valid.jsonl`로 다시 계산한 Recall@100도 0.3543(22,203 / 62,669)입니다. 기록과 일치합니다.
- 같은 README의 "Current Performance" 표에는 **Recall@100 0.3574, MRR 0.1024**가 따로 적혀 있습니다. 어느 split, 어느 모델의 값인지는 **코드에서 확인 불가**입니다(위 표와 맞지 않음).
- valid와 test 쿼리 수가 62,669로 정확히 같습니다. 원래 리뷰 수가 70,152와 70,154로 비슷하긴 하지만, 20자 필터를 거친 뒤에도 완전히 같다는 점은 **코드에서 확인 불가**입니다. test 후보 파일이 로컬에 없어서 대조하지 못했습니다.

**파인튜닝 데이터** ([기록] 코드와 모델카드로 맞춰 봄)
- `create_training_pairs.py QUERY_TYPE=both`는 리뷰 1건마다 (키워드 쿼리, 리뷰 원문 쿼리) 쌍을 2개 만듭니다. keywords_train 538,013건이면 약 1.0M쌍입니다.
- 여기서 `val_ratio=0.1`을 빼면 약 905K이고, 모델카드의 **905,650쌍**과 맞습니다. 따라서 배포 모델은 `both` 데이터의 90%로 학습한 것으로 보입니다.
- 지금 로컬의 `training_pairs.jsonl`(518,393줄)은 Beam이 만든 키워드 전용 파일로 덮어쓴 것이라 학습 당시 파일이 아닙니다.
- 스크립트 기본값(lr 2e-5, warmup 10%, epoch 1, batch 16)과 모델카드 값(lr 5e-5, warmup 0, epoch 2, batch 32)이 다릅니다. 실제 실행 인자는 **코드에서 확인 불가**입니다. 모델카드 값이 실제 학습 기록에 더 가깝습니다.

### 3.3 Reranking 수치

| 출처 | NDCG@5 | NDCG@10 | 비고 |
|---|---|---|---|
| [문서] README | 0.3655 | 0.4015 | valid, 정답이 Top-100 안에 있는 쿼리만 |
| pkl `_best_score` | 0.36562 | 0.40159 | iter 70 (최고점) |
| pkl `_evals_result` iter 1 | 0.3470 | 0.3836 | 트리 1개 |
| XGBoost (`run_xgb_reranker.py`가 MLflow에 기록) | **코드에서 확인 불가** | **코드에서 확인 불가** | 결과가 MLflow에만 있음 |
| 피처 확장 실험 (`Reranker_Feature_Expansion`: 13개, 9개, 10개, 검색 점수만) | **코드에서 확인 불가** | | 결과가 MLflow에만 있음 |

참고: [기록] 피처 확장 실험의 텍스트 매칭 피처(`overlap_count`, `jaccard`, `coverage`)는 `sample.query_keywords`와 후보의 `title`을 씁니다.
그런데 후보 파일에는 `keywords` 키도 `title`도 없어서 이 피처들은 **항상 0**이었을 것입니다([기록] `1e9f188:backend/ml/item_ranker/features_lgbm.py`).
그 실험의 "텍스트 매칭 제거" ablation은 의미 있는 비교가 아니었을 가능성이 높습니다.

---

## 4. Baseline과 비교

### 4.1 Retrieval

| 모델 | Valid R@100 | Test R@100 | 근거 |
|---|---|---|---|
| `BAAI/bge-m3` (파인튜닝 전) | 0.2015 | **코드에서 확인 불가** | [문서] README |
| **파인튜닝 (Approach A, 채택)** | **0.3543** | **0.3728** | [기록] 출력 원문 |
| Approach B (LLM 생성 쿼리, 약 100K쌍) 전 → 후 | 0.0543 → 0.1092 | 0.1587 | [문서]. 평가셋과 평가 코드는 **코드에서 확인 불가**이고, A와 같은 평가셋인지도 알 수 없음 |
| BM25 / hybrid | — | — | 없음. TODO에 "BM25-only baseline 추가"가 미완료로 남아 있음 |

파인튜닝 전 모델의 Recall@K 전체와 MRR, 그리고 test 수치는 **코드에서 확인 불가**입니다(README의 단일 값만 있음).

### 4.2 Reranking

기준선 코드([기록] `evaluate_retrieval_baseline.py`)는 있지만 출력 기록이 없습니다. 그래서 같은 검증 파일과 같은 쿼리 필터(22,203개)로 직접 계산했습니다.

| 정렬 방식 | NDCG@5 | NDCG@10 | 근거 |
|---|---|---|---|
| 리트리버 점수 순서 (리랭커 없음) | 0.2165 | 0.2535 | [측정] |
| 리뷰 수 순서, **train 기간 리뷰만** (leakage 없음) | 0.1758 | 0.2059 | [측정] |
| 리뷰 수 순서, **CSV 값** (미래 리뷰 포함) | 0.2704 | 0.3076 | [측정] |
| LightGBM 10개 피처 (CSV 값 사용) | **0.3656** | **0.4016** | pkl |
| LightGBM 10개 피처 (leakage 없는 피처로 재학습) | **코드에서 확인 불가** | | 미측정 |

읽는 방법:
- **leakage 없는** 인기도 정렬(0.1758)은 리트리버 점수 순서(0.2165)보다 **낮습니다**.
  그런데 **CSV 값**으로 인기도 정렬을 하면 0.2704로, 리트리버를 **이깁니다**. 차이 0.0946은 미래 리뷰 정보가 만든 것입니다.
- 따라서 리랭커의 +0.149(0.2165 → 0.3656) 중 상당 부분은 leakage에서 왔을 가능성이 높습니다.
  정확한 비율은 깨끗한 피처로 다시 학습해야 알 수 있습니다.
- 계산 방법: 정답 1개, 이진 gain, `1/log2(rank+1)`, 리뷰 수가 같으면 리트리버 점수로 순서를 정함.
  sklearn `ndcg_score`는 동점을 평균 처리하므로 소수점 아래에서 약간 다를 수 있습니다.

---

## 5. 정리: 평가를 믿을 수 있게 하려면

1. **평가 코드를 레포로 되살리기.** `evaluate_recall.py`, `retrieve_candidates.py`, `evaluate_retrieval_baseline.py`를 `ml/evaluation/`에 다시 넣으면 됩니다.
   지금 `main`만으로는 README의 어떤 수치도 다시 만들 수 없습니다.
2. **리랭커 피처를 시간 기준으로 다시 계산하기.** 쿼리 시점 이전(최소한 train 기간) 리뷰로만 통계를 내고 다시 학습해서 NDCG를 다시 보고해야 합니다.
   CSV를 만든 코드도 레포에 넣어야 합니다.
3. **valid는 모델 선택에만 쓰고, test로 최종 보고하기.** `retrieval_candidates_test.jsonl`로 리랭커 NDCG를 따로 보고하면 됩니다.
4. **지표를 함께 보고하기.**
   - 전체 쿼리를 분모로 한 end-to-end 지표(예: 최종 Top-5 HitRate@5)
   - 리트리버 순서와 인기도 순서 기준선
   - 신규 상품(train 기간 리뷰 0건) 구간의 지표
5. **서빙 쿼리 분포로 평가하기.** 한국어 상황 문장 쿼리 셋이 소량이라도 있으면, 오프라인 지표와 서비스 품질 사이의 차이를 가늠할 수 있습니다.
