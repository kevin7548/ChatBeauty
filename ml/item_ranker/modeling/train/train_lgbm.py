import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from typing import Optional, Dict, Tuple
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.dataset.base import iter_samples

def train_reranker(
    train_path: str,
    valid_path: str,
    model_path: str,
    feature_builder: TreeFeatureBuilder,
    limit: Optional[int] = None,
):

    X_train, y_train, group_train = build_dataset(train_path, feature_builder, limit)
    X_valid, y_valid, group_valid = build_dataset(valid_path, feature_builder)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        importance_type="gain",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )

    print("[Train] Start LightGBM training...")

    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[5, 10],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] Model saved to {model_path}")
    return model

def build_dataset(data_path, feature_builder, limit=None):
    X_list, y_list, group = [], [], []

    for sample in iter_samples(data_path, limit=limit):
        df = feature_builder.build(sample)
        labels = sample.labels

        if not labels or sum(labels) == 0:
            continue

        X_list.append(df)
        y_list.extend(labels)
        group.append(len(df))

    X = pd.concat(X_list, ignore_index=True)
    y = pd.Series(y_list).astype("float32").values
    return X, y, group