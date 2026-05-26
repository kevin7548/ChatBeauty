import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.dataset.base import iter_samples


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
    y = np.array(y_list, dtype=np.float32)

    return X, y, group


def train_reranker_xgb(
    train_path: str,
    valid_path: str,
    model_path: str,
    feature_builder: TreeFeatureBuilder,
    limit: Optional[int] = None,
) -> Tuple[xgb.Booster, Dict, Dict]:

    params = {
        "objective": "rank:ndcg",
        "eval_metric": ["ndcg@5", "ndcg@10"],
        "eta": 0.05,
        "max_depth": 5,
        "tree_method": "hist",
        "seed": 42,
    }

    X_train, y_train, group_train = build_dataset(train_path, feature_builder, limit)
    X_valid, y_valid, group_valid = build_dataset(valid_path, feature_builder)

    dtrain = xgb.DMatrix(X_train.values, label=y_train)
    dtrain.set_group(group_train)

    dvalid = xgb.DMatrix(X_valid.values, label=y_valid)
    dvalid.set_group(group_valid)

    print("[Train] Start XGBoost training...")

    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=50
    )

    model.save_model(model_path)
    print(f"[OK] Model saved to {model_path}")
    print(f"[Best Iteration] {model.best_iteration}")

    return model, params, evals_result
