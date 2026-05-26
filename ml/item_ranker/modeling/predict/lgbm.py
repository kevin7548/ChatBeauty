import pickle
import lightgbm as lgb
from typing import List

from item_ranker.dataset.base import RerankSample
from item_ranker.features.tree import TreeFeatureBuilder
from .base_tree import BaseTreeReranker


class LGBMTreeReranker(BaseTreeReranker):
    def __init__(self, model_path: str, feature_builder: TreeFeatureBuilder):
        super().__init__(feature_builder)

        with open(model_path, "rb") as f:
            self.model: lgb.LGBMRanker = pickle.load(f)

    def score(self, sample: RerankSample) -> List[float]:
        X = self._build_features(sample)
        return self.model.predict(X).tolist()
