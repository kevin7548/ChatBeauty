import xgboost as xgb
import numpy as np
from typing import List
from item_ranker.dataset import RerankSample
from .base import Reranker
from .base_tree import BaseTreeReranker

class XGBTreeReranker(BaseTreeReranker, Reranker):
    def __init__(self, model: xgb.Booster, feature_builder):
        super().__init__(feature_builder)
        self.model = model

    def score(self, sample: RerankSample) -> List[float]:
        df = self._build_features(sample)
        dtest = xgb.DMatrix(df.values, feature_names=df.columns.tolist())
        return self.model.predict(dtest).tolist()
