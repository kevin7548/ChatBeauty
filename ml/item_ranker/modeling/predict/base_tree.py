from item_ranker.features.tree import TreeFeatureBuilder
from item_ranker.dataset.base import RerankSample
import pandas as pd

class BaseTreeReranker:
    def __init__(self, feature_builder: TreeFeatureBuilder):
        self.feature_builder = feature_builder

    def _build_features(self, sample: RerankSample) -> pd.DataFrame:
        return self.feature_builder.build(sample)