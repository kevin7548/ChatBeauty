from typing import List
from item_ranker.dataset import RerankSample

class Reranker:
    def score(self, sample: RerankSample) -> List[float]:
        raise NotImplementedError