"""
Training pair creation via CoGroupByKey.

Replicates the logic from create_training_pairs.py:37-51 in Beam.
"""

import apache_beam as beam
from apache_beam.metrics import Metrics

from ml.pipeline.schemas import TrainingPair


class CreateTrainingPairs(beam.PTransform):
    """
    Create (query, positive) training pairs for BGE-M3 fine-tuning.

    Joins per-review keywords with per-product items via CoGroupByKey,
    then emits one TrainingPair per review.

    Input: (keywords PCollection[KeywordRecord], items PCollection[(str, ItemForEmbedding)])
    Output: PCollection[TrainingPair]
    """

    def __init__(self, max_keywords: int = 20):
        super().__init__()
        self.max_keywords = max_keywords

    def expand(self, inputs):
        keywords, items = inputs

        kw_keyed = keywords | "KeyKWByProduct" >> beam.Map(
            lambda kw: (kw.parent_asin, kw)
        )

        joined = (
            {"keywords": kw_keyed, "items": items}
            | "CoGroupForPairs" >> beam.CoGroupByKey()
        )

        return joined | "EmitPairs" >> beam.ParDo(
            EmitTrainingPairsFn(self.max_keywords)
        )


class EmitTrainingPairsFn(beam.DoFn):
    """
    Emit TrainingPair for each review's keywords matched with its product.

    Query text construction matches create_training_pairs.py:
    - Take top N keywords (already sorted by frequency in the source)
    - Join with comma separator
    - Skip if no keywords or no item
    """

    def __init__(self, max_keywords: int):
        self.max_keywords = max_keywords
        self.pairs_created = Metrics.counter("pairs", "pairs_created")
        self.skipped_no_item = Metrics.counter("pairs", "skipped_no_item")
        self.skipped_no_kw = Metrics.counter("pairs", "skipped_no_keywords")
        self.kw_per_query = Metrics.distribution("pairs", "keywords_per_query")

    def process(self, element):
        parent_asin, groups = element

        items_list = groups.get("items", [])
        kw_list = groups.get("keywords", [])

        if not items_list:
            if kw_list:
                self.skipped_no_item.inc(len(kw_list))
            return

        item = items_list[0]
        positive_text = item.embedding_text
        if not positive_text:
            self.skipped_no_item.inc(len(kw_list))
            return

        for kw_record in kw_list:
            keywords = kw_record.keywords
            if not keywords:
                self.skipped_no_kw.inc()
                continue

            selected = keywords[: self.max_keywords]
            query_text = ", ".join(str(kw) for kw in selected if kw is not None)
            if not query_text:
                self.skipped_no_kw.inc()
                continue

            self.kw_per_query.update(len(selected))
            self.pairs_created.inc()
            yield TrainingPair(
                query=query_text,
                positive=positive_text,
                parent_asin=parent_asin,
            )
