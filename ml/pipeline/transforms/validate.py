"""Validation DoFns with Beam Metrics for data quality monitoring."""

import apache_beam as beam
from apache_beam.metrics import Metrics

from ml.pipeline.schemas import KeywordRecord, Metadata, Review


class ValidateReviewFn(beam.DoFn):
    """Filter invalid reviews and emit quality metrics."""

    def __init__(self):
        self.valid = Metrics.counter("validate", "valid_reviews")
        self.invalid = Metrics.counter("validate", "invalid_reviews")
        self.short_text = Metrics.counter("validate", "short_text_reviews")
        self.missing_asin = Metrics.counter("validate", "missing_parent_asin")
        self.text_len_dist = Metrics.distribution("validate", "review_text_length")
        self.rating_dist = Metrics.distribution("validate", "review_rating")

    def process(self, review: Review):
        if not review.parent_asin:
            self.missing_asin.inc()
            self.invalid.inc()
            return

        if not review.text or len(review.text) < 10:
            self.short_text.inc()
            self.invalid.inc()
            return

        self.text_len_dist.update(len(review.text))
        self.rating_dist.update(int(review.rating * 10))
        self.valid.inc()
        yield review


class ValidateMetadataFn(beam.DoFn):
    """Filter invalid metadata and emit quality metrics."""

    def __init__(self):
        self.valid = Metrics.counter("validate", "valid_metadata")
        self.invalid = Metrics.counter("validate", "invalid_metadata")
        self.missing_price = Metrics.counter("validate", "metadata_missing_price")
        self.rating_num_dist = Metrics.distribution("validate", "metadata_rating_number")

    def process(self, meta: Metadata):
        if not meta.parent_asin or not meta.title:
            self.invalid.inc()
            return

        if meta.price is None:
            self.missing_price.inc()

        self.rating_num_dist.update(meta.rating_number)
        self.valid.inc()
        yield meta


class ValidateKeywordFn(beam.DoFn):
    """Filter keyword records with empty keyword lists."""

    def __init__(self):
        self.valid = Metrics.counter("validate", "valid_keywords")
        self.empty = Metrics.counter("validate", "empty_keywords")
        self.kw_count_dist = Metrics.distribution("validate", "keyword_count")

    def process(self, kw: KeywordRecord):
        if not kw.keywords:
            self.empty.inc()
            return

        self.kw_count_dist.update(len(kw.keywords))
        self.valid.inc()
        yield kw
