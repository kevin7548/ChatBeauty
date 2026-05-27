"""
CoGroupByKey joins for metadata + keywords → ItemForEmbedding.

Demonstrates Beam's multi-way join pattern and composite PTransform.
"""

import json

import apache_beam as beam
from apache_beam.metrics import Metrics

from ml.pipeline.schemas import ItemForEmbedding, ProductRecord


class JoinMetadataAndKeywords(beam.PTransform):
    """
    Join aggregated keywords with product metadata via CoGroupByKey.

    Input: (agg_keywords keyed by parent_asin, metadata keyed by parent_asin)
    Output: PCollection[(str, ItemForEmbedding)] keyed by parent_asin
    """

    def expand(self, inputs):
        agg_keywords, metadata = inputs

        metadata_keyed = metadata | "KeyMetadata" >> beam.Map(
            lambda m: (m.parent_asin, m)
        )

        joined = (
            {"keywords": agg_keywords, "metadata": metadata_keyed}
            | "CoGroupMetaKW" >> beam.CoGroupByKey()
        )

        return joined | "BuildItems" >> beam.ParDo(BuildEmbeddingTextFn())


class BuildEmbeddingTextFn(beam.DoFn):
    """
    Build ItemForEmbedding from joined metadata + aggregated keywords.

    Constructs embedding_text in the format:
      [Title] {title} [Review Keywords] {kw1, kw2} [Description Summary] {desc} [Features] {feat}

    This format must match the existing items_for_embedding.jsonl exactly.
    """

    def __init__(self):
        self.with_kw = Metrics.counter("join", "items_with_keywords")
        self.without_kw = Metrics.counter("join", "items_without_keywords")
        self.orphan = Metrics.counter("join", "orphan_keywords_no_metadata")
        self.text_len = Metrics.distribution("join", "embedding_text_length")

    def process(self, element):
        parent_asin, groups = element

        metadata_list = groups.get("metadata", [])
        keywords_list = groups.get("keywords", [])

        if not metadata_list:
            if keywords_list:
                self.orphan.inc()
            return

        meta = metadata_list[0]
        review_keywords = keywords_list[0].review_keywords if keywords_list else []

        if review_keywords:
            self.with_kw.inc()
        else:
            self.without_kw.inc()

        # Build embedding_text matching existing format
        parts = [f"[Title] {meta.title}"]

        if review_keywords:
            parts.append(f"[Review Keywords] {', '.join(str(kw) for kw in review_keywords)}")

        desc = " ".join(meta.description) if meta.description else ""
        if desc:
            parts.append(f"[Description Summary] {desc}")

        if meta.features:
            parts.append(f"[Features] {', '.join(meta.features)}")

        embedding_text = " ".join(parts)
        self.text_len.update(len(embedding_text))

        # Aggregate top reviews (done in metadata update step in original pipeline)
        # Here we pass empty since reviews are aggregated separately
        top_reviews = ""

        item = ItemForEmbedding(
            asin=parent_asin,
            title=meta.title,
            review_keywords=review_keywords,
            description_summary=meta.description,
            features=meta.features,
            embedding_text=embedding_text,
            price=meta.price,
            average_rating=meta.average_rating,
            rating_number=meta.rating_number,
            store=meta.store,
            categories=meta.categories,
            main_category=meta.main_category,
            image=meta.image,
            details=meta.details,
            top_reviews=top_reviews,
        )
        yield (parent_asin, item)


class MergeItemAndFeatures(beam.PTransform):
    """
    Merge ItemForEmbedding with RerankingFeatures into ProductRecord.

    Uses CoGroupByKey on parent_asin to join the two PCollections.
    Output: PCollection[ProductRecord] ready for PostgreSQL write.
    """

    def expand(self, inputs):
        items, features = inputs

        joined = (
            {"items": items, "features": features}
            | "CoGroupItemsFeatures" >> beam.CoGroupByKey()
        )

        return joined | "MergeToProduct" >> beam.ParDo(MergeProductFn())


class MergeProductFn(beam.DoFn):
    """Merge item and features into a single ProductRecord."""

    def __init__(self):
        self.counter = Metrics.counter("merge", "products_merged")

    def process(self, element):
        parent_asin, groups = element

        items_list = groups.get("items", [])
        features_list = groups.get("features", [])

        if not items_list:
            return

        item = items_list[0]
        feat = features_list[0] if features_list else None

        self.counter.inc()
        yield ProductRecord(
            parent_asin=parent_asin,
            title=item.title,
            embedding_text=item.embedding_text,
            description=" ".join(item.description_summary) if item.description_summary else "",
            features=", ".join(item.features) if item.features else "",
            top_reviews=item.top_reviews,
            details=item.details,
            image=item.image,
            store=item.store,
            price=item.price,
            average_rating=item.average_rating,
            rating_number=item.rating_number,
            review_cnt=feat.review_cnt if feat else 0,
            vp_review_cnt=feat.vp_review_cnt if feat else 0,
            vp_ratio=feat.vp_ratio if feat else 0.0,
            recent_review_cnt=feat.recent_review_cnt if feat else 0,
            avg_rating=feat.avg_rating if feat else 0.0,
            rating_std=feat.rating_std if feat else 0.0,
            avg_review_len=feat.avg_review_len if feat else 0.0,
            log_median_price=feat.log_median_price if feat else 0.0,
            price_cnt=feat.price_cnt if feat else 0,
        )
