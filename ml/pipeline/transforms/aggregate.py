"""
Aggregation transforms using CombineFn and CombinePerKey.

Demonstrates Beam's distributed aggregation patterns:
- MergeKeywordsFn: CombineFn with Counter accumulator
- ReviewStatsCombineFn: CombineFn computing 8 statistical features
"""

import collections
import math

import apache_beam as beam
from apache_beam.metrics import Metrics

from ml.pipeline.schemas import AggregatedKeywords, RerankingFeatures


class MergeKeywordsFn(beam.CombineFn):
    """
    Merge keywords across all reviews for a single product.

    Accumulator is a Counter that tracks keyword frequency.
    Output is the top-50 most frequent keywords, deduplicated.
    """

    def create_accumulator(self):
        return collections.Counter()

    def add_input(self, accumulator, keywords):
        accumulator.update(keywords)
        return accumulator

    def merge_accumulators(self, accumulators):
        merged = collections.Counter()
        for acc in accumulators:
            merged.update(acc)
        return merged

    def extract_output(self, accumulator):
        return [kw for kw, _ in accumulator.most_common(50)]


class AggregateKeywordsPerProduct(beam.PTransform):
    """
    Aggregate keywords per product using CombinePerKey.

    Input: PCollection[KeywordRecord]
    Output: PCollection[(str, AggregatedKeywords)]  keyed by parent_asin
    """

    def expand(self, keywords):
        return (
            keywords
            | "KeyByProduct" >> beam.Map(lambda kw: (kw.parent_asin, kw.keywords))
            | "CombineKeywords" >> beam.CombinePerKey(MergeKeywordsFn())
            | "ToAggregated"
            >> beam.MapTuple(
                lambda asin, kws: (
                    asin,
                    AggregatedKeywords(parent_asin=asin, review_keywords=kws),
                )
            )
        )


class ReviewStatsCombineFn(beam.CombineFn):
    """
    Compute 8 reranking features from reviews for a single product.

    Uses online algorithms for mean and variance to avoid storing all values.
    The "recent" cutoff timestamp is passed via constructor.

    Features computed:
        review_cnt, vp_review_cnt, vp_ratio, recent_review_cnt,
        avg_rating, rating_std, avg_review_len
    """

    def __init__(self, recent_cutoff: int = 0):
        self.recent_cutoff = recent_cutoff

    def create_accumulator(self):
        return {
            "count": 0,
            "vp_count": 0,
            "recent_count": 0,
            "sum_rating": 0.0,
            "sum_rating_sq": 0.0,
            "sum_text_len": 0,
        }

    def add_input(self, acc, review):
        acc["count"] += 1
        if review.verified_purchase:
            acc["vp_count"] += 1
        if self.recent_cutoff > 0 and review.timestamp >= self.recent_cutoff:
            acc["recent_count"] += 1
        acc["sum_rating"] += review.rating
        acc["sum_rating_sq"] += review.rating ** 2
        acc["sum_text_len"] += len(review.text) if review.text else 0
        return acc

    def merge_accumulators(self, accumulators):
        merged = self.create_accumulator()
        for acc in accumulators:
            merged["count"] += acc["count"]
            merged["vp_count"] += acc["vp_count"]
            merged["recent_count"] += acc["recent_count"]
            merged["sum_rating"] += acc["sum_rating"]
            merged["sum_rating_sq"] += acc["sum_rating_sq"]
            merged["sum_text_len"] += acc["sum_text_len"]
        return merged

    def extract_output(self, acc):
        n = acc["count"]
        if n == 0:
            return {
                "review_cnt": 0,
                "vp_review_cnt": 0,
                "vp_ratio": 0.0,
                "recent_review_cnt": 0,
                "avg_rating": 0.0,
                "rating_std": 0.0,
                "avg_review_len": 0.0,
            }

        mean_rating = acc["sum_rating"] / n
        variance = max(0.0, acc["sum_rating_sq"] / n - mean_rating ** 2)

        return {
            "review_cnt": n,
            "vp_review_cnt": acc["vp_count"],
            "vp_ratio": acc["vp_count"] / n if n > 0 else 0.0,
            "recent_review_cnt": acc["recent_count"],
            "avg_rating": mean_rating,
            "rating_std": math.sqrt(variance),
            "avg_review_len": acc["sum_text_len"] / n,
        }


class PercentileCombineFn(beam.CombineFn):
    """Compute an approximate percentile from a PCollection of numbers."""

    def __init__(self, percentile: int):
        self.percentile = percentile

    def create_accumulator(self):
        return []

    def add_input(self, acc, value):
        acc.append(value)
        return acc

    def merge_accumulators(self, accumulators):
        merged = []
        for acc in accumulators:
            merged.extend(acc)
        return merged

    def extract_output(self, acc):
        if not acc:
            return 0
        acc.sort()
        idx = int(len(acc) * self.percentile / 100)
        return acc[min(idx, len(acc) - 1)]


class ComputeRerankingFeatures(beam.PTransform):
    """
    Compute all 10 reranking features per product.

    Joins review-based stats (8 features from ReviewStatsCombineFn)
    with metadata-based stats (log_median_price, price_cnt).

    Input: (reviews PCollection, metadata PCollection)
    Output: PCollection[(str, RerankingFeatures)] keyed by parent_asin
    """

    def expand(self, inputs):
        reviews, metadata = inputs

        # Compute 80th-percentile timestamp as "recent" cutoff (side input)
        timestamps = reviews | "ExtractTimestamps" >> beam.Map(lambda r: r.timestamp)
        cutoff = timestamps | "ComputeCutoff" >> beam.CombineGlobally(
            PercentileCombineFn(80)
        )

        # Tag reviews with recent flag and compute stats per product
        review_stats = (
            reviews
            | "KeyReviewsByProduct" >> beam.Map(lambda r: (r.parent_asin, r))
            | "ComputeStats"
            >> beam.CombinePerKey(ReviewStatsCombineFn(recent_cutoff=0))
        )

        # Actually, CombinePerKey doesn't support side inputs directly.
        # Instead, compute stats without recent, then fix recent_count separately.
        # For simplicity, use a DoFn with side input to compute recent_count.
        recent_counts = (
            reviews
            | "KeyForRecent" >> beam.Map(lambda r: (r.parent_asin, r.timestamp))
            | "GroupForRecent" >> beam.GroupByKey()
            | "CountRecent"
            >> beam.ParDo(CountRecentFn(), cutoff=beam.pvalue.AsSingleton(cutoff))
        )

        # Extract price info from metadata
        price_info = metadata | "ExtractPrice" >> beam.Map(
            lambda m: (
                m.parent_asin,
                {
                    "log_median_price": math.log(m.price) if m.price and m.price > 0 else 0.0,
                    "price_cnt": 1 if m.price is not None else 0,
                },
            )
        )

        # Join review_stats + recent_counts + price_info
        joined = (
            {
                "review_stats": review_stats,
                "recent_counts": recent_counts,
                "price_info": price_info,
            }
            | "JoinFeatures" >> beam.CoGroupByKey()
        )

        return joined | "BuildFeatures" >> beam.ParDo(BuildRerankingFeaturesFn())


class CountRecentFn(beam.DoFn):
    """Count recent timestamps using a side input cutoff."""

    def process(self, element, cutoff):
        parent_asin, timestamps = element
        recent = sum(1 for ts in timestamps if ts >= cutoff)
        yield (parent_asin, recent)


class BuildRerankingFeaturesFn(beam.DoFn):
    """Merge review stats, recent counts, and price info into RerankingFeatures."""

    def __init__(self):
        self.counter = Metrics.counter("features", "features_computed")

    def process(self, element):
        parent_asin, groups = element

        stats_list = groups.get("review_stats", [])
        recent_list = groups.get("recent_counts", [])
        price_list = groups.get("price_info", [])

        stats = stats_list[0] if stats_list else {
            "review_cnt": 0, "vp_review_cnt": 0, "vp_ratio": 0.0,
            "recent_review_cnt": 0, "avg_rating": 0.0, "rating_std": 0.0,
            "avg_review_len": 0.0,
        }
        recent_count = recent_list[0] if recent_list else 0
        price = price_list[0] if price_list else {"log_median_price": 0.0, "price_cnt": 0}

        self.counter.inc()
        yield (
            parent_asin,
            RerankingFeatures(
                parent_asin=parent_asin,
                review_cnt=stats["review_cnt"],
                vp_review_cnt=stats["vp_review_cnt"],
                vp_ratio=stats["vp_ratio"],
                recent_review_cnt=recent_count,
                avg_rating=stats["avg_rating"],
                rating_std=stats["rating_std"],
                avg_review_len=stats["avg_review_len"],
                log_median_price=price["log_median_price"],
                price_cnt=price["price_cnt"],
            ),
        )
