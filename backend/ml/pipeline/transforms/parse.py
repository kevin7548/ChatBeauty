"""Parsing DoFns: raw JSON dict → typed NamedTuples with Beam metrics."""

import json
import logging

import apache_beam as beam
from apache_beam.metrics import Metrics

from ml.pipeline.schemas import KeywordRecord, Metadata, Review

logger = logging.getLogger(__name__)


class ParseReviewFn(beam.DoFn):
    """Parse a raw JSON dict into a Review NamedTuple."""

    def __init__(self):
        self.success = Metrics.counter("parse", "review_parse_success")
        self.error = Metrics.counter("parse", "review_parse_error")

    def process(self, element):
        try:
            if isinstance(element, str):
                element = json.loads(element)
            self.success.inc()
            yield Review(
                asin=element.get("asin", ""),
                parent_asin=element.get("parent_asin") or element.get("asin", ""),
                title=element.get("title", ""),
                text=element.get("text", ""),
                rating=float(element.get("rating", 0.0)),
                timestamp=int(element.get("timestamp", 0)),
                helpful_vote=int(element.get("helpful_vote", 0)),
                verified_purchase=bool(element.get("verified_purchase", False)),
            )
        except Exception as e:
            self.error.inc()
            logger.debug(f"Failed to parse review: {e}")


class ParseMetadataFn(beam.DoFn):
    """Parse a raw JSON dict into a Metadata NamedTuple."""

    def __init__(self):
        self.success = Metrics.counter("parse", "metadata_parse_success")
        self.error = Metrics.counter("parse", "metadata_parse_error")

    def process(self, element):
        try:
            if isinstance(element, str):
                element = json.loads(element)

            # Extract main image URL
            images = element.get("images", [])
            image_url = ""
            for img in images:
                if isinstance(img, dict) and img.get("variant") == "MAIN":
                    image_url = img.get("large", "") or img.get("thumb", "")
                    break

            # Serialize details dict to JSON string
            details = element.get("details", {})
            details_str = json.dumps(details, ensure_ascii=False) if details else ""

            self.success.inc()
            yield Metadata(
                parent_asin=element.get("parent_asin") or element.get("asin", ""),
                title=element.get("title", ""),
                average_rating=float(element.get("average_rating", 0.0)),
                rating_number=int(element.get("rating_number", 0)),
                features=element.get("features", []) or [],
                description=element.get("description", []) or [],
                price=float(element["price"]) if element.get("price") is not None else None,
                store=element.get("store", "") or "",
                categories=element.get("categories", []) or [],
                main_category=element.get("main_category", ""),
                image=image_url,
                details=details_str,
            )
        except Exception as e:
            self.error.inc()
            logger.debug(f"Failed to parse metadata: {e}")


class ParseKeywordFn(beam.DoFn):
    """Parse a raw JSON dict into a KeywordRecord NamedTuple."""

    def __init__(self):
        self.success = Metrics.counter("parse", "keyword_parse_success")
        self.error = Metrics.counter("parse", "keyword_parse_error")
        self.kw_dist = Metrics.distribution("parse", "keywords_per_review")

    def process(self, element):
        try:
            if isinstance(element, str):
                element = json.loads(element)
            keywords = element.get("keywords", []) or []
            self.kw_dist.update(len(keywords))
            self.success.inc()
            yield KeywordRecord(
                asin=element.get("asin", ""),
                parent_asin=element.get("parent_asin") or element.get("asin", ""),
                title=element.get("title", ""),
                review_text=element.get("review_text", ""),
                rating=float(element.get("rating", 0.0)),
                keywords=keywords,
            )
        except Exception as e:
            self.error.inc()
            logger.debug(f"Failed to parse keyword record: {e}")
