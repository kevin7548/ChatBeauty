"""
Type-safe data containers for Beam PCollections.

NamedTuples are used because they are immutable, picklable by default
(critical for Beam serialization across workers), and work with Beam's
type hint system.
"""

from typing import List, NamedTuple, Optional


class Review(NamedTuple):
    asin: str
    parent_asin: str
    title: str
    text: str
    rating: float
    timestamp: int
    helpful_vote: int
    verified_purchase: bool


class Metadata(NamedTuple):
    parent_asin: str
    title: str
    average_rating: float
    rating_number: int
    features: List[str]
    description: List[str]
    price: Optional[float]
    store: str
    categories: List[str]
    main_category: str
    image: str
    details: str


class KeywordRecord(NamedTuple):
    asin: str
    parent_asin: str
    title: str
    review_text: str
    rating: float
    keywords: List[str]


class AggregatedKeywords(NamedTuple):
    parent_asin: str
    review_keywords: List[str]


class ItemForEmbedding(NamedTuple):
    asin: str
    title: str
    review_keywords: List[str]
    description_summary: List[str]
    features: List[str]
    embedding_text: str
    price: Optional[float]
    average_rating: float
    store: str
    categories: List[str]
    main_category: str
    rating_number: int
    image: str
    details: str
    top_reviews: str


class TrainingPair(NamedTuple):
    query: str
    positive: str
    parent_asin: str


class RerankingFeatures(NamedTuple):
    parent_asin: str
    review_cnt: int
    vp_review_cnt: int
    vp_ratio: float
    recent_review_cnt: int
    avg_rating: float
    rating_std: float
    avg_review_len: float
    log_median_price: float
    price_cnt: int


class ProductRecord(NamedTuple):
    """Merged item + features for PostgreSQL write."""
    parent_asin: str
    title: str
    embedding_text: str
    description: str
    features: str
    top_reviews: str
    details: str
    image: str
    store: str
    price: Optional[float]
    average_rating: float
    rating_number: int
    # reranking features
    review_cnt: int
    vp_review_cnt: int
    vp_ratio: float
    recent_review_cnt: int
    avg_rating: float
    rating_std: float
    avg_review_len: float
    log_median_price: float
    price_cnt: int
