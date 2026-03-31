"""
LightGBM reranking service backed by PostgreSQL for feature lookup.

Replaces the CSV-based TreeFeatureBuilder with direct PostgreSQL queries.
"""

import os
import pickle
import logging

import pandas as pd
import lightgbm as lgb
from pathlib import Path

from app.services.retrieval_resources import get_db_connection, release_db_connection

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

RERANK_MODEL_PATH = os.environ.get(
    "RERANK_MODEL_PATH",
    str(BASE_DIR / "ml" / "model-gcs" / "reranking" / "lgbm_reranker_current_features_v1.pkl"),
)

FEATURE_NAMES = [
    "retrieval_score",
    "review_cnt",
    "vp_review_cnt",
    "vp_ratio",
    "recent_review_cnt",
    "avg_rating",
    "rating_std",
    "avg_review_len",
    "log_median_price",
    "price_cnt",
]

FEATURE_SQL = """
    SELECT parent_asin, review_cnt, vp_review_cnt, vp_ratio,
           recent_review_cnt, avg_rating, rating_std, avg_review_len,
           log_median_price, price_cnt
    FROM products
    WHERE parent_asin = ANY(%s)
"""

# Load LightGBM model once at startup
with open(RERANK_MODEL_PATH, "rb") as f:
    _model: lgb.LGBMRanker = pickle.load(f)


def _fetch_features(parent_asins: list[str]) -> dict[str, dict]:
    """Fetch reranking features from PostgreSQL for a batch of ASINs."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(FEATURE_SQL, (parent_asins,))
        rows = cur.fetchall()
        cur.close()
    finally:
        release_db_connection(conn)

    feat_map = {}
    for row in rows:
        feat_map[row[0]] = {
            "review_cnt": float(row[1] or 0),
            "vp_review_cnt": float(row[2] or 0),
            "vp_ratio": float(row[3] or 0.0),
            "recent_review_cnt": float(row[4] or 0),
            "avg_rating": float(row[5] or 0.0),
            "rating_std": float(row[6] or 0.0),
            "avg_review_len": float(row[7] or 0.0),
            "log_median_price": float(row[8] or 0.0),
            "price_cnt": float(row[9] or 0),
        }
    return feat_map


def rerank_items(query: str, candidates: list[dict], top_k: int):
    if not candidates:
        return []

    parent_asins = [c["item_id"] for c in candidates]
    feat_map = _fetch_features(parent_asins)

    rows = []
    for c in candidates:
        db_feat = feat_map.get(c["item_id"], {})
        rows.append({
            "retrieval_score": float(c.get("score", 0.0)),
            "review_cnt": db_feat.get("review_cnt", 0.0),
            "vp_review_cnt": db_feat.get("vp_review_cnt", 0.0),
            "vp_ratio": db_feat.get("vp_ratio", 0.0),
            "recent_review_cnt": db_feat.get("recent_review_cnt", 0.0),
            "avg_rating": db_feat.get("avg_rating", 0.0),
            "rating_std": db_feat.get("rating_std", 0.0),
            "avg_review_len": db_feat.get("avg_review_len", 0.0),
            "log_median_price": db_feat.get("log_median_price", 0.0),
            "price_cnt": db_feat.get("price_cnt", 0.0),
        })

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    scores = _model.predict(X).tolist()

    reranked = []
    for c, s in zip(candidates, scores):
        item = c.copy()
        item["score"] = float(s)
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]
