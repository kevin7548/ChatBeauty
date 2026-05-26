import pandas as pd
from item_ranker.dataset.base import RerankSample

class TreeFeatureBuilder:
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

    def __init__(self, item_feat_path: str):
        item_feat_df = pd.read_csv(item_feat_path).fillna(0)
        self.item_feat_map = item_feat_df.set_index("parent_asin").to_dict(orient="index")

    def build(self, sample: RerankSample) -> pd.DataFrame:
        rows = []

        for idx, c in enumerate(sample.candidates):
            parent_asin = c.item_id
            csv_feat = self.item_feat_map.get(parent_asin, {})

            row = {
                "retrieval_score": float(c.retrieval_score),
                "review_cnt": float(csv_feat.get("review_cnt", 0)),
                "vp_review_cnt": float(csv_feat.get("vp_review_cnt", 0)),
                "vp_ratio": float(csv_feat.get("vp_ratio", 0.0)),
                "recent_review_cnt": float(csv_feat.get("recent_review_cnt", 0)),
                "avg_rating": float(csv_feat.get("avg_rating", 0.0)),
                "rating_std": float(csv_feat.get("rating_std", 0.0)),
                "avg_review_len": float(csv_feat.get("avg_review_len", 0.0)),
                "log_median_price": float(csv_feat.get("log_median_price", 0.0)),
                "price_cnt": float(csv_feat.get("price_cnt", 0)),
            }
            rows.append(row)

        return pd.DataFrame(rows, columns=self.FEATURE_NAMES)