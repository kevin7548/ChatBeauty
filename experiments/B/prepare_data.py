"""
Cache valid/test queries and train-period review counts for experiment B.

Uses ml/evaluation/retrieve_candidates.load_and_split_reviews (80/10/10 by timestamp,
reviews >= 20 chars) so queries match B0 (retrieval_candidates_valid.jsonl) exactly.

Outputs (experiments/B/cache/):
    {SPLIT}_queries.jsonl     {"parent_asin", "text"} in split order
    train_review_counts.json  parent_asin -> # reviews in the train 80%

Usage:
    python experiments/B/prepare_data.py
    SPLIT=test python experiments/B/prepare_data.py
"""

import json
import sys
from collections import Counter

from common import CACHE, ML, QUERIES, REVIEWS_PATH, SPLIT, TRAIN_COUNTS

sys.path.insert(0, str(ML / 'evaluation'))
from retrieve_candidates import load_and_split_reviews  # noqa: E402


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    valid = load_and_split_reviews(str(REVIEWS_PATH), SPLIT)
    with open(QUERIES, 'w', encoding='utf-8') as f:
        for r in valid:
            f.write(json.dumps({'parent_asin': r['parent_asin'], 'text': r['text']}, ensure_ascii=False) + '\n')
    print(f"Wrote {len(valid)} queries -> {QUERIES}")

    # Sanity check: same order/ground truth as the B0 candidates file
    b0_path = ML / f'data/evaluation/retrieval_candidates_{SPLIT}.jsonl'
    if b0_path.exists():
        with open(b0_path, 'r', encoding='utf-8') as f:
            b0_gt = [json.loads(line)['parent_asin'] for line in f]
        same = sum(a == r['parent_asin'] for a, r in zip(b0_gt, valid))
        print(f"B0 order check: {same}/{len(valid)} match (B0 has {len(b0_gt)})")

    train = load_and_split_reviews(str(REVIEWS_PATH), 'train')
    counts = Counter(r['parent_asin'] for r in train)
    with open(TRAIN_COUNTS, 'w', encoding='utf-8') as f:
        json.dump(counts, f)
    print(f"Wrote train counts for {len(counts)} products -> {TRAIN_COUNTS}")


if __name__ == '__main__':
    main()
