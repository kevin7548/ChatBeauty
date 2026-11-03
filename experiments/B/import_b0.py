"""
Reference source S1ann: convert the B0 serving-HNSW candidates
(ml/data/evaluation/retrieval_candidates_valid.jsonl, read only) to corpus row indices.

Candidates not in items_for_embedding.jsonl get row -1.

Usage:
    python experiments/B/import_b0.py
"""

import json

import numpy as np

from common import CACHE, ML, TOP_K, load_corpus


def main():
    asin_to_row = {it['asin']: i for i, it in enumerate(load_corpus())}
    rows, gt_rank, missing = [], [], 0
    with open(ML / 'data/evaluation/retrieval_candidates_valid.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            asins = [c['item_asin'] for c in item['candidates']][:TOP_K]
            r = [asin_to_row.get(a, -1) for a in asins] + [-1] * (TOP_K - len(asins))
            missing += sum(x == -1 for x in r)
            rows.append(r)
            gt_rank.append(asins.index(item['parent_asin']) + 1 if item['parent_asin'] in asins else 0)
    np.save(CACHE / 'topk_valid_S1ann.npy', np.array(rows, dtype=np.int32))
    np.save(CACHE / 'gt_rank_valid_S1ann.npy', np.array(gt_rank, dtype=np.int32))
    n = len(rows)
    print(f"{n} queries, Recall@100 {sum(1 for g in gt_rank if g) / n:.4f}, "
          f"candidates outside corpus: {missing / (n * TOP_K):.2%}")


if __name__ == '__main__':
    main()
