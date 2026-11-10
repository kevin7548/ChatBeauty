"""
Merge S1 (fine-tuned BGE-M3) and S3 (BM25) top-100 lists into one candidate list of size K.

- union: S1 top-100 + S3 top-100 deduplicated (K varies, up to 200)
- quota a/b: S1 top-a, then S3 in rank order skipping duplicates until K
- RRF: score = sum 1 / (k + rank) over sources, top-K (k = 60)

No reranker here: Recall@K of the merged list is the ceiling a ranker could reach.

Output: experiments/results/B_fusion.json

Usage:
    python experiments/B/fusion.py
"""

import numpy as np

from common import TRAIN_COUNTS, cache_file, load_corpus, load_queries, save_result
import json

RRF_K = 60


def merged_hits(a, b, gt, K, how, quota=None):
    hits = np.zeros(len(gt), dtype=bool)
    for i in range(len(gt)):
        if how == 'quota':
            cand = list(a[i][:quota])
            seen = set(cand)
            for x in b[i]:
                if len(cand) >= K:
                    break
                if x not in seen:
                    cand.append(x)
                    seen.add(x)
            for x in a[i][quota:]:
                if len(cand) >= K:
                    break
                if x not in seen:
                    cand.append(x)
                    seen.add(x)
        else:  # rrf
            score = {}
            for lst in (a[i], b[i]):
                for r, x in enumerate(lst, 1):
                    score[x] = score.get(x, 0.0) + 1.0 / (RRF_K + r)
            cand = sorted(score, key=score.get, reverse=True)[:K]
        hits[i] = gt[i] in cand
    return hits


def main():
    items = load_corpus()
    row = {it['asin']: i for i, it in enumerate(items)}
    queries = load_queries()
    gt = np.array([row.get(q['parent_asin'], -1) for q in queries])
    s1, s3 = np.load(cache_file('topk', 'S1')), np.load(cache_file('topk', 'S3'))
    with open(TRAIN_COUNTS) as f:
        tc = json.load(f)
    bucket = np.digitize([tc.get(q['parent_asin'], 0) for q in queries], [1, 5, 20])
    names = ['0', '1-4', '5-19', '20+']

    hit1 = (s1 == gt[:, None]).any(1)
    hit3 = (s3 == gt[:, None]).any(1)
    union_size = np.array([len(set(a.tolist()) | set(b.tolist())) for a, b in zip(s1, s3)])

    runs = {'S1 only (K=100)': hit1, 'S3 only (K=100)': hit3, 'union (K≈%d)' % round(union_size.mean()): hit1 | hit3}
    for q in (90, 80, 70, 50):
        runs[f'quota S1 {q} + S3 {100 - q} (K=100)'] = merged_hits(s1, s3, gt, 100, 'quota', q)
    runs['RRF k=60 (K=100)'] = merged_hits(s1, s3, gt, 100, 'rrf')
    runs['RRF k=60 (K=50)'] = merged_hits(s1, s3, gt, 50, 'rrf')
    s1_50 = (s1[:, :50] == gt[:, None]).any(1)
    runs['S1 only (K=50)'] = s1_50

    result = {
        'note': 'Recall@K of the merged candidate list before any reranker; union K varies per query.',
        'union_size': {'mean': float(union_size.mean()), 'min': int(union_size.min()), 'max': int(union_size.max())},
        'recall': {name: float(h.mean()) for name, h in runs.items()},
        'recall_by_train_reviews': {
            name: {names[k]: float(h[bucket == k].mean()) for k in range(4)} for name, h in runs.items()
        },
    }
    # Candidate budget curve: S1 alone at K vs S1 top-(K-b) + S3 top-b (duplicates count as slots, so a lower bound)
    r1 = np.load(cache_file('gt_rank', 'S1'))
    s1_at = lambda k: (r1 > 0) & (r1 <= k)
    curve = {}
    for K in (50, 100, 150, 184, 200, 300, 500, 1000):
        best = max(((s1_at(K - b) | (s3[:, :b] == gt[:, None]).any(1)).mean(), b)
                   for b in (0, 10, 20, 30, 50, 100) if b <= min(K, 100))
        curve[K] = {'S1_alone': float(s1_at(K).mean()), 'S1_plus_S3_best': float(best[0]), 'S3_slots': best[1]}
    result['budget_curve'] = curve

    save_result('fusion', result)
    for name, h in runs.items():
        by = ' '.join(f'{h[bucket == k].mean():.3f}' for k in range(4))
        print(f'{name:32s} {h.mean():.4f} | 0/1-4/5-19/20+: {by}')


if __name__ == '__main__':
    main()
