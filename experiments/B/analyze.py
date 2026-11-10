"""
B1 / B2 / B4 analysis over cached top-100 results (no GPU needed).

Inputs (experiments/B/cache/): topk_valid_{S}.npy, gt_rank_valid_{S}.npy for S1, S2, S3
(+ optional references S1ann, S3tdc), valid_queries.jsonl, train_review_counts.json.
Optional: experiments/B/miss_labels.json (manual type labels for the 50 sampled S1 misses).

Outputs:
    experiments/results/B_B1_recall.json
    experiments/results/B_B2_overlap.json
    experiments/results/B_B4_misses.json
    experiments/B/cache/miss_samples.jsonl   (the 50 sampled misses, for reading/labeling)

Usage:
    python experiments/B/analyze.py
"""

import json
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from common import (B_DIR, CACHE, ML, SEED, SOURCES, SPLIT, TOP_K, TRAIN_COUNTS, cache_file, load_corpus,
                    load_queries, load_topk, save_result)

MAIN = ['S1', 'S2', 'S3']
REFERENCE = {
    'S1ann': 'fine-tuned BGE-M3 (serving HNSW, B0)',
    'S3emb': 'BM25 (embedding_text: LLM review keywords + description summary)',
    'S3rev20raw': 'BM25 (S3 with only the 20 most recent train reviews per product)',
    'S3tdcraw': 'BM25 (title + raw description + features + categories, no reviews)',
}
BUCKETS = [('0', 0, 0), ('1-4', 1, 4), ('5-19', 5, 19), ('20+', 20, 10**9)]
N_MISS_SAMPLE = 50
FT_MODEL = ML / 'model/retrieval/bge-m3-finetuned-20260202-120852'


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name


def recall_table(gt_rank, n):
    hit = lambda k: float(((gt_rank > 0) & (gt_rank <= k)).sum() / n)
    rr = np.where((gt_rank > 0) & (gt_rank <= TOP_K), 1.0 / np.maximum(gt_rank, 1), 0.0)
    return {'recall@10': hit(10), 'recall@50': hit(50), 'recall@100': hit(100), 'mrr@100': float(rr.mean())}


def mean_jaccard(a, b):
    total = 0.0
    for ra, rb in zip(a, b):
        sa, sb = set(ra.tolist()), set(rb.tolist())
        total += len(sa & sb) / len(sa | sb)
    return total / len(a)


def main():
    items = load_corpus()
    queries = load_queries()
    n = len(queries)
    with open(TRAIN_COUNTS) as f:
        train_counts = json.load(f)
    asin_to_row = {it['asin']: i for i, it in enumerate(items)}
    gt_row = np.array([asin_to_row.get(q['parent_asin'], -1) for q in queries])
    item_counts = np.array([train_counts.get(it['asin'], 0) for it in items])
    item_bucket = np.array([bucket_of(c) for c in item_counts])
    query_bucket = np.array([bucket_of(train_counts.get(q['parent_asin'], 0)) for q in queries])

    available = [s for s in MAIN + list(REFERENCE) if cache_file('topk', s).exists()]
    main_avail = [s for s in MAIN if s in available]
    print(f"Sources available: {available}")
    topk = {s: load_topk(s) for s in available}
    gt_rank = {s: np.load(cache_file('gt_rank', s)) for s in available}
    hit100 = {s: (gt_rank[s] > 0) & (gt_rank[s] <= TOP_K) for s in available}

    # ---------------- B1 ----------------
    b1 = {
        'split': f'{SPLIT} (eval split: reviews >= 20 chars, timestamp 80/10/10)',
        'n_queries': n,
        'n_items': len(items),
        'gt_in_corpus': int((gt_row >= 0).sum()),
        'sources': {s: {'desc': SOURCES.get(s) or REFERENCE[s], **recall_table(gt_rank[s], n)} for s in available},
        'runs': {s: json.load(open(cache_file('run', s).with_suffix('.json'))) for s in available
                 if cache_file('run', s).with_suffix('.json').exists()},
    }
    save_result('B1_recall', b1)

    # ---------------- B2 ----------------
    jaccard = {f'{a}-{b}': mean_jaccard(topk[a], topk[b]) for a, b in combinations(main_avail, 2)}
    union = {}
    for r in range(2, len(main_avail) + 1):
        for combo in combinations(main_avail, r):
            mask = np.logical_or.reduce([hit100[s] for s in combo])
            union['∪'.join(combo)] = float(mask.sum() / n)
    any_hit = np.logical_or.reduce([hit100[s] for s in main_avail])
    unique = {}
    for s in main_avail:
        others = [o for o in main_avail if o != s]
        only = hit100[s] & ~np.logical_or.reduce([hit100[o] for o in others]) if others else hit100[s]
        unique[s] = {
            'queries': int(only.sum()),
            'share_of_all_queries': float(only.sum() / n),
            'share_of_union_hits': float(only.sum() / max(any_hit.sum(), 1)),
        }
    n_bucket_items = Counter(item_bucket.tolist())
    coverage = {}
    for s in available:
        seen = np.zeros(len(items), dtype=bool)
        seen[topk[s][topk[s] >= 0]] = True
        seen10 = np.zeros(len(items), dtype=bool)
        seen10[topk[s][:, :10][topk[s][:, :10] >= 0]] = True
        slots = item_bucket[topk[s][topk[s] >= 0]]
        coverage[s] = {
            'items_seen': int(seen.sum()),
            'share_of_corpus': float(seen.mean()),
            'share_of_items_by_train_reviews': {
                b: float(seen[item_bucket == b].mean()) for b, _, _ in BUCKETS
            },
            'top10_share_of_corpus': float(seen10.mean()),
            'top10_share_of_items_by_train_reviews': {b: float(seen10[item_bucket == b].mean()) for b, _, _ in BUCKETS},
            # exposure: share of all top-100 slots taken by each bucket (compare with corpus share)
            'slot_share_by_train_reviews': {b: float((slots == b).mean()) for b, _, _ in BUCKETS},
        }
    seen_all = np.zeros(len(items), dtype=bool)
    for s in main_avail:
        seen_all[topk[s][topk[s] >= 0]] = True
    coverage['∪'.join(main_avail)] = {
        'items_seen': int(seen_all.sum()),
        'share_of_corpus': float(seen_all.mean()),
        'share_of_items_by_train_reviews': {b: float(seen_all[item_bucket == b].mean()) for b, _, _ in BUCKETS},
    }
    b2 = {
        'mean_top100_jaccard': jaccard,
        'union_recall@100': union,
        'unique_contribution': unique,
        'coverage': coverage,
        'items_by_train_reviews': {b: int(n_bucket_items[b]) for b, _, _ in BUCKETS},
        'corpus_share_by_train_reviews': {b: n_bucket_items[b] / len(items) for b, _, _ in BUCKETS},
    }
    save_result('B2_overlap', b2)

    # ---------------- B4: review-count buckets ----------------
    buckets = {}
    for b, _, _ in BUCKETS:
        mask = query_bucket == b
        entry = {'n_queries': int(mask.sum()), 'share_of_queries': float(mask.mean())}
        for s in available:
            entry[s] = float(hit100[s][mask].mean()) if mask.any() else None
        entry['∪'.join(main_avail)] = float(any_hit[mask].mean()) if mask.any() else None
        buckets[b] = entry

    # ---------------- B4: S1 misses ----------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(FT_MODEL))
    q_tokens = np.array([len(x) for x in tok([q['text'] for q in queries], add_special_tokens=True)['input_ids']])

    miss = ~hit100['S1'] & (gt_row >= 0)
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(np.nonzero(miss)[0], size=N_MISS_SAMPLE, replace=False))
    s1_scores = np.load(cache_file('topk_scores', 'S1')) if cache_file('topk_scores', 'S1').exists() else None

    def brief(row):
        it = items[row]
        return {'asin': it['asin'], 'title': it['title'], 'store': it.get('store'),
                'categories': it.get('categories'), 'train_reviews': int(item_counts[row]),
                'embedding_text_chars': len(it['embedding_text']), 'n_review_keywords': len(it['review_keywords'])}

    samples = []
    for qi in sample:
        g = int(gt_row[qi])
        rec = {
            'query_idx': int(qi),
            'query': queries[qi]['text'],
            'query_chars': len(queries[qi]['text']),
            'query_tokens': int(q_tokens[qi]),
            'gt': brief(g),
            'gt_embedding_text': items[g]['embedding_text'][:600],
            'rank': {s: int(gt_rank[s][qi]) for s in available},
            'rescued_by': [s for s in main_avail if s != 'S1' and hit100[s][qi]],
            'S1_top3': [brief(int(r)) for r in topk['S1'][qi][:3]],
            'S1_top1_score': float(s1_scores[qi][0]) if s1_scores is not None else None,
            'S1_top1_same_store': bool(items[int(topk['S1'][qi][0])].get('store') == items[g].get('store')
                                       and items[g].get('store')),
        }
        samples.append(rec)
    with open(CACHE / f'miss_samples_{SPLIT}.jsonl', 'w', encoding='utf-8') as f:
        for rec in samples:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    labels_path = B_DIR / 'miss_labels.json'
    by_type = None
    if labels_path.exists() and SPLIT == 'valid':
        labels = json.load(open(labels_path))['labels']
        by_type = defaultdict(lambda: {'n': 0, 'rescued_by_any': 0, **{f'rescued_by_{s}': 0 for s in main_avail if s != 'S1'}})
        for rec in samples:
            t = labels[str(rec['query_idx'])]['type']
            rec['type'] = t
            by_type[t]['n'] += 1
            by_type[t]['rescued_by_any'] += bool(rec['rescued_by'])
            for s in rec['rescued_by']:
                by_type[t][f'rescued_by_{s}'] += 1
        by_type = dict(sorted(by_type.items(), key=lambda kv: -kv[1]['n']))

    # Whole-population stats for all S1 misses (automatic, no manual labels)
    hit_mask = hit100['S1']
    pop = {
        'n_s1_misses': int(miss.sum()),
        'query_over_512_tokens': {'misses': float((q_tokens[miss] > 512).mean()), 'hits': float((q_tokens[hit_mask] > 512).mean())},
        'median_query_tokens': {'misses': float(np.median(q_tokens[miss])), 'hits': float(np.median(q_tokens[hit_mask]))},
        'gt_zero_train_reviews': {'misses': float((item_counts[gt_row[miss]] == 0).mean()),
                                  'hits': float((item_counts[gt_row[hit_mask]] == 0).mean())},
        'rescued_by_other_sources': {s: float(hit100[s][miss].mean()) for s in main_avail if s != 'S1'},
        'rescued_by_any_other': float(np.logical_or.reduce([hit100[s][miss] for s in main_avail if s != 'S1']).mean())
        if len(main_avail) > 1 else None,
    }
    if 'S1' in gt_rank and gt_rank['S1'][miss].max() > TOP_K:
        r = gt_rank['S1'][miss]
        pop['s1_rank_of_missed_gt'] = {'median': float(np.median(r)), 'share_101_200': float((r <= 200).mean()),
                                       'share_101_500': float((r <= 500).mean())}

    b4 = {
        'note': 'Eval split is per review, so "0 train reviews" is rare; report as low-review products, not cold-start. '
                'Train review counts come from the eval split train 80%.',
        'recall@100_by_train_reviews': buckets,
        'all_s1_misses': pop,
        'sample': {'seed': SEED, 'n': N_MISS_SAMPLE, 'type_counts': by_type,
                   'cases': [{k: v for k, v in rec.items() if k not in ('gt_embedding_text', 'S1_top3')} for rec in samples]},
    }
    save_result('B4_misses', b4)


if __name__ == '__main__':
    main()
