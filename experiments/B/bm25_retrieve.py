"""
S3: BM25 over product text for valid (or SPLIT=test) queries (bm25s, English stopwords + Snowball stemmer).

FIELDS=embedding_text: the same text the dense models embed (LLM review keywords, summarized description)
FIELDS=tdc: title + description_summary + features + categories (no review keywords)
FIELDS=tdcr (default): tdc + raw text of the product's train-period reviews (eval split train 80%,
             so no valid/test review leaks in). MAX_REVIEWS=N keeps only the N most recent.
DESC=raw (default): use the original meta_All_Beauty description instead of description_summary
          (the summary exists to keep BGE-M3 inputs short; BM25 has no length limit).

Outputs (experiments/B/cache/):
    topk_valid_{SOURCE}.npy  (n_queries, 100) int32 corpus row indices
    gt_rank_valid_{SOURCE}.npy  rank within top-100 (0 = not retrieved)

Usage:
    python experiments/B/bm25_retrieve.py                                  # S3: raw desc + all train reviews
    python experiments/B/bm25_retrieve.py FIELDS=embedding_text SOURCE=S3emb
    python experiments/B/bm25_retrieve.py MAX_REVIEWS=20 SOURCE=S3rev20raw
    python experiments/B/bm25_retrieve.py FIELDS=tdc SOURCE=S3tdcraw
"""

import json
import sys
import time
from collections import defaultdict

import bm25s
import numpy as np
import Stemmer

from common import CACHE, ML, REVIEWS_PATH, TOP_K, cache_file, load_corpus, load_queries

sys.path.insert(0, str(ML / 'evaluation'))
from retrieve_candidates import load_and_split_reviews, parse_args  # noqa: E402


def train_reviews_by_item(max_reviews):
    """parent_asin -> train-period review texts (oldest first, last `max_reviews` kept)."""
    by_item = defaultdict(list)
    for r in load_and_split_reviews(str(REVIEWS_PATH), 'train'):
        by_item[r['parent_asin']].append(r['text'])
    if max_reviews:
        by_item = {a: texts[-max_reviews:] for a, texts in by_item.items()}
    return by_item


def raw_descriptions():
    desc = {}
    with open(ML / 'data/raw/meta_All_Beauty.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            desc[m['parent_asin']] = m.get('description') or []
    return desc


def item_text(item, fields, reviews=None, raw_desc=None):
    if fields == 'embedding_text':
        return item['embedding_text']
    parts = [item.get('title') or '']
    parts += raw_desc.get(item['asin'], []) if raw_desc is not None else item.get('description_summary') or []
    parts += item.get('features') or []
    parts += item.get('categories') or []
    if fields == 'tdcr':
        parts += reviews.get(item['asin'], [])
    return ' '.join(str(p) for p in parts)


def main():
    args = parse_args()
    fields = args.get('FIELDS', 'tdcr')
    source = args.get('SOURCE', 'S3')
    max_reviews = int(args.get('MAX_REVIEWS', 0))
    reviews = train_reviews_by_item(max_reviews) if fields == 'tdcr' else None
    desc = args.get('DESC', 'raw')
    raw_desc = raw_descriptions() if desc == 'raw' else None

    items = load_corpus()
    asin_to_row = {it['asin']: i for i, it in enumerate(items)}
    stemmer = Stemmer.Stemmer('english')

    t0 = time.time()
    corpus_tokens = bm25s.tokenize([item_text(it, fields, reviews, raw_desc) for it in items], stopwords='en', stemmer=stemmer)
    retriever = bm25s.BM25()  # default: Lucene variant, k1=1.5, b=0.75
    retriever.index(corpus_tokens)
    t_index = time.time() - t0

    queries = load_queries()
    t0 = time.time()
    query_tokens = bm25s.tokenize([q['text'] for q in queries], stopwords='en', stemmer=stemmer)
    topk_idx, _ = retriever.retrieve(query_tokens, k=TOP_K, n_threads=-1)
    t_search = time.time() - t0
    topk_idx = topk_idx.astype(np.int32)

    gt_rank = np.zeros(len(queries), dtype=np.int32)
    for i, q in enumerate(queries):
        row = asin_to_row.get(q['parent_asin'], -1)
        hit = np.nonzero(topk_idx[i] == row)[0]
        if hit.size:
            gt_rank[i] = hit[0] + 1

    np.save(cache_file('topk', source), topk_idx)
    np.save(cache_file('gt_rank', source), gt_rank)
    run = {
        'source': source,
        'fields': fields,
        'max_reviews': max_reviews or None,
        'description': desc,
        'bm25': 'bm25s Lucene k1=1.5 b=0.75, en stopwords, Snowball english stemmer',
        'n_items': len(items),
        'n_queries': len(queries),
        'seconds': {'index': round(t_index, 1), 'search': round(t_search, 1)},
        'recall@100': float((gt_rank > 0).sum() / len(queries)),
    }
    with open(cache_file('run', source).with_suffix('.json'), 'w') as f:
        json.dump(run, f, indent=2)
    print(json.dumps(run, indent=2))


if __name__ == '__main__':
    main()
