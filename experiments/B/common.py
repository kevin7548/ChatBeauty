"""Shared paths and loaders for experiment B (retrieval source analysis)."""

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ML = REPO / 'ml'
B_DIR = Path(__file__).resolve().parent
CACHE = B_DIR / 'cache'
RESULTS = REPO / 'experiments' / 'results'

ITEMS_PATH = ML / 'data/processed/items_for_embedding.jsonl'
REVIEWS_PATH = ML / 'data/raw/All_Beauty.jsonl'
# SPLIT=test python ... runs every script on the test split (default: valid)
SPLIT = os.environ.get('SPLIT', 'valid')
QUERIES = CACHE / f'{SPLIT}_queries.jsonl'
TRAIN_COUNTS = CACHE / 'train_review_counts.json'

SEED = 42
TOP_K = 100
SOURCES = {
    'S1': 'fine-tuned BGE-M3 (flat)',
    'S2': 'zero-shot BAAI/bge-m3 (flat)',
    'S3': 'BM25 (title + raw description + features + categories + raw train-period reviews)',
}


def load_corpus():
    """Items in items_for_embedding.jsonl order (same order build_flat_index uses)."""
    items = []
    with open(ITEMS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('embedding_text'):
                items.append(item)
    return items


def load_queries():
    with open(QUERIES, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_topk(source):
    """Top-K corpus row indices, shape (n_queries, TOP_K), int32."""
    return np.load(cache_file('topk', source))


def cache_file(kind, source):
    """e.g. cache_file('topk', 'S1') -> cache/topk_valid_S1.npy"""
    return CACHE / f'{kind}_{SPLIT}_{source}.npy'


def save_result(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / (f'B_{name}.json' if SPLIT == 'valid' else f'B_{SPLIT}_{name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {path}")
