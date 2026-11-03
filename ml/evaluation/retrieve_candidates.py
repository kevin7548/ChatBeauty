"""
Retrieve top-K candidates for evaluation and re-ranking.

Reads directly from All_Beauty.jsonl, splits by timestamp (80/10/10),
encodes review text with BGE-M3, and searches a FAISS index.

Note: this split (80/10/10 of reviews with >= 20 chars) is NOT the same as the
keywords_{split}.jsonl split (fixed timestamp cutoffs). The existing
retrieval_candidates_valid.jsonl (62,669 queries) was produced by this split.

Index:
    INDEX=ann   serving HNSW index (model/retrieval/ann/). Built from the fine-tuned
                model's product embeddings, so only valid with that model.
    INDEX=flat  exact search. Encodes items_for_embedding.jsonl with MODEL_PATH
                (GPU recommended), caches the matrix under data/evaluation/.
                Use this for any other model (e.g. zero-shot BAAI/bge-m3).

Output format is lean (re-ranking features only, no large text fields).
Candidate metadata comes from meta_All_Beauty.jsonl. Output is written to
retrieval_candidates_{SPLIT}_{TAG}.jsonl so the original files are never overwritten.

Usage:
    python ml/evaluation/retrieve_candidates.py SPLIT=valid
    python ml/evaluation/retrieve_candidates.py MODEL_PATH=BAAI/bge-m3 INDEX=flat SPLIT=valid
    python ml/evaluation/retrieve_candidates.py SPLIT=train TOP_K=100
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> Dict[str, str]:
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def load_and_split_reviews(reviews_path: str, split: str) -> List[dict]:
    """
    Load reviews from All_Beauty.jsonl and split by timestamp.

    Split ratios (by sort_timestamp):
        - train: first 80%
        - valid: middle 10%
        - test: last 10%

    Returns list of reviews for the requested split.
    """
    print(f"Loading reviews from {reviews_path}...")

    # Load all reviews with timestamp
    reviews = []
    skipped = 0

    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                review = json.loads(line)

                # Skip reviews without required fields
                text = review.get('text', '')
                if not text or len(text.strip()) < 20:
                    skipped += 1
                    continue

                parent_asin = review.get('parent_asin') or review.get('asin')
                if not parent_asin:
                    skipped += 1
                    continue

                timestamp = review.get('timestamp', 0)

                reviews.append({
                    'parent_asin': parent_asin,
                    'text': text,
                    'timestamp': timestamp,
                })
            except json.JSONDecodeError:
                skipped += 1

    print(f"Loaded {len(reviews)} reviews (skipped {skipped})")

    # Sort by timestamp
    print("Sorting by timestamp...")
    reviews.sort(key=lambda x: x['timestamp'])

    # Split 80/10/10
    total = len(reviews)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    if split == 'train':
        selected = reviews[:train_end]
    elif split == 'valid':
        selected = reviews[train_end:valid_end]
    elif split == 'test':
        selected = reviews[valid_end:]
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'valid', or 'test'")

    print(f"Split '{split}': {len(selected)} reviews")
    print(f"  (train: {train_end}, valid: {valid_end - train_end}, test: {total - valid_end})")

    return selected


def load_item_metadata(meta_path: Path) -> Dict[str, dict]:
    """Re-ranking metadata per parent_asin (previously stored in ChromaDB)."""
    metadata = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading metadata"):
            item = json.loads(line)
            parent_asin = item.get('parent_asin')
            if not parent_asin:
                continue
            metadata[parent_asin] = {
                'price': float(item.get('price') or 0.0),
                'average_rating': float(item.get('average_rating') or 0.0),
                'rating_number': int(item.get('rating_number') or 0),
                'store': item.get('store') or '',
            }
    return metadata


def load_ann_index(ann_dir: Path) -> Tuple[faiss.Index, List[str]]:
    """Serving HNSW index (L2 over unit vectors) + row index -> parent_asin."""
    index = faiss.read_index(str(ann_dir / 'faiss_hnsw_cosine.index'))
    index.hnsw.efSearch = max(index.hnsw.efSearch, 128)
    with open(ann_dir / 'asins.json', 'r', encoding='utf-8') as f:
        asins = json.load(f)
    return index, asins


def build_flat_index(
    model: SentenceTransformer,
    model_path: str,
    items_path: Path,
    cache_dir: Path,
    batch_size: int,
) -> Tuple[faiss.Index, List[str]]:
    """Exact inner-product index over items_for_embedding.jsonl encoded with `model`."""
    cache_name = Path(model_path).name
    emb_path = cache_dir / f'item_embeddings_{cache_name}.npy'
    asins_path = cache_dir / f'item_asins_{cache_name}.json'

    if emb_path.exists() and asins_path.exists():
        print(f"Loading cached item embeddings: {emb_path}")
        embeddings = np.load(emb_path)
        with open(asins_path, 'r', encoding='utf-8') as f:
            asins = json.load(f)
    else:
        asins, texts = [], []
        with open(items_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading items"):
                item = json.loads(line)
                if item.get('embedding_text'):
                    # 'asin' in items_for_embedding.jsonl is actually parent_asin
                    asins.append(item['asin'])
                    texts.append(item['embedding_text'])
        print(f"Encoding {len(texts)} items (cached to {emb_path})")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype('float32')
        np.save(emb_path, embeddings)
        with open(asins_path, 'w', encoding='utf-8') as f:
            json.dump(asins, f)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, asins


def main():
    cli_args = parse_args()

    base_dir = Path(__file__).parent.parent  # ml/

    MODEL_PATH = cli_args.get(
        'MODEL_PATH', str(base_dir / 'model/retrieval/bge-m3-finetuned-20260202-120852')
    )
    INDEX = cli_args.get('INDEX', 'ann').lower()
    TOP_K = int(cli_args.get('TOP_K', 100))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 256))
    SPLIT = cli_args.get('SPLIT', 'valid').lower()
    TAG = cli_args.get('TAG', f'{INDEX}-{Path(MODEL_PATH).name}')

    if INDEX not in ('ann', 'flat'):
        print(f"Error: INDEX must be 'ann' or 'flat', got '{INDEX}'")
        return

    reviews_path = base_dir / 'data/raw/All_Beauty.jsonl'
    meta_path = base_dir / 'data/raw/meta_All_Beauty.jsonl'
    items_path = base_dir / 'data/processed/items_for_embedding.jsonl'
    ann_dir = base_dir / 'model/retrieval/ann'
    output_dir = base_dir / 'data/evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'retrieval_candidates_{SPLIT}_{TAG}.jsonl'

    print("=" * 60)
    print("Retrieve Top-K Candidates (FAISS)")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Index: {INDEX}")
    print(f"Split: {SPLIT}")
    print(f"Top-K: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output: {output_path}")
    print()

    # Check inputs
    required = [reviews_path, meta_path] + ([ann_dir] if INDEX == 'ann' else [items_path])
    for path in required:
        if not path.exists():
            print(f"Error: {path} not found")
            return

    # Step 1: Load and split reviews
    print("=== Step 1: Loading and splitting reviews ===")
    reviews = load_and_split_reviews(str(reviews_path), SPLIT)

    # Step 2: Load model
    print("\n=== Step 2: Loading model ===")
    model = SentenceTransformer(MODEL_PATH)
    print(f"Model loaded. Max seq length: {model.max_seq_length}")

    # Step 3: Load / build the FAISS index
    print(f"\n=== Step 3: Preparing FAISS index ({INDEX}) ===")
    if INDEX == 'ann':
        index, index_asins = load_ann_index(ann_dir)
    else:
        index, index_asins = build_flat_index(model, MODEL_PATH, items_path, output_dir, BATCH_SIZE)
    print(f"Index has {index.ntotal} items")
    item_meta = load_item_metadata(meta_path)

    # Step 4: Encode all queries
    print("\n=== Step 4: Encoding queries ===")
    query_texts = [r['text'] for r in reviews]
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype('float32')
    print(f"Encoded {len(query_embeddings)} queries")

    # Step 5: Search and write results (lean format)
    print(f"\n=== Step 5: Retrieving top-{TOP_K} candidates ===")
    search_batch_size = 1024
    written = 0

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(reviews), search_batch_size), desc="Searching FAISS"):
            batch_end = min(i + search_batch_size, len(reviews))
            distances, indices = index.search(query_embeddings[i:batch_end], TOP_K)

            for j, review in enumerate(reviews[i:batch_end]):
                candidates = []
                for idx, dist in zip(indices[j], distances[j]):
                    if idx == -1:
                        continue
                    item_asin = index_asins[idx]
                    # ann: squared L2 of unit vectors -> cosine; flat: inner product = cosine
                    score = 1.0 - float(dist) / 2.0 if INDEX == 'ann' else float(dist)
                    meta = item_meta.get(item_asin, {})

                    # Lean format: only re-ranking features (no large text fields)
                    candidates.append({
                        'item_asin': item_asin,
                        'score': round(score, 6),
                        # Re-ranking features only
                        'price': meta.get('price', 0.0),
                        'average_rating': meta.get('average_rating', 0.0),
                        'rating_number': meta.get('rating_number', 0),
                        'store': meta.get('store', ''),
                    })

                output = {
                    'parent_asin': review['parent_asin'],  # ground truth
                    'query_text': review['text'],  # for re-ranking text features
                    'candidates': candidates,
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                written += 1

    print(f"\n=== Complete ===")
    print(f"  Queries processed: {written}")
    print(f"  Output saved to: {output_path}")
    print()
    print(f"Evaluate: python ml/evaluation/evaluate_recall.py SPLIT={SPLIT} TAG={TAG}")


if __name__ == '__main__':
    main()
