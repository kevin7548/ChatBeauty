"""
Create training pairs for fine-tuning BGE-M3.

Each training sample consists of:
- query: keywords extracted from a review (what the user is looking for)
- positive: the item's embedding_text (the item the reviewer actually purchased)
- negative: (optional) hard negatives from other items

Output format compatible with sentence-transformers training.

Usage:
    python ml/retrieval/create_training_pairs.py
    python ml/retrieval/create_training_pairs.py OUTPUT_FORMAT=triplet  # query, positive, negative
    python ml/retrieval/create_training_pairs.py OUTPUT_FORMAT=pair     # query, positive only

The served model (bge-m3-finetuned-20260202-120852) was trained on QUERY_TYPE=both:
518,393 keyword pairs + 487,885 review-text pairs = 1,006,278 (90% = the 905,650 in its model card).
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import random


def load_items_for_embedding(items_path: str) -> Dict[str, dict]:
    """Load items and create lookup by parent_asin"""
    items = {}
    with open(items_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading items"):
            item = json.loads(line)
            # Key is 'asin' in items_for_embedding.jsonl but it's actually parent_asin
            asin = item.get('asin')
            items[asin] = item
    return items


def create_query_text(keywords: List, max_keywords: int = 20) -> str:
    """
    Create query text from keywords.

    Options for query format:
    1. Comma-separated keywords (current)
    2. Natural language synthesis (future enhancement with LLM)
    """
    if not keywords:
        return ""

    # Take top N keywords (already sorted by frequency)
    selected = keywords[:max_keywords]
    # Convert all to strings (some keywords may be integers)
    return ", ".join(str(kw) for kw in selected if kw is not None)


def create_training_pairs(
    keywords_path: str,
    items_path: str,
    output_path: str,
    output_format: str = "pair",
    max_keywords: int = 20,
    include_review_text: bool = False,
    query_type: str = "keywords"
):
    """
    Create training pairs from keywords and items.

    Args:
        keywords_path: Path to keywords_train.jsonl
        items_path: Path to items_for_embedding.jsonl
        output_path: Output path for training pairs
        output_format: "pair" (query, positive) or "triplet" (query, positive, negative)
        max_keywords: Maximum keywords to use in query
        include_review_text: If True, also include original review text as query variant
        query_type: "keywords", "review_text", or "both"
    """

    # Step 1: Load items
    print("\n=== Step 1: Loading items ===")
    items = load_items_for_embedding(items_path)
    print(f"Loaded {len(items)} items")

    # Get list of all item asins for negative sampling
    all_asins = list(items.keys())

    # Step 2: Process keywords and create pairs
    print(f"\n=== Step 2: Creating training pairs ===")

    pairs_created = 0
    skipped_no_item = 0
    skipped_no_keywords = 0

    with open(keywords_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Creating pairs"):
            review = json.loads(line)

            parent_asin = review.get('parent_asin') or review.get('asin')
            keywords = review.get('keywords', [])
            review_text = review.get('review_text', '')

            # Skip if item not found
            if parent_asin not in items:
                skipped_no_item += 1
                continue

            item = items[parent_asin]
            positive_text = item.get('embedding_text', '')

            if not positive_text:
                skipped_no_item += 1
                continue

            # Create query based on query_type
            if query_type == "review_text":
                if not review_text or len(review_text) < 20:
                    skipped_no_keywords += 1
                    continue
                query_text = review_text
            else:
                if not keywords:
                    skipped_no_keywords += 1
                    continue
                query_text = create_query_text(keywords, max_keywords)

            if output_format == "pair":
                # Simple pair format for contrastive learning
                pair = {
                    "query": query_text,
                    "positive": positive_text,
                    "parent_asin": parent_asin
                }
                f_out.write(json.dumps(pair, ensure_ascii=False) + '\n')
                pairs_created += 1

            elif output_format == "triplet":
                # Triplet format with random negative
                # Note: Hard negatives from ChromaDB would be better
                negative_asin = random.choice(all_asins)
                while negative_asin == parent_asin:
                    negative_asin = random.choice(all_asins)

                negative_text = items[negative_asin].get('embedding_text', '')

                triplet = {
                    "query": query_text,
                    "positive": positive_text,
                    "negative": negative_text,
                    "parent_asin": parent_asin,
                    "negative_asin": negative_asin
                }
                f_out.write(json.dumps(triplet, ensure_ascii=False) + '\n')
                pairs_created += 1

            # Optionally create variant with review text as query
            if include_review_text and review_text and len(review_text) >= 20:
                if output_format == "pair":
                    pair = {
                        "query": review_text,
                        "positive": positive_text,
                        "parent_asin": parent_asin,
                        "query_type": "review_text"
                    }
                    f_out.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    pairs_created += 1

    print(f"\n=== Complete ===")
    print(f"  Training pairs created: {pairs_created}")
    print(f"  Skipped (no item): {skipped_no_item}")
    print(f"  Skipped (no keywords): {skipped_no_keywords}")
    print(f"  Output saved to: {output_path}")


def parse_args() -> Dict[str, str]:
    """Parse KEY=VALUE arguments from command line"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def main():
    cli_args = parse_args()

    # Configuration
    OUTPUT_FORMAT = cli_args.get('OUTPUT_FORMAT', 'pair').lower()
    MAX_KEYWORDS = int(cli_args.get('MAX_KEYWORDS', 20))
    INCLUDE_REVIEW_TEXT = cli_args.get('INCLUDE_REVIEW_TEXT', 'false').lower() == 'true'
    # QUERY_TYPE: "keywords" (default), "review_text", "both"
    QUERY_TYPE = cli_args.get('QUERY_TYPE', 'keywords').lower()
    if QUERY_TYPE == 'review_text':
        INCLUDE_REVIEW_TEXT = False  # handled by QUERY_TYPE directly
    elif QUERY_TYPE == 'both':
        INCLUDE_REVIEW_TEXT = True

    # File paths
    base_dir = Path(__file__).parent.parent  # ml/

    keywords_path = base_dir / 'data/processed/keywords_train.jsonl'
    items_path = base_dir / 'data/processed/items_for_embedding.jsonl'
    output_path = base_dir / 'data/processed/training_pairs.jsonl'

    print("=" * 60)
    print("BGE-M3 Training Pair Generator")
    print("=" * 60)
    print(f"Output format: {OUTPUT_FORMAT}")
    print(f"Query type: {QUERY_TYPE}")
    print(f"Max keywords: {MAX_KEYWORDS}")
    print(f"Include review text: {INCLUDE_REVIEW_TEXT}")
    print()

    # Check input files exist
    if not keywords_path.exists():
        print(f"Error: Keywords file not found: {keywords_path}")
        return

    if not items_path.exists():
        print(f"Error: Items file not found: {items_path}")
        return

    create_training_pairs(
        keywords_path=str(keywords_path),
        items_path=str(items_path),
        output_path=str(output_path),
        output_format=OUTPUT_FORMAT,
        max_keywords=MAX_KEYWORDS,
        include_review_text=INCLUDE_REVIEW_TEXT,
        query_type=QUERY_TYPE
    )


if __name__ == '__main__':
    main()