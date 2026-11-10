"""
Split keywords_output.jsonl into train/valid/test by timestamp.

Reads All_Beauty.jsonl in parallel to get timestamps and parent_asin,
since keywords_output.jsonl was generated from All_Beauty.jsonl in the same order
(filtering reviews with text length >= 10).

Usage:
    python ml/preprocess/split_keywords.py
    python ml/preprocess/split_keywords.py TRAIN_END=1621867671638 VALID_END=1643212573029
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm


def parse_args():
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def main():
    cli_args = parse_args()

    # Timestamp boundaries
    TRAIN_END = int(cli_args.get('TRAIN_END', '1621867671638'))
    VALID_END = int(cli_args.get('VALID_END', '1643212573029'))

    base_dir = Path(__file__).parent.parent  # ml/
    reviews_path = base_dir / 'data/raw/All_Beauty.jsonl'
    keywords_path = base_dir / 'data/processed/keywords_output.jsonl'
    output_dir = base_dir / 'data/processed'

    train_path = output_dir / 'keywords_train.jsonl'
    valid_path = output_dir / 'keywords_valid.jsonl'
    test_path = output_dir / 'keywords_test.jsonl'

    print("=" * 60)
    print("Keyword Splitter (by timestamp)")
    print("=" * 60)
    print(f"Train end:  {TRAIN_END}")
    print(f"Valid end:  {VALID_END}")
    print()

    # Count lines in keywords file for progress bar
    total_keywords = sum(1 for _ in open(keywords_path, 'r'))
    print(f"Total keywords records: {total_keywords}")

    counts = {'train': 0, 'valid': 0, 'test': 0}

    with open(reviews_path, 'r', encoding='utf-8') as f_reviews, \
         open(keywords_path, 'r', encoding='utf-8') as f_keywords, \
         open(train_path, 'w', encoding='utf-8') as f_train, \
         open(valid_path, 'w', encoding='utf-8') as f_valid, \
         open(test_path, 'w', encoding='utf-8') as f_test:

        keywords_iter = iter(f_keywords)
        pbar = tqdm(total=total_keywords, desc="Splitting")

        for review_line in f_reviews:
            try:
                review = json.loads(review_line)
            except json.JSONDecodeError:
                continue

            text = review.get('text', '')
            if not text or len(text.strip()) < 10:
                continue

            # This review has a corresponding keywords record
            try:
                keyword_line = next(keywords_iter)
            except StopIteration:
                break

            try:
                keyword_record = json.loads(keyword_line)
            except json.JSONDecodeError:
                pbar.update(1)
                continue

            # Add parent_asin from the original review
            keyword_record['parent_asin'] = review.get('parent_asin', review.get('asin', ''))

            timestamp = review.get('timestamp', 0)
            output_line = json.dumps(keyword_record, ensure_ascii=False) + '\n'

            if timestamp <= TRAIN_END:
                f_train.write(output_line)
                counts['train'] += 1
            elif timestamp <= VALID_END:
                f_valid.write(output_line)
                counts['valid'] += 1
            else:
                f_test.write(output_line)
                counts['test'] += 1

            pbar.update(1)

        pbar.close()

    print(f"\n=== Split Complete ===")
    print(f"  Train: {counts['train']} ({train_path.name})")
    print(f"  Valid: {counts['valid']} ({valid_path.name})")
    print(f"  Test:  {counts['test']} ({test_path.name})")
    print(f"  Total: {sum(counts.values())}")


if __name__ == '__main__':
    main()
