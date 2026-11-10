"""
Evaluate retrieval quality (Recall@K, MRR) from retrieval_candidates_{split}.jsonl.

Usage:
    python ml/evaluation/evaluate_recall.py SPLIT=valid
    python ml/evaluation/evaluate_recall.py SPLIT=test
    python ml/evaluation/evaluate_recall.py SPLIT=valid TAG=flat-bge-m3  # retrieval_candidates_valid_flat-bge-m3.jsonl
"""

import json
import sys
from pathlib import Path


def parse_args():
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def main():
    cli_args = parse_args()
    split = cli_args.get('SPLIT', 'valid')
    tag = cli_args.get('TAG', '')

    base_dir = Path(__file__).parent.parent  # ml/
    suffix = f'_{tag}' if tag else ''
    candidates_path = base_dir / f'data/evaluation/retrieval_candidates_{split}{suffix}.jsonl'

    if not candidates_path.exists():
        print(f"Error: {candidates_path} not found")
        print(f"Run retrieve_candidates.py first with SPLIT={split}")
        return

    k_values = [1, 5, 10, 20, 50, 100]
    hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    total = 0

    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            ground_truth = item['parent_asin']
            candidate_asins = [c['item_asin'] for c in item['candidates']]

            # Find rank of ground truth (1-indexed)
            rank = None
            for i, asin in enumerate(candidate_asins):
                if asin == ground_truth:
                    rank = i + 1
                    break

            for k in k_values:
                if rank is not None and rank <= k:
                    hits[k] += 1

            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            total += 1

    if total == 0:
        print("No queries found.")
        return

    print("=" * 50)
    print(f"Retrieval Evaluation - Split: {split}" + (f" ({tag})" if tag else ""))
    print("=" * 50)
    print(f"Total queries: {total:,}")
    print()
    print("Recall@K:")
    for k in k_values:
        recall = hits[k] / total
        print(f"  Recall@{k:<4d} {recall:.4f}  ({hits[k]:,}/{total:,})")
    mrr = sum(reciprocal_ranks) / total
    print()
    print(f"MRR:         {mrr:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
