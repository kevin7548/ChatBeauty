"""
Fine-tune BGE-M3 for beauty product retrieval using sentence-transformers.

Uses MultipleNegativesRankingLoss with in-batch negatives for contrastive learning.

Usage:
    python ml/retrieval/finetune_bge_m3.py
    python ml/retrieval/finetune_bge_m3.py EPOCHS=3 BATCH_SIZE=32
    python ml/retrieval/finetune_bge_m3.py USE_FP16=true  # Mixed precision training

The served model (bge-m3-finetuned-20260202-120852) used EPOCHS=2 BATCH_SIZE=32 on
QUERY_TYPE=both pairs (see create_training_pairs.py); the defaults below are smaller.

Requirements:
    pip install sentence-transformers>=2.2.0
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from tqdm import tqdm


def load_training_pairs(
    train_path: str,
    val_path: str = None,
    val_ratio: float = 0.1,
    max_query_length: int = 256,
    max_doc_length: int = 512,
    seed: int = 42
) -> Tuple[List[InputExample], List[InputExample]]:
    """
    Load training pairs and split into train/val sets.

    Args:
        train_path: Path to training_pairs.jsonl
        val_path: Optional separate validation file
        val_ratio: Ratio for validation split if no val_path
        max_query_length: Max characters for query (will be tokenized later)
        max_doc_length: Max characters for document
        seed: Random seed for reproducibility

    Returns:
        (train_examples, val_examples)
    """
    random.seed(seed)

    all_examples = []

    print(f"Loading training pairs from {train_path}...")
    skipped = 0
    with open(train_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading"), 1):
            try:
                item = json.loads(line)
                query = item.get('query', '')[:max_query_length]
                positive = item.get('positive', '')[:max_doc_length]

                if query and positive:
                    all_examples.append(InputExample(texts=[query, positive]))
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"\nWarning: Skipping malformed line {line_num}: {e}")

    if skipped > 0:
        print(f"Skipped {skipped} malformed lines")

    print(f"Loaded {len(all_examples)} training pairs")

    # Shuffle
    random.shuffle(all_examples)

    # Split
    if val_path:
        val_examples = []
        val_skipped = 0
        with open(val_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    query = item.get('query', '')[:max_query_length]
                    positive = item.get('positive', '')[:max_doc_length]
                    if query and positive:
                        val_examples.append(InputExample(texts=[query, positive]))
                except json.JSONDecodeError:
                    val_skipped += 1
        if val_skipped > 0:
            print(f"Skipped {val_skipped} malformed lines in validation file")
        train_examples = all_examples
    else:
        split_idx = int(len(all_examples) * (1 - val_ratio))
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]

    print(f"Train: {len(train_examples)}, Validation: {len(val_examples)}")

    return train_examples, val_examples


def create_evaluator(
    val_examples: List[InputExample],
    name: str = "val"
) -> evaluation.InformationRetrievalEvaluator:
    """
    Create an evaluator for validation during training.

    Uses InformationRetrievalEvaluator which computes:
    - MRR (Mean Reciprocal Rank)
    - Recall@K
    - NDCG@K
    """
    # Build queries, corpus, and relevant_docs
    queries = {}
    corpus = {}
    relevant_docs = {}

    for i, example in enumerate(val_examples):
        query_id = f"q{i}"
        doc_id = f"d{i}"

        queries[query_id] = example.texts[0]
        corpus[doc_id] = example.texts[1]
        relevant_docs[query_id] = {doc_id}

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
        batch_size=32,
    )

    return evaluator


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
    MODEL_NAME = cli_args.get('MODEL', 'BAAI/bge-m3')
    EPOCHS = int(cli_args.get('EPOCHS', 1))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 16))  # Adjust based on GPU memory
    LEARNING_RATE = float(cli_args.get('LR', 2e-5))
    WARMUP_RATIO = float(cli_args.get('WARMUP_RATIO', 0.1))
    USE_FP16 = cli_args.get('USE_FP16', 'true').lower() == 'true'
    MAX_QUERY_LENGTH = int(cli_args.get('MAX_QUERY_LENGTH', 256))
    MAX_DOC_LENGTH = int(cli_args.get('MAX_DOC_LENGTH', 512))
    EVAL_STEPS = int(cli_args.get('EVAL_STEPS', 1000))
    SAVE_STEPS = int(cli_args.get('SAVE_STEPS', 5000))

    # File paths
    base_dir = Path(__file__).parent.parent  # ml/
    train_path = base_dir / 'data/processed/training_pairs.jsonl'
    output_dir = base_dir / 'model/retrieval' / f'bge-m3-finetuned-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    print("=" * 60)
    print("BGE-M3 Fine-tuning with Sentence-Transformers")
    print("=" * 60)
    print(f"Base model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Use FP16: {USE_FP16}")
    print(f"Max query length: {MAX_QUERY_LENGTH}")
    print(f"Max doc length: {MAX_DOC_LENGTH}")
    print(f"Output: {output_dir}")
    print()

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow!")
        USE_FP16 = False
    print()

    # Check input file
    if not train_path.exists():
        print(f"Error: Training file not found: {train_path}")
        print("Run create_training_pairs.py first")
        return

    # Load model
    print("=== Loading model ===")
    model = SentenceTransformer(MODEL_NAME)

    # Set max sequence length (BGE-M3 supports up to 8192)
    model.max_seq_length = max(MAX_QUERY_LENGTH, MAX_DOC_LENGTH)
    print(f"Max sequence length: {model.max_seq_length}")

    # Load training data
    print("\n=== Loading training data ===")
    train_examples, val_examples = load_training_pairs(
        train_path=str(train_path),
        max_query_length=MAX_QUERY_LENGTH,
        max_doc_length=MAX_DOC_LENGTH
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    # Create loss function
    # MultipleNegativesRankingLoss uses in-batch negatives
    # For each (query, positive) pair, all other positives in batch are negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Create evaluator (sample for speed)
    print("\n=== Creating evaluator ===")
    eval_samples = min(len(val_examples), 5000)  # Limit eval samples for speed
    evaluator = create_evaluator(val_examples[:eval_samples])

    # Calculate training steps
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"\n=== Training configuration ===")
    print(f"Total training examples: {len(train_examples)}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train!
    print("\n=== Starting training ===")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        #evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=str(output_dir),
        save_best_model=False,
        show_progress_bar=True,
        use_amp=USE_FP16,  # Automatic Mixed Precision
        #evaluation_steps=EVAL_STEPS,
        #checkpoint_save_steps=SAVE_STEPS,
        #checkpoint_save_total_limit=1,
        #checkpoint_path=str(output_dir / 'checkpoints'),
    )

    print(f"\n=== Training complete ===")
    print(f"Model saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. Retrieve candidates with the fine-tuned model:")
    print(f"   python ml/evaluation/retrieve_candidates.py MODEL_PATH={output_dir} INDEX=flat SPLIT=valid")
    print("2. Evaluate: python ml/evaluation/evaluate_recall.py SPLIT=valid")


if __name__ == '__main__':
    main()