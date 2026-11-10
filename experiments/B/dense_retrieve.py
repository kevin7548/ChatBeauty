"""
S1 / S2: exact (flat) dense retrieval over items_for_embedding.jsonl for valid (or SPLIT=test) queries.

Item embeddings are built with ml/evaluation/retrieve_candidates.build_flat_index
(cached under experiments/B/cache/). Both models use max_seq_length=512 so S1 and S2
are compared under the same conditions. GPU (Colab T4) recommended.

Outputs (experiments/B/cache/):
    topk_valid_{SOURCE}.npy         (n_queries, 100) int32 corpus row indices
    topk_scores_valid_{SOURCE}.npy  (n_queries, 100) float16 cosine scores
    gt_rank_valid_{SOURCE}.npy      (n_queries,) int32 full rank of the ground truth (0 = not in corpus)
    run_{SOURCE}.json               model, timing

Usage:
    python experiments/B/dense_retrieve.py SOURCE=S1 MODEL_PATH=<fine-tuned dir>
    python experiments/B/dense_retrieve.py SOURCE=S2 MODEL_PATH=BAAI/bge-m3
"""

import json
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import CACHE, ITEMS_PATH, ML, SEED, TOP_K, cache_file, load_corpus, load_queries

sys.path.insert(0, str(ML / 'evaluation'))
from retrieve_candidates import build_flat_index, parse_args  # noqa: E402


def main():
    args = parse_args()
    source = args['SOURCE']
    model_path = args['MODEL_PATH']
    batch_size = int(args.get('BATCH_SIZE', 128))

    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = 512
    if device == 'cuda':
        model.half()
    print(f"{source}: {model_path} on {device}, max_seq_length={model.max_seq_length}")

    corpus_asins = [it['asin'] for it in load_corpus()]
    asin_to_row = {a: i for i, a in enumerate(corpus_asins)}

    t0 = time.time()
    index, index_asins = build_flat_index(model, model_path, ITEMS_PATH, CACHE, batch_size)
    assert index_asins == corpus_asins, "item order differs from items_for_embedding.jsonl"
    item_emb = torch.from_numpy(index.reconstruct_n(0, index.ntotal)).to(device)
    t_items = time.time() - t0

    queries = load_queries()
    t0 = time.time()
    q_emb = model.encode(
        [q['text'] for q in queries],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype('float32')
    t_queries = time.time() - t0

    gt_rows = torch.tensor([asin_to_row.get(q['parent_asin'], -1) for q in queries], device=device)
    topk_idx = np.zeros((len(queries), TOP_K), dtype=np.int32)
    topk_sc = np.zeros((len(queries), TOP_K), dtype=np.float16)
    gt_rank = np.zeros(len(queries), dtype=np.int32)

    t0 = time.time()
    step = 1024
    for i in range(0, len(queries), step):
        q = torch.from_numpy(q_emb[i:i + step]).to(device)
        scores = q @ item_emb.T
        sc, idx = scores.topk(TOP_K, dim=1)
        topk_idx[i:i + step] = idx.cpu().numpy()
        topk_sc[i:i + step] = sc.cpu().numpy()
        rows = gt_rows[i:i + step]
        valid = rows >= 0
        gt_sc = scores.gather(1, rows.clamp(min=0).unsqueeze(1))
        rank = (scores > gt_sc).sum(dim=1) + 1
        gt_rank[i:i + step] = torch.where(valid, rank, torch.zeros_like(rank)).cpu().numpy()
    t_search = time.time() - t0

    np.save(cache_file('topk', source), topk_idx)
    np.save(cache_file('topk_scores', source), topk_sc)
    np.save(cache_file('gt_rank', source), gt_rank)

    in_corpus = gt_rank > 0
    run = {
        'source': source,
        'model_path': model_path,
        'device': device,
        'fp16': device == 'cuda',
        'max_seq_length': model.max_seq_length,
        'n_items': len(corpus_asins),
        'n_queries': len(queries),
        'seconds': {'items': round(t_items, 1), 'queries': round(t_queries, 1), 'search': round(t_search, 1)},
        'recall@100': float(((gt_rank > 0) & (gt_rank <= 100)).sum() / len(queries)),
        'gt_in_corpus': int(in_corpus.sum()),
    }
    with open(cache_file('run', source).with_suffix('.json'), 'w') as f:
        json.dump(run, f, indent=2)
    print(json.dumps(run, indent=2))


if __name__ == '__main__':
    main()
