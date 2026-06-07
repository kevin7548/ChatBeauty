"""
Retrieve candidate products using an in-memory FAISS ANN index.

Vector search runs in-process (the embeddings + index don't fit the Supabase
free-tier storage cap): the FAISS HNSW index returns the Top-N parent_asins, then
a single batched query fetches just those rows' metadata from PostgreSQL.
"""

import logging

import faiss

from app.services.retrieval_resources import (
    model,
    ann_index,
    ann_asins,
    get_db_connection,
    release_db_connection,
)

logger = logging.getLogger(__name__)

# Fetch metadata for the ANN-selected candidates in one batched query.
FETCH_SQL = """
    SELECT
        parent_asin,
        title,
        price,
        average_rating,
        rating_number,
        store,
        features,
        description,
        top_reviews,
        details,
        image,
        embedding_text
    FROM products
    WHERE parent_asin = ANY(%s)
"""


def retrieve_candidates(query: str, n: int = 100):
    if not query or not query.strip():
        return []

    # Encode + L2-normalize so HNSW (L2) ranking == cosine ranking.
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)

    distances, indices = ann_index.search(vec, n)
    # indices/distances are shape (1, n); -1 marks empty slots.
    ranked = [
        (ann_asins[idx], 1.0 - float(dist) / 2.0)  # squared-L2 of unit vecs -> cosine
        for idx, dist in zip(indices[0], distances[0])
        if idx != -1
    ]
    if not ranked:
        return []

    asins = [asin for asin, _ in ranked]
    score_by_asin = {asin: score for asin, score in ranked}

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(FETCH_SQL, (asins,))
        rows = cur.fetchall()
        cur.close()
    finally:
        release_db_connection(conn)

    by_asin = {row[0]: row for row in rows}

    # Reassemble in ANN rank order, attaching the ANN cosine score.
    candidates = []
    for asin in asins:
        row = by_asin.get(asin)
        if row is None:
            continue
        candidates.append({
            "item_id": row[0],
            "title": row[1],
            "price": float(row[2]) if row[2] is not None else 0.0,
            "average_rating": float(row[3]) if row[3] is not None else 0.0,
            "rating_number": int(row[4]) if row[4] is not None else 0,
            "store": row[5] or "",
            "features": row[6] or "",
            "description": row[7] or "",
            "top_reviews": row[8] or "",
            "details": row[9] or "",
            "image": row[10] or "",
            "embedding_text": row[11] or "",
            "score": round(score_by_asin[asin], 6),
        })

    return candidates
