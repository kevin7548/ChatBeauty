"""
Retrieve candidate products using pgvector cosine similarity.

Replaces the previous ChromaDB-based retrieval with PostgreSQL + pgvector.
"""

import logging

from app.services.retrieval_resources import model, get_db_connection, release_db_connection

logger = logging.getLogger(__name__)

RETRIEVE_SQL = """
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
        embedding_text,
        1 - (embedding <=> %s::vector) AS score
    FROM products
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""


def retrieve_candidates(query: str, n: int = 100):
    if not query or not query.strip():
        return []

    embedding = model.encode([query], convert_to_numpy=True)[0].tolist()
    embedding_str = str(embedding)

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(RETRIEVE_SQL, (embedding_str, embedding_str, n))
        rows = cur.fetchall()
        cur.close()
    finally:
        release_db_connection(conn)

    candidates = []
    for row in rows:
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
            "score": round(float(row[12]), 6),
        })

    return candidates
