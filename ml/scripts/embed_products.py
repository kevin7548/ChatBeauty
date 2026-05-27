"""
Compute BGE-M3 embeddings for all products in PostgreSQL and write them back.

Run locally with Cloud SQL Proxy, or in Colab with a GPU:

    python -m ml.scripts.embed_products \
        --database-url=postgresql://postgres:PASSWORD@localhost:5433/chatbeauty \
        --model-path=ml/model/retrieval/bge-m3-finetuned-20260202-120852 \
        --batch-size=256
"""

import argparse
import logging

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--model-path", default="ml/model/retrieval/bge-m3-finetuned-20260202-120852")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = SentenceTransformer(args.model_path)

    # Fetch all products without embeddings
    conn = psycopg2.connect(args.database_url)
    cur = conn.cursor()

    cur.execute("""
        SELECT parent_asin, embedding_text
        FROM products
        WHERE embedding IS NULL AND embedding_text IS NOT NULL
        ORDER BY parent_asin
    """)
    rows = cur.fetchall()
    logger.info(f"Found {len(rows)} products to embed")

    if not rows:
        logger.info("Nothing to do")
        cur.close()
        conn.close()
        return

    asins = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    # Encode in batches
    all_embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Write embeddings back in batches
    update_sql = "UPDATE products SET embedding = %s::vector WHERE parent_asin = %s"
    batch_size = 500

    for i in tqdm(range(0, len(asins), batch_size), desc="Writing to DB"):
        batch_asins = asins[i:i + batch_size]
        batch_embs = all_embeddings[i:i + batch_size]

        data = [
            (str(emb.tolist()), asin)
            for asin, emb in zip(batch_asins, batch_embs)
        ]

        cur.executemany(update_sql, data)
        conn.commit()

    cur.close()
    conn.close()
    logger.info(f"Done. Embedded {len(asins)} products.")
    logger.info("Now create the IVFFlat index in Cloud SQL Studio:")
    logger.info("  CREATE INDEX idx_products_embedding ON products")
    logger.info("  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")


if __name__ == "__main__":
    main()
