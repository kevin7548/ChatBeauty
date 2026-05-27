"""
PostgreSQL batch write sink using psycopg2.

Demonstrates a custom Beam DoFn that batches elements and writes
to a relational database with ON CONFLICT for idempotent upserts.
"""

import logging
from typing import List

import apache_beam as beam
from apache_beam.metrics import Metrics

from ml.pipeline.schemas import ProductRecord

logger = logging.getLogger(__name__)

UPSERT_SQL = """
INSERT INTO products (
    parent_asin, title, embedding_text, description, features,
    top_reviews, details, image, store, price, average_rating,
    rating_number, review_cnt, vp_review_cnt, vp_ratio,
    recent_review_cnt, avg_rating, rating_std, avg_review_len,
    log_median_price, price_cnt
) VALUES %s
ON CONFLICT (parent_asin) DO UPDATE SET
    title = EXCLUDED.title,
    embedding_text = EXCLUDED.embedding_text,
    description = EXCLUDED.description,
    features = EXCLUDED.features,
    top_reviews = EXCLUDED.top_reviews,
    details = EXCLUDED.details,
    image = EXCLUDED.image,
    store = EXCLUDED.store,
    price = EXCLUDED.price,
    average_rating = EXCLUDED.average_rating,
    rating_number = EXCLUDED.rating_number,
    review_cnt = EXCLUDED.review_cnt,
    vp_review_cnt = EXCLUDED.vp_review_cnt,
    vp_ratio = EXCLUDED.vp_ratio,
    recent_review_cnt = EXCLUDED.recent_review_cnt,
    avg_rating = EXCLUDED.avg_rating,
    rating_std = EXCLUDED.rating_std,
    avg_review_len = EXCLUDED.avg_review_len,
    log_median_price = EXCLUDED.log_median_price,
    price_cnt = EXCLUDED.price_cnt
"""


class WriteToPostgreSQL(beam.DoFn):
    """
    Batch-write ProductRecords to PostgreSQL.

    Uses psycopg2's execute_values for efficient bulk inserts.
    ON CONFLICT ensures idempotent writes (safe to re-run pipeline).
    """

    def __init__(self, database_url: str, batch_size: int = 1000):
        self.database_url = database_url
        self.batch_size = batch_size
        self.rows_written = Metrics.counter("sink", "rows_written")
        self.write_errors = Metrics.counter("sink", "write_errors")
        self.batch_count = Metrics.counter("sink", "batches_written")

    def setup(self):
        import psycopg2

        self.conn = psycopg2.connect(self.database_url)
        self.conn.autocommit = False
        self.buffer: List[tuple] = []

    def process(self, record: ProductRecord):
        self.buffer.append((
            record.parent_asin,
            record.title,
            record.embedding_text,
            record.description,
            record.features,
            record.top_reviews,
            record.details,
            record.image,
            record.store,
            record.price,
            record.average_rating,
            record.rating_number,
            record.review_cnt,
            record.vp_review_cnt,
            record.vp_ratio,
            record.recent_review_cnt,
            record.avg_rating,
            record.rating_std,
            record.avg_review_len,
            record.log_median_price,
            record.price_cnt,
        ))

        if len(self.buffer) >= self.batch_size:
            self._flush()

    def finish_bundle(self):
        if self.buffer:
            self._flush()

    def _flush(self):
        from psycopg2.extras import execute_values

        try:
            cur = self.conn.cursor()
            execute_values(cur, UPSERT_SQL, self.buffer, page_size=self.batch_size)
            self.conn.commit()
            self.rows_written.inc(len(self.buffer))
            self.batch_count.inc()
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.write_errors.inc(len(self.buffer))
            logger.error(f"PostgreSQL write failed: {e}")
        finally:
            self.buffer.clear()

    def teardown(self):
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
