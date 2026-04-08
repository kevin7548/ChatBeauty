"""
Shared resources for the retrieval service.

Loads the BGE-M3 model once at startup and provides a PostgreSQL
connection pool for pgvector similarity search.
"""

import os
import logging

import psycopg2
from psycopg2 import pool
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = os.environ.get(
    "BGE_MODEL_PATH",
    str(BASE_DIR / "ml" / "model-gcs" / "retrieval" / "bge-m3-finetuned-20260202-120852"),
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty",
)

# Load embedding model once at startup
model = SentenceTransformer(MODEL_PATH)

# PostgreSQL connection pool (min 1, max 5 connections)
# TCP keepalives help detect dead sockets faster when Cloud Run pauses
# instances between requests.
db_pool = pool.ThreadedConnectionPool(
    1,
    5,
    DATABASE_URL,
    keepalives=1,
    keepalives_idle=30,
    keepalives_interval=10,
    keepalives_count=3,
)


def get_db_connection():
    """
    Get a live connection from the pool.

    Cloud Run pauses instances between requests, which can silently kill
    pooled TCP connections. We run a cheap liveness check (SELECT 1) before
    handing the connection back to the caller; if it fails we discard the
    dead connection and fetch a fresh one.
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return conn
    except psycopg2.OperationalError:
        logger.warning("Dead DB connection detected, replacing it")
        try:
            db_pool.putconn(conn, close=True)
        except Exception:
            pass
        return db_pool.getconn()


def release_db_connection(conn):
    """Return a connection to the pool."""
    db_pool.putconn(conn)
