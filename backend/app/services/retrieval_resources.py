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
db_pool = pool.ThreadedConnectionPool(1, 5, DATABASE_URL)


def get_db_connection():
    """Get a connection from the pool."""
    return db_pool.getconn()


def release_db_connection(conn):
    """Return a connection to the pool."""
    db_pool.putconn(conn)
