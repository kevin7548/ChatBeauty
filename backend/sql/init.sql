CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS products (
    parent_asin       VARCHAR(20) PRIMARY KEY,
    title             TEXT NOT NULL,
    embedding_text    TEXT,
    embedding         VECTOR(1024),

    -- metadata for explanation
    description       TEXT,
    features          TEXT,
    top_reviews       TEXT,
    details           TEXT,
    image             TEXT,
    store             VARCHAR(255),

    -- reranking features (9 of 10; retrieval_score computed at query time)
    review_cnt        INTEGER DEFAULT 0,
    vp_review_cnt     INTEGER DEFAULT 0,
    vp_ratio          FLOAT DEFAULT 0.0,
    recent_review_cnt INTEGER DEFAULT 0,
    avg_rating        FLOAT DEFAULT 0.0,
    rating_std        FLOAT DEFAULT 0.0,
    avg_review_len    FLOAT DEFAULT 0.0,
    log_median_price  FLOAT DEFAULT 0.0,
    price_cnt         INTEGER DEFAULT 0,

    -- filter fields
    price             NUMERIC,
    average_rating    FLOAT,
    rating_number     INTEGER DEFAULT 0
);

-- pgvector index for cosine similarity search (build after data is loaded)
-- IVFFlat requires data to be present; run this after loading embeddings:
-- CREATE INDEX idx_products_embedding ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_products_price ON products (price);
CREATE INDEX IF NOT EXISTS idx_products_rating ON products (average_rating);
CREATE INDEX IF NOT EXISTS idx_products_store ON products (store);
