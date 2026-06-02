"""Test fixtures for the FastAPI backend.

Importing ``app.main`` transitively triggers heavy import-time side effects that
require live infrastructure:

* ``app.services.retrieval_resources`` loads the BGE-M3 model and opens a psycopg2
  connection pool.
* ``app.services.reranking`` ``pickle.load``s the LightGBM model from disk.
* ``app.services.explanation`` raises ``RuntimeError`` unless ``GEMINI_API_KEY`` is set
  and builds a Gemini client.

To run the integration test offline and deterministically, we replace those service
modules with lightweight stubs in ``sys.modules`` *before* the app is ever imported.
The route module (``app.api.routes.recommend``) stays real, so the test still exercises
the full retrieve -> rerank -> explain orchestration and the response formatting — we
only mock at the service-function boundary (Gemini included, so no network calls).
"""

import sys
import types

import pytest


# --- Deterministic stub data ------------------------------------------------------

def _make_candidate(i: int) -> dict:
    """A retrieval candidate with the full metadata the route/explanation read."""
    return {
        "item_id": f"ASIN{i:04d}",
        "title": f"Test Product {i}",
        "price": 10.0 + i,
        "average_rating": 4.2,
        "rating_number": 100 + i,
        "store": "TestStore",
        "features": "hydrating, lightweight",
        "description": "A test beauty product.",
        "top_reviews": "Reviewers liked it.",
        "details": "details blob",
        "image": f"https://example.com/{i}.jpg",
        "embedding_text": "embedding text",
        # descending so the stub reranker's order is deterministic
        "score": round(0.9 - i * 0.001, 6),
    }


DEFAULT_CANDIDATES = [_make_candidate(i) for i in range(100)]


def _stub_retrieve_candidates(query: str, n: int = 100):
    if not query or not query.strip():
        return []
    return [dict(c) for c in DEFAULT_CANDIDATES[:n]]


def _stub_rerank_items(query: str, candidates: list[dict], top_k: int):
    if not candidates:
        return []
    return [dict(c) for c in candidates[:top_k]]


def _stub_generate_explanation(explanation_input: dict) -> dict:
    """Mock Gemini: return one explanation per item, keyed by item_id."""
    return {
        "explanations": [
            {"item_id": item["item_id"], "explanation": f"추천 이유 {item['item_id']}"}
            for item in explanation_input["items"]
        ]
    }


# --- Install service stubs before the app is imported -----------------------------

def _install_service_stubs() -> None:
    retrieval = types.ModuleType("app.services.retrieval")
    retrieval.retrieve_candidates = _stub_retrieve_candidates

    reranking = types.ModuleType("app.services.reranking")
    reranking.rerank_items = _stub_rerank_items

    explanation = types.ModuleType("app.services.explanation")
    explanation.generate_explanation = _stub_generate_explanation

    # The real services import this at module load; stub it so nothing tries to
    # build the model or open a DB pool even if a real service slips through.
    resources = types.ModuleType("app.services.retrieval_resources")
    resources.model = None
    resources.get_db_connection = lambda: None
    resources.release_db_connection = lambda conn: None

    sys.modules.update(
        {
            "app.services.retrieval": retrieval,
            "app.services.reranking": reranking,
            "app.services.explanation": explanation,
            "app.services.retrieval_resources": resources,
        }
    )


# Run at collection time, before any test module imports the app.
_install_service_stubs()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
