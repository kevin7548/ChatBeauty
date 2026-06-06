"""Integration test for ``POST /recommend``.

Exercises the full retrieve -> rerank -> explain path through the real route
orchestration, with the three service stages stubbed at their function boundary
(see ``conftest.py``). This is the CI gate for the recommendation endpoint.
"""

from app.models.schemas import ItemScore

LATENCY_KEYS = {"retrieval_ms", "reranking_ms", "explanation_ms", "total_ms"}


def test_recommend_returns_top5_with_explanations(client):
    resp = client.post("/recommend", json={"user_input": "hydrating serum"})

    assert resp.status_code == 200
    body = resp.json()

    # Top-5 is hardcoded in the route.
    recs = body["recommendations"]
    assert len(recs) == 5

    # Every recommendation conforms to the ItemScore schema.
    for rec in recs:
        ItemScore(**rec)

    # Explanations are joined back onto items by item_id (the route builds an
    # explanation_map keyed on item_id).
    for rec in recs:
        assert rec["explanation"] == f"추천 이유 {rec['item_id']}"

    # Latency breakdown is reported with all four stages.
    assert set(body["latency"].keys()) == LATENCY_KEYS

    # LatencyMiddleware stamps the total-latency header on every response.
    assert "X-Total-Latency-Ms" in resp.headers


def test_recommend_blank_input_returns_empty(client):
    """Blank query short-circuits retrieval -> empty recommendations (api-spec)."""
    resp = client.post("/recommend", json={"user_input": "   "})

    assert resp.status_code == 200
    assert resp.json()["recommendations"] == []


def test_recommend_requires_user_input(client):
    """Missing the required field is a 422 validation error."""
    resp = client.post("/recommend", json={})

    assert resp.status_code == 422
