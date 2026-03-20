import logging
import time

from fastapi import APIRouter
from app.models.schemas import RecommendRequest, RecommendResponse, ItemScore
from app.services.retrieval import retrieve_candidates
from app.services.reranking import rerank_items
from app.services.explanation import generate_explanation

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger(__name__)

@router.post("", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    t0 = time.perf_counter()
    candidates = retrieve_candidates(request.user_input)
    t1 = time.perf_counter()

    ranked_items = rerank_items(
        query=request.user_input,
        candidates=candidates,
        top_k=5,
    )
    t2 = time.perf_counter()

    explanation_input = build_explanation_input(
        user_query=request.user_input,
        ranked_items=ranked_items,
    )

    explanations_data = generate_explanation(explanation_input)
    t3 = time.perf_counter()

    latency = {
        "retrieval_ms": round((t1 - t0) * 1000, 1),
        "reranking_ms": round((t2 - t1) * 1000, 1),
        "explanation_ms": round((t3 - t2) * 1000, 1),
        "total_ms": round((t3 - t0) * 1000, 1),
    }
    logger.info(f"Latency breakdown: {latency}")

    explanations = explanations_data.get("explanations", [])
    explanation_map = {exp["item_id"]: exp["explanation"] for exp in explanations}
    formatted_recommendations = [
        {
            "item_id": item["item_id"],
            "item_name": item.get("title", "Unknown Item"),
            "score": item.get("score", 0.0),
            "explanation": explanation_map.get(item["item_id"]),
            "image": item.get("image", ""),
            "price": item.get("price", 0.0),
            "average_rating": item.get("average_rating", 0.0),
            "rating_number": int(item.get("rating_number", 0)),
            "store": item.get("store", ""),
        }
        for item in ranked_items
    ]

    return RecommendResponse(
        recommendations=formatted_recommendations,
        latency=latency,
    )
    
def build_explanation_input(user_query: str, ranked_items: list[dict]):
    return {
        "user_query": user_query,
        "items": [
            {
                "item_id": item["item_id"],
                "title": item.get("title", ""),
                "price": item.get("price", 0.0),
                "average_rating": item.get("average_rating", 0.0),
                "features": item.get("features", ""),
                "top_reviews": item.get("top_reviews", ""),
                "details": item.get("details", ""),
                "description": item.get("description", ""),
            }
            for item in ranked_items
        ]
    }