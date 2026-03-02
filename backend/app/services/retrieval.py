from app.services.retrieval_resources import model, collection

def retrieve_candidates(query: str, n: int = 100):
    if not query or not query.strip():
        return []

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=n,
        include=["distances", "metadatas", "documents"] 
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    candidates = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i] or {}
        distance = results["distances"][0][i]

        candidates.append({
            "item_id": results["ids"][0][i],
            "score": round(1.0 - distance, 6),
            "title": meta.get("title", ""),
            "price": meta.get("price", 0.0),
            "average_rating": meta.get("average_rating", 0.0),
            "rating_number": meta.get("rating_number", 0),
            "store": meta.get("store", ""),
            "categories": meta.get("categories", ""),
            "features": meta.get("features", []),
            "description_summary": meta.get("description_summary", ""),
            "review_keywords": meta.get("review_keywords", []),
            "top_reviews": meta.get("top_reviews", ""),
            "description": meta.get("description", ""),
            "details": meta.get("details", ""),
            "image": meta.get("image", ""),
        })

    return candidates
