import type { RecommendRequest, RecommendResponse } from "../types/recommend";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchRecommend(
  payload: RecommendRequest
): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error("Recommendation API failed");
  }

  return res.json();
}