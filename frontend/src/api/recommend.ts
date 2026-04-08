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

/**
 * Warm up the Cloud Run backend.
 *
 * The backend runs with min-instances=0, so after a period of inactivity
 * the instance is shut down. The first request after that pays the full
 * cold-start cost (~60s) because the embedding model has to load into RAM.
 *
 * We call /health as soon as the page loads so the instance wakes up
 * while the user is still reading the hero section. By the time they
 * actually submit a query, the server is usually already warm.
 *
 * Resolves on success, rejects on failure (the caller can ignore errors).
 */
export async function warmUp(): Promise<void> {
  const res = await fetch(`${API_BASE}/health`, { method: "GET" });
  if (!res.ok) {
    throw new Error("Warmup failed");
  }
}