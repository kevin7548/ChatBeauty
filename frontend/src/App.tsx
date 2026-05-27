import { useEffect, useRef, useState } from "react";
import { fetchRecommend, warmUp } from "./api/recommend";
import type { RecommendResponse } from "./types/recommend";
import { useLoadingMessage } from "./hooks/useLoadingMessage";
import { Header } from "./components/Header";
import { WarmupBanner, type WarmupStatus } from "./components/WarmupBanner";
import { Hero } from "./components/Hero";
import { ProductCard } from "./components/ProductCard";
import { SkeletonCard } from "./components/SkeletonCard";
import { SearchFooter } from "./components/SearchFooter";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warmupStatus, setWarmupStatus] = useState<WarmupStatus>("warming");
  const warmupStartedAt = useRef<number>(Date.now());
  const loadingMessage = useLoadingMessage(isLoading);

  useEffect(() => {
    let cancelled = false;
    warmupStartedAt.current = Date.now();
    warmUp()
      .then(() => {
        if (!cancelled) setWarmupStatus("warm");
      })
      .catch(() => {
        if (!cancelled) setWarmupStatus("failed");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = async (input?: string) => {
    const q = input ?? query;
    if (!q.trim() || isLoading) return;

    setSubmittedQuery(q);
    setResult(null);
    setError(null);
    setIsLoading(true);
    try {
      const data = await fetchRecommend({
        user_input: q,
        top_k: 5,
      });
      setResult(data);
    } catch {
      setError("추천 결과를 가져오지 못했어요. 잠시 후 다시 시도해 주세요.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleChipClick = (text: string) => {
    setQuery(text);
    handleSubmit(text);
  };

  const handleReset = () => {
    setResult(null);
    setQuery("");
    setSubmittedQuery("");
  };

  const showHero = !isLoading && !result && !error;

  return (
    <div className="app">
      <Header onLogoClick={handleReset} />
      <WarmupBanner status={warmupStatus} />

      <main className="main">
        {showHero && <Hero onChipClick={handleChipClick} />}

        {isLoading && (
          <div className="results">
            <div className="query-display">
              <div className="query-label">나의 상황</div>
              <div className="query-text">{submittedQuery}</div>
            </div>
            <div className="results-header">
              <h2>{loadingMessage}</h2>
            </div>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        )}

        {error && !isLoading && (
          <div className="results">
            <div className="query-display">
              <div className="query-label">나의 상황</div>
              <div className="query-text">{submittedQuery}</div>
            </div>
            <div className="error-state">
              <p className="error-message">{error}</p>
              <button
                className="retry-btn"
                onClick={() => handleSubmit(submittedQuery)}
              >
                다시 시도
              </button>
            </div>
          </div>
        )}

        {result && (
          <div className="results">
            <div className="query-display">
              <div className="query-label">나의 상황</div>
              <div className="query-text">{submittedQuery}</div>
            </div>
            <div className="results-header">
              <h2>추천 결과</h2>
              <span className="results-count">
                {result.recommendations.length}개 제품
              </span>
            </div>
            {result.recommendations.map((item, index) => (
              <ProductCard
                key={item.item_id}
                item={item}
                rank={index + 1}
              />
            ))}
          </div>
        )}
      </main>

      <SearchFooter
        query={query}
        onQueryChange={setQuery}
        onSubmit={() => handleSubmit()}
        isLoading={isLoading}
      />
    </div>
  );
}

export default App;
