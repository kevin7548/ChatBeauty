import { useState } from "react";
import { fetchRecommend } from "./api/recommend";
import type { RecommendResponse } from "./types/recommend";
import "./App.css";

const EXAMPLE_QUERIES = [
  "건조한 겨울철 보습 크림 추천해주세요",
  "얇은 머리카락에 볼륨감을 주는 샴푸",
  "민감성 피부용 선크림",
  "I need a gentle cleanser for acne-prone skin",
];

function renderStars(rating: number) {
  const full = Math.floor(rating);
  const empty = 5 - full;
  return "\u2605".repeat(full) + "\u2606".repeat(empty);
}

function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <div className="skeleton-inner">
        <div className="skeleton-img" />
        <div className="skeleton-body">
          <div className="skeleton-line w60" />
          <div className="skeleton-line w80" />
          <div className="skeleton-line w40" />
          <div className="skeleton-explanation">
            <div className="skeleton-line w100" style={{ marginBottom: 6 }} />
            <div className="skeleton-line w80" />
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const showHero = !isLoading && !result && !error;

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div
          className="logo"
          onClick={() => {
            setResult(null);
            setQuery("");
            setSubmittedQuery("");
          }}
          style={{ cursor: "pointer" }}
        >
          <div className="logo-text">
            <span className="chat">Chat</span>
            <span className="beauty">Beauty</span>
          </div>
          <span className="sparkle">{"\u2728"}</span>
        </div>
        <span className="header-badge">AI Powered</span>
      </header>

      {/* MAIN */}
      <main className="main">
        {/* HERO / EMPTY STATE */}
        {showHero && (
          <div className="hero">
            <h1>
              어떤 뷰티 제품을
              <br />
              찾고 계신가요?
            </h1>
            <p>상황을 설명해 주시면, AI가 딱 맞는 제품을 추천해 드릴게요.</p>
            <div className="example-chips">
              {EXAMPLE_QUERIES.map((text) => (
                <button
                  key={text}
                  className="chip"
                  onClick={() => handleChipClick(text)}
                >
                  {text}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* LOADING STATE */}
        {isLoading && (
          <div className="results">
            <div className="query-display">
              <div className="query-label">나의 상황</div>
              <div className="query-text">{submittedQuery}</div>
            </div>
            <div className="results-header">
              <h2>추천 결과를 가져오고 있어요...</h2>
            </div>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        )}

        {/* ERROR STATE */}
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

        {/* RESULTS STATE */}
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
              <div key={item.item_id} className="product-card">
                <div className="card-inner">
                  <div className="card-image">
                    {item.image ? (
                      <img
                        src={item.image}
                        alt={item.item_name}
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = "none";
                          (
                            e.target as HTMLImageElement
                          ).parentElement!.innerHTML =
                            '<div class="image-placeholder">No Image</div>';
                        }}
                      />
                    ) : (
                      <div className="image-placeholder">No Image</div>
                    )}
                  </div>
                  <div className="card-body">
                    <div className="card-rank-row">
                      <div className={`rank-badge rank-${index + 1}`}>
                        {index + 1}
                      </div>
                      {item.store && (
                        <span className="store-tag">{item.store}</span>
                      )}
                    </div>
                    <div className="product-name">{item.item_name}</div>
                    <div className="meta-row">
                      {item.price != null && item.price > 0 && (
                        <span className="price">${item.price.toFixed(2)}</span>
                      )}
                      {item.average_rating != null &&
                        item.average_rating > 0 && (
                          <span className="rating">
                            <span className="stars">
                              {renderStars(item.average_rating)}
                            </span>
                            {item.average_rating.toFixed(1)}
                            {item.rating_number != null && (
                              <span className="rating-count">
                                ({item.rating_number.toLocaleString()})
                              </span>
                            )}
                          </span>
                        )}
                    </div>
                    <div className="explanation">
                      <div className="explanation-label">
                        이 제품을 추천하는 이유
                      </div>
                      <div className="explanation-text">
                        {item.explanation ?? "설명을 생성하지 못했어요."}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* INPUT FOOTER */}
      <footer className="footer">
        <div className="input-container">
          <textarea
            className="input-box"
            value={query}
            placeholder="어떤 뷰티 상품을 찾으시나요? 상황을 자유롭게 설명해 주세요..."
            rows={1}
            onChange={(e) => {
              setQuery(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = e.target.scrollHeight + "px";
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
          />
          <button
            className="send-btn"
            onClick={() => handleSubmit()}
            disabled={isLoading}
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          </button>
        </div>
        <div className="footer-hint">
          Enter to send &middot; Shift+Enter for new line
        </div>
      </footer>
    </div>
  );
}

export default App;
