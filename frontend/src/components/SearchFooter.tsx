import "./SearchFooter.css";

interface SearchFooterProps {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export function SearchFooter({
  query,
  onQueryChange,
  onSubmit,
  isLoading,
}: SearchFooterProps) {
  return (
    <footer className="footer">
      <div className="input-container">
        <textarea
          className="input-box"
          value={query}
          placeholder="어떤 뷰티 상품을 찾으시나요? 상황을 자유롭게 설명해 주세요..."
          rows={1}
          onChange={(e) => {
            onQueryChange(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = e.target.scrollHeight + "px";
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit();
            }
          }}
        />
        <button
          className="send-btn"
          onClick={onSubmit}
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
  );
}
