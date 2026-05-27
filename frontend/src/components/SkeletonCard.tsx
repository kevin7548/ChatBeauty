import "./SkeletonCard.css";

export function SkeletonCard() {
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
