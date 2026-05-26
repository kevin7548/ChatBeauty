import type { ItemScore } from "../types/recommend";
import "./ProductCard.css";

function renderStars(rating: number) {
  const full = Math.floor(rating);
  const empty = 5 - full;
  return "★".repeat(full) + "☆".repeat(empty);
}

interface ProductCardProps {
  item: ItemScore;
  rank: number;
}

export function ProductCard({ item, rank }: ProductCardProps) {
  return (
    <div className="product-card">
      <div className="card-inner">
        <div className="card-image">
          {item.image ? (
            <img
              src={item.image}
              alt={item.item_name}
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
                (e.target as HTMLImageElement).parentElement!.innerHTML =
                  '<div class="image-placeholder">No Image</div>';
              }}
            />
          ) : (
            <div className="image-placeholder">No Image</div>
          )}
        </div>
        <div className="card-body">
          <div className="card-rank-row">
            <div className={`rank-badge rank-${rank}`}>{rank}</div>
            {item.store && <span className="store-tag">{item.store}</span>}
          </div>
          <div className="product-name">{item.item_name}</div>
          <div className="meta-row">
            {item.price != null && item.price > 0 && (
              <span className="price">${item.price.toFixed(2)}</span>
            )}
            {item.average_rating != null && item.average_rating > 0 && (
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
            <div className="explanation-label">이 제품을 추천하는 이유</div>
            <div className="explanation-text">
              {item.explanation ?? "설명을 생성하지 못했어요."}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
