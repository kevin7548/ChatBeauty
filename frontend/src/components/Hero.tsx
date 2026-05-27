import "./Hero.css";

const EXAMPLE_QUERIES = [
  "건조한 겨울철 보습 크림 추천해주세요",
  "얇은 머리카락에 볼륨감을 주는 샴푸",
  "민감성 피부용 선크림",
  "I need a gentle cleanser for acne-prone skin",
];

interface HeroProps {
  onChipClick: (text: string) => void;
}

export function Hero({ onChipClick }: HeroProps) {
  return (
    <div className="hero">
      <h1>
        어떤 뷰티 제품을
        <br />
        찾고 계신가요?
      </h1>
      <p>상황을 설명해 주시면, AI가 딱 맞는 제품을 추천해 드릴게요.</p>
      <div className="example-chips">
        {EXAMPLE_QUERIES.map((text) => (
          <button key={text} className="chip" onClick={() => onChipClick(text)}>
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}
