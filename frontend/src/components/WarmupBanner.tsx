import "./WarmupBanner.css";

export type WarmupStatus = "warming" | "warm" | "failed";

interface WarmupBannerProps {
  status: WarmupStatus;
}

export function WarmupBanner({ status }: WarmupBannerProps) {
  if (status === "warm") return null;

  if (status === "failed") {
    return (
      <div className="warmup-banner warmup-banner-failed">
        서버 준비 중 문제가 있었어요. 검색은 가능하지만 첫 요청이 느릴 수 있어요.
      </div>
    );
  }

  return (
    <div className="warmup-banner">
      <span className="warmup-dot" />
      서버를 준비 중이에요. 첫 요청은 최대 1분 정도 걸릴 수 있어요.
    </div>
  );
}
