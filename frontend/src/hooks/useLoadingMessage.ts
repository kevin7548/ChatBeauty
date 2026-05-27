import { useEffect, useState } from "react";

export function useLoadingMessage(isLoading: boolean): string {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!isLoading) {
      setElapsed(0);
      return;
    }
    const start = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 500);
    return () => clearInterval(id);
  }, [isLoading]);

  if (elapsed < 5) return "추천 결과를 가져오고 있어요...";
  if (elapsed < 15) return "상품을 검색하고 있어요...";
  if (elapsed < 30) return "서버를 깨우고 있어요... 잠시만 기다려 주세요";
  if (elapsed < 50) return "추천 모델을 불러오는 중이에요...";
  return "거의 다 됐어요...";
}
