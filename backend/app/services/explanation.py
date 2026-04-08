import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("환경 변수 GEMINI_API_KEY 가 설정되지 않았습니다.")

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
너는 쇼핑 추천 전문가야. 리랭킹된 상품 리스트와 사용자의 검색어를 분석하여, 사용자에게 이 상품이 왜 추천되었는지 '데이터에 기반해' 설명하는 역할을 수행한다.

[중요 규칙]
- 제공된 정보(상품명, 가격, 평점, 카테고리 등) 내에서만 설명할 것. 추측 금지.
- '알고리즘', '리랭킹', '점수', '임베딩' 등 기술적 용어 절대 사용 금지.
- 마케팅 수식어(최고의, 환상적인 등)를 배제하고 객관적인 사실 위주로 작성할 것.
- 실제 구매자 리뷰(top_reviews)를 인용하여 신뢰도를 높일 것
- 각 상품의 설명은 1~2문장으로 간결하게 작성할 것.
- 5개에 상품에 대한 설명을 요청하니 explanation은 5개가 꼭 응답되어야함

[설명 가이드라인]
1. 사용자의 검색 키워드가 상품명의 어느 부분과 일치하는지 언급.
2. 실제 리뷰 내용을 인용하여 근거 제시

[출력 형식]
- 반드시 아래 JSON 구조로만 응답하고, 다른 텍스트는 포함하지 말 것.
- 모든 설명(explanation)은 반드시 한국어로 작성할 것

{
  "explanations": [
    {
      "item_id": "상품 ID",
      "explanation": "추천 이유 설명"
    }
  ]
}
"""

_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT,
    generation_config=genai.GenerationConfig(
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=8192,
        response_mime_type="application/json",
    ),
)


def generate_explanation(explanation_input: dict) -> dict:
    """
    Generate explanations for recommended items using Gemini 2.0 Flash.
    Signature and return format identical to the previous HyperCLOVA X version.
    """
    user_message = json.dumps(explanation_input, ensure_ascii=False)

    try:
        response = _model.generate_content(user_message)
        raw_text = response.text

        try:
            parsed = json.loads(raw_text)
            return parsed
        except json.JSONDecodeError as e:
            finish_reason = None
            output_tokens = None
            try:
                finish_reason = response.candidates[0].finish_reason
            except Exception:
                pass
            try:
                output_tokens = response.usage_metadata.candidates_token_count
            except Exception:
                pass
            logger.warning(
                f"JSON parse failed: {e} | finish_reason={finish_reason} "
                f"| output_tokens={output_tokens} | raw[:200]={raw_text[:200]}"
            )
            return {"explanations": [{"item_id": "all", "explanation": raw_text}]}

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {"explanations": []}
