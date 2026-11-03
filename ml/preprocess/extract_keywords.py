"""
Extract keywords and use scenarios from All_Beauty.jsonl using LLaMA via vLLM.

Usage:
    python ml/preprocess/extract_keywords.py
    python ml/preprocess/extract_keywords.py MAX_ITEMS=100
    python ml/preprocess/extract_keywords.py MODEL=meta-llama/Llama-3.1-8B-Instruct GPU=0,1
    python ml/preprocess/extract_keywords.py BATCH_SIZE=32

Record only: utils/llm_client.py (vLLM wrapper) is not in the repo; restore it with
    git show f568da2^:backend/ml/utils/llm_client.py
"""

import json
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_client import LLMClient


class KeywordExtractor:
    """Extract keywords and scenarios from product reviews using LLM"""

    def __init__(self, client: LLMClient):
        self.client = client

    def _build_prompt(self, product_title: str, review_text: str) -> str:
        """Build prompt for contextual keyword extraction with Llama 3 formatting"""

        system_content = """You are a product recommendation expert. Analyze this product review to extract contextual keywords that describe:
- Product features and characteristics
- Benefits and improvements provided
- Use cases and situations where it's useful
- Target user types and their needs

Combine all these aspects into a single efficient list of keywords that capture WHO would use this product, WHEN, and WHY.

Please respond with ONLY valid JSON, no markdown, no extra text:
{
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}"""

        user_content = f"""Product: {product_title}
Review: {review_text}"""

        # Llama 3.1 Chat Template
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return formatted_prompt

    def extract_keywords_batch(
        self, items: List[Dict]
    ) -> List[List[str]]:
        """Extract keywords from multiple reviews (batched for vLLM efficiency)"""
        prompts = [
            self._build_prompt(item["title"], item["text"])
            for item in items
        ]

        responses = self.client.generate_batch(prompts)

        results = []
        for response in responses:
            if response:
                try:
                    text = response
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    parsed = json.loads(text.strip())
                    results.append(parsed.get("keywords", []))
                except json.JSONDecodeError:
                    results.append([])
            else:
                results.append([])

        return results

    def process_jsonl_file(
        self,
        input_file: str,
        output_file: str,
        max_items: Optional[int] = None,
        batch_size: int = 64
    ) -> None:
        """Process all products in a JSONL file and extract keywords"""
        input_path = Path(input_file).resolve()
        output_path = Path(output_file).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            return

        # Load all items first
        items_to_process = []
        with open(input_path, "r") as inf:
            for line in inf:
                try:
                    product = json.loads(line)
                    text = product.get("text", "")
                    if text and len(text.strip()) >= 10:
                        items_to_process.append(product)
                        if max_items and len(items_to_process) >= max_items:
                            break
                except json.JSONDecodeError:
                    continue

        print(f"Processing {len(items_to_process)} items from {input_file}")
        print(f"Results will be saved to {output_file}")
        print(f"Batch size: {batch_size}\n")

        processed = 0
        successful = 0

        with open(output_path, "w") as outf:
            for i in range(0, len(items_to_process), batch_size):
                batch = items_to_process[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1} ({len(batch)} items)...", end=" ", flush=True)

                batch_items = [
                    {"title": p.get("title", "Unknown"), "text": p.get("text", "")}
                    for p in batch
                ]
                keywords_list = self.extract_keywords_batch(batch_items)

                batch_successful = 0
                for j, (product, keywords) in enumerate(zip(batch, keywords_list)):
                    final_keywords = keywords if keywords is not None else []

                    output_record = {
                        "asin": product.get("asin", ""),
                        "title": product.get("title", ""),
                        "review_text": product.get("text", "")[:500],
                        "rating": product.get("rating"),
                        "keywords": keywords,
                    }
                    outf.write(json.dumps(output_record) + "\n")

                    if final_keywords:
                        batch_successful += 1
                    successful += 1

                processed += len(batch)
                print(f"OK ({batch_successful}/{len(batch)})")

                if processed % 100 == 0:
                    print(f"   Progress: {processed}/{len(items_to_process)} ({successful} successful)")

        print(f"\n" + "=" * 60)
        print(f"Processing Complete!")
        print(f"  Total processed: {processed}")
        print(f"  Successful: {successful}")
        print(f"  Results saved to: {output_path}")
        print("=" * 60)


def parse_args() -> Dict[str, str]:
    """Parse KEY=VALUE arguments from command line"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def main():
    """
    Usage:
        python ml/preprocess/extract_keywords.py
        python ml/preprocess/extract_keywords.py GPU=0,1
        python ml/preprocess/extract_keywords.py MAX_ITEMS=100
        python ml/preprocess/extract_keywords.py BATCH_SIZE=32
    """
    cli_args = parse_args()

    MODEL = cli_args.get('MODEL')
    GPU = cli_args.get('GPU')

    gpu_ids = None
    if GPU:
        gpu_ids = [int(g.strip()) for g in GPU.split(',')]

    # File paths
    script_dir = Path(__file__).parent.parent  # ml/
    INPUT_FILE = script_dir / "data/raw/All_Beauty.jsonl"
    OUTPUT_FILE = script_dir / "data/processed/keywords_output.jsonl"

    MAX_ITEMS = int(cli_args['MAX_ITEMS']) if 'MAX_ITEMS' in cli_args else None
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 64))

    print("=" * 60)
    print("LLaMA Keyword & Scenario Extractor (vLLM)")
    print("=" * 60)
    if MODEL:
        print(f"Model: {MODEL}")
    if gpu_ids:
        print(f"GPUs: {gpu_ids}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    client_kwargs = {"backend": "vllm"}
    if MODEL:
        client_kwargs["model"] = MODEL
    if gpu_ids:
        client_kwargs["gpu_ids"] = gpu_ids

    client = LLMClient(**client_kwargs)

    if not client.verify_connection():
        print("Failed to connect to vLLM backend. Exiting.")
        return

    extractor = KeywordExtractor(client)
    extractor.process_jsonl_file(
        input_file=str(INPUT_FILE),
        output_file=str(OUTPUT_FILE),
        max_items=MAX_ITEMS,
        batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    main()
