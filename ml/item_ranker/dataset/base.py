import json
from dataclasses import dataclass
from typing import List, Optional, Generator


@dataclass
class Candidate:
    item_id: str
    retrieval_score: float
    metadata: dict


@dataclass
class RerankSample:
    query_keywords: List[str]
    candidates: List[Candidate]
    labels: Optional[List[float]] = None


def iter_samples(
    data_path: str,
    limit: Optional[int] = None
) -> Generator[RerankSample, None, None]:

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            row = json.loads(line)

            target_id = row.get("parent_asin")

            keywords = [
                str(k).lower()
                for k in row.get("keywords", [])
                if isinstance(k, str)
            ]

            candidates = []
            labels = []

            for c in row.get("candidates", []):
                item_id = c["item_asin"]

                candidates.append(
                    Candidate(
                        item_id=item_id,
                        retrieval_score=float(c.get("score", 0.0)),
                        metadata=c,
                    )
                )

                if target_id is not None:
                    labels.append(1.0 if item_id == target_id else 0.0)

            yield RerankSample(
                query_keywords=keywords,
                candidates=candidates,
                labels=labels if target_id else None,
            )
