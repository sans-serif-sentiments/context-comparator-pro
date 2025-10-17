"""Heuristic quality scoring for model generations."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_WEIGHTS = {
    "faithfulness": 0.4,
    "completeness": 0.3,
    "structure": 0.2,
    "conciseness": 0.1,
}

HALLUCINATION_PHRASES = [
    "i think",
    "i believe",
    "it seems",
    "probably",
    "maybe",
    "might be",
    "cannot access the internet",
    "as an ai language model",
]


@dataclass(frozen=True)
class RubricRequest:
    """Input payload for scoring."""

    prompt: str
    response: str
    expected_structure: Optional[str] = None
    keywords: Optional[Sequence[str]] = None
    target_word_count: int = 200
    weights: Optional[Mapping[str, float]] = None


def normalise_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _extract_keywords(prompt: str, limit: int = 6) -> List[str]:
    words = re.findall(r"[A-Za-z]{5,}", prompt.lower())
    most_common = [word for word, _ in Counter(words).most_common(limit)]
    return most_common


def score_faithfulness(response: str) -> float:
    lowered = response.lower()
    penalty = sum(lowered.count(phrase) for phrase in HALLUCINATION_PHRASES)
    score = max(0.0, 5.0 - penalty)
    return round(score, 2)


def score_completeness(response: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 5.0
    lowered = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lowered)
    ratio = hits / len(keywords)
    score = ratio * 5.0
    return round(score, 2)


def score_structure(response: str, expected: Optional[str]) -> float:
    if not expected:
        return 5.0
    lowered = response.lower()
    if expected == "bullet_list":
        score = 5.0 if lowered.count("\n-") >= 3 or lowered.count("\n*") >= 3 else 2.5
    elif expected == "numbered_list":
        score = 5.0 if re.search(r"\n\d+\.", response) else 2.0
    elif expected == "json":
        score = 5.0 if response.strip().startswith("{") and response.strip().endswith("}") else 0.0
    elif expected == "paragraph_then_bullets":
        score = 5.0 if ("\n\n" in response and lowered.count("\n-") >= 2) else 2.5
    elif expected == "headings_and_code":
        score = 5.0 if "overview" in lowered and "how it works" in lowered else 2.5
    else:
        score = 3.0
    return round(score, 2)


def score_conciseness(response: str, target: int) -> float:
    words = re.findall(r"\w+", response)
    count = len(words)
    if count <= target:
        return 5.0
    overshoot_ratio = (count - target) / max(target, 1)
    penalty = min(5.0, overshoot_ratio * 5.0)
    score = max(0.0, 5.0 - penalty)
    return round(score, 2)


def score_similarity(responses: Iterable[str]) -> Optional[float]:
    """Optional similarity helper for variance analysis."""

    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    responses = list(responses)
    if len(responses) < 2:
        return None
    pairs = []
    for idx in range(len(responses)):
        for jdx in range(idx + 1, len(responses)):
            pairs.append(fuzz.token_sort_ratio(responses[idx], responses[jdx]))
    if not pairs:
        return None
    return round(sum(pairs) / len(pairs), 2)


def score_response(request: RubricRequest) -> Dict[str, float]:
    """Compute rubric scores for a single response."""

    weights = normalise_weights(request.weights or DEFAULT_WEIGHTS)
    keywords = request.keywords or _extract_keywords(request.prompt)
    scores = {
        "faithfulness": score_faithfulness(request.response),
        "completeness": score_completeness(request.response, keywords),
        "structure": score_structure(request.response, request.expected_structure),
        "conciseness": score_conciseness(request.response, request.target_word_count),
    }
    aggregate = sum(weights[key] * scores[key] for key in scores)
    scores["aggregate"] = round(aggregate, 2)
    return scores
