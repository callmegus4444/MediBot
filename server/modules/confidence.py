from dataclasses import dataclass
from typing import Iterable, List, Optional

INTERNAL_ANSWER_THRESHOLD = 0.85
INTERNAL_FALLBACK_THRESHOLD = 0.50
RELEVANT_MATCH_THRESHOLD = 0.55


@dataclass
class InternalConfidence:
    score: float
    top_score: float
    relevant_matches: int
    answerable: bool
    needs_external: bool


def _normalize_match_score(raw: float) -> float:
    if raw <= 0:
        return 0.0
    if raw >= 1:
        return 1.0
    return float(raw)


def score_pinecone_matches(matches: Iterable[dict]) -> InternalConfidence:
    scores: List[float] = []
    for m in matches or []:
        if not isinstance(m, dict):
            continue
        s = m.get("score")
        if s is None:
            continue
        scores.append(_normalize_match_score(float(s)))

    if not scores:
        return InternalConfidence(
            score=0.0,
            top_score=0.0,
            relevant_matches=0,
            answerable=False,
            needs_external=True,
        )

    scores.sort(reverse=True)
    top = scores[0]
    relevant = sum(1 for s in scores if s >= RELEVANT_MATCH_THRESHOLD)

    # Aggregate: weighted by top match plus volume of relevant matches.
    volume_factor = min(relevant, 3) / 3.0
    aggregate = 0.7 * top + 0.3 * volume_factor

    answerable = top >= INTERNAL_ANSWER_THRESHOLD and relevant >= 2
    needs_external = aggregate < INTERNAL_FALLBACK_THRESHOLD or relevant == 0

    return InternalConfidence(
        score=round(aggregate, 4),
        top_score=round(top, 4),
        relevant_matches=relevant,
        answerable=answerable,
        needs_external=needs_external,
    )


def to_percent(score: float) -> int:
    return max(0, min(100, int(round(score * 100))))


def combined_confidence(internal: float, external: float, *, conflicts: bool, unsupported: int) -> int:
    base = max(internal, external)
    if internal > 0 and external > 0:
        base = 0.5 * internal + 0.5 * external
    pct = to_percent(base)
    if conflicts:
        pct -= 15
    pct -= min(unsupported, 5) * 4
    return max(0, min(100, pct))


def threshold_band(percent: int) -> str:
    if percent >= 85:
        return "answer_full"
    if percent >= 70:
        return "answer_strong_only"
    if percent >= 50:
        return "answer_partial_or_abstain"
    return "abstain"
