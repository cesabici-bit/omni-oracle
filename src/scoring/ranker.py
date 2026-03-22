"""Hypothesis scoring and ranking.

Score composito 0-10, weighted average of 4 components:
- MI strength (25%)
- Direction significance (25%)
- OOS robustness (35%) — highest weight, strongest anti-spurious filter
- Effect size (15%)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    SCORE_WEIGHT_DIRECTION,
    SCORE_WEIGHT_EFFECT,
    SCORE_WEIGHT_MI,
    SCORE_WEIGHT_OOS,
)
from src.models import PairResult


@dataclass
class ScoredPair:
    """A pair with its composite score and breakdown."""

    pair: PairResult
    score: float  # 0-10
    mi_component: float
    direction_component: float
    oos_component: float
    effect_component: float
    confidence: str  # "high" | "medium" | "low"


def compute_score(
    pair: PairResult,
    max_mi: float = 1.0,
    max_f: float = 20.0,
) -> ScoredPair:
    """Compute composite score for a pair.

    Components (all normalized to 0-1 before weighting):
    1. MI strength (25%):     normalize(mi, 0, max_mi)
    2. Direction sig (25%):   normalize(-log10(direction_pvalue), 0, 10)
    3. OOS robustness (35%):  normalize(oos_r2, 0, 0.3)
    4. Effect size (15%):     normalize(-log10(direction_pvalue) * oos_r2, 0, 3)
    """
    # Component 1: MI strength
    mi_norm = _clip_normalize(pair.mi, 0.0, max_mi)

    # Component 2: Direction significance (-log10 transform)
    dir_log = -np.log10(max(pair.direction_pvalue, 1e-15))
    dir_norm = _clip_normalize(dir_log, 0.0, 10.0)

    # Component 3: OOS robustness
    oos_norm = _clip_normalize(pair.oos_r2, 0.0, 0.3)

    # Component 4: Effect size (combined significance x predictive power)
    effect = dir_log * max(pair.oos_r2, 0.0)
    effect_norm = _clip_normalize(effect, 0.0, 3.0)

    # Weighted sum -> scale to 0-10
    raw_score = (
        SCORE_WEIGHT_MI * mi_norm
        + SCORE_WEIGHT_DIRECTION * dir_norm
        + SCORE_WEIGHT_OOS * oos_norm
        + SCORE_WEIGHT_EFFECT * effect_norm
    )
    score = round(raw_score * 10.0, 2)

    # Confidence classification
    if score >= 7.0 and pair.oos_valid and pair.fdr_significant:
        confidence = "high"
    elif score >= 4.0 and (pair.oos_valid or pair.fdr_significant):
        confidence = "medium"
    else:
        confidence = "low"

    return ScoredPair(
        pair=pair,
        score=score,
        mi_component=round(mi_norm * SCORE_WEIGHT_MI * 10, 2),
        direction_component=round(dir_norm * SCORE_WEIGHT_DIRECTION * 10, 2),
        oos_component=round(oos_norm * SCORE_WEIGHT_OOS * 10, 2),
        effect_component=round(effect_norm * SCORE_WEIGHT_EFFECT * 10, 2),
        confidence=confidence,
    )


def rank_pairs(
    pairs: list[PairResult],
    max_mi: float | None = None,
    max_f: float = 20.0,
) -> list[ScoredPair]:
    """Score and rank all pairs. Returns sorted list (highest score first)."""
    if not pairs:
        return []

    # Auto-compute max_mi from data if not provided
    if max_mi is None:
        max_mi = max(p.mi for p in pairs) if pairs else 1.0
        max_mi = max(max_mi, 0.01)  # avoid division by zero

    scored = [compute_score(p, max_mi=max_mi, max_f=max_f) for p in pairs]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def _clip_normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] with clipping."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))
