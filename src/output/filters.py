"""Post-discovery filters to remove autocorrelations, duplicates, and seasonality artifacts.

Three filters as per F5 review:
1. Identity filter: remove pairs where X and Y are the same series (different names)
2. Seasonality penalty: remove self-referential lag-12/lag-4 relationships
3. Cross-validation on sub-periods for robustness checking
"""

from __future__ import annotations

import numpy as np

from src.models import Hypothesis


def compute_identity_score(h: Hypothesis) -> float:
    """Compute similarity between X and Y variable IDs/names.

    Returns a score 0-1 where 1 = definitely the same series.
    Uses multiple heuristics:
    - Same base ID (e.g. BUSLOANS vs TOTCI both = C&I loans)
    - Same name (fuzzy)
    - High correlation at lag 0 (checked separately with data)
    """
    x_id = h.x.variable_id.split(":")[-1].upper()
    y_id = h.y.variable_id.split(":")[-1].upper()
    x_name = h.x.name.upper()
    y_name = h.y.name.upper()

    # Exact ID match (after stripping source prefix)
    if x_id == y_id:
        return 1.0

    # Same name = likely duplicate
    if x_name == y_name:
        return 0.95

    # High name overlap (Jaccard similarity on words)
    x_words = set(x_name.split())
    y_words = set(y_name.split())
    if x_words and y_words:
        jaccard = len(x_words & y_words) / len(x_words | y_words)
        if jaccard > 0.7:
            return jaccard

    return 0.0


def is_seasonality_artifact(h: Hypothesis) -> bool:
    """Check if relationship is a seasonality artifact.

    Flags lag-12 (annual) or lag-4 (quarterly) self-referential patterns.
    Only flags if X and Y have high identity score.
    """
    if h.lag not in (4, 12):
        return False

    identity = compute_identity_score(h)
    return identity > 0.5


def filter_hypotheses(
    hypotheses: list[Hypothesis],
    identity_threshold: float = 0.5,
    remove_seasonality: bool = True,
) -> list[Hypothesis]:
    """Apply all filters to hypothesis list.

    Removes:
    1. Identity duplicates (X ~ Y with similarity > threshold)
    2. Seasonality artifacts (lag 12/4 with self-referential pattern)

    Returns filtered list, re-ranked.
    """
    filtered: list[Hypothesis] = []

    for h in hypotheses:
        # Filter 1: Identity / duplicate detection
        identity = compute_identity_score(h)
        if identity > identity_threshold:
            continue

        # Filter 2: Seasonality artifact
        if remove_seasonality and is_seasonality_artifact(h):
            continue

        filtered.append(h)

    # Re-rank
    for i, h in enumerate(filtered, 1):
        h.rank = i

    return filtered


def cross_validate_subperiods(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    split_year_frac: float = 0.5,
) -> dict:
    """Cross-validate a relationship on two sub-periods.

    Splits data into first half and second half, runs OOS validation
    on each independently. If R2 is positive in both halves,
    the relationship is considered robust.

    Returns dict with r2_first, r2_second, robust (bool).
    """
    from src.validation.temporal_oos import validate_oos

    n = len(x)
    mid = int(n * split_year_frac)

    results = {"r2_first": 0.0, "r2_second": 0.0, "robust": False}

    # First half
    if mid > lag + 20:
        try:
            oos1 = validate_oos(x[:mid], y[:mid], lag=lag)
            results["r2_first"] = oos1.r2_incremental
        except Exception:
            results["r2_first"] = 0.0

    # Second half
    if n - mid > lag + 20:
        try:
            oos2 = validate_oos(x[mid:], y[mid:], lag=lag)
            results["r2_second"] = oos2.r2_incremental
        except Exception:
            results["r2_second"] = 0.0

    # Robust if both halves show positive R2
    results["robust"] = (
        results["r2_first"] > 0.01
        and results["r2_second"] > 0.01
    )

    return results
