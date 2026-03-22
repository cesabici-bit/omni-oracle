"""Post-discovery filters to remove autocorrelations, duplicates, and seasonality artifacts.

Three filters as per F5 review:
1. Identity filter: remove pairs where X and Y are the same series (different names)
2. Seasonality penalty: remove self-referential lag-12/lag-4 relationships
3. Cross-validation on sub-periods for robustness checking
"""

from __future__ import annotations

import numpy as np

from src.models import Hypothesis

# Series that are derived/forward-looking models and should not be used as
# targets in causal discovery.  Including them creates circular reasoning
# (their inputs trivially "predict" them) or look-ahead bias (they embed
# forecasts of the next 12 months).
# See KNOWN_ISSUES.md EC-003.
BLACKLISTED_SERIES_PREFIXES: list[str] = [
    "STLPPM",   # St. Louis Fed Price Pressures family (STLPPM, STLPPMDEF, etc.)
]


def is_blacklisted(variable_id: str) -> bool:
    """Check if a variable belongs to a blacklisted family."""
    series_id = variable_id.split(":")[-1].upper()
    return any(series_id.startswith(p) for p in BLACKLISTED_SERIES_PREFIXES)


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
    remove_blacklisted: bool = True,
) -> list[Hypothesis]:
    """Apply all filters to hypothesis list.

    Removes:
    1. Blacklisted series (derived/forward-looking models)
    2. Identity duplicates (X ~ Y with similarity > threshold)
    3. Seasonality artifacts (lag 12/4 with self-referential pattern)

    Returns filtered list, re-ranked.
    """
    filtered: list[Hypothesis] = []

    for h in hypotheses:
        # Filter 0: Blacklisted series (circular / look-ahead bias)
        if remove_blacklisted and (
            is_blacklisted(h.x.variable_id)
            or is_blacklisted(h.y.variable_id)
        ):
            continue

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
    """Cross-validate a relationship on two sub-periods (legacy).

    Splits data into first half and second half, runs OOS validation
    on each independently. If R2 is positive in both halves,
    the relationship is considered robust.

    Returns dict with r2_first, r2_second, robust (bool).

    NOTE: Superseded by walk_forward_validate() which has better
    statistical power. Kept for backward compatibility.
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


def walk_forward_validate(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    train_months: int = 60,
    test_months: int = 24,
    step_months: int = 12,
    min_pass_ratio: float = 0.6,
    r2_threshold: float = 0.01,
    regime_break_threshold: float = -2.0,
    min_non_regime_windows: int = 3,
) -> dict:
    """Walk-forward cross-validation with rolling windows and regime-aware filtering.

    Instead of a single 50/50 split, uses multiple overlapping windows:
    - Train on [t, t+train_months), test on [t+train_months, t+train_months+test_months)
    - Step forward by step_months, repeat
    - ROBUST if R2 > threshold in >= min_pass_ratio of non-regime-break windows

    Regime-aware filtering (handles COVID-like structural breaks):
    - Windows with R2 < regime_break_threshold are flagged as regime breaks
    - These windows are excluded from the pass_ratio calculation
    - Requires at least min_non_regime_windows valid windows
    - Both raw and adjusted metrics are reported

    Returns dict with window_r2s, n_windows, n_pass, pass_ratio, robust,
    plus regime-aware fields: n_regime_breaks, n_valid_windows,
    adjusted_pass_ratio, adjusted_robust.
    """
    from src.validation.temporal_oos import validate_oos

    n = len(x)
    min_window = train_months + test_months + lag

    empty_result = {
        "window_r2s": [],
        "n_windows": 0,
        "n_pass": 0,
        "pass_ratio": 0.0,
        "robust": False,
        "n_regime_breaks": 0,
        "n_valid_windows": 0,
        "adjusted_pass_ratio": 0.0,
        "adjusted_robust": False,
    }

    if n < min_window:
        return empty_result

    window_r2s: list[float] = []
    start = 0

    while start + min_window <= n:
        train_end = start + train_months
        test_end = min(train_end + test_months, n)

        # Need enough test points
        if test_end - train_end < lag + 10:
            start += step_months
            continue

        x_window = x[start:test_end]
        y_window = y[start:test_end]

        # train_ratio maps to the proportion of this window used for training
        window_train_ratio = train_months / (test_end - start)

        try:
            oos = validate_oos(
                x_window, y_window,
                lag=lag,
                train_ratio=window_train_ratio,
                r2_threshold=r2_threshold,
            )
            window_r2s.append(oos.r2_incremental)
        except Exception:
            window_r2s.append(0.0)

        start += step_months

    n_windows = len(window_r2s)
    if n_windows == 0:
        return empty_result

    # Raw metrics (unchanged from before)
    n_pass = sum(1 for r2 in window_r2s if r2 > r2_threshold)
    pass_ratio = n_pass / n_windows

    # Regime-aware metrics: exclude extreme outlier windows
    regime_breaks = [r2 < regime_break_threshold for r2 in window_r2s]
    n_regime_breaks = sum(regime_breaks)
    valid_r2s = [r2 for r2, is_break in zip(window_r2s, regime_breaks) if not is_break]
    n_valid_windows = len(valid_r2s)

    if n_valid_windows >= min_non_regime_windows:
        adjusted_n_pass = sum(1 for r2 in valid_r2s if r2 > r2_threshold)
        adjusted_pass_ratio = adjusted_n_pass / n_valid_windows
        adjusted_robust = adjusted_pass_ratio >= min_pass_ratio
    else:
        adjusted_pass_ratio = 0.0
        adjusted_robust = False

    return {
        "window_r2s": window_r2s,
        "n_windows": n_windows,
        "n_pass": n_pass,
        "pass_ratio": pass_ratio,
        "robust": pass_ratio >= min_pass_ratio,
        "n_regime_breaks": n_regime_breaks,
        "n_valid_windows": n_valid_windows,
        "adjusted_pass_ratio": adjusted_pass_ratio,
        "adjusted_robust": adjusted_robust,
    }
