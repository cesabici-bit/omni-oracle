"""Lagged Mutual Information for non-linear directional discovery.

Replaces Granger causality with a fully non-linear approach:
- Direction: compare MI(X_{t-k}, Y_t) vs MI(Y_{t-k}, X_t)
- Lag selection: argmax MI over k=1..max_lag
- Significance: permutation test on lagged MI at best lag

This is the information-theoretic analogue of Granger causality,
capturing non-linear predictive relationships that linear VAR models miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.discovery.mi_screening import compute_mi, compute_mi_with_pvalue


@dataclass
class LaggedMIResult:
    """Result of lagged MI directional analysis between two variables."""

    direction: str  # "x->y" | "y->x" | "bidirectional" | "none"
    best_lag: int  # lag of the significant direction (or best forward)
    mi_forward: float  # MI(X_{t-k}, Y_t) at best forward lag
    mi_reverse: float  # MI(Y_{t-k}, X_t) at best reverse lag
    pvalue_forward: float  # permutation p-value for X->Y
    pvalue_reverse: float  # permutation p-value for Y->X
    best_pvalue: float  # min of the two relevant p-values
    lag_profile_forward: list[float] = field(default_factory=list)
    lag_profile_reverse: list[float] = field(default_factory=list)


def compute_lagged_mi(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    n_neighbors: int = 7,
) -> float:
    """Compute MI(X_{t-lag}, Y_t) — does X's past predict Y's present?

    Slices arrays so that x is shifted back by `lag` periods:
    x_past = x[:n-lag], y_present = y[lag:]

    Args:
        x: 1D array (the potential cause).
        y: 1D array (the potential effect), same length as x.
        lag: Number of periods to shift x back. Must be >= 1.
        n_neighbors: KSG estimator parameter.

    Returns:
        MI estimate in nats (non-negative).
    """
    assert lag >= 1, f"Lag must be >= 1, got {lag}"
    n = len(x)
    assert n == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    assert n > lag + 20, f"Series too short ({n}) for lag {lag}"

    x_past = x[: n - lag]
    y_present = y[lag:]
    return compute_mi(x_past, y_present, n_neighbors=n_neighbors)


def select_best_lag(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 12,
    n_neighbors: int = 7,
) -> tuple[int, list[float]]:
    """Find the lag k that maximizes MI(X_{t-k}, Y_t).

    Args:
        x: 1D array (potential cause).
        y: 1D array (potential effect).
        max_lag: Maximum lag to test.
        n_neighbors: KSG parameter.

    Returns:
        (best_lag, lag_profile) where lag_profile[i] = MI at lag i+1.
    """
    n = len(x)
    # Cap max_lag to ensure enough data points
    effective_max = min(max_lag, n - 21)
    if effective_max < 1:
        return 1, [0.0]

    lag_profile: list[float] = []
    for k in range(1, effective_max + 1):
        try:
            mi_k = compute_lagged_mi(x, y, lag=k, n_neighbors=n_neighbors)
        except (AssertionError, Exception):
            mi_k = 0.0
        lag_profile.append(mi_k)

    best_idx = int(np.argmax(lag_profile))
    best_lag = best_idx + 1  # lags are 1-indexed
    return best_lag, lag_profile


def detect_direction_lagged_mi(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 12,
    n_permutations: int = 100,
    threshold: float = 0.05,
    n_neighbors: int = 7,
) -> LaggedMIResult:
    """Test directional predictive relationship using lagged MI.

    Non-linear replacement for Granger causality. Algorithm:
    1. For k=1..max_lag: compute MI(X_{t-k}, Y_t) -> forward profile
    2. For k=1..max_lag: compute MI(Y_{t-k}, X_t) -> reverse profile
    3. Select best lag for each direction (argmax MI)
    4. Compute permutation p-values only at best lags (efficient)
    5. Determine direction from p-values

    Args:
        x: 1D array of observations (potential cause).
        y: 1D array of observations (potential effect).
        max_lag: Maximum lag to test (months).
        n_permutations: Permutations for p-value at best lag.
        threshold: p-value threshold for significance.
        n_neighbors: KSG estimator parameter.

    Returns:
        LaggedMIResult with direction, lag, MI values, p-values.
    """
    n = len(x)
    assert n == len(y), f"Length mismatch: {len(x)} vs {len(y)}"

    # Step 1-2: Find best lag for each direction
    best_lag_fwd, profile_fwd = select_best_lag(
        x, y, max_lag=max_lag, n_neighbors=n_neighbors
    )
    best_lag_rev, profile_rev = select_best_lag(
        y, x, max_lag=max_lag, n_neighbors=n_neighbors
    )

    mi_forward = profile_fwd[best_lag_fwd - 1] if profile_fwd else 0.0
    mi_reverse = profile_rev[best_lag_rev - 1] if profile_rev else 0.0

    # Step 3: Permutation p-values at best lags only
    # For X->Y: shuffle X, recompute MI(X_shuffled_{t-k}, Y_t)
    x_past_fwd = x[: n - best_lag_fwd]
    y_present_fwd = y[best_lag_fwd:]

    result_fwd = compute_mi_with_pvalue(
        x_past_fwd,
        y_present_fwd,
        n_permutations=n_permutations,
        threshold=threshold,
        n_neighbors=n_neighbors,
    )

    # For Y->X: shuffle Y, recompute MI(Y_shuffled_{t-k}, X_t)
    y_past_rev = y[: n - best_lag_rev]
    x_present_rev = x[best_lag_rev:]

    result_rev = compute_mi_with_pvalue(
        y_past_rev,
        x_present_rev,
        n_permutations=n_permutations,
        threshold=threshold,
        n_neighbors=n_neighbors,
    )

    pvalue_forward = result_fwd.pvalue
    pvalue_reverse = result_rev.pvalue

    # Step 4: Determine direction
    fwd_sig = pvalue_forward < threshold
    rev_sig = pvalue_reverse < threshold

    if fwd_sig and rev_sig:
        direction = "bidirectional"
        best_lag = best_lag_fwd  # use forward lag as primary
        best_pvalue = min(pvalue_forward, pvalue_reverse)
    elif fwd_sig:
        direction = "x->y"
        best_lag = best_lag_fwd
        best_pvalue = pvalue_forward
    elif rev_sig:
        direction = "y->x"
        best_lag = best_lag_rev
        best_pvalue = pvalue_reverse
    else:
        direction = "none"
        best_lag = best_lag_fwd  # default
        best_pvalue = min(pvalue_forward, pvalue_reverse)

    return LaggedMIResult(
        direction=direction,
        best_lag=best_lag,
        mi_forward=mi_forward,
        mi_reverse=mi_reverse,
        pvalue_forward=pvalue_forward,
        pvalue_reverse=pvalue_reverse,
        best_pvalue=best_pvalue,
        lag_profile_forward=profile_fwd,
        lag_profile_reverse=profile_rev,
    )
