"""Granger causality testing with BIC-based lag selection.

Tests whether past values of X help predict Y beyond Y's own past.
Uses VAR model for lag selection and F-test for significance.

Supports cointegration-aware testing: if both series are I(1) and
cointegrated, tests Granger on levels (preserving long-run relationship)
instead of differenced series. See Engle & Granger (1987).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import coint, grangercausalitytests

from src.config import GRANGER_MAX_LAG, GRANGER_PVALUE_THRESHOLD


@dataclass
class GrangerResult:
    """Result of Granger causality test for a pair."""

    direction: str  # "x→y" | "y→x" | "bidirectional" | "none"
    pvalue_xy: float  # p-value for X→Y (X Granger-causes Y)
    pvalue_yx: float  # p-value for Y→X
    best_pvalue: float  # min of the two relevant p-values
    lag: int  # optimal lag (BIC)
    cointegrated: bool = False  # whether pair was tested on levels


def select_lag_bic(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = GRANGER_MAX_LAG,
) -> int:
    """Select optimal lag using BIC on a VAR model.

    Args:
        x, y: 1D arrays (must be stationary).
        max_lag: Maximum lag to consider.

    Returns:
        Optimal lag (>= 1).
    """
    data = pd.DataFrame({"x": x, "y": y})
    try:
        model = VAR(data)
        # Limit max_lag to a reasonable fraction of sample size
        effective_max = min(max_lag, len(x) // 5)
        if effective_max < 1:
            effective_max = 1
        result = model.select_order(maxlags=effective_max)
        lag = result.bic
        return max(lag, 1)  # at least lag 1
    except Exception:
        return 1  # fallback to lag 1


def test_granger(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
) -> float:
    """Test if X Granger-causes Y at given lag. Returns p-value.

    Uses statsmodels grangercausalitytests with F-test.
    The function tests whether past values of x[column 1] help predict
    y[column 0] — i.e., the SECOND column Granger-causes the FIRST.
    """
    # grangercausalitytests expects: column 0 = Y (response), column 1 = X (cause)
    data = np.column_stack([y, x])
    try:
        results = grangercausalitytests(data, maxlag=lag, verbose=False)
        # Get p-value at the specified lag
        f_test = results[lag][0]["ssr_ftest"]
        return float(f_test[1])  # p-value
    except Exception:
        return 1.0  # non-significant on error


def test_cointegration(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """Test if x and y are cointegrated using Engle-Granger test.

    Returns (is_cointegrated, pvalue).
    """
    try:
        _, pvalue, _ = coint(x, y)
        return pvalue < alpha, float(pvalue)
    except Exception:
        return False, 1.0


def test_granger_bidirectional(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = GRANGER_MAX_LAG,
    threshold: float = GRANGER_PVALUE_THRESHOLD,
    x_raw: np.ndarray | None = None,
    y_raw: np.ndarray | None = None,
) -> GrangerResult:
    """Test Granger causality in both directions.

    1. If x_raw/y_raw provided: test cointegration on levels.
       If cointegrated: run Granger on LEVELS (not differenced).
       This catches long-run causal relationships that differencing destroys.
    2. Select optimal lag via BIC on VAR(x, y).
    3. Test X->Y (does X help predict Y?).
    4. Test Y->X (does Y help predict X?).
    5. Determine direction based on p-values.
    """
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    assert len(x) >= 30, f"Too few observations for Granger: {len(x)}"

    cointegrated = False

    # Cointegration check: if raw (undifferenced) series provided, test them
    if x_raw is not None and y_raw is not None and len(x_raw) == len(y_raw):
        is_coint, _ = test_cointegration(x_raw, y_raw)
        if is_coint:
            # Use levels instead of differenced — Granger on levels is valid
            # when series are cointegrated (Engle & Granger 1987, Toda & Phillips 1993)
            cointegrated = True
            x = x_raw
            y = y_raw

    lag = select_lag_bic(x, y, max_lag)

    pvalue_xy = test_granger(x, y, lag)  # X Granger-causes Y
    pvalue_yx = test_granger(y, x, lag)  # Y Granger-causes X

    xy_sig = pvalue_xy < threshold
    yx_sig = pvalue_yx < threshold

    if xy_sig and yx_sig:
        direction = "bidirectional"
    elif xy_sig:
        direction = "x->y"
    elif yx_sig:
        direction = "y->x"
    else:
        direction = "none"

    best_pvalue = min(pvalue_xy, pvalue_yx)

    return GrangerResult(
        direction=direction,
        pvalue_xy=pvalue_xy,
        pvalue_yx=pvalue_yx,
        best_pvalue=best_pvalue,
        lag=lag,
        cointegrated=cointegrated,
    )
