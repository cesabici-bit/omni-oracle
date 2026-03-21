"""Alternative Granger causality implementation using numpy OLS + scipy F-test.

M4 cross-tool verification: this implements Granger causality from scratch
using numpy for OLS and scipy.stats.f for the F-test p-value, instead of
statsmodels.tsa.stattools.grangercausalitytests used in src/discovery/granger.py.

Both should produce very similar p-values (same mathematical test,
different implementation).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _ols_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS via normal equations. Returns residuals."""
    # Add intercept
    n = len(y)
    X_aug = np.column_stack([np.ones(n), X])
    # beta = (X'X)^-1 X'y
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    residuals = y - X_aug @ beta
    return residuals


def _build_lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    """Build matrix of lagged values [x(t-1), x(t-2), ..., x(t-lag)]."""
    n = len(series)
    mat = np.zeros((n - lag, lag))
    for i in range(lag):
        mat[:, i] = series[lag - 1 - i : n - 1 - i]
    return mat


def test_granger_manual(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
) -> float:
    """Test if X Granger-causes Y using manual F-test.

    Restricted model:  Y(t) = a0 + a1*Y(t-1) + ... + ap*Y(t-p) + e
    Unrestricted model: Y(t) = a0 + a1*Y(t-1) + ... + ap*Y(t-p)
                              + b1*X(t-1) + ... + bp*X(t-p) + e

    F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))

    Parameters
    ----------
    x, y : 1D arrays of same length (X potentially causes Y)
    lag : number of lags to test

    Returns
    -------
    p-value from F-distribution
    """
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    n_full = len(x)
    assert n_full >= lag + 10, f"Too few observations: {n_full}, lag={lag}"

    # Build lag matrices
    y_lags = _build_lag_matrix(y, lag)  # shape: (n-lag, lag)
    x_lags = _build_lag_matrix(x, lag)  # shape: (n-lag, lag)
    y_target = y[lag:]  # shape: (n-lag,)

    n = len(y_target)
    p = lag

    # Restricted model: Y ~ Y_lags only
    resid_r = _ols_residuals(y_lags, y_target)
    rss_r = np.sum(resid_r**2)

    # Unrestricted model: Y ~ Y_lags + X_lags
    xy_lags = np.column_stack([y_lags, x_lags])
    resid_u = _ols_residuals(xy_lags, y_target)
    rss_u = np.sum(resid_u**2)

    # F-statistic
    df1 = p  # number of restrictions
    df2 = n - 2 * p - 1  # residual df for unrestricted model

    if df2 <= 0 or rss_u <= 0:
        return 1.0

    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

    if f_stat < 0:
        return 1.0

    pvalue = float(stats.f.sf(f_stat, df1, df2))
    return pvalue


def select_lag_bic_manual(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 12,
) -> int:
    """Select optimal lag via BIC on VAR model (manual implementation).

    BIC = n * ln(RSS/n) + k * ln(n)
    where k = number of parameters per equation.
    """
    n_full = len(x)
    effective_max = min(max_lag, n_full // 5)
    effective_max = max(effective_max, 1)

    best_lag = 1
    best_bic = np.inf

    for lag in range(1, effective_max + 1):
        y_lags = _build_lag_matrix(y, lag)
        x_lags = _build_lag_matrix(x, lag)
        y_target = y[lag:]
        x_target = x[lag:]

        n = len(y_target)
        if n <= 2 * lag + 1:
            continue

        # VAR(p): each equation has 2*lag + 1 parameters (intercept + lags of both vars)
        xy_lags = np.column_stack([y_lags, x_lags])
        k = 2 * lag + 1  # parameters per equation

        # Equation for Y
        resid_y = _ols_residuals(xy_lags, y_target)
        rss_y = np.sum(resid_y**2) / n

        # Equation for X
        resid_x = _ols_residuals(xy_lags, x_target)
        rss_x = np.sum(resid_x**2) / n

        # BIC for VAR system (sum of log-determinant approximation)
        if rss_y <= 0 or rss_x <= 0:
            continue
        bic = n * (np.log(rss_y) + np.log(rss_x)) + 2 * k * np.log(n)

        if bic < best_bic:
            best_bic = bic
            best_lag = lag

    return best_lag


def test_granger_bidirectional_manual(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 12,
    threshold: float = 0.05,
) -> dict:
    """Bidirectional Granger test (manual implementation).

    Same logic as src/discovery/granger.py but using manual OLS + scipy F-test.
    """
    lag = select_lag_bic_manual(x, y, max_lag)

    pvalue_xy = test_granger_manual(x, y, lag)  # X -> Y
    pvalue_yx = test_granger_manual(y, x, lag)  # Y -> X

    sig_xy = pvalue_xy < threshold
    sig_yx = pvalue_yx < threshold

    if sig_xy and sig_yx:
        direction = "bidirectional"
    elif sig_xy:
        direction = "x->y"
    elif sig_yx:
        direction = "y->x"
    else:
        direction = "none"

    return {
        "direction": direction,
        "pvalue_xy": pvalue_xy,
        "pvalue_yx": pvalue_yx,
        "best_pvalue": min(pvalue_xy, pvalue_yx),
        "lag": lag,
        "method": "manual_ols_ftest",
    }
