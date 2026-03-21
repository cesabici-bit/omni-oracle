"""Temporal Out-of-Sample validation.

CRITICAL: No future data must contaminate training.
All normalization and lag selection are done ONLY on the training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class OOSResult:
    """Result of out-of-sample validation."""

    r2_incremental: float  # R² improvement of Granger model over AR-only
    mse_ar: float  # MSE of AR-only model on test set
    mse_augmented: float  # MSE of Granger (AR + X) model on test set
    valid: bool  # r2_incremental > threshold
    train_size: int
    test_size: int


def validate_oos(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lag: int,
    train_ratio: float = 0.7,
    r2_threshold: float = 0.02,
) -> OOSResult:
    """Validate Granger relationship out-of-sample.

    Steps:
    1. Split temporally: train = first 70%, test = last 30%
    2. Normalize x and y using ONLY train statistics
    3. Fit AR(lag) on y_train → predict on y_test → MSE_ar
    4. Fit AR(lag) + X_lagged on y_train → predict on y_test → MSE_augmented
    5. R²_incremental = 1 - (MSE_augmented / MSE_ar)

    ANTI-LEAKAGE:
    - z-score computed ONLY on train
    - Model coefficients frozen from train
    - No interpolation using future points
    """
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    n = len(x)
    assert n > lag + 10, f"Series too short: {n} points, lag={lag}"

    split = int(n * train_ratio)
    assert split > lag + 5, "Train set too small after lag"
    assert n - split > lag + 5, "Test set too small after lag"

    # ANTI-LEAKAGE: normalize using ONLY train statistics
    x_mean, x_std = np.mean(x[:split]), np.std(x[:split])
    y_mean, y_std = np.mean(y[:split]), np.std(y[:split])
    x_std = max(x_std, 1e-10)
    y_std = max(y_std, 1e-10)

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # Build lagged matrices
    # For AR model: Y(t) = a1*Y(t-1) + ... + ap*Y(t-p)
    # For augmented: Y(t) = a1*Y(t-1) + ... + ap*Y(t-p) + b1*X(t-1) + ... + bp*X(t-p)
    y_ar_features_train, y_target_train = _build_lagged_matrix(
        y_norm[:split], lag
    )
    y_ar_features_test, y_target_test = _build_lagged_matrix(
        y_norm[split - lag :], lag  # include lag points before split for context
    )
    # Remove the first `lag` rows of test that overlap with train
    # Actually, we built from split-lag, so the first valid target is at index lag
    # which corresponds to the split point

    x_lagged_train = _build_lagged_features(x_norm[:split], lag)
    x_lagged_test = _build_lagged_features(x_norm[split - lag :], lag)

    # Trim to matching sizes
    min_train = min(len(y_ar_features_train), len(x_lagged_train))
    y_ar_train = y_ar_features_train[-min_train:]
    y_tgt_train = y_target_train[-min_train:]
    x_lag_train = x_lagged_train[-min_train:]

    min_test = min(len(y_ar_features_test), len(x_lagged_test))
    y_ar_test = y_ar_features_test[-min_test:]
    y_tgt_test = y_target_test[-min_test:]
    x_lag_test = x_lagged_test[-min_test:]

    # Fit AR-only model on train
    ar_coeffs = _ols_fit(y_ar_train, y_tgt_train)
    ar_pred_test = y_ar_test @ ar_coeffs
    mse_ar = float(np.mean((y_tgt_test - ar_pred_test) ** 2))

    # Fit augmented model (AR + X_lagged) on train
    aug_features_train = np.hstack([y_ar_train, x_lag_train])
    aug_coeffs = _ols_fit(aug_features_train, y_tgt_train)
    aug_features_test = np.hstack([y_ar_test, x_lag_test])
    aug_pred_test = aug_features_test @ aug_coeffs
    mse_augmented = float(np.mean((y_tgt_test - aug_pred_test) ** 2))

    # R² incremental
    if mse_ar < 1e-15:
        r2_inc = 0.0  # AR already perfect
    else:
        r2_inc = 1.0 - (mse_augmented / mse_ar)

    return OOSResult(
        r2_incremental=r2_inc,
        mse_ar=mse_ar,
        mse_augmented=mse_augmented,
        valid=r2_inc > r2_threshold,
        train_size=min_train,
        test_size=min_test,
    )


def _build_lagged_matrix(
    series: NDArray[np.float64], lag: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build lagged feature matrix and target vector.

    For series [y0, y1, y2, y3, y4] with lag=2:
    Features: [[y0, y1], [y1, y2], [y2, y3]]
    Target:   [y2, y3, y4]
    """
    n = len(series)
    features = np.column_stack(
        [series[lag - i - 1 : n - i - 1] for i in range(lag)]
    )
    target = series[lag:]
    return features, target


def _build_lagged_features(
    series: NDArray[np.float64], lag: int
) -> NDArray[np.float64]:
    """Build lagged feature matrix (no target)."""
    n = len(series)
    return np.column_stack(
        [series[lag - i - 1 : n - i - 1] for i in range(lag)]
    )


def _ols_fit(
    X: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Simple OLS fit. Returns coefficient vector."""
    # Use pseudoinverse for numerical stability
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs
