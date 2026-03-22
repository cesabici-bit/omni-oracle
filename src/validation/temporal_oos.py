"""Temporal Out-of-Sample validation with multi-model support.

CRITICAL: No future data must contaminate training.
All normalization and lag selection are done ONLY on the training set.

The augmented model (AR + X_lagged) uses a best-of-3 approach:
OLS, Ridge, and Random Forest. The model with the lowest training MSE
wins, capturing both linear and non-linear predictive relationships.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class OOSResult:
    """Result of out-of-sample validation."""

    r2_incremental: float  # R² improvement of augmented model over AR-only
    mse_ar: float  # MSE of AR-only model on test set
    mse_augmented: float  # MSE of augmented (AR + X) model on test set
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
    """Validate predictive relationship out-of-sample using multi-model.

    Steps:
    1. Split temporally: train = first 70%, test = last 30%
    2. Normalize x and y using ONLY train statistics
    3. Fit AR(lag) on y_train → predict on y_test → MSE_ar
    4. Fit multi-model (OLS/Ridge/RF) on y_train with X_lagged → MSE_augmented
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

    # Cap effective lag to ensure sufficient degrees of freedom in test set.
    test_n = n - split
    effective_lag = min(lag, max(test_n // 4, 1))
    lag = effective_lag

    # ANTI-LEAKAGE: normalize using ONLY train statistics
    x_mean, x_std = np.mean(x[:split]), np.std(x[:split])
    y_mean, y_std = np.mean(y[:split]), np.std(y[:split])
    x_std = max(x_std, 1e-10)
    y_std = max(y_std, 1e-10)

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # Build lagged matrices
    y_ar_features_train, y_target_train = _build_lagged_matrix(
        y_norm[:split], lag
    )
    y_ar_features_test, y_target_test = _build_lagged_matrix(
        y_norm[split - lag :], lag
    )

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

    # Fit AR-only model on train (OLS baseline — linear AR is fine as null model)
    ar_coeffs = _ols_fit(y_ar_train, y_tgt_train)
    ar_pred_test = y_ar_test @ ar_coeffs
    mse_ar = float(np.mean((y_tgt_test - ar_pred_test) ** 2))

    # Fit augmented model (AR + X_lagged) on train using multi-model
    aug_features_train = np.hstack([y_ar_train, x_lag_train])
    aug_features_test = np.hstack([y_ar_test, x_lag_test])
    aug_pred_test = _best_model_predict(
        aug_features_train, y_tgt_train, aug_features_test
    )
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


def _best_model_predict(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict using Ridge (default) or Random Forest when it outperforms.

    Strategy:
    - Ridge replaces OLS as default: handles collinearity and regularizes
      without overfitting on small samples (60-100 points typical).
    - RF attempted when training set has >= 60 points with adaptive
      hyperparameters based on sample size. RF must beat Ridge on a
      temporal hold-out by >= 10% to be selected.

    Falls back to OLS if Ridge fails.
    """
    n_train = len(X_train)
    n_features = X_train.shape[1] if X_train.ndim > 1 else 1

    # Default: Ridge regression (better than OLS for small-sample,
    # multi-feature settings due to L2 regularization)
    try:
        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
    except Exception:
        # Fallback to OLS
        coeffs = _ols_fit(X_train, y_train)
        return X_test @ coeffs

    # Attempt RF when we have enough samples relative to features
    # Need at least 60 points AND at least 3x features for RF to be viable
    if n_train < 60 or n_train < 3 * n_features:
        return ridge_pred

    # Temporal hold-out to compare Ridge vs RF
    val_size = max(n_train // 5, 10)
    fit_end = n_train - val_size
    X_fit, y_fit = X_train[:fit_end], y_train[:fit_end]
    X_val, y_val = X_train[fit_end:], y_train[fit_end:]

    try:
        from sklearn.linear_model import Ridge as Ridge2

        ridge_val = Ridge2(alpha=1.0)
        ridge_val.fit(X_fit, y_fit)
        ridge_val_mse = float(np.mean((y_val - ridge_val.predict(X_val)) ** 2))
    except Exception:
        return ridge_pred

    try:
        from sklearn.ensemble import RandomForestRegressor

        # Adaptive RF hyperparams: more conservative for smaller samples
        if n_train < 100:
            rf_params = dict(
                n_estimators=30, max_depth=3,
                min_samples_leaf=max(5, n_train // 10),
            )
        elif n_train < 200:
            rf_params = dict(
                n_estimators=50, max_depth=4,
                min_samples_leaf=max(5, n_train // 15),
            )
        else:
            rf_params = dict(
                n_estimators=50, max_depth=5,
                min_samples_leaf=10,
            )

        rf = RandomForestRegressor(
            **rf_params, random_state=42, n_jobs=1,
        )
        rf.fit(X_fit, y_fit)
        rf_val_mse = float(np.mean((y_val - rf.predict(X_val)) ** 2))

        if rf_val_mse < ridge_val_mse * 0.9:  # RF must be meaningfully better
            # Re-fit RF on full training set
            rf_full = RandomForestRegressor(
                **rf_params, random_state=42, n_jobs=1,
            )
            rf_full.fit(X_train, y_train)
            return rf_full.predict(X_test)
    except Exception:
        pass

    return ridge_pred
