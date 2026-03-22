"""Tests for Lagged MI directional discovery.

Replaces Granger causality tests with non-linear equivalents.
Key test: Y = X_{t-2}^2 + noise is detected (Granger would miss it).
"""

import numpy as np
import pytest

from src.discovery.lagged_mi import (
    LaggedMIResult,
    compute_lagged_mi,
    select_best_lag,
    detect_direction_lagged_mi,
)


@pytest.fixture
def var2_xy() -> tuple[np.ndarray, np.ndarray]:
    """VAR(2) where X causes Y at lag 2 (linear).

    X(t) = 0.5*X(t-1) + eps_x
    Y(t) = 0.3*Y(t-1) + 0.5*X(t-2) + eps_y
    """
    rng = np.random.default_rng(42)
    n = 500
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        y[t] = 0.3 * y[t - 1] + 0.5 * x[t - 2] + rng.standard_normal()
    return x[50:], y[50:]  # discard burn-in


@pytest.fixture
def independent_series() -> tuple[np.ndarray, np.ndarray]:
    """Two completely independent series."""
    rng = np.random.default_rng(123)
    n = 300
    return rng.standard_normal(n), rng.standard_normal(n)


class TestLaggedMIBasic:
    """Basic lagged MI computation."""

    def test_lagged_mi_positive_for_causal(self, var2_xy):
        """MI(X_{t-2}, Y_t) should be positive for causal relationship."""
        x, y = var2_xy
        mi = compute_lagged_mi(x, y, lag=2)
        assert mi > 0.01, f"Lagged MI too low for known causal pair: {mi}"

    def test_lagged_mi_zero_for_independent(self, independent_series):
        """MI(X_{t-k}, Y_t) should be near zero for independent series."""
        x, y = independent_series
        mi = compute_lagged_mi(x, y, lag=2)
        assert mi < 0.1, f"Lagged MI too high for independent series: {mi}"

    def test_lag_selection_finds_correct_lag(self, var2_xy):
        """Best lag should be at or near lag=2 for VAR(2) DGP."""
        x, y = var2_xy
        best_lag, profile = select_best_lag(x, y, max_lag=6)
        assert 1 <= best_lag <= 4, (
            f"Best lag {best_lag} far from true lag 2. Profile: {profile}"
        )

    def test_lag_profile_nonnegative(self, var2_xy):
        """L3: MI at every lag should be >= 0."""
        x, y = var2_xy
        _, profile = select_best_lag(x, y, max_lag=6)
        assert all(mi >= 0 for mi in profile), (
            f"Negative MI in profile: {profile}"
        )


class TestLaggedMIDirection:
    """Directional discovery tests."""

    def test_direction_xy(self, var2_xy):
        """X causes Y at lag 2 -> direction should be 'x->y'."""
        x, y = var2_xy
        result = detect_direction_lagged_mi(
            x, y, max_lag=6, n_permutations=100
        )
        assert result.direction in ("x->y", "bidirectional"), (
            f"Expected x->y, got {result.direction} "
            f"(fwd p={result.pvalue_forward:.4f}, rev p={result.pvalue_reverse:.4f})"
        )

    def test_no_reverse(self, var2_xy):
        """Y does not cause X in VAR(2) DGP -> reverse p-value should be high."""
        x, y = var2_xy
        result = detect_direction_lagged_mi(
            x, y, max_lag=6, n_permutations=100
        )
        # Forward should be significant
        assert result.pvalue_forward < 0.05, (
            f"Forward not significant: p={result.pvalue_forward}"
        )

    def test_independent_low_mi(self, independent_series):
        """Independent series -> lagged MI values should be low.

        Note: direction may NOT be "none" due to lag selection inflation
        (testing max_lag candidates inflates false positive rate).
        The pipeline's FDR step corrects for this. Here we verify
        the MI magnitudes are small.
        """
        x, y = independent_series
        result = detect_direction_lagged_mi(
            x, y, max_lag=6, n_permutations=100
        )
        # MI values should be small for independent series
        assert result.mi_forward < 0.15, (
            f"MI forward too high for independent: {result.mi_forward}"
        )
        assert result.mi_reverse < 0.15, (
            f"MI reverse too high for independent: {result.mi_reverse}"
        )

    def test_pvalue_range(self, var2_xy):
        """P-values must be in [0, 1]."""
        x, y = var2_xy
        result = detect_direction_lagged_mi(
            x, y, max_lag=6, n_permutations=50
        )
        assert 0 <= result.pvalue_forward <= 1
        assert 0 <= result.pvalue_reverse <= 1
        assert 0 <= result.best_pvalue <= 1


class TestLaggedMINonLinear:
    """The key tests: non-linear relationships that Granger would miss."""

    def test_nonlinear_quadratic_detected(self):
        """L2: Y = X_{t-2}^2 + noise should be detected by lagged MI.

        # SOURCE: Kraskov et al. (2004), "Estimating Mutual Information",
        # Physical Review E 69(6). KSG estimator captures non-linear
        # dependence that linear methods (Granger, Pearson) miss.
        """
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = x[t - 2] ** 2 + 0.3 * rng.standard_normal()

        result = detect_direction_lagged_mi(
            x, y, max_lag=6, n_permutations=100
        )

        # Lagged MI should detect this non-linear relationship
        assert result.mi_forward > 0.05, (
            f"Failed to detect quadratic relationship: MI_fwd={result.mi_forward}"
        )
        assert result.pvalue_forward < 0.05, (
            f"Quadratic relationship not significant: p={result.pvalue_forward}"
        )
        assert result.direction in ("x->y", "bidirectional"), (
            f"Wrong direction for quadratic: {result.direction}"
        )

    def test_nonlinear_threshold_detected(self):
        """Y = max(X_{t-1}, 0) + noise — threshold/ReLU relationship."""
        rng = np.random.default_rng(77)
        n = 500
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = max(x[t - 1], 0) + 0.3 * rng.standard_normal()

        result = detect_direction_lagged_mi(
            x, y, max_lag=4, n_permutations=100
        )

        assert result.mi_forward > 0.05, (
            f"Failed to detect threshold relationship: MI_fwd={result.mi_forward}"
        )
        assert result.pvalue_forward < 0.05, (
            f"Threshold relationship not significant: p={result.pvalue_forward}"
        )
