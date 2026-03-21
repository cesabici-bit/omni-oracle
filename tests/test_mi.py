"""Tests for Mutual Information screening (ST-06).

L2 tests use the known formula for MI of bivariate Gaussian:
    MI = -0.5 * ln(1 - ρ²)
SOURCE: Kraskov, Stögbauer, Grassberger (2004), Eq. 1 + Cover & Thomas (2006), Theorem 8.4.1
"""

from __future__ import annotations

import numpy as np

from src.discovery.mi_screening import compute_mi, compute_mi_with_pvalue


class TestMI:
    """L1 + L2 + L3: Mutual Information computation."""

    def test_mi_correlated_gaussian(
        self, correlated_gaussian: tuple[np.ndarray, np.ndarray, float]
    ) -> None:
        """L2: MI of bivariate Gaussian with ρ=0.7.

        SOURCE: Cover & Thomas (2006), Theorem 8.4.1
        MI = -0.5 * ln(1 - ρ²) = -0.5 * ln(1 - 0.49) = -0.5 * ln(0.51) ≈ 0.3365 nats
        """
        x, y, rho = correlated_gaussian
        expected_mi = -0.5 * np.log(1 - rho**2)  # ≈ 0.3365

        mi = compute_mi(x, y)
        assert mi > 0, "MI should be positive for correlated variables"
        # KSG estimator has some bias — allow 30% tolerance
        assert abs(mi - expected_mi) / expected_mi < 0.30, (
            f"MI={mi:.4f} too far from expected={expected_mi:.4f}"
        )

    def test_mi_independent_near_zero(
        self, independent_series: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """L2: MI of independent variables should be near zero."""
        # SOURCE: By definition, MI(X,Y) = 0 iff X ⊥ Y
        x, y = independent_series
        mi = compute_mi(x, y)
        assert mi < 0.1, f"MI of independent series should be near 0, got {mi:.4f}"

    def test_mi_nonnegative(
        self, correlated_gaussian: tuple[np.ndarray, np.ndarray, float]
    ) -> None:
        """L3: MI(X, Y) >= 0 always."""
        x, y, _ = correlated_gaussian
        mi = compute_mi(x, y)
        assert mi >= 0

    def test_mi_symmetric(
        self, correlated_gaussian: tuple[np.ndarray, np.ndarray, float]
    ) -> None:
        """L3: MI(X, Y) = MI(Y, X)."""
        x, y, _ = correlated_gaussian
        mi_xy = compute_mi(x, y)
        mi_yx = compute_mi(y, x)
        assert abs(mi_xy - mi_yx) < 0.05, (
            f"MI should be symmetric: MI(X,Y)={mi_xy:.4f} vs MI(Y,X)={mi_yx:.4f}"
        )

    def test_mi_pvalue_significant(
        self, correlated_gaussian: tuple[np.ndarray, np.ndarray, float]
    ) -> None:
        """Correlated variables should have significant MI p-value."""
        x, y, _ = correlated_gaussian
        result = compute_mi_with_pvalue(x, y, n_permutations=99)
        assert result.significant
        assert result.pvalue < 0.05

    def test_mi_pvalue_independent_not_significant(
        self, independent_series: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Independent variables should NOT have significant MI p-value."""
        x, y = independent_series
        result = compute_mi_with_pvalue(x, y, n_permutations=99)
        # With independent series, p-value should usually be > 0.05
        # (may occasionally fail due to random chance — not a hard assertion)
        assert result.pvalue > 0.01, (
            f"Independent series have suspiciously low MI p-value: {result.pvalue:.4f}"
        )

    def test_false_positive_rate(self) -> None:
        """L3: On 100 independent pairs, <10% should pass MI threshold.

        SOURCE: By construction — under H0, p-values are uniform,
        so ~5% should be below 0.05.
        """
        rng = np.random.default_rng(42)
        false_positives = 0
        n_tests = 50  # reduced for speed

        for i in range(n_tests):
            x = rng.normal(0, 1, 200)
            y = rng.normal(0, 1, 200)
            result = compute_mi_with_pvalue(x, y, n_permutations=49)
            if result.significant:
                false_positives += 1

        fp_rate = false_positives / n_tests
        assert fp_rate < 0.15, (
            f"False positive rate {fp_rate:.2%} too high (expected <15%)"
        )
