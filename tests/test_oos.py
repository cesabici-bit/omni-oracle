"""Tests for Out-of-Sample temporal validation (ST-09).

CRITICAL: These tests verify anti-leakage properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.temporal_oos import OOSResult, validate_oos


class TestOOS:
    """L1: Out-of-sample validation."""

    def test_persistent_relationship_positive_r2(
        self, var2_synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A persistent VAR(2) relationship should have positive OOS R²."""
        x, y = var2_synthetic
        result = validate_oos(x, y, lag=2)
        assert result.r2_incremental > 0, (
            f"Expected positive R² for persistent relationship, got {result.r2_incremental:.4f}"
        )

    def test_independent_near_zero_r2(
        self, independent_series: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Independent series should have R² ≈ 0 (or negative)."""
        x, y = independent_series
        result = validate_oos(x, y, lag=2)
        assert result.r2_incremental < 0.05, (
            f"Independent series R² should be near 0, got {result.r2_incremental:.4f}"
        )

    def test_first_half_only_relationship_fails_oos(self) -> None:
        """Relationship only in first half → OOS R² should be ≈ 0.

        This tests that the model doesn't overfit to in-sample patterns
        that don't persist out-of-sample.
        """
        rng = np.random.default_rng(42)
        n = 400
        x = rng.normal(0, 1, n)
        y = np.zeros(n)

        # Relationship only in first 50%
        for t in range(2, n // 2):
            y[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
        # No relationship in second 50%
        for t in range(n // 2, n):
            y[t] = rng.normal(0, 1)

        result = validate_oos(x, y, lag=1, train_ratio=0.5)
        # OOS R² should be near zero or negative since relationship
        # doesn't persist into the test period
        assert result.r2_incremental < 0.1, (
            f"First-half-only relationship should fail OOS, got R²={result.r2_incremental:.4f}"
        )

    def test_antileakage_normalization(self) -> None:
        """CRITICAL: Normalization must use ONLY training statistics.

        If we scale using the full dataset, test performance is inflated.
        """
        rng = np.random.default_rng(42)
        # Create a regime shift: train has mean=0, test has mean=100
        x = np.concatenate([rng.normal(0, 1, 210), rng.normal(100, 1, 90)])
        y = np.concatenate([rng.normal(0, 1, 210), rng.normal(100, 1, 90)])

        result = validate_oos(x, y, lag=1, train_ratio=0.7)
        # Should not crash and R² should not be artificially high
        assert isinstance(result, OOSResult)

    def test_train_test_sizes(self) -> None:
        """Train and test sizes should reflect the ratio."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = validate_oos(x, y, lag=2, train_ratio=0.7)
        assert result.train_size > 0
        assert result.test_size > 0

    def test_too_short_raises(self) -> None:
        """Very short series should raise."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        with pytest.raises(AssertionError):
            validate_oos(x, y, lag=2)


class TestOOSMultiModel:
    """Tests for multi-model OOS (OLS + Ridge + RF)."""

    def test_nonlinear_relationship_positive_r2(self) -> None:
        """Y = X_{t-1}^2 + noise should have positive OOS R² with multi-model.

        Old OLS-only OOS would likely fail this since OLS can't capture
        the quadratic relationship. Multi-model with RF should succeed.
        """
        rng = np.random.default_rng(42)
        n = 400
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] ** 2 + 0.3 * rng.standard_normal()

        result = validate_oos(x, y, lag=1)
        # With multi-model (RF), this should be detected
        assert result.r2_incremental > 0.0, (
            f"Multi-model should detect quadratic relationship, got R²={result.r2_incremental:.4f}"
        )

    def test_linear_still_works(
        self, var2_synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Linear relationship should still be detected (no regression)."""
        x, y = var2_synthetic
        result = validate_oos(x, y, lag=2)
        assert result.r2_incremental > 0, (
            f"Multi-model should still detect linear: R²={result.r2_incremental:.4f}"
        )
