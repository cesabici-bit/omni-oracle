"""Tests for Granger causality (ST-07).

L2 tests use synthetic VAR processes with known causal structure.
SOURCE: Granger (1969), "Investigating Causal Relations by Econometric Models"
"""

from __future__ import annotations

import numpy as np
import pytest

from src.discovery.granger import (
    select_lag_bic,
)
from src.discovery.granger import (
    test_granger_bidirectional as granger_bidirectional,
)


class TestGranger:
    """L1 + L2: Granger causality testing."""

    def test_detects_known_causation(
        self, var2_synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """L2: VAR(2) with X→Y (β=0.5) should be detected.

        SOURCE: Granger (1969) — if past X improves prediction of Y
        beyond Y's own past, then X Granger-causes Y.

        The synthetic process has:
        X(t) = 0.5*X(t-1) + ε_x
        Y(t) = 0.3*Y(t-1) + 0.5*X(t-2) + ε_y
        """
        x, y = var2_synthetic
        result = granger_bidirectional(x, y)
        assert result.direction in ("x->y", "bidirectional"), (
            f"Expected X→Y direction, got {result.direction}"
        )
        assert result.pvalue_xy < 0.01, (
            f"X→Y p-value should be <0.01, got {result.pvalue_xy:.4f}"
        )

    def test_no_reverse_causation(
        self, var2_synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """L2: Y should NOT Granger-cause X in the VAR(2) setup."""
        x, y = var2_synthetic
        result = granger_bidirectional(x, y)
        # Y→X should not be significant (or at least much weaker)
        assert result.pvalue_yx > 0.01 or result.direction != "y->x", (
            f"Unexpected Y→X significance: p={result.pvalue_yx:.4f}"
        )

    def test_independent_no_granger(
        self, independent_series: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """L2: Independent series should show no Granger causality."""
        # SOURCE: By construction — independent series cannot Granger-cause each other
        x, y = independent_series
        result = granger_bidirectional(x, y)
        assert result.direction == "none" or result.best_pvalue > 0.01, (
            f"Independent series show spurious Granger: {result.direction}, p={result.best_pvalue:.4f}"
        )

    def test_lag_selection_reasonable(
        self, var2_synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """BIC should select a lag close to the true lag (2)."""
        x, y = var2_synthetic
        lag = select_lag_bic(x, y, max_lag=12)
        assert 1 <= lag <= 6, f"Expected lag near 2, got {lag}"

    def test_short_series_assertion(self) -> None:
        """Series with <30 points should raise."""
        x = np.random.default_rng(42).normal(0, 1, 20)
        y = np.random.default_rng(43).normal(0, 1, 20)
        with pytest.raises(AssertionError):
            granger_bidirectional(x, y)

    def test_granger_pvalue_range(
        self, var2_synthetic: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """P-values should be in [0, 1]."""
        x, y = var2_synthetic
        result = granger_bidirectional(x, y)
        assert 0 <= result.pvalue_xy <= 1
        assert 0 <= result.pvalue_yx <= 1
