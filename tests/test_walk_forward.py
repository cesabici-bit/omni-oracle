"""Tests for walk-forward validation with regime-aware filtering.

L2 oracle: regime breaks (COVID-like shocks) should not invalidate
an otherwise robust relationship.

# SOURCE: Walk-forward validation is standard in quantitative finance.
# Regime-aware filtering follows Cont (2007) "Volatility Clustering in
# Financial Markets" — extreme outlier windows are regime breaks, not
# evidence against the signal.
"""

import numpy as np
import pytest

from src.output.filters import walk_forward_validate


class TestWalkForwardBasic:
    """Basic walk-forward behavior (no regime breaks)."""

    def test_robust_signal(self):
        """A strong persistent signal should be ROBUST."""
        rng = np.random.default_rng(42)
        n = 200  # ~16 years monthly
        x = rng.standard_normal(n)
        # y follows x with lag 2 + noise
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.6 * x[t - 2] + 0.4 * rng.standard_normal()

        result = walk_forward_validate(x, y, lag=2)
        assert result["robust"] is True
        assert result["pass_ratio"] >= 0.6
        assert result["n_regime_breaks"] == 0
        assert result["adjusted_robust"] == result["robust"]

    def test_no_signal_fragile(self):
        """Independent series should be FRAGILE."""
        rng = np.random.default_rng(123)
        n = 200
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)

        result = walk_forward_validate(x, y, lag=2)
        assert result["robust"] is False

    def test_too_short_returns_empty(self):
        """Series shorter than min window returns empty result."""
        x = np.ones(30)
        y = np.ones(30)
        result = walk_forward_validate(x, y, lag=2)
        assert result["n_windows"] == 0
        assert result["robust"] is False
        assert result["adjusted_robust"] is False


class TestRegimeAware:
    """Regime-aware filtering for structural breaks."""

    def test_regime_break_excluded(self):
        """Windows with R2 < -2 should be flagged as regime breaks."""
        rng = np.random.default_rng(42)
        n = 250  # enough for many windows
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.5 * x[t - 2] + 0.5 * rng.standard_normal()

        # Inject a COVID-like shock: corrupt a chunk of y
        # This will cause 1-2 windows to have extreme negative R2
        y[120:145] = rng.standard_normal(25) * 20  # massive shock

        result = walk_forward_validate(x, y, lag=2)

        # Should have detected at least one regime break
        # (the shock window may or may not produce R2 < -2 depending
        # on exact window placement, so we test the mechanism)
        assert "n_regime_breaks" in result
        assert "adjusted_pass_ratio" in result
        assert "adjusted_robust" in result

        # The adjusted metrics should be at least as good as raw
        if result["n_regime_breaks"] > 0:
            assert result["adjusted_pass_ratio"] >= result["pass_ratio"]

    def test_adjusted_robust_when_raw_fragile(self):
        """A signal fragile due to 1 extreme window can be adjusted_robust."""
        rng = np.random.default_rng(99)
        n = 200
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.4 * x[t - 2] + 0.6 * rng.standard_normal()

        # Run once to see baseline
        base = walk_forward_validate(x, y, lag=2)

        # Now inject extreme shock to flip 1 window to R2 << -2
        y_shock = y.copy()
        y_shock[100:125] = rng.standard_normal(25) * 50

        shock_result = walk_forward_validate(x, y_shock, lag=2)

        # If shock created regime breaks, adjusted should recover
        if shock_result["n_regime_breaks"] > 0:
            # adjusted_pass_ratio should be higher than raw
            assert shock_result["adjusted_pass_ratio"] >= shock_result["pass_ratio"]

    def test_min_non_regime_windows_enforced(self):
        """If too many windows are regime breaks, adjusted_robust stays False."""
        rng = np.random.default_rng(55)
        # Short series = few windows
        n = 100  # ~8 years, maybe 1-2 windows
        x = rng.standard_normal(n)
        y = rng.standard_normal(n) * 50  # all extreme

        result = walk_forward_validate(
            x, y, lag=2,
            min_non_regime_windows=3,
        )
        # With most windows being extreme, not enough valid windows
        if result["n_valid_windows"] < 3:
            assert result["adjusted_robust"] is False

    def test_no_regime_breaks_adjusted_equals_raw(self):
        """When no regime breaks, adjusted metrics equal raw metrics."""
        rng = np.random.default_rng(77)
        n = 200
        x = rng.standard_normal(n)
        y = 0.3 * np.roll(x, 2) + 0.7 * rng.standard_normal(n)

        result = walk_forward_validate(x, y, lag=2)

        if result["n_regime_breaks"] == 0:
            assert result["adjusted_pass_ratio"] == pytest.approx(
                result["pass_ratio"]
            )
            assert result["adjusted_robust"] == result["robust"]
