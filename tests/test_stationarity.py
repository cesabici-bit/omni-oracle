"""Tests for stationarity module (ST-05).

L2 tests use known statistical properties:
- Random walk → non-stationary → should be differenced
- White noise → stationary → no transformation
- Trend + noise → trend-stationary → should be detrended
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess.quality import check_quality
from src.preprocess.stationarity import check_and_transform


class TestStationarity:
    """L1 + L2: Stationarity detection and transformation."""

    def test_white_noise_is_stationary(self, white_noise: pd.Series) -> None:
        """L2: White noise should be identified as stationary, no transformation."""
        # SOURCE: By construction — iid N(0,1) is stationary by definition
        transformed, result = check_and_transform(white_noise)
        assert result.is_stationary
        assert result.original_order == 0
        assert len(result.transformations) == 0

    def test_random_walk_is_differenced(self, random_walk: pd.Series) -> None:
        """L2: Random walk should be differenced (I(1) process)."""
        # SOURCE: Hamilton (1994), Ch.15 — random walk is I(1)
        transformed, result = check_and_transform(random_walk)
        assert result.original_order >= 1
        assert "differenced_1" in result.transformations

    def test_trend_stationary(self) -> None:
        """L2: Linear trend + noise should be detrended or differenced."""
        # SOURCE: By construction — deterministic trend is not unit root
        rng = np.random.default_rng(42)
        t = np.arange(300)
        trend = 0.1 * t + rng.normal(0, 1, 300)
        idx = pd.date_range("2000-01-01", periods=300, freq="MS")
        series = pd.Series(trend, index=idx)

        transformed, result = check_and_transform(series)
        # Should either detrend or difference
        assert len(result.transformations) > 0

    def test_transformed_length(self, random_walk: pd.Series) -> None:
        """Transformed series should lose at most max_diffs points."""
        original_len = len(random_walk.dropna())
        transformed, result = check_and_transform(random_walk)
        assert len(transformed) >= original_len - 2  # max 2 differencing steps

    def test_short_series_raises(self) -> None:
        """Series < 20 points should raise assertion."""
        short = pd.Series(np.ones(10))
        with pytest.raises(AssertionError):
            check_and_transform(short)


class TestQuality:
    """L1: Quality filter checks."""

    def test_good_series_passes(self, white_noise: pd.Series) -> None:
        result = check_quality(white_noise, min_obs=100)
        assert result.passed

    def test_constant_series_fails(self, constant_series: pd.Series) -> None:
        result = check_quality(constant_series)
        assert not result.passed
        assert "constant" in result.reason.lower() or "variance" in result.reason.lower()

    def test_too_short_fails(self) -> None:
        short = pd.Series(np.arange(10, dtype=float))
        result = check_quality(short, min_obs=120)
        assert not result.passed
        assert "obs" in result.reason.lower() or "valid" in result.reason.lower()

    def test_too_many_nans_fails(self) -> None:
        series = pd.Series([np.nan] * 80 + list(range(20)))
        result = check_quality(series, min_obs=10, max_nan=0.2)
        assert not result.passed
        assert "nan" in result.reason.lower()

    def test_empty_series_fails(self) -> None:
        result = check_quality(pd.Series([], dtype=float))
        assert not result.passed
