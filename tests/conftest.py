"""Shared fixtures for OmniOracle tests.

Provides synthetic time series with known properties for deterministic testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import TimeSeries
from src.storage.repo import TimeSeriesRepo


@pytest.fixture
def repo() -> TimeSeriesRepo:
    """In-memory DuckDB repo."""
    r = TimeSeriesRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def sample_series_meta() -> list[TimeSeries]:
    """10 sample TimeSeries metadata objects."""
    from datetime import date

    series = []
    for i in range(10):
        series.append(
            TimeSeries(
                variable_id=f"test:VAR{i:02d}",
                source="test",
                name=f"Test Variable {i}",
                frequency="monthly",
                unit="index",
                geo="US",
                observations=240,
                start_date=date(2000, 1, 1),
                end_date=date(2019, 12, 1),
            )
        )
    return series


@pytest.fixture
def sample_observations() -> pd.DataFrame:
    """240 monthly observations (2000-2019)."""
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    rng = np.random.default_rng(42)
    values = rng.normal(100, 10, size=240)
    return pd.DataFrame({"ts": dates.date, "value": values})


@pytest.fixture
def var2_synthetic() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic VAR(2) process where X Granger-causes Y but not vice versa.

    X(t) = 0.5*X(t-1) + eps_x
    Y(t) = 0.3*Y(t-1) + 0.5*X(t-2) + eps_y

    So X→Y with lag 2, beta=0.5. Y does NOT cause X.
    """
    rng = np.random.default_rng(42)
    n = 500
    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
        y[t] = 0.3 * y[t - 1] + 0.5 * x[t - 2] + rng.normal(0, 1)

    # Discard burn-in
    return x[50:], y[50:]


@pytest.fixture
def independent_series() -> tuple[np.ndarray, np.ndarray]:
    """Two independent random series."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, size=300)
    y = rng.normal(0, 1, size=300)
    return x, y


@pytest.fixture
def correlated_gaussian() -> tuple[np.ndarray, np.ndarray, float]:
    """Bivariate Gaussian with known correlation ρ=0.7.

    Returns (x, y, rho).
    """
    rng = np.random.default_rng(42)
    rho = 0.7
    n = 500
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    x = z1
    y = rho * z1 + np.sqrt(1 - rho**2) * z2
    return x, y, rho


@pytest.fixture
def random_walk() -> pd.Series:
    """Random walk (non-stationary) series."""
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 1, 300)
    walk = np.cumsum(steps)
    idx = pd.date_range("2000-01-01", periods=300, freq="MS")
    return pd.Series(walk, index=idx)


@pytest.fixture
def white_noise() -> pd.Series:
    """White noise (stationary) series."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 300)
    idx = pd.date_range("2000-01-01", periods=300, freq="MS")
    return pd.Series(noise, index=idx)


@pytest.fixture
def constant_series() -> pd.Series:
    """Constant series (zero variance)."""
    idx = pd.date_range("2000-01-01", periods=200, freq="MS")
    return pd.Series(np.ones(200) * 42.0, index=idx)
