"""Stationarity testing and transformation.

Uses the confirmatory strategy (Pfaff 2008):
ADF (H0: unit root) + KPSS (H0: stationarity) together to decide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    """Result of stationarity analysis."""

    is_stationary: bool
    adf_pvalue: float
    kpss_pvalue: float
    transformations: list[str]  # e.g. ["differenced_1"]
    original_order: int  # 0 = already stationary, 1 = I(1), etc.


def check_and_transform(
    series: pd.Series,
    alpha: float = 0.05,
    max_diffs: int = 2,
) -> tuple[pd.Series, StationarityResult]:
    """Test stationarity and transform if needed.

    Confirmatory strategy (Pfaff 2008):
    1. ADF reject + KPSS not reject → stationary
    2. ADF not reject + KPSS reject → unit root → difference
    3. Both reject → trend-stationary → detrend
    4. Neither reject → inconclusive → difference (conservative)

    Returns (transformed_series, metadata).
    """
    assert len(series.dropna()) >= 20, (
        f"Series too short for stationarity test: {len(series.dropna())} points"
    )

    current = series.dropna().copy()
    transformations: list[str] = []
    n_diffs = 0

    for _ in range(max_diffs + 1):
        adf_pval = _safe_adf(current)
        kpss_pval = _safe_kpss(current)

        adf_rejects = adf_pval < alpha
        kpss_rejects = kpss_pval < alpha

        if adf_rejects and not kpss_rejects:
            # Case 1: stationary
            return current, StationarityResult(
                is_stationary=True,
                adf_pvalue=adf_pval,
                kpss_pvalue=kpss_pval,
                transformations=transformations,
                original_order=n_diffs,
            )

        if not adf_rejects and kpss_rejects:
            # Case 2: unit root → difference
            current = current.diff().dropna()
            n_diffs += 1
            transformations.append(f"differenced_{n_diffs}")
            continue

        if adf_rejects and kpss_rejects:
            # Case 3: trend-stationary → detrend
            current = _detrend(current)
            transformations.append("detrended")
            # Re-check after detrending
            adf_pval = _safe_adf(current)
            kpss_pval = _safe_kpss(current)
            return current, StationarityResult(
                is_stationary=True,
                adf_pvalue=adf_pval,
                kpss_pvalue=kpss_pval,
                transformations=transformations,
                original_order=0,
            )

        # Case 4: neither rejects → inconclusive → difference (conservative)
        current = current.diff().dropna()
        n_diffs += 1
        transformations.append(f"differenced_{n_diffs}")

    # Exhausted max_diffs — return what we have
    adf_pval = _safe_adf(current)
    kpss_pval = _safe_kpss(current)
    return current, StationarityResult(
        is_stationary=adf_pval < alpha,
        adf_pvalue=adf_pval,
        kpss_pvalue=kpss_pval,
        transformations=transformations,
        original_order=n_diffs,
    )


def _safe_adf(series: pd.Series) -> float:
    """Run ADF test, return p-value. Returns 1.0 on failure."""
    try:
        result = adfuller(series, autolag="AIC")
        return float(result[1])
    except Exception:
        return 1.0


def _safe_kpss(series: pd.Series) -> float:
    """Run KPSS test, return p-value. Returns 0.0 on failure (conservative)."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(series, regression="c", nlags="auto")
        return float(result[1])
    except Exception:
        return 0.0


def _detrend(series: pd.Series) -> pd.Series:
    """Remove linear trend from series."""
    x = np.arange(len(series), dtype=float)
    coeffs = np.polyfit(x, series.values, 1)
    trend = np.polyval(coeffs, x)
    return pd.Series(series.values - trend, index=series.index)
