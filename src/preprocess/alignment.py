"""Frequency alignment and temporal matching for time series pairs."""

from __future__ import annotations

import pandas as pd


def align_pair(
    x: pd.Series,
    y: pd.Series,
    method: str = "inner",
) -> tuple[pd.Series, pd.Series]:
    """Align two time series by their DatetimeIndex.

    Args:
        x: First series (DatetimeIndex expected).
        y: Second series (DatetimeIndex expected).
        method: "inner" (only common dates) or "outer" (union with NaN fill).

    Returns:
        Aligned (x, y) tuple with matching indices.
    """
    if method == "inner":
        common = x.index.intersection(y.index)
        return x.loc[common], y.loc[common]
    elif method == "outer":
        combined = x.index.union(y.index)
        return x.reindex(combined), y.reindex(combined)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def downsample_to_lower_frequency(
    high_freq: pd.Series,
    low_freq: pd.Series,
    agg: str = "mean",
) -> tuple[pd.Series, pd.Series]:
    """Downsample the higher-frequency series to match the lower-frequency one.

    Uses the lower-frequency series' dates as reference points and
    aggregates the higher-frequency data accordingly.
    """
    # Infer frequencies
    high_freq_days = _median_gap_days(high_freq)
    low_freq_days = _median_gap_days(low_freq)

    if high_freq_days >= low_freq_days:
        # high_freq is actually lower or same frequency — just align
        return align_pair(high_freq, low_freq)

    # Resample high_freq to match low_freq's approximate frequency
    freq_map = {
        (25, 35): "MS",  # monthly
        (85, 95): "QS",  # quarterly
        (360, 370): "YS",  # annual
    }
    target_rule = "MS"  # default to monthly
    for (lo, hi), rule in freq_map.items():
        if lo <= low_freq_days <= hi:
            target_rule = rule
            break

    resampled = high_freq.resample(target_rule).agg(agg)
    return align_pair(resampled, low_freq)


def _median_gap_days(series: pd.Series) -> float:
    """Compute median gap in days between consecutive observations."""
    if len(series) < 2:
        return 0.0
    diffs = pd.Series(series.index).diff().dropna()
    return float(diffs.dt.days.median())
