"""Data quality checks: min length, variance, NaN ratio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import MAX_NAN_RATIO, MIN_OBSERVATIONS, MIN_VARIANCE


@dataclass
class QualityResult:
    """Result of quality checks on a series."""

    passed: bool
    reason: str  # "" if passed, else why it failed
    n_observations: int
    nan_ratio: float
    variance: float


def check_quality(
    series: pd.Series,
    min_obs: int = MIN_OBSERVATIONS,
    min_var: float = MIN_VARIANCE,
    max_nan: float = MAX_NAN_RATIO,
) -> QualityResult:
    """Check if a series meets minimum quality requirements."""
    total = len(series)
    n_nan = int(series.isna().sum())
    nan_ratio = n_nan / total if total > 0 else 1.0
    n_valid = total - n_nan
    variance = float(np.nanvar(series.values)) if n_valid > 1 else 0.0

    if total == 0:
        return QualityResult(False, "empty series", 0, 1.0, 0.0)

    if nan_ratio > max_nan:
        return QualityResult(
            False,
            f"NaN ratio {nan_ratio:.2%} > {max_nan:.2%}",
            n_valid,
            nan_ratio,
            variance,
        )

    if n_valid < min_obs:
        return QualityResult(
            False,
            f"only {n_valid} valid obs < {min_obs}",
            n_valid,
            nan_ratio,
            variance,
        )

    if variance < min_var:
        return QualityResult(
            False,
            f"variance {variance:.2e} < {min_var:.2e} (constant series)",
            n_valid,
            nan_ratio,
            variance,
        )

    return QualityResult(True, "", n_valid, nan_ratio, variance)
