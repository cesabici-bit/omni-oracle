"""Core data models for OmniOracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class TimeSeries:
    """Metadata for an ingested time series variable."""

    variable_id: str  # e.g. "fred:CPIAUCSL"
    source: str  # "fred" | "worldbank" | "noaa"
    name: str  # "Consumer Price Index for All Urban Consumers"
    frequency: str  # "monthly" | "quarterly" | "annual"
    unit: str  # "index" | "percent" | "dollars"
    geo: str  # "US" | "IT" | "GLOBAL"
    observations: int  # number of data points
    start_date: date
    end_date: date


@dataclass
class PairResult:
    """Result of pairwise statistical analysis between two variables."""

    x: str  # variable_id of X
    y: str  # variable_id of Y
    mi: float  # Mutual Information (nats)
    mi_pvalue: float  # p-value from permutation test
    granger_direction: str  # "x→y" | "y→x" | "bidirectional" | "none"
    granger_pvalue: float  # p-value Granger (best direction)
    granger_lag: int  # optimal lag (BIC)
    fdr_significant: bool  # survives BH correction?
    oos_r2: float  # incremental R² out-of-sample
    oos_valid: bool  # OOS R² > threshold


@dataclass
class Hypothesis:
    """A ranked hypothesis ready for presentation."""

    rank: int
    score: float  # composite 0-10
    x: TimeSeries
    y: TimeSeries
    direction: str  # "x→y" | "y→x" | "bidirectional"
    lag: int
    mi: float
    granger_pvalue: float
    oos_r2: float
    confidence: str  # "high" | "medium" | "low"
    caveats: list[str] = field(default_factory=list)
