"""Global configuration, constants, and thresholds for OmniOracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "omnioracle.duckdb"


# --- Preprocessing thresholds ---
MIN_OBSERVATIONS = 120  # minimum data points for a series to be included
MIN_VARIANCE = 1e-10  # series with variance below this are constant → discard
MAX_NAN_RATIO = 0.20  # series with >20% NaN are discarded
STATIONARITY_ALPHA = 0.05  # significance level for ADF/KPSS tests


# --- MI Screening ---
MI_PERMUTATIONS = 200  # number of permutations for MI p-value
MI_PVALUE_THRESHOLD = 0.05  # p-value threshold for MI screening
MI_TOP_FRACTION = 0.10  # keep top 10% of pairs by MI (alternative to p-value)


# --- Directional Discovery (Lagged MI) ---
MAX_LAG = 12  # maximum lag to test (months)
DIRECTION_PERMUTATIONS = 100  # permutations for lagged MI p-value
DIRECTION_PVALUE_THRESHOLD = 0.05  # p-value threshold before FDR

# Backward-compatible aliases (deprecated)
GRANGER_MAX_LAG = MAX_LAG
GRANGER_PVALUE_THRESHOLD = DIRECTION_PVALUE_THRESHOLD


# --- FDR ---
FDR_ALPHA = 0.05  # Benjamini-Hochberg target FDR


# --- OOS Validation ---
OOS_TRAIN_RATIO = 0.70  # temporal train/test split
OOS_R2_THRESHOLD = 0.02  # minimum incremental R² to consider OOS valid


# --- Scoring weights ---
SCORE_WEIGHT_MI = 0.25
SCORE_WEIGHT_DIRECTION = 0.25
SCORE_WEIGHT_OOS = 0.35
SCORE_WEIGHT_EFFECT = 0.15

# Backward-compatible alias (deprecated)
SCORE_WEIGHT_GRANGER = SCORE_WEIGHT_DIRECTION


@dataclass
class PipelineConfig:
    """Runtime configuration for a pipeline run."""

    db_path: Path = DEFAULT_DB_PATH
    sources: list[str] = field(default_factory=lambda: ["fred"])
    limit: int = 100  # max series per source
    min_observations: int = MIN_OBSERVATIONS
    mi_permutations: int = MI_PERMUTATIONS
    mi_pvalue_threshold: float = MI_PVALUE_THRESHOLD
    max_lag: int = MAX_LAG
    direction_permutations: int = DIRECTION_PERMUTATIONS
    direction_pvalue_threshold: float = DIRECTION_PVALUE_THRESHOLD
    fdr_alpha: float = FDR_ALPHA
    oos_train_ratio: float = OOS_TRAIN_RATIO
    oos_r2_threshold: float = OOS_R2_THRESHOLD
    spearman_prescreen: float = 0.0  # if > 0, skip MI for pairs with |rho| < this
