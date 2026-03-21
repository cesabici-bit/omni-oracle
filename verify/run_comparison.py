"""M4 Cross-Tool Verification: compare main pipeline vs alternative implementations.

This script:
1. Loads aligned pairs from the DuckDB database
2. Runs MI and Granger using BOTH the main pipeline (src/) and alternative (verify/)
3. Compares results and reports agreement/disagreement

The alternative implementations use:
- MI: histogram binning (vs sklearn KSG k-NN in src/)
- Granger: manual OLS + scipy F-test (vs statsmodels in src/)

Expected behavior:
- Granger p-values should agree closely (same test, different OLS implementation)
- MI values may differ in magnitude (different estimators) but ranking should agree
- Direction decisions should agree for strong signals
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for src/ imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alt_granger import test_granger_bidirectional_manual  # noqa: E402
from alt_mi import compute_mi_with_pvalue_histogram  # noqa: E402

from src.discovery.granger import test_granger_bidirectional  # noqa: E402
from src.discovery.mi_screening import compute_mi_with_pvalue  # noqa: E402
from src.preprocess.alignment import align_pair  # noqa: E402
from src.preprocess.stationarity import check_and_transform  # noqa: E402
from src.storage.repo import TimeSeriesRepo  # noqa: E402

# --- Configuration ---
DB_PATH = PROJECT_ROOT / "data" / "omnioracle.duckdb"
# Test pairs: known relationships + extra pairs for ranking comparison
TEST_PAIRS = [
    ("fred:FEDFUNDS", "fred:GS10"),       # Fed Funds -> Treasury
    ("fred:DCOILBRENTEU", "fred:CPIAUCSL"),  # Oil -> CPI
    ("fred:GDP", "fred:UNRATE"),          # GDP <-> Unemployment
    ("fred:ICSA", "fred:PAYEMS"),         # Initial Claims -> Employment
    ("fred:INDPRO", "fred:UNRATE"),       # Industrial Prod <-> Unemployment
    ("fred:M2SL", "fred:CPIAUCSL"),       # Money Supply -> CPI
    ("fred:TB3MS", "fred:FEDFUNDS"),      # 3-mo Treasury <-> Fed Funds
    ("fred:PPIACO", "fred:CPIAUCSL"),     # PPI -> CPI
]
# Tolerance for Granger p-value comparison
GRANGER_PVALUE_ATOL = 0.05  # absolute tolerance
# MI ranking agreement: Spearman correlation threshold
MI_RANK_THRESHOLD = 0.5


def load_stationary_pair(
    repo: TimeSeriesRepo, var_x: str, var_y: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load, align, and make stationary a pair of series."""
    df_x = repo.get_observations(var_x)
    df_y = repo.get_observations(var_y)

    if df_x.empty or df_y.empty:
        return None

    sx = pd.Series(df_x["value"].values, index=pd.to_datetime(df_x["ts"]))
    sy = pd.Series(df_y["value"].values, index=pd.to_datetime(df_y["ts"]))

    # Resample to monthly if needed (same logic as pipeline)
    for s_name, s in [("x", sx), ("y", sy)]:
        median_gap = s.index.to_series().diff().dropna().dt.days.median()
        if median_gap < 25:
            if s_name == "x":
                sx = s.resample("MS").mean().dropna()
            else:
                sy = s.resample("MS").mean().dropna()

    # Align
    sx_a, sy_a = align_pair(sx, sy, method="inner")
    if len(sx_a) < 30:
        return None

    # Make stationary
    sx_stat, _ = check_and_transform(sx_a)
    sy_stat, _ = check_and_transform(sy_a)

    # Re-align after differencing (may lose observations)
    sx_final, sy_final = align_pair(sx_stat, sy_stat, method="inner")
    if len(sx_final) < 30:
        return None

    return sx_final.values.astype(float), sy_final.values.astype(float)


def compare_granger(x: np.ndarray, y: np.ndarray, pair_name: str) -> dict:
    """Compare Granger results between src/ and verify/ implementations."""
    # Main pipeline
    gr_main = test_granger_bidirectional(x, y)
    # Alternative
    gr_alt = test_granger_bidirectional_manual(x, y)

    # Compare p-values
    pv_xy_diff = abs(gr_main.pvalue_xy - gr_alt["pvalue_xy"])
    pv_yx_diff = abs(gr_main.pvalue_yx - gr_alt["pvalue_yx"])
    direction_match = gr_main.direction == gr_alt["direction"]

    # Tolerance: p-values should be very close (same test)
    pv_ok = pv_xy_diff < GRANGER_PVALUE_ATOL and pv_yx_diff < GRANGER_PVALUE_ATOL

    return {
        "pair": pair_name,
        "main_direction": gr_main.direction,
        "alt_direction": gr_alt["direction"],
        "main_pv_xy": gr_main.pvalue_xy,
        "alt_pv_xy": gr_alt["pvalue_xy"],
        "main_pv_yx": gr_main.pvalue_yx,
        "alt_pv_yx": gr_alt["pvalue_yx"],
        "main_lag": gr_main.lag,
        "alt_lag": gr_alt["lag"],
        "pvalue_agree": pv_ok,
        "direction_agree": direction_match,
        "PASS": pv_ok and direction_match,
    }


def compare_mi(x: np.ndarray, y: np.ndarray, pair_name: str) -> dict:
    """Compare MI results between src/ (KSG) and verify/ (histogram)."""
    # Main pipeline (KSG k-NN)
    mi_main = compute_mi_with_pvalue(x, y, n_permutations=200, threshold=0.05)
    # Alternative (histogram)
    mi_alt = compute_mi_with_pvalue_histogram(x, y, n_permutations=200, threshold=0.05)

    # MI values will differ (different estimators), compare significance decision
    sig_match = mi_main.significant == mi_alt["significant"]

    return {
        "pair": pair_name,
        "main_mi": mi_main.mi,
        "alt_mi": mi_alt["mi"],
        "main_pvalue": mi_main.pvalue,
        "alt_pvalue": mi_alt["pvalue"],
        "main_significant": mi_main.significant,
        "alt_significant": mi_alt["significant"],
        "significance_agree": sig_match,
        "PASS": sig_match,
    }


def run_comparison() -> bool:
    """Run full M4 comparison. Returns True if all checks pass."""
    print("=" * 70)
    print("M4 CROSS-TOOL VERIFICATION")
    print("Main pipeline (src/) vs Alternative implementations (verify/)")
    print("=" * 70)

    repo = TimeSeriesRepo(DB_PATH)
    all_pass = True
    granger_results = []
    mi_results = []

    for var_x, var_y in TEST_PAIRS:
        pair_name = f"{var_x} <-> {var_y}"
        print(f"\n--- {pair_name} ---")

        pair_data = load_stationary_pair(repo, var_x, var_y)
        if pair_data is None:
            print("  SKIP: insufficient data")
            continue

        x, y = pair_data
        print(f"  Observations: {len(x)}")

        # Granger comparison
        gr = compare_granger(x, y, pair_name)
        granger_results.append(gr)
        status = "[OK]" if gr["PASS"] else "[FAIL]"
        print(f"  Granger {status}:")
        print(
            f"    Main:  dir={gr['main_direction']},"
            f" pv_xy={gr['main_pv_xy']:.4f},"
            f" pv_yx={gr['main_pv_yx']:.4f}, lag={gr['main_lag']}"
        )
        print(
            f"    Alt:   dir={gr['alt_direction']},"
            f" pv_xy={gr['alt_pv_xy']:.4f},"
            f" pv_yx={gr['alt_pv_yx']:.4f}, lag={gr['alt_lag']}"
        )
        if not gr["PASS"]:
            all_pass = False

        # MI comparison
        mi = compare_mi(x, y, pair_name)
        mi_results.append(mi)
        status = "[OK]" if mi["PASS"] else "[FAIL]"
        print(f"  MI {status}:")
        print(
            f"    Main (KSG):      mi={mi['main_mi']:.4f},"
            f" pv={mi['main_pvalue']:.4f},"
            f" sig={mi['main_significant']}"
        )
        print(
            f"    Alt (histogram): mi={mi['alt_mi']:.4f},"
            f" pv={mi['alt_pvalue']:.4f},"
            f" sig={mi['alt_significant']}"
        )
        if not mi["PASS"]:
            all_pass = False

    # --- MI Ranking Comparison (informational) ---
    if len(mi_results) >= 3:
        main_mis = [r["main_mi"] for r in mi_results]
        alt_mis = [r["alt_mi"] for r in mi_results]
        from scipy.stats import spearmanr
        rho, _ = spearmanr(main_mis, alt_mis)
        print("\n--- MI Ranking Agreement (informational) ---")
        print(f"  Spearman rho: {rho:.3f}")
        print("  NOTE: KSG and histogram estimators have different biases.")
        print("  Ranking divergence on weak signals is expected and acceptable.")

    # --- Summary ---
    n_granger_pass = sum(1 for r in granger_results if r["PASS"])
    n_mi_pass = sum(1 for r in mi_results if r["PASS"])
    n_total = len(granger_results) + len(mi_results)
    n_pass = n_granger_pass + n_mi_pass

    # Pass criteria: Granger must be 100%, MI >= 75% (different estimators)
    granger_ok = n_granger_pass == len(granger_results)
    mi_ok = n_mi_pass >= len(mi_results) * 0.75

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    print(f"  Granger: {n_granger_pass}/{len(granger_results)} (required: 100%)")
    print(f"  MI:      {n_mi_pass}/{len(mi_results)} (required: >=75%)")

    all_pass = granger_ok and mi_ok
    if all_pass:
        print("[OK] M4 CROSS-TOOL VERIFICATION PASSED")
    else:
        print("[FAIL] M4 CROSS-TOOL VERIFICATION FAILED")
    print(f"{'=' * 70}")

    return all_pass


if __name__ == "__main__":
    success = run_comparison()
    sys.exit(0 if success else 1)
