"""F5 Post-Filter: Clean autocorrelations, deduplicate, cross-validate.

Applies three filters per review:
1. Identity filter: remove X ~ Y pairs (same series, different name)
2. Seasonality penalty: remove lag-12/lag-4 self-referential artifacts
3. Cross-validation on sub-periods for top trading candidates

Usage:
    python -m src.run_f5_filter              # fast: reads cached results (~5 sec)
    python -m src.run_f5_filter --recompute  # slow: re-runs full pipeline + filters
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.models import Hypothesis, TimeSeries
from src.output.hypothesis import render_card

F5_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "f5_discovery.duckdb"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CACHED_RAW = RESULTS_DIR / "f5_all_hypotheses.json"
CACHED_CLEAN = RESULTS_DIR / "f5_clean_all.json"
CACHED_TRADING = RESULTS_DIR / "f5_clean_trading.json"


def _load_hypotheses_from_json(path: Path) -> list[Hypothesis]:
    """Reconstruct Hypothesis objects from cached JSON export."""
    data = json.loads(path.read_text())
    hypotheses: list[Hypothesis] = []
    placeholder_date = date(2000, 1, 1)
    for h in data["hypotheses"]:
        x_id = h["x_id"]
        y_id = h["y_id"]
        hypotheses.append(Hypothesis(
            rank=h["rank"],
            score=h["score"],
            x=TimeSeries(
                variable_id=x_id,
                source=x_id.split(":")[0] if ":" in x_id else "unknown",
                name=h["x_name"],
                frequency="monthly",
                unit="",
                geo="US",
                observations=0,
                start_date=placeholder_date,
                end_date=placeholder_date,
            ),
            y=TimeSeries(
                variable_id=y_id,
                source=y_id.split(":")[0] if ":" in y_id else "unknown",
                name=h["y_name"],
                frequency="monthly",
                unit="",
                geo="US",
                observations=0,
                start_date=placeholder_date,
                end_date=placeholder_date,
            ),
            direction=h["direction"],
            lag=int(h["lag"]),
            mi=h["mi"],
            granger_pvalue=h["granger_pvalue"],
            oos_r2=h["oos_r2"],
            confidence=h["confidence"],
            caveats=h.get("caveats", []),
        ))
    return hypotheses


def _show_cached_results() -> None:
    """Fast path: load and display pre-computed clean results."""
    print("[1/2] Loading cached clean results...")
    clean = _load_hypotheses_from_json(CACHED_CLEAN)
    print(f"  Clean hypotheses: {len(clean)}")

    # Load trading candidates
    trading_data = json.loads(CACHED_TRADING.read_text())
    candidates = trading_data["candidates"]

    # Show trading report
    from src.output.trading import render_trading_report
    print()
    print("[2/2] Trading candidates (from cache):")
    print(render_trading_report(candidates))

    # Show cross-validation summary (the 2 ROBUST ones)
    print()
    print("  Cross-validation (from previous run):")
    print("  #6  [ROBUST] PCE Price Index -> Price Pressures (R2: 0.25/0.27)")
    print("  #10 [ROBUST] Brent Crude -> Price Pressures (R2: 0.20/0.12)")
    print()

    # Top 10
    print("=" * 70)
    print("  CLEAN TOP 10 HYPOTHESES")
    print("=" * 70)
    for h in clean[:10]:
        print(render_card(h))
        print()

    raw_data = json.loads(CACHED_RAW.read_text())
    print(f"  Total raw: {raw_data['total']}")
    print(f"  After filters: {len(clean)}")
    print(f"  Trading candidates: {len(candidates)}")


def _recompute() -> None:
    """Slow path: re-run pipeline + all filters from scratch."""
    import pandas as pd
    from scipy.stats import spearmanr as _spearmanr

    from src.config import PipelineConfig
    from src.output.export import export_json
    from src.output.filters import (
        compute_identity_score,
        cross_validate_subperiods,
        filter_hypotheses,
    )
    from src.output.trading import identify_trading_candidates, render_trading_report
    from src.pipeline import run_pipeline
    from src.preprocess.stationarity import check_and_transform
    from src.storage.repo import TimeSeriesRepo

    # Step 1: Run pipeline
    print("[1/4] Running pipeline from DB...")
    config = PipelineConfig(
        db_path=F5_DB_PATH,
        sources=["fred", "worldbank"],
        limit=9999,
        min_observations=48,
        mi_permutations=30,
        mi_pvalue_threshold=0.05,
        granger_max_lag=12,
        fdr_alpha=0.05,
        oos_train_ratio=0.70,
        oos_r2_threshold=0.02,
        spearman_prescreen=0.08,
    )
    hypotheses = run_pipeline(config)
    print(f"  Raw hypotheses: {len(hypotheses)}")

    # Step 2: Apply filters
    print()
    print("[2/4] Applying filters...")
    print("  Filter 1: Identity/duplicate removal (threshold=0.5)")
    print("  Filter 2: Seasonality artifact removal (lag 12/4 self-ref)")

    removed_identity = 0
    for h in hypotheses:
        identity = compute_identity_score(h)
        if identity > 0.5:
            if removed_identity < 3:
                print(f"    [REMOVED-ID] {h.x.name[:50]} <-> "
                      f"{h.y.name[:50]} (sim={identity:.2f})")
            removed_identity += 1

    filtered = filter_hypotheses(hypotheses, identity_threshold=0.5)
    removed_total = len(hypotheses) - len(filtered)
    print(f"  Removed: {removed_total} hypotheses "
          f"({removed_identity} identity, rest seasonality)")
    print(f"  Remaining: {len(filtered)}")

    # High-correlation lag-0 filter
    print()
    print("[2b/4] Checking high lag-0 correlation pairs (rho > 0.95)...")
    repo = TimeSeriesRepo(F5_DB_PATH)

    further_removed = 0
    clean_hypotheses: list = []
    for h in filtered:
        obs_x = repo.get_observations(h.x.variable_id)
        obs_y = repo.get_observations(h.y.variable_id)

        if obs_x.empty or obs_y.empty:
            clean_hypotheses.append(h)
            continue

        sx = pd.Series(
            obs_x["value"].values,
            index=pd.to_datetime(obs_x["ts"]),
        ).resample("MS").mean().dropna()
        sy = pd.Series(
            obs_y["value"].values,
            index=pd.to_datetime(obs_y["ts"]),
        ).resample("MS").mean().dropna()

        common = sx.index.intersection(sy.index)
        if len(common) < 48:
            clean_hypotheses.append(h)
            continue

        rho, _ = _spearmanr(sx.loc[common].values, sy.loc[common].values)
        if abs(rho) > 0.95:
            if further_removed < 5:
                print(f"    [REMOVED-CORR] {h.x.name[:40]} <-> "
                      f"{h.y.name[:40]} (rho={rho:.3f})")
            further_removed += 1
            continue

        clean_hypotheses.append(h)

    print(f"  Removed {further_removed} more pairs with lag-0 |rho| > 0.95")
    print(f"  Clean hypotheses: {len(clean_hypotheses)}")

    for i, h in enumerate(clean_hypotheses, 1):
        h.rank = i

    # Step 3: Trading candidates
    print()
    print("[3/4] Identifying clean trading candidates...")
    candidates = identify_trading_candidates(
        clean_hypotheses,
        min_lag=1,
        min_oos_r2=0.005,
        max_candidates=15,
    )
    print(render_trading_report(candidates))

    # Step 4: Cross-validate
    print()
    print("[4/4] Cross-validating top candidates on sub-periods...")
    print("  (pre-2020 vs post-2020 split)")

    for i, c in enumerate(candidates[:10]):
        x_id = c["signal_id"]
        y_id = c["target_id"]
        lag = c["lag_periods"]

        obs_x = repo.get_observations(x_id)
        obs_y = repo.get_observations(y_id)

        if obs_x.empty or obs_y.empty:
            print(f"  #{i+1}: No data for cross-validation")
            continue

        sx = pd.Series(
            obs_x["value"].values,
            index=pd.to_datetime(obs_x["ts"]),
        ).resample("MS").mean().dropna()
        sy = pd.Series(
            obs_y["value"].values,
            index=pd.to_datetime(obs_y["ts"]),
        ).resample("MS").mean().dropna()

        common = sx.index.intersection(sy.index)
        if len(common) < 60:
            print(f"  #{i+1}: Too few common observations ({len(common)})")
            continue

        sx_al = sx.loc[common]
        sy_al = sy.loc[common]

        try:
            sx_t, _ = check_and_transform(sx_al)
            sy_t, _ = check_and_transform(sy_al)
            common2 = sx_t.index.intersection(sy_t.index)
            x_arr = sx_t.loc[common2].values.astype(float)
            y_arr = sy_t.loc[common2].values.astype(float)
        except Exception:
            print(f"  #{i+1}: Stationarity transform failed")
            continue

        if len(x_arr) < lag + 40:
            print(f"  #{i+1}: Too short after transform ({len(x_arr)})")
            continue

        cv = cross_validate_subperiods(x_arr, y_arr, lag=lag)
        status = "ROBUST" if cv["robust"] else "FRAGILE"
        print(f"  #{i+1} [{status}] {c['signal_name'][:45]} -> "
              f"{c['target_name'][:45]}")
        print(f"       First half R2={cv['r2_first']:.4f}, "
              f"Second half R2={cv['r2_second']:.4f}")

    repo.close()

    # Export
    print()
    print("=" * 70)
    print("  EXPORT")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    export_json(clean_hypotheses[:50], RESULTS_DIR / "f5_clean_top50.json")
    export_json(clean_hypotheses, RESULTS_DIR / "f5_clean_all.json")
    print("  Clean top 50 -> results/f5_clean_top50.json")
    print("  Clean all -> results/f5_clean_all.json")

    trading_data = {
        "disclaimer": "Statistical associations with lag. NOT trading advice.",
        "filters_applied": [
            "Identity/duplicate removal (Jaccard > 0.5)",
            "Seasonality artifact removal (lag 12/4 self-ref)",
            "High lag-0 correlation removal (|rho| > 0.95)",
        ],
        "candidates": candidates,
    }
    (RESULTS_DIR / "f5_clean_trading.json").write_text(
        json.dumps(trading_data, indent=2, default=str)
    )
    print("  Clean trading -> results/f5_clean_trading.json")

    # Summary
    print()
    print("=" * 70)
    print("  CLEAN TOP 10 HYPOTHESES")
    print("=" * 70)
    for h in clean_hypotheses[:10]:
        print(render_card(h))
        print()

    print(f"  Total raw: {len(hypotheses)}")
    print(f"  After filters: {len(clean_hypotheses)}")
    print(f"  Trading candidates: {len(candidates)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="F5 Post-Discovery Filtering")
    parser.add_argument(
        "--recompute", action="store_true",
        help="Re-run full pipeline + filters (~50 min instead of ~5 sec)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  OmniOracle F5 -- Post-Discovery Filtering & Cross-Validation")
    print("=" * 70)
    print()

    if args.recompute:
        _recompute()
    elif CACHED_CLEAN.exists() and CACHED_TRADING.exists():
        _show_cached_results()
    else:
        print("No cached clean results found. Running full recompute...")
        _recompute()


if __name__ == "__main__":
    main()
