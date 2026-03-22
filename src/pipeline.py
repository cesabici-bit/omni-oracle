"""Pipeline orchestrator: connects all modules end-to-end.

Usage:
    python -m src.pipeline --source fred --limit 50
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.discovery.lagged_mi import detect_direction_lagged_mi
from src.discovery.mi_screening import compute_mi_with_pvalue
from src.models import Hypothesis, PairResult, TimeSeries
from src.output.export import export_json, export_stdout
from src.preprocess.alignment import align_pair
from src.preprocess.quality import check_quality
from src.preprocess.stationarity import check_and_transform
from src.scoring.ranker import rank_pairs
from src.storage.repo import TimeSeriesRepo
from src.validation.fdr import benjamini_hochberg
from src.validation.temporal_oos import validate_oos


def run_pipeline(config: PipelineConfig) -> list[Hypothesis]:
    """Execute the full discovery pipeline.

    Steps:
    1. Load series from DB
    2. Quality filter
    3. Stationarity transform
    4. MI screening (all pairs)
    5. Granger causality (surviving pairs)
    6. FDR correction
    7. OOS validation
    8. Score and rank
    """
    print("=== OmniOracle Pipeline ===")
    print(f"DB: {config.db_path}")
    print(f"Sources: {config.sources}")
    print()

    # Ensure data directory exists
    config.db_path.parent.mkdir(parents=True, exist_ok=True)

    repo = TimeSeriesRepo(config.db_path)

    # Step 1: Load all series
    all_series = repo.list_series()
    if config.sources:
        all_series = [s for s in all_series if s.source in config.sources]
    print(f"[1/7] Loaded {len(all_series)} series")

    if len(all_series) < 2:
        print("ERROR: Need at least 2 series. Run ingest first.")
        repo.close()
        return []

    # Step 2-3: Quality filter + resample to monthly + stationarity transform
    processed: dict[str, pd.Series] = {}  # keeps DatetimeIndex for alignment
    series_meta: dict[str, TimeSeries] = {}
    for s in all_series:
        obs_df = repo.get_observations(s.variable_id)
        if obs_df.empty:
            continue

        series_values = pd.Series(
            obs_df["value"].values,
            index=pd.to_datetime(obs_df["ts"]),
        )

        # Resample to monthly (handles daily, weekly, quarterly, annual)
        series_values = series_values.sort_index()
        if len(series_values) >= 2:
            median_gap = series_values.index.to_series().diff().dropna().dt.days.median()
            if median_gap < 25:
                # Sub-monthly (daily/weekly) -> resample to monthly mean
                series_values = series_values.resample("MS").mean().dropna()
            elif median_gap > 80:
                # Quarterly or annual -> keep as-is but note lower freq
                pass  # will have fewer points, alignment will handle it

        # Quality check
        qr = check_quality(series_values)
        if not qr.passed:
            continue

        # Stationarity transform
        try:
            clean_values = series_values.dropna()
            transformed, stat_result = check_and_transform(clean_values)
            if len(transformed) < config.min_observations:
                continue
            processed[s.variable_id] = transformed  # keep as pd.Series with index
            series_meta[s.variable_id] = s
        except Exception:
            continue

    print(f"[2/7] After quality + stationarity: {len(processed)} series")

    if len(processed) < 2:
        print("ERROR: Less than 2 series after preprocessing.")
        repo.close()
        return []

    # Step 4: MI screening — all pairs
    var_ids = sorted(processed.keys())
    n_pairs_total = len(var_ids) * (len(var_ids) - 1) // 2

    # Optional Spearman pre-screen for scalability
    # Checks multiple lags (0, 1, 3, 6, 12) to avoid killing lag-only relationships
    candidate_pairs: set[tuple[int, int]] | None = None
    if config.spearman_prescreen > 0 and len(var_ids) > 50:
        from scipy.stats import spearmanr

        spearman_lags = [0, 1, 3, 6, 12]
        print(f"[3a/7] Lagged Spearman pre-screen (|rho| > {config.spearman_prescreen}, "
              f"lags={spearman_lags}) on {n_pairs_total} pairs...")
        candidate_pairs = set()
        spearman_done = 0
        for i, j in itertools.combinations(range(len(var_ids)), 2):
            x_id, y_id = var_ids[i], var_ids[j]
            x_ser, y_ser = processed[x_id], processed[y_id]
            x_al, y_al = align_pair(x_ser, y_ser, method="inner")
            if len(x_al) < config.min_observations:
                spearman_done += 1
                continue

            x_vals = x_al.values
            y_vals = y_al.values
            n_al = len(x_vals)
            passed = False

            for lag_k in spearman_lags:
                if lag_k == 0:
                    rho, _ = spearmanr(x_vals, y_vals)
                elif n_al > lag_k + 20:
                    # x leads y (x[:-lag] vs y[lag:])
                    rho_xy, _ = spearmanr(x_vals[:-lag_k], y_vals[lag_k:])
                    # y leads x (y[:-lag] vs x[lag:])
                    rho_yx, _ = spearmanr(y_vals[:-lag_k], x_vals[lag_k:])
                    rho = max(abs(rho_xy), abs(rho_yx))
                else:
                    continue

                if abs(rho) >= config.spearman_prescreen:
                    passed = True
                    break

            if passed:
                candidate_pairs.add((i, j))
            spearman_done += 1
            if spearman_done % 500 == 0:
                print(f"  ... Spearman: {spearman_done}/{n_pairs_total} "
                      f"({len(candidate_pairs)} pass)", flush=True)
        print(f"  Lagged Spearman pre-screen: {len(candidate_pairs)}/{n_pairs_total} "
              f"pairs pass (filtered {n_pairs_total - len(candidate_pairs)})")

    # Determine which pairs to screen with MI
    if candidate_pairs is not None:
        pairs_to_screen = sorted(candidate_pairs)
        n_pairs = len(pairs_to_screen)
    else:
        pairs_to_screen = list(itertools.combinations(range(len(var_ids)), 2))
        n_pairs = n_pairs_total

    print(f"[3/7] MI screening on {n_pairs} pairs...")

    mi_results: list[tuple[str, str, float, float]] = []  # (x, y, mi, pval)
    pairs_done = 0
    for i, j in pairs_to_screen:
        x_id, y_id = var_ids[i], var_ids[j]
        x_ser, y_ser = processed[x_id], processed[y_id]

        # Align by date (inner join on DatetimeIndex)
        x_aligned, y_aligned = align_pair(x_ser, y_ser, method="inner")
        if len(x_aligned) < config.min_observations:
            pairs_done += 1
            continue

        x_arr = x_aligned.values.astype(float)
        y_arr = y_aligned.values.astype(float)

        result = compute_mi_with_pvalue(
            x_arr, y_arr,
            n_permutations=config.mi_permutations,
            threshold=config.mi_pvalue_threshold,
        )
        if result.significant:
            mi_results.append((x_id, y_id, result.mi, result.pvalue))

        pairs_done += 1
        if pairs_done % 50 == 0:
            n_sig = len(mi_results)
            print(f"  ... MI: {pairs_done}/{n_pairs} ({n_sig} sig)", flush=True)

    print(f"[4/7] MI significant pairs: {len(mi_results)}")

    if not mi_results:
        print("No significant MI pairs found.")
        repo.close()
        return []

    # Step 5: Lagged MI directional discovery on surviving pairs
    print(f"[5/7] Lagged MI direction on {len(mi_results)} pairs...")

    # Prepare aligned arrays for parallel processing
    lmi_inputs: list[tuple[str, str, float, float, np.ndarray, np.ndarray]] = []
    for x_id, y_id, mi_val, mi_pval in mi_results:
        x_ser, y_ser = processed[x_id], processed[y_id]
        x_aligned, y_aligned = align_pair(x_ser, y_ser, method="inner")
        x_cut = x_aligned.values.astype(float)
        y_cut = y_aligned.values.astype(float)
        lmi_inputs.append((x_id, y_id, mi_val, mi_pval, x_cut, y_cut))

    def _run_lagged_mi(
        args: tuple[str, str, float, float, np.ndarray, np.ndarray],
    ) -> PairResult | None:
        x_id, y_id, mi_val, mi_pval, x_cut, y_cut = args
        try:
            lmi = detect_direction_lagged_mi(
                x_cut, y_cut,
                max_lag=config.max_lag,
                n_permutations=config.direction_permutations,
                threshold=config.direction_pvalue_threshold,
            )
        except Exception:
            return None
        if lmi.direction == "none":
            return None
        return PairResult(
            x=x_id, y=y_id,
            mi=mi_val, mi_pvalue=mi_pval,
            direction=lmi.direction,
            direction_pvalue=lmi.best_pvalue,
            best_lag=lmi.best_lag,
            fdr_significant=False,
            oos_r2=0.0, oos_valid=False,
        )

    # Parallel execution (use all available CPUs)
    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(_run_lagged_mi)(inp) for inp in lmi_inputs
        )
        pair_results = [r for r in results if r is not None]
    except ImportError:
        # Fallback to sequential if joblib not available
        pair_results = []
        for idx, inp in enumerate(lmi_inputs):
            r = _run_lagged_mi(inp)
            if r is not None:
                pair_results.append(r)
            if (idx + 1) % 20 == 0:
                print(f"  ... Lagged MI: {idx + 1}/{len(lmi_inputs)} "
                      f"({len(pair_results)} directional)", flush=True)

    print(f"[5/7] Directional pairs: {len(pair_results)}")

    if not pair_results:
        print("No directional pairs found.")
        repo.close()
        return []

    # Step 6: FDR correction
    pvalues = [p.direction_pvalue for p in pair_results]
    fdr_mask = benjamini_hochberg(pvalues, alpha=config.fdr_alpha)
    for pr, is_sig in zip(pair_results, fdr_mask):
        pr.fdr_significant = is_sig

    fdr_survivors = [p for p in pair_results if p.fdr_significant]
    print(f"[6/7] After FDR: {len(fdr_survivors)} pairs")

    # Step 7: OOS validation on FDR survivors
    for pr in fdr_survivors:
        x_ser, y_ser = processed[pr.x], processed[pr.y]
        x_aligned, y_aligned = align_pair(x_ser, y_ser, method="inner")
        x_cut = x_aligned.values.astype(float)
        y_cut = y_aligned.values.astype(float)

        try:
            oos = validate_oos(
                x_cut, y_cut,
                lag=pr.best_lag,
                train_ratio=config.oos_train_ratio,
                r2_threshold=config.oos_r2_threshold,
            )
            pr.oos_r2 = oos.r2_incremental
            pr.oos_valid = oos.valid
        except Exception:
            pr.oos_r2 = 0.0
            pr.oos_valid = False

    # Score and rank all FDR survivors (including those that failed OOS)
    scored = rank_pairs(fdr_survivors)

    # Build Hypothesis objects
    hypotheses: list[Hypothesis] = []
    for rank, sp in enumerate(scored, 1):
        pr = sp.pair
        x_meta = series_meta.get(pr.x)
        y_meta = series_meta.get(pr.y)
        if x_meta is None or y_meta is None:
            continue

        caveats: list[str] = []
        if not pr.oos_valid:
            caveats.append("Failed out-of-sample validation")
        if pr.direction == "bidirectional":
            caveats.append("Bidirectional -- may indicate common driver")

        hypotheses.append(Hypothesis(
            rank=rank,
            score=sp.score,
            x=x_meta,
            y=y_meta,
            direction=pr.direction,
            lag=pr.best_lag,
            mi=pr.mi,
            direction_pvalue=pr.direction_pvalue,
            oos_r2=pr.oos_r2,
            confidence=sp.confidence,
            caveats=caveats,
        ))

    print(f"[7/7] Final hypotheses: {len(hypotheses)}")
    print()

    repo.close()
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniOracle Discovery Pipeline")
    parser.add_argument("--source", nargs="+", default=["fred"], help="Data sources")
    parser.add_argument("--limit", type=int, default=50, help="Max series per source")
    parser.add_argument("--db", type=str, default=None, help="DuckDB path")
    parser.add_argument("--output-json", type=str, default=None, help="Export JSON path")
    args = parser.parse_args()

    config = PipelineConfig(
        sources=args.source,
        limit=args.limit,
    )
    if args.db:
        config.db_path = Path(args.db)

    hypotheses = run_pipeline(config)

    if hypotheses:
        export_stdout(hypotheses)
        if args.output_json:
            export_json(hypotheses, args.output_json)
            print(f"\nExported to {args.output_json}")
    else:
        print("No hypotheses generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
