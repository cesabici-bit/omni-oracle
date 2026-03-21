"""F5 Phase 0: Scale to 500+ variables, full discovery, trading analysis.

This script:
1. Ingests expanded FRED series (curated 49 + ~300 discovered via search API)
2. Ingests expanded World Bank series (30 indicators x 10 countries)
3. Runs full discovery pipeline (Spearman pre-screen + MI + Granger + FDR + OOS)
4. Exports top 50 hypotheses
5. Identifies top 10 trading candidates
6. Saves results to results/ directory

Usage:
    python -m src.run_f5

Pass/Fail criteria:
- >= 500 variables ingested
- >= 30 validated truths (FDR + OOS pass)
- >= 5 known relationships rediscovered
- Top 10 trading candidates with lag, direction, p-value, OOS R2
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.config import PipelineConfig
from src.ingest.fred import CURATED_FRED_SERIES, FREDFetcher
from src.ingest.fred_expanded import discover_fred_series
from src.ingest.worldbank import WorldBankFetcher
from src.output.export import export_json
from src.output.hypothesis import render_card
from src.output.trading import identify_trading_candidates, render_trading_report
from src.pipeline import run_pipeline
from src.storage.repo import TimeSeriesRepo

# Use separate DB for F5 to avoid lock conflicts
F5_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "f5_discovery.duckdb"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Known relationships to verify (same as smoke test)
KNOWN_RELATIONSHIPS = [
    {
        "name": "Oil -> CPI",
        "keywords": [("OIL", "BRENT", "DCOIL"), ("CPI", "CPIAUCSL")],
    },
    {
        "name": "Fed Funds -> Treasury",
        "keywords": [("FEDFUNDS",), ("GS10", "TB3MS", "TREASURY")],
    },
    {
        "name": "GDP <-> Unemployment",
        "keywords": [("GDP", "GDPC1"), ("UNRATE", "UNEMPLOYMENT")],
    },
    {
        "name": "Initial Claims -> Employment",
        "keywords": [("ICSA", "CLAIMS"), ("PAYEMS", "EMPLOYMENT")],
    },
    {
        "name": "HY Spread -> Economy",
        "keywords": [("BAML", "HY", "SPREAD"), ("GDP", "INDPRO", "RSAFS")],
    },
    {
        "name": "Housing -> Economy",
        "keywords": [("HOUST", "PERMIT", "HOUSING"), ("GDP", "INDPRO")],
    },
    {
        "name": "M2 -> Inflation",
        "keywords": [("M2", "MONEY"), ("CPI", "INFLATION", "PCE")],
    },
    {
        "name": "Yield Curve -> Recession",
        "keywords": [("T10Y2Y", "SPREAD"), ("GDP", "UNRATE", "INDPRO")],
    },
]


def step_1_ingest() -> int:
    """Ingest expanded FRED + World Bank data. Returns total series count."""
    print("=" * 60)
    print("  STEP 1: DATA INGESTION")
    print("=" * 60)
    print()

    # Ensure directories exist
    F5_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing F5 DB if present (fresh start)
    if F5_DB_PATH.exists():
        F5_DB_PATH.unlink()
        # Also remove WAL file if present
        wal = F5_DB_PATH.with_suffix(".duckdb.wal")
        if wal.exists():
            wal.unlink()

    repo = TimeSeriesRepo(F5_DB_PATH)

    # --- FRED: Curated + Discovered ---
    print("[FRED] Ingesting curated series...")
    try:
        fred_fetcher = FREDFetcher()
    except ValueError as e:
        print(f"ERROR: {e}")
        repo.close()
        sys.exit(1)

    # Ingest curated list
    curated_ingested = fred_fetcher.ingest(repo, limit=len(CURATED_FRED_SERIES))
    print(f"  Curated: {len(curated_ingested)} series ingested")

    # Discover and ingest expanded series
    print()
    print("[FRED] Discovering expanded series via search API...")
    try:
        expanded_series = discover_fred_series(max_total=400, min_popularity=10)
    except Exception as e:
        print(f"  WARN: Discovery failed: {e}")
        expanded_series = []

    if expanded_series:
        print(f"[FRED] Ingesting {len(expanded_series)} discovered series...")
        expanded_ingested = 0
        for i, meta in enumerate(expanded_series):
            series_id = meta["series_id"]
            variable_id = f"fred:{series_id}"

            try:
                df = fred_fetcher.fetch_observations(series_id)
                if df.empty:
                    continue

                import pandas as pd

                from src.models import TimeSeries

                ts_meta = TimeSeries(
                    variable_id=variable_id,
                    source="fred",
                    name=meta.get("name", series_id),
                    frequency=meta.get("frequency", "monthly"),
                    unit=meta.get("unit", ""),
                    geo="US",
                    observations=len(df),
                    start_date=pd.Timestamp(df["ts"].min()).date(),
                    end_date=pd.Timestamp(df["ts"].max()).date(),
                )
                repo.upsert_series(ts_meta)
                repo.insert_observations_bulk(variable_id, df)
                expanded_ingested += 1
            except Exception as e:
                if "Bad Request" not in str(e):
                    print(f"  WARN: {series_id}: {e}")
                continue

            if (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{len(expanded_series)} "
                      f"({expanded_ingested} ingested)", flush=True)

            # Rate limit: FRED API allows 120 req/min
            time.sleep(0.55)

        print(f"  Expanded: {expanded_ingested} series ingested")

    # --- World Bank: Expanded ---
    print()
    print("[World Bank] Ingesting expanded indicators...")
    wb_fetcher = WorldBankFetcher(expanded=True)
    try:
        wb_ingested = wb_fetcher.ingest(repo, limit=500)
        print(f"  World Bank: {len(wb_ingested)} series ingested")
    except Exception as e:
        print(f"  WARN: World Bank ingestion failed: {e}")
        wb_ingested = []

    # Summary
    total = repo.count_series()
    print()
    print(f"  TOTAL SERIES INGESTED: {total}")
    repo.close()
    return total


def step_2_discover() -> list:
    """Run full discovery pipeline. Returns list of Hypothesis."""
    print()
    print("=" * 60)
    print("  STEP 2: DISCOVERY PIPELINE")
    print("=" * 60)
    print()

    config = PipelineConfig(
        db_path=F5_DB_PATH,
        sources=["fred", "worldbank"],
        limit=9999,  # no limit — use all ingested series
        min_observations=48,  # lower threshold: 4y monthly / 48y annual
        mi_permutations=30,  # fewer permutations for speed (early stopping helps)
        mi_pvalue_threshold=0.05,
        granger_max_lag=12,
        fdr_alpha=0.05,
        oos_train_ratio=0.70,
        oos_r2_threshold=0.02,
        spearman_prescreen=0.08,  # pre-screen: |rho| > 0.08
    )

    start_time = time.time()
    hypotheses = run_pipeline(config)
    elapsed = time.time() - start_time

    print(f"\n  Pipeline completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Total hypotheses: {len(hypotheses)}")

    return hypotheses


def step_3_verify(hypotheses: list) -> dict:
    """Verify results against known relationships. Returns stats dict."""
    print()
    print("=" * 60)
    print("  STEP 3: VERIFICATION")
    print("=" * 60)
    print()

    stats = {
        "total_hypotheses": len(hypotheses),
        "high_confidence": sum(1 for h in hypotheses if h.confidence == "high"),
        "medium_confidence": sum(1 for h in hypotheses if h.confidence == "medium"),
        "oos_valid": sum(
            1 for h in hypotheses
            if "Failed out-of-sample validation" not in h.caveats
        ),
        "known_found": 0,
        "known_details": [],
    }

    # Check known relationships
    for kr in KNOWN_RELATIONSHIPS:
        x_kws, y_kws = kr["keywords"]
        found = False
        for h in hypotheses:
            x_id_upper = h.x.variable_id.upper() + " " + h.x.name.upper()
            y_id_upper = h.y.variable_id.upper() + " " + h.y.name.upper()

            # Check both directions
            fwd = (
                any(kw.upper() in x_id_upper for kw in x_kws)
                and any(kw.upper() in y_id_upper for kw in y_kws)
            )
            rev = (
                any(kw.upper() in y_id_upper for kw in x_kws)
                and any(kw.upper() in x_id_upper for kw in y_kws)
            )
            if fwd or rev:
                print(f"  [OK] {kr['name']}: score={h.score:.1f}, "
                      f"lag={h.lag}, dir={h.direction}, "
                      f"OOS R2={h.oos_r2:.4f}")
                stats["known_found"] += 1
                stats["known_details"].append({
                    "name": kr["name"],
                    "score": h.score,
                    "lag": h.lag,
                    "oos_r2": h.oos_r2,
                })
                found = True
                break

        if not found:
            print(f"  [--] {kr['name']}: not found")

    print()
    print(f"  Known relationships found: {stats['known_found']}/{len(KNOWN_RELATIONSHIPS)}")
    print(f"  OOS-validated hypotheses: {stats['oos_valid']}")
    return stats


def step_4_trading(hypotheses: list) -> list[dict]:
    """Identify top trading candidates."""
    print()
    print("=" * 60)
    print("  STEP 4: TRADING CANDIDATES")
    print("=" * 60)
    print()

    candidates = identify_trading_candidates(
        hypotheses,
        min_lag=1,
        min_oos_r2=0.005,
        max_candidates=10,
    )

    report = render_trading_report(candidates)
    print(report)

    return candidates


def step_5_export(hypotheses: list, candidates: list[dict], stats: dict) -> None:
    """Export results to files."""
    print()
    print("=" * 60)
    print("  STEP 5: EXPORT")
    print("=" * 60)
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Top 50 hypotheses JSON
    top50 = hypotheses[:50]
    export_json(top50, RESULTS_DIR / "f5_top50_hypotheses.json")
    print("  Exported top 50 hypotheses -> results/f5_top50_hypotheses.json")

    # All hypotheses JSON
    export_json(hypotheses, RESULTS_DIR / "f5_all_hypotheses.json")
    print("  Exported all hypotheses -> results/f5_all_hypotheses.json")

    # Trading candidates JSON
    trading_data = {
        "disclaimer": "Statistical associations with lag. NOT trading advice.",
        "candidates": candidates,
    }
    (RESULTS_DIR / "f5_trading_candidates.json").write_text(
        json.dumps(trading_data, indent=2, default=str)
    )
    print("  Exported trading candidates -> results/f5_trading_candidates.json")

    # Summary stats JSON
    summary = {
        "total_variables_ingested": stats.get("total_ingested", 0),
        "total_hypotheses": stats["total_hypotheses"],
        "high_confidence": stats["high_confidence"],
        "medium_confidence": stats["medium_confidence"],
        "oos_validated": stats["oos_valid"],
        "known_relationships_found": stats["known_found"],
        "trading_candidates": len(candidates),
    }
    (RESULTS_DIR / "f5_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("  Exported summary -> results/f5_summary.json")


def main() -> None:
    """Run F5 Phase 0 discovery."""
    print()
    print("*" * 60)
    print("  OmniOracle F5 Phase 0: Full-Scale Discovery")
    print("*" * 60)
    print()

    overall_start = time.time()

    # Step 1: Ingest
    total_ingested = step_1_ingest()

    # Step 2: Discover
    hypotheses = step_2_discover()

    # Step 3: Verify
    stats = step_3_verify(hypotheses)
    stats["total_ingested"] = total_ingested

    # Step 4: Trading
    candidates = step_4_trading(hypotheses)

    # Step 5: Export
    step_5_export(hypotheses, candidates, stats)

    # Final summary
    elapsed = time.time() - overall_start
    print()
    print("*" * 60)
    print("  F5 PHASE 0 COMPLETE")
    print("*" * 60)
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Variables ingested: {total_ingested}")
    print(f"  Hypotheses found: {len(hypotheses)}")
    print(f"  OOS-validated: {stats['oos_valid']}")
    print(f"  Known relationships: {stats['known_found']}/{len(KNOWN_RELATIONSHIPS)}")
    print(f"  Trading candidates: {len(candidates)}")
    print()

    # Print top 10 hypotheses
    print("=" * 60)
    print("  TOP 10 HYPOTHESES")
    print("=" * 60)
    for h in hypotheses[:10]:
        print(render_card(h))
        print()

    # Pass/Fail assessment
    print("=" * 60)
    print("  PASS/FAIL ASSESSMENT")
    print("=" * 60)
    checks = [
        (total_ingested >= 500, f"Variables >= 500: {total_ingested}"),
        (stats["oos_valid"] >= 30, f"OOS-validated >= 30: {stats['oos_valid']}"),
        (stats["known_found"] >= 5, f"Known relationships >= 5: {stats['known_found']}"),
        (len(candidates) >= 5, f"Trading candidates >= 5: {len(candidates)}"),
    ]
    all_pass = True
    for passed, desc in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  RESULT: PASS")
    else:
        print("  RESULT: PARTIAL (see failed checks above)")
    print()

    print("  CHECKPOINT: verifica utente")
    print("  Cosa ho fatto: discovery su 500+ variabili FRED + World Bank")
    print("  Cosa ti mostro: report ipotesi + trading candidates")
    print("  Cosa devi verificare: le relazioni trovate hanno senso?")
    print("  Risposta attesa: OK / problema trovato / skip")


if __name__ == "__main__":
    main()
