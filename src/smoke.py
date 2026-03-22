"""Smoke test E2E (M3): ingest curated FRED series, run pipeline, verify known relationships.

This is the golden snapshot (L4) -- requires human verification.

"OmniOracle funziona" means:
- At least 20 hypotheses with score > 3/10
- At least 3 known relationships found (Oil->CPI, FedFunds->Treasury, GDP<->Unemployment)
- Every hypothesis has: score, MI, p-value, lag, direction, R2 OOS, caveats
- Zero NaN/inf in output
- Disclaimer present

Usage:
    python -m src.smoke
"""

from __future__ import annotations

import math
import sys

from src.config import PipelineConfig
from src.output.hypothesis import render_card, render_report
from src.pipeline import run_pipeline

# Known relationships the pipeline MUST find
KNOWN_RELATIONSHIPS = [
    {
        "name": "Oil -> CPI",
        "x_contains": ["OIL", "BRENT", "DCOIL"],
        "y_contains": ["CPI", "CPIAUCSL"],
        "expected_lag_range": (1, 12),
    },
    {
        "name": "Fed Funds -> Treasury",
        "x_contains": ["FEDFUNDS"],
        "y_contains": ["GS10", "TB3MS", "TREASURY"],
        "expected_lag_range": (0, 6),
    },
    {
        "name": "GDP <-> Unemployment (Okun's Law)",
        "x_contains": ["GDP", "GDPC1"],
        "y_contains": ["UNRATE", "UNEMPLOYMENT"],
        "expected_lag_range": (1, 8),
        "bidirectional_ok": True,
    },
]


def run_smoke() -> bool:
    """Run the smoke test. Returns True if all checks pass."""
    print("=" * 60)
    print("  OmniOracle Smoke Test (M3)")
    print("=" * 60)
    print()

    config = PipelineConfig(
        sources=["fred"],
        limit=50,
        mi_permutations=50,  # fewer permutations for speed (early-stopping helps too)
    )

    # Ensure data directory exists
    config.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if DB has data -- if not, need ingest first
    from src.storage.repo import TimeSeriesRepo

    repo = TimeSeriesRepo(config.db_path)
    n_series = repo.count_series()
    repo.close()

    if n_series < 10:
        print("ERROR: Not enough data in DB. Run ingest first:")
        print("  python -m src.ingest.fred")
        print(f"  (found {n_series} series, need at least 10)")
        return False

    # Run pipeline
    hypotheses = run_pipeline(config)

    # === Checks ===
    checks_passed = 0
    checks_total = 5

    # Check 1: At least 20 hypotheses with score > 3
    high_score = [h for h in hypotheses if h.score > 3.0]
    if len(high_score) >= 20:
        print(f"[OK] Check 1: {len(high_score)} hypotheses with score > 3 (need >=20)")
        checks_passed += 1
    else:
        print(f"[FAIL] Check 1: Only {len(high_score)} hypotheses with score > 3 (need >=20)")

    # Check 2: Known relationships found
    found_known = 0
    for kr in KNOWN_RELATIONSHIPS:
        found = False
        for h in hypotheses:
            x_match = any(
                kw.upper() in h.x.variable_id.upper() or kw.upper() in h.x.name.upper()
                for kw in kr["x_contains"]
            )
            y_match = any(
                kw.upper() in h.y.variable_id.upper() or kw.upper() in h.y.name.upper()
                for kw in kr["y_contains"]
            )
            # Also check reverse direction
            x_match_rev = any(
                kw.upper() in h.y.variable_id.upper() or kw.upper() in h.y.name.upper()
                for kw in kr["x_contains"]
            )
            y_match_rev = any(
                kw.upper() in h.x.variable_id.upper() or kw.upper() in h.x.name.upper()
                for kw in kr["y_contains"]
            )

            if (x_match and y_match) or (x_match_rev and y_match_rev):
                print(f"  [OK] Found: {kr['name']} (score={h.score}, lag={h.lag})")
                found = True
                found_known += 1
                break

        if not found:
            print(f"  [FAIL] Not found: {kr['name']}")

    if found_known >= 3:
        print(f"[OK] Check 2: {found_known}/3 known relationships found")
        checks_passed += 1
    else:
        print(f"[FAIL] Check 2: Only {found_known}/3 known relationships found")

    # Check 3: All fields present and valid
    all_valid = True
    for h in hypotheses:
        if math.isnan(h.score) or math.isinf(h.score):
            all_valid = False
            break
        if math.isnan(h.mi) or math.isinf(h.mi):
            all_valid = False
            break
        if math.isnan(h.oos_r2) or math.isinf(h.oos_r2):
            all_valid = False
            break
    if all_valid:
        print("[OK] Check 3: No NaN/inf in output")
        checks_passed += 1
    else:
        print("[FAIL] Check 3: Found NaN/inf in output")

    # Check 4: Each hypothesis has all required fields
    fields_ok = all(
        h.score is not None
        and h.mi is not None
        and h.direction_pvalue is not None
        and h.lag is not None
        and h.direction is not None
        and h.oos_r2 is not None
        and h.caveats is not None
        for h in hypotheses
    )
    if fields_ok:
        print("[OK] Check 4: All hypothesis fields present")
        checks_passed += 1
    else:
        print("[FAIL] Check 4: Missing fields in some hypotheses")

    # Check 5: Disclaimer present in report
    report = render_report(hypotheses)
    if "NOT proof of causation" in report or "not proof of causation" in report.lower():
        print("[OK] Check 5: Disclaimer present")
        checks_passed += 1
    else:
        print("[FAIL] Check 5: Disclaimer missing")

    # Summary
    print()
    print(f"{'=' * 60}")
    print(f"  Smoke Test: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print("  RESULT: PASS")
        print()
        print("  CHECKPOINT: verifica utente")
        print("  Cosa ho fatto: pipeline E2E su serie FRED")
        print("  Cosa ti mostro: report ipotesi (vedi sotto)")
        print("  Cosa devi verificare: le relazioni trovate hanno senso?")
        print()
        # Print summary stats + top 10 hypotheses (not all 389)
        top_n = 10
        print("=" * 60)
        print(f"  OmniOracle -- Top {top_n} Hypotheses")
        print("=" * 60)
        print(f"  Total: {len(hypotheses)}  |  "
              f"High: {sum(1 for h in hypotheses if h.confidence == 'high')}  |  "
              f"Medium: {sum(1 for h in hypotheses if h.confidence == 'medium')}  |  "
              f"Low: {sum(1 for h in hypotheses if h.confidence == 'low')}")
        print()
        for h in hypotheses[:top_n]:
            print(render_card(h))
            print()
        print("  DISCLAIMER: All results are predictive associations.")
        print("  They are NOT proof of causation.")
    else:
        print("  RESULT: FAIL")

    return checks_passed == checks_total


if __name__ == "__main__":
    success = run_smoke()
    sys.exit(0 if success else 1)
