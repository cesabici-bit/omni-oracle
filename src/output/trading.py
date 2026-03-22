"""Trading candidate identification from discovery results.

Identifies hypotheses with predictive lag relationships suitable
for trading signals. Ranks by out-of-sample predictive power.
"""

from __future__ import annotations

from src.models import Hypothesis


def identify_trading_candidates(
    hypotheses: list[Hypothesis],
    min_lag: int = 1,
    min_oos_r2: float = 0.01,
    max_candidates: int = 10,
) -> list[dict]:
    """Identify and rank trading candidates from hypotheses.

    Criteria for trading candidates:
    - Unidirectional direction (x->y or y->x, not bidirectional)
    - Lag >= min_lag (signal must lead target by at least 1 period)
    - OOS R2 > min_oos_r2 (out-of-sample predictive power)
    - FDR-significant (survives multiple testing correction)

    Returns list of dicts sorted by OOS R2 (descending), each with:
    - signal, target, direction, lag, oos_r2, direction_pvalue,
      mi, score, confidence, signal_description
    """
    candidates: list[dict] = []

    for h in hypotheses:
        # Skip bidirectional (common driver, not predictive)
        if h.direction == "bidirectional":
            continue

        # Must have positive lag
        if h.lag < min_lag:
            continue

        # Must have positive OOS R2
        if h.oos_r2 < min_oos_r2:
            continue

        # Must not have OOS failure caveat
        if "Failed out-of-sample validation" in h.caveats:
            continue

        # Determine signal and target based on direction
        if "x->y" in h.direction or "x→y" in h.direction:
            signal = h.x
            target = h.y
        elif "y->x" in h.direction or "y→x" in h.direction:
            signal = h.y
            target = h.x
        else:
            continue

        candidates.append({
            "signal_id": signal.variable_id,
            "signal_name": signal.name,
            "target_id": target.variable_id,
            "target_name": target.name,
            "direction": h.direction,
            "lag_periods": h.lag,
            "oos_r2": h.oos_r2,
            "direction_pvalue": h.direction_pvalue,
            "mi": h.mi,
            "score": h.score,
            "confidence": h.confidence,
            "signal_description": (
                f"{signal.name} predicts {target.name} "
                f"with {h.lag}-period lag (OOS R2={h.oos_r2:.4f}, "
                f"p={h.direction_pvalue:.2e})"
            ),
        })

    # Sort by OOS R2 (highest predictive power first)
    candidates.sort(key=lambda c: c["oos_r2"], reverse=True)
    return candidates[:max_candidates]


def render_trading_report(candidates: list[dict]) -> str:
    """Render trading candidates as human-readable report."""
    if not candidates:
        return "No trading candidates found."

    lines = [
        "=" * 70,
        "  OmniOracle -- Top Trading Candidates",
        "  (Unidirectional lagged MI with predictive lag)",
        "=" * 70,
        "",
    ]

    for i, c in enumerate(candidates, 1):
        lines.extend([
            f"  #{i}  [Score: {c['score']:.1f}/10]  [{c['confidence'].upper()}]",
            f"  Signal: {c['signal_name']}",
            f"    --> Target: {c['target_name']}",
            f"    Lag: {c['lag_periods']} periods  |  "
            f"OOS R2: {c['oos_r2']:.4f}  |  "
            f"Dir p: {c['direction_pvalue']:.2e}  |  "
            f"MI: {c['mi']:.4f}",
            f"    {c['signal_description']}",
            "",
        ])

    lines.extend([
        "-" * 70,
        "  DISCLAIMER: These are statistical associations with predictive lag.",
        "  They are NOT guaranteed to persist. Use as hypotheses for backtest.",
        "  Past statistical relationships do not guarantee future performance.",
        "-" * 70,
    ])

    return "\n".join(lines)
