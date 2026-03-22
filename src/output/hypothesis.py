"""Hypothesis card rendering for human-readable output."""

from __future__ import annotations

from src.models import Hypothesis


def render_card(h: Hypothesis) -> str:
    """Render a single hypothesis as a human-readable card."""
    caveats_str = ""
    if h.caveats:
        caveats_str = "\n".join(f"    [!] {c}" for c in h.caveats)
        caveats_str = f"\n  Caveats:\n{caveats_str}"

    return (
        f"+-----------------------------------------------------------+\n"
        f"| #{h.rank}  Score: {h.score:.1f}/10  [{h.confidence.upper()}]\n"
        f"+-----------------------------------------------------------+\n"
        f"|  {h.x.name}\n"
        f"|    {h.direction}  (lag: {h.lag} periods)\n"
        f"|  {h.y.name}\n"
        f"|\n"
        f"|  MI: {h.mi:.4f}  |  Dir p: {h.direction_pvalue:.2e}  |  OOS R2: {h.oos_r2:.4f}\n"
        f"|  Sources: {h.x.source}:{h.x.variable_id}"
        f" -> {h.y.source}:{h.y.variable_id}{caveats_str}\n"
        f"|\n"
        f"|  Predictive association, NOT proof of causation.\n"
        f"+-----------------------------------------------------------+"
    )


def render_report(hypotheses: list[Hypothesis]) -> str:
    """Render a full report of ranked hypotheses."""
    if not hypotheses:
        return "No significant hypotheses found."

    header = (
        "=" * 60 + "\n"
        "  OmniOracle -- Hypothesis Report\n"
        "=" * 60 + "\n\n"
        f"  Total hypotheses: {len(hypotheses)}\n"
        f"  High confidence: {sum(1 for h in hypotheses if h.confidence == 'high')}\n"
        f"  Medium confidence: {sum(1 for h in hypotheses if h.confidence == 'medium')}\n"
        f"  Low confidence: {sum(1 for h in hypotheses if h.confidence == 'low')}\n\n"
        "  DISCLAIMER: All results are predictive associations.\n"
        "  They are NOT proof of causation. Use as hypotheses for\n"
        "  further investigation.\n\n"
        "-" * 60 + "\n"
    )

    cards = "\n\n".join(render_card(h) for h in hypotheses)
    return header + cards
