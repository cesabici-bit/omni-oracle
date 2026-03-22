"""Export hypotheses to JSON, CSV, and stdout."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.models import Hypothesis
from src.output.hypothesis import render_report


def to_dict(h: Hypothesis) -> dict:
    """Convert a Hypothesis to a plain dict for serialization."""
    return {
        "rank": h.rank,
        "score": h.score,
        "x_id": h.x.variable_id,
        "x_name": h.x.name,
        "y_id": h.y.variable_id,
        "y_name": h.y.name,
        "direction": h.direction,
        "lag": h.lag,
        "mi": h.mi,
        "direction_pvalue": h.direction_pvalue,
        "oos_r2": h.oos_r2,
        "confidence": h.confidence,
        "caveats": h.caveats,
    }


def export_json(
    hypotheses: list[Hypothesis],
    path: str | Path,
) -> None:
    """Export hypotheses to a JSON file."""
    data = {
        "disclaimer": "Predictive associations, NOT proof of causation.",
        "total": len(hypotheses),
        "hypotheses": [to_dict(h) for h in hypotheses],
    }
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def export_csv(
    hypotheses: list[Hypothesis],
    path: str | Path,
) -> None:
    """Export hypotheses to a CSV file."""
    if not hypotheses:
        Path(path).write_text("")
        return

    fieldnames = [
        "rank", "score", "x_id", "x_name", "y_id", "y_name",
        "direction", "lag", "mi", "direction_pvalue", "oos_r2",
        "confidence", "caveats",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in hypotheses:
            row = to_dict(h)
            row["caveats"] = "; ".join(row["caveats"])
            writer.writerow(row)


def export_stdout(hypotheses: list[Hypothesis]) -> None:
    """Print hypothesis report to stdout."""
    print(render_report(hypotheses))
