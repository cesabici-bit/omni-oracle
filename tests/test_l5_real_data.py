"""L5 validation: verify pipeline rediscovers known economic relationships from real FRED data.

These tests use the actual DuckDB with ingested FRED series. They verify that
the discovery engine finds well-documented economic relationships WITHOUT any
hints about which variables to pair.

# SOURCE: All expected relationships are documented in economics literature:
# - Okun's Law: Okun (1962) "Potential GNP: Its Measurement and Significance"
# - Oil -> CPI: Hamilton (1983) "Oil and the Macroeconomy since World War II"
# - Fed Funds -> Treasury: Mishkin (2019) "The Economics of Money, Banking, and Financial Markets"

Requirements:
- data/omnioracle.duckdb must exist with ingested FRED data (49 series)
- Run: FRED_API_KEY=... python -m src.smoke  (to populate DB first)
"""

from __future__ import annotations

import pytest

from src.config import DEFAULT_DB_PATH, PipelineConfig
from src.models import Hypothesis
from src.pipeline import run_pipeline

DB_PATH = DEFAULT_DB_PATH

# Skip all tests if DB doesn't exist or has no data
pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason=f"Real data DB not found at {DB_PATH}. Run smoke test first.",
)


@pytest.fixture(scope="module")
def hypotheses() -> list[Hypothesis]:
    """Run the full pipeline once and cache results for all L5 tests."""
    config = PipelineConfig(
        db_path=DB_PATH,
        sources=["fred"],
        mi_permutations=200,
    )
    results = run_pipeline(config)
    assert len(results) > 0, "Pipeline returned no hypotheses -- DB may be empty"
    return results


def _find_relationship(
    hypotheses: list[Hypothesis],
    x_keywords: list[str],
    y_keywords: list[str],
    bidirectional_ok: bool = False,
) -> Hypothesis | None:
    """Find a hypothesis matching x->y (or y->x if bidirectional_ok)."""
    for h in hypotheses:
        x_match = any(
            kw.upper() in h.x.variable_id.upper() or kw.upper() in h.x.name.upper()
            for kw in x_keywords
        )
        y_match = any(
            kw.upper() in h.y.variable_id.upper() or kw.upper() in h.y.name.upper()
            for kw in y_keywords
        )
        if x_match and y_match:
            return h
        # Check reverse direction
        if bidirectional_ok:
            x_match_rev = any(
                kw.upper() in h.y.variable_id.upper() or kw.upper() in h.y.name.upper()
                for kw in x_keywords
            )
            y_match_rev = any(
                kw.upper() in h.x.variable_id.upper() or kw.upper() in h.x.name.upper()
                for kw in y_keywords
            )
            if x_match_rev and y_match_rev:
                return h
    return None


class TestL5KnownRelationships:
    """L5: Pipeline must rediscover known economic relationships without hints."""

    def test_fed_funds_predicts_treasury(self, hypotheses: list[Hypothesis]) -> None:
        """L5: Fed Funds Rate should Granger-cause Treasury yields.
        # SOURCE: Mishkin (2019) Ch.5 -- short-term rates lead long-term rates.
        """
        h = _find_relationship(
            hypotheses,
            x_keywords=["FEDFUNDS"],
            y_keywords=["GS10", "TB3MS", "TREASURY"],
            bidirectional_ok=True,
        )
        assert h is not None, (
            "Pipeline failed to discover Fed Funds -> Treasury relationship"
        )
        assert h.direction_pvalue < 0.05, (
            f"Fed Funds -> Treasury not significant: p={h.direction_pvalue:.4f}"
        )
        assert 0 <= h.lag <= 6, (
            f"Unexpected lag for Fed Funds -> Treasury: {h.lag} (expected 0-6 months)"
        )

    def test_oil_predicts_cpi(self, hypotheses: list[Hypothesis]) -> None:
        """L5: Oil prices should Granger-cause CPI with 1-12 month lag.
        # SOURCE: Hamilton (1983), "Oil and the Macroeconomy since World War II"
        """
        h = _find_relationship(
            hypotheses,
            x_keywords=["OIL", "BRENT", "DCOIL", "MCOILBRENT"],
            y_keywords=["CPI", "CPIAUCSL"],
            bidirectional_ok=True,
        )
        assert h is not None, (
            "Pipeline failed to discover Oil -> CPI relationship"
        )
        assert h.direction_pvalue < 0.05, (
            f"Oil -> CPI not significant: p={h.direction_pvalue:.4f}"
        )
        assert 1 <= h.lag <= 12, (
            f"Unexpected lag for Oil -> CPI: {h.lag} (expected 1-12 months)"
        )

    def test_gdp_unemployment_okun(self, hypotheses: list[Hypothesis]) -> None:
        """L5: GDP and Unemployment should be linked (Okun's Law).
        # SOURCE: Okun (1962) "Potential GNP: Its Measurement and Significance"
        """
        h = _find_relationship(
            hypotheses,
            x_keywords=["GDP", "GDPC1"],
            y_keywords=["UNRATE", "UNEMPLOYMENT"],
            bidirectional_ok=True,
        )
        assert h is not None, (
            "Pipeline failed to discover GDP <-> Unemployment (Okun's Law)"
        )
        assert h.direction_pvalue < 0.05, (
            f"GDP <-> Unemployment not significant: p={h.direction_pvalue:.4f}"
        )
        assert 1 <= h.lag <= 8, (
            f"Unexpected lag for GDP <-> Unemployment: {h.lag} (expected 1-8)"
        )

    def test_initial_claims_predicts_employment(
        self, hypotheses: list[Hypothesis]
    ) -> None:
        """L5: Initial claims should predict nonfarm payrolls.
        # SOURCE: Montgomery et al. (1998) "The time series behavior of initial
        # unemployment claims", Journal of Business & Economic Statistics.
        """
        h = _find_relationship(
            hypotheses,
            x_keywords=["ICSA", "INITIAL CLAIMS"],
            y_keywords=["PAYEMS", "NONFARM"],
            bidirectional_ok=True,
        )
        assert h is not None, (
            "Pipeline failed to discover Initial Claims -> Employment"
        )
        assert h.direction_pvalue < 0.05

    def test_hy_spread_predicts_activity(
        self, hypotheses: list[Hypothesis]
    ) -> None:
        """L5: High-yield spread predicts economic activity.
        # SOURCE: Gilchrist & Zakrajsek (2012) "Credit Spreads and Business
        # Cycle Fluctuations", AER.
        """
        h = _find_relationship(
            hypotheses,
            x_keywords=["BAMLH0A0HYM2", "HIGH YIELD"],
            y_keywords=["PAYEMS", "USSLIND", "LEADING"],
            bidirectional_ok=True,
        )
        assert h is not None, (
            "Pipeline failed to discover HY Spread -> Economic Activity"
        )
        assert h.direction_pvalue < 0.05


class TestL5PipelineStats:
    """L5: Verify pipeline output statistics are reasonable on real data."""

    def test_minimum_hypotheses(self, hypotheses: list[Hypothesis]) -> None:
        """Pipeline should find at least 20 hypotheses from 49 FRED series."""
        assert len(hypotheses) >= 20, (
            f"Too few hypotheses: {len(hypotheses)} (expected >= 20)"
        )

    def test_score_range(self, hypotheses: list[Hypothesis]) -> None:
        """All scores should be in [0, 10]."""
        for h in hypotheses:
            assert 0.0 <= h.score <= 10.0, (
                f"Score out of range: {h.score} for {h.x.name} -> {h.y.name}"
            )

    def test_no_self_pairs(self, hypotheses: list[Hypothesis]) -> None:
        """No variable should be paired with itself."""
        for h in hypotheses:
            assert h.x.variable_id != h.y.variable_id, (
                f"Self-pair found: {h.x.variable_id}"
            )

    def test_medium_or_high_confidence_exists(
        self, hypotheses: list[Hypothesis]
    ) -> None:
        """At least some hypotheses should be medium or high confidence."""
        med_high = [h for h in hypotheses if h.confidence in ("medium", "high")]
        assert len(med_high) >= 5, (
            f"Only {len(med_high)} medium/high confidence hypotheses (expected >= 5)"
        )

    def test_ranks_are_unique_and_ordered(
        self, hypotheses: list[Hypothesis]
    ) -> None:
        """Ranks should be 1..N with no duplicates, scores descending."""
        ranks = [h.rank for h in hypotheses]
        assert ranks == list(range(1, len(hypotheses) + 1)), "Ranks not sequential"
        scores = [h.score for h in hypotheses]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores not descending: rank {i+1}={scores[i]}, rank {i+2}={scores[i+1]}"
            )
