"""Tests for EIA fetcher.

Tests cover: curated list structure, series_id parsing, period parsing,
observation processing, and BaseFetcher integration.
Uses mocked HTTP responses (no real API key needed for unit tests).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingest.eia import (
    CURATED_EIA_SERIES,
    EIAFetcher,
    _parse_period,
)


# ---------- M2: External Oracle ----------
# SOURCE: EIA API v2 docs — https://www.eia.gov/opendata/documentation.php
# The response format uses {"response": {"data": [{"period": ..., "value": ...}]}}

MOCK_EIA_RESPONSE = {
    "response": {
        "total": 3,
        "dateFormat": "YYYY-MM",
        "frequency": "monthly",
        "data": [
            {"period": "2024-01", "value": 2.57, "series-description": "NG futures"},
            {"period": "2024-02", "value": 1.75, "series-description": "NG futures"},
            {"period": "2024-03", "value": 1.76, "series-description": "NG futures"},
        ],
    },
    "request": {},
    "apiVersion": "2.1.7",
}


class TestParsePerid:
    """Test EIA period string parsing."""

    def test_monthly_period(self) -> None:
        assert _parse_period("2024-01") == date(2024, 1, 1)

    def test_annual_period(self) -> None:
        assert _parse_period("2024") == date(2024, 1, 1)

    def test_daily_period(self) -> None:
        assert _parse_period("2024-01-15") == date(2024, 1, 15)

    def test_invalid_period(self) -> None:
        assert _parse_period("not-a-date") is None

    def test_empty_period(self) -> None:
        assert _parse_period("") is None


class TestEIAFetcherInit:
    """Test EIA fetcher initialization."""

    def test_raises_without_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="EIA_API_KEY"):
                EIAFetcher(api_key="")

    def test_init_with_explicit_key(self) -> None:
        fetcher = EIAFetcher(api_key="test-key-123")
        assert fetcher.api_key == "test-key-123"
        assert fetcher.source_name == "eia"
        fetcher.close()

    def test_init_from_env(self) -> None:
        with patch.dict("os.environ", {"EIA_API_KEY": "env-key-456"}):
            fetcher = EIAFetcher()
            assert fetcher.api_key == "env-key-456"
            fetcher.close()


class TestCuratedList:
    """Test curated series list."""

    def test_curated_list_not_empty(self) -> None:
        assert len(CURATED_EIA_SERIES) >= 10

    def test_all_entries_have_required_fields(self) -> None:
        required = {"series_id", "name", "route", "unit", "frequency"}
        for entry in CURATED_EIA_SERIES:
            missing = required - set(entry.keys())
            assert not missing, f"{entry['series_id']} missing: {missing}"

    def test_fetch_series_list_returns_domain_energy(self) -> None:
        fetcher = EIAFetcher(api_key="test-key")
        series = fetcher.fetch_series_list(limit=5)
        assert len(series) == 5
        for s in series:
            assert s["domain"] == "energy"
            assert s["geo"] == "US"
        fetcher.close()

    def test_fetch_series_list_respects_limit(self) -> None:
        fetcher = EIAFetcher(api_key="test-key")
        series = fetcher.fetch_series_list(limit=3)
        assert len(series) == 3
        fetcher.close()

    def test_series_ids_unique(self) -> None:
        ids = [s["series_id"] for s in CURATED_EIA_SERIES]
        assert len(ids) == len(set(ids)), "Duplicate series_id found"


class TestFetchObservations:
    """Test observation fetching with mocked HTTP."""

    def test_fetch_known_series(self) -> None:
        """Fetch mocked NG futures data and check DataFrame shape."""
        fetcher = EIAFetcher(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_EIA_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            df = fetcher.fetch_observations("NG_RNGC1")

        assert len(df) == 3
        assert list(df.columns) == ["ts", "value"]
        assert df["value"].iloc[0] == 2.57
        assert df["ts"].iloc[0] == date(2024, 1, 1)
        fetcher.close()

    def test_fetch_unknown_series_returns_empty(self) -> None:
        fetcher = EIAFetcher(api_key="test-key")
        df = fetcher.fetch_observations("NONEXISTENT_SERIES")
        assert df.empty
        assert list(df.columns) == ["ts", "value"]
        fetcher.close()

    def test_fetch_handles_null_values(self) -> None:
        """API sometimes returns null values — these should be skipped."""
        fetcher = EIAFetcher(api_key="test-key")
        response = {
            "response": {
                "data": [
                    {"period": "2024-01", "value": 2.5},
                    {"period": "2024-02", "value": None},
                    {"period": "2024-03", "value": 3.0},
                ],
            },
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            df = fetcher.fetch_observations("NG_RNGC1")

        assert len(df) == 2  # null row skipped
        fetcher.close()

    def test_fetch_handles_api_error(self) -> None:
        """HTTP errors should return empty DataFrame, not raise."""
        import httpx

        fetcher = EIAFetcher(api_key="test-key")

        with patch.object(
            fetcher._client,
            "get",
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            ),
        ):
            df = fetcher.fetch_observations("NG_RNGC1")

        assert df.empty
        fetcher.close()

    def test_fetch_handles_empty_response(self) -> None:
        fetcher = EIAFetcher(api_key="test-key")
        response = {"response": {"data": []}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            df = fetcher.fetch_observations("NG_RNGC1")

        assert df.empty
        fetcher.close()


class TestBaseFetcherIntegration:
    """Test that EIAFetcher works correctly with BaseFetcher.ingest()."""

    def test_ingest_stores_series_with_energy_domain(self, repo) -> None:
        """Verify ingest creates series with domain='energy'."""
        fetcher = EIAFetcher(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_EIA_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            ingested = fetcher.ingest(repo, limit=1)

        assert len(ingested) == 1
        assert ingested[0] == "eia:NG_RNGC1"

        # Verify stored metadata
        meta = repo.get_series("eia:NG_RNGC1")
        assert meta is not None
        assert meta.domain == "energy"
        assert meta.source == "eia"
        fetcher.close()
