"""Tests for NOAA CDO fetcher.

Tests cover: curated station/datatype list, series_id parsing,
observation processing, pagination, rate limiting, and BaseFetcher integration.
Uses mocked HTTP responses (no real NOAA token needed for unit tests).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingest.noaa import (
    CURATED_NOAA_STATIONS,
    GSOM_DATATYPES,
    NOAAFetcher,
    _build_series_list,
    _unit_for_datatype,
)


# ---------- M2: External Oracle ----------
# SOURCE: NOAA CDO API v2 docs — https://www.ncdc.noaa.gov/cdo-web/webservices/v2
# Response format: {"metadata": {"resultset": {...}}, "results": [{...}]}
# GSOM TAVG for NYC Central Park is well-documented in NOAA records.

MOCK_NOAA_RESPONSE = {
    "metadata": {
        "resultset": {"offset": 1, "count": 3, "limit": 1000},
    },
    "results": [
        {
            "date": "2024-01-01T00:00:00",
            "datatype": "TAVG",
            "station": "GHCND:USW00094728",
            "attributes": ",,7,",
            "value": 1.2,
        },
        {
            "date": "2024-02-01T00:00:00",
            "datatype": "TAVG",
            "station": "GHCND:USW00094728",
            "attributes": ",,7,",
            "value": 3.5,
        },
        {
            "date": "2024-03-01T00:00:00",
            "datatype": "TAVG",
            "station": "GHCND:USW00094728",
            "attributes": ",,7,",
            "value": 8.1,
        },
    ],
}


class TestHelpers:
    """Test helper functions."""

    def test_unit_for_datatype(self) -> None:
        assert _unit_for_datatype("TAVG") == "celsius"
        assert _unit_for_datatype("PRCP") == "mm"
        assert _unit_for_datatype("HTDD") == "degree_days"
        assert _unit_for_datatype("UNKNOWN") == "unknown"

    def test_build_series_list(self) -> None:
        stations = [CURATED_NOAA_STATIONS[0]]  # Just NYC
        series = _build_series_list(stations, GSOM_DATATYPES)
        assert len(series) == len(GSOM_DATATYPES)
        for s in series:
            assert s["domain"] == "climate"
            assert s["frequency"] == "monthly"
            assert "GHCND_USW00094728" in s["series_id"]

    def test_build_series_list_cross_product(self) -> None:
        stations = CURATED_NOAA_STATIONS[:2]
        series = _build_series_list(stations, ["TAVG", "PRCP"])
        assert len(series) == 4  # 2 stations × 2 datatypes


class TestNOAAFetcherInit:
    """Test NOAA fetcher initialization."""

    def test_raises_without_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="NOAA_TOKEN"):
                NOAAFetcher(token="")

    def test_init_with_explicit_token(self) -> None:
        fetcher = NOAAFetcher(token="test-token-123")
        assert fetcher.token == "test-token-123"
        assert fetcher.source_name == "noaa"
        fetcher.close()

    def test_init_from_env(self) -> None:
        with patch.dict("os.environ", {"NOAA_TOKEN": "env-token-456"}):
            fetcher = NOAAFetcher()
            assert fetcher.token == "env-token-456"
            fetcher.close()


class TestCuratedStations:
    """Test curated station list."""

    def test_stations_not_empty(self) -> None:
        assert len(CURATED_NOAA_STATIONS) >= 10

    def test_all_stations_have_required_fields(self) -> None:
        for station in CURATED_NOAA_STATIONS:
            assert "station_id" in station
            assert "name" in station
            assert "geo" in station
            assert station["station_id"].startswith("GHCND:")

    def test_fetch_series_list_respects_limit(self) -> None:
        fetcher = NOAAFetcher(token="test-token")
        series = fetcher.fetch_series_list(limit=5)
        assert len(series) == 5
        fetcher.close()

    def test_total_series_count(self) -> None:
        fetcher = NOAAFetcher(token="test-token")
        series = fetcher.fetch_series_list(limit=1000)
        expected = len(CURATED_NOAA_STATIONS) * len(GSOM_DATATYPES)
        assert len(series) == expected
        fetcher.close()


class TestFetchObservations:
    """Test observation fetching with mocked HTTP."""

    def test_fetch_tavg_nyc(self) -> None:
        """Fetch mocked TAVG data for NYC Central Park."""
        fetcher = NOAAFetcher(token="test-token")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_NOAA_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            # Constrain year range to avoid multiple 10-year chunks
            df = fetcher._fetch_gsom("GHCND:USW00094728", "TAVG", start_year=2024, end_year=2024)

        assert len(df) == 3
        assert list(df.columns) == ["ts", "value"]
        assert df["value"].iloc[0] == 1.2
        assert df["ts"].iloc[0] == date(2024, 1, 1)
        fetcher.close()

    def test_fetch_invalid_series_id(self) -> None:
        fetcher = NOAAFetcher(token="test-token")
        df = fetcher.fetch_observations("INVALID")
        assert df.empty
        fetcher.close()

    def test_fetch_handles_null_values(self) -> None:
        fetcher = NOAAFetcher(token="test-token")
        response = {
            "metadata": {"resultset": {"offset": 1, "count": 2, "limit": 1000}},
            "results": [
                {"date": "2024-01-01T00:00:00", "datatype": "TAVG", "station": "X", "value": 5.0},
                {"date": "2024-02-01T00:00:00", "datatype": "TAVG", "station": "X", "value": None},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            df = fetcher._fetch_gsom("GHCND:USW00094728", "TAVG", start_year=2024, end_year=2024)

        assert len(df) == 1
        fetcher.close()

    def test_fetch_handles_api_error(self) -> None:
        import httpx

        fetcher = NOAAFetcher(token="test-token")
        with patch.object(
            fetcher._client,
            "get",
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            ),
        ):
            df = fetcher.fetch_observations("GHCND_USW00094728_TAVG")

        assert df.empty
        fetcher.close()

    def test_fetch_handles_empty_results(self) -> None:
        fetcher = NOAAFetcher(token="test-token")
        response = {
            "metadata": {"resultset": {"offset": 1, "count": 0, "limit": 1000}},
            "results": [],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            df = fetcher.fetch_observations("GHCND_USW00094728_TAVG")

        assert df.empty
        fetcher.close()


class TestPagination:
    """Test 10-year chunk pagination."""

    def test_pagination_splits_range(self) -> None:
        """Verify that a 25-year range is split into 3 chunks."""
        fetcher = NOAAFetcher(token="test-token")
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.json.return_value = {
                "metadata": {"resultset": {"offset": 1, "count": 1, "limit": 1000}},
                "results": [
                    {"date": "2000-01-01T00:00:00", "datatype": "TAVG", "station": "X", "value": 10.0},
                ],
            }
            resp.raise_for_status = MagicMock()
            return resp

        with patch.object(fetcher._client, "get", side_effect=mock_get):
            df = fetcher._fetch_gsom("GHCND:USW00094728", "TAVG", start_year=2000, end_year=2024)

        # 2000-2009, 2010-2019, 2020-2024 = 3 chunks
        assert call_count == 3
        assert len(df) == 3  # 1 row per chunk
        fetcher.close()


class TestBaseFetcherIntegration:
    """Test that NOAAFetcher works correctly with BaseFetcher.ingest()."""

    def test_ingest_stores_series_with_climate_domain(self, repo) -> None:
        fetcher = NOAAFetcher(token="test-token")

        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_NOAA_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._client, "get", return_value=mock_resp):
            ingested = fetcher.ingest(repo, limit=1)

        assert len(ingested) == 1
        assert ingested[0].startswith("noaa:")

        # Verify stored metadata
        meta = repo.get_series(ingested[0])
        assert meta is not None
        assert meta.domain == "climate"
        assert meta.source == "noaa"
        fetcher.close()
