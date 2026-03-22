"""NOAA Climate Data Online (CDO) fetcher — API v2.

Requires NOAA_TOKEN environment variable.
API docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
Uses httpx for HTTP (no wrapper library).
"""

from __future__ import annotations

import os
import time
from datetime import date

import httpx
import pandas as pd

from src.ingest.base import BaseFetcher

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"

# Datatypes we fetch from GSOM (Global Summary of the Month)
GSOM_DATATYPES = ["TAVG", "PRCP", "TMAX", "TMIN", "HTDD", "CLDD"]

# Curated US stations with long, reliable monthly records.
# Selected for geographic coverage and data completeness.
CURATED_NOAA_STATIONS = [
    {"station_id": "GHCND:USW00094728", "name": "New York Central Park", "geo": "US-NY"},
    {"station_id": "GHCND:USW00023174", "name": "Los Angeles Intl Airport", "geo": "US-CA"},
    {"station_id": "GHCND:USW00094846", "name": "Chicago O'Hare", "geo": "US-IL"},
    {"station_id": "GHCND:USW00012839", "name": "Houston Hobby Airport", "geo": "US-TX"},
    {"station_id": "GHCND:USW00023183", "name": "Phoenix Sky Harbor", "geo": "US-AZ"},
    {"station_id": "GHCND:USW00013874", "name": "Atlanta Hartsfield", "geo": "US-GA"},
    {"station_id": "GHCND:USW00024233", "name": "Seattle-Tacoma Intl", "geo": "US-WA"},
    {"station_id": "GHCND:USW00014732", "name": "Boston Logan", "geo": "US-MA"},
    {"station_id": "GHCND:USW00013881", "name": "Miami Intl Airport", "geo": "US-FL"},
    {"station_id": "GHCND:USW00023169", "name": "Denver Stapleton", "geo": "US-CO"},
]


def _build_series_list(stations: list[dict], datatypes: list[str]) -> list[dict]:
    """Build series list: one series per station × datatype."""
    series = []
    for station in stations:
        for dtype in datatypes:
            sid = station["station_id"].replace(":", "_")
            series.append({
                "series_id": f"{sid}_{dtype}",
                "station_id": station["station_id"],
                "datatype": dtype,
                "name": f"{station['name']} — {dtype}",
                "unit": _unit_for_datatype(dtype),
                "frequency": "monthly",
                "geo": station["geo"],
                "domain": "climate",
            })
    return series


def _unit_for_datatype(dtype: str) -> str:
    """Return unit string for a GSOM datatype."""
    units = {
        "TAVG": "celsius",
        "TMAX": "celsius",
        "TMIN": "celsius",
        "PRCP": "mm",
        "HTDD": "degree_days",
        "CLDD": "degree_days",
    }
    return units.get(dtype, "unknown")


class NOAAFetcher(BaseFetcher):
    """Fetcher for NOAA Climate Data Online (CDO) API v2.

    Fetches GSOM (Global Summary of the Month) data for curated US stations.
    Rate limit: 5 requests/second.
    """

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get("NOAA_TOKEN", "")
        if not self.token:
            raise ValueError(
                "NOAA_TOKEN not set. Set via env var or pass token. "
                "Get a free token at https://www.ncdc.noaa.gov/cdo-web/token"
            )
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"token": self.token},
            timeout=30.0,
        )
        self._last_request_time: float = 0.0

    @property
    def source_name(self) -> str:
        return "noaa"

    def fetch_series_list(self, limit: int = 100) -> list[dict]:
        """Return station × datatype combinations, up to limit."""
        all_series = _build_series_list(CURATED_NOAA_STATIONS, GSOM_DATATYPES)
        return all_series[:limit]

    def fetch_observations(self, series_id: str) -> pd.DataFrame:
        """Fetch GSOM observations for a station/datatype pair.

        series_id format: "GHCND_USW00094728_TAVG"
        """
        # Parse series_id back to station_id and datatype
        # Format: GHCND_STATIONNUM_DATATYPE
        parts = series_id.rsplit("_", 1)
        if len(parts) != 2:
            return pd.DataFrame(columns=["ts", "value"])

        station_raw, datatype = parts
        # Reconstruct station_id: GHCND_USW00094728 -> GHCND:USW00094728
        station_id = station_raw.replace("_", ":", 1)

        return self._fetch_gsom(station_id, datatype)

    def _fetch_gsom(
        self,
        station_id: str,
        datatype: str,
        start_year: int = 1970,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """Fetch GSOM data, paginating over 10-year chunks (API limit)."""
        if end_year is None:
            end_year = date.today().year

        all_records: list[dict] = []

        # NOAA CDO allows max 10-year range per request for GSOM
        year = start_year
        while year <= end_year:
            chunk_end = min(year + 9, end_year)
            start_date = f"{year}-01-01"
            end_date = f"{chunk_end}-12-31"

            records = self._fetch_page(station_id, datatype, start_date, end_date)
            all_records.extend(records)
            year = chunk_end + 1

        if not all_records:
            return pd.DataFrame(columns=["ts", "value"])

        df = pd.DataFrame(all_records)
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def _fetch_page(
        self,
        station_id: str,
        datatype: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch a single paginated chunk from the CDO API."""
        records: list[dict] = []
        offset = 1  # CDO uses 1-based offset

        while True:
            self._rate_limit()

            params = {
                "datasetid": "GSOM",
                "stationid": station_id,
                "datatypeid": datatype,
                "startdate": start_date,
                "enddate": end_date,
                "units": "metric",
                "limit": 1000,
                "offset": offset,
            }

            try:
                resp = self._client.get("/data", params=params)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                print(f"  WARN: NOAA API error for {station_id}/{datatype}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for row in results:
                value = row.get("value")
                date_str = row.get("date", "")
                if value is None:
                    continue
                try:
                    ts = pd.Timestamp(date_str).date()
                    records.append({"ts": ts, "value": float(value)})
                except (ValueError, TypeError):
                    continue

            # Check if there are more pages
            metadata = data.get("metadata", {}).get("resultset", {})
            total = metadata.get("count", 0)
            if offset + 1000 > total:
                break
            offset += 1000

        return records

    def _rate_limit(self) -> None:
        """Enforce 5 requests/second rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < 0.2:  # 1/5 second
            time.sleep(0.2 - elapsed)
        self._last_request_time = time.monotonic()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
