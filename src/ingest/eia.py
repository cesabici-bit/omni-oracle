"""EIA (Energy Information Administration) fetcher — API v2.

Requires EIA_API_KEY environment variable.
API docs: https://www.eia.gov/opendata/documentation.php
Uses httpx for HTTP (no wrapper library).
"""

from __future__ import annotations

import os
from datetime import date

import httpx
import pandas as pd

from src.ingest.base import BaseFetcher

BASE_URL = "https://api.eia.gov/v2"

# Curated list of key energy series for cross-domain discovery.
# Each entry maps to a v2 route + facet filters.
# Format: route is the API path, facets narrow the series.
CURATED_EIA_SERIES = [
    # Natural Gas
    {
        "series_id": "NG_RNGC1",
        "name": "Natural Gas Futures (Henry Hub)",
        "route": "natural-gas/pri/fut",
        "facets": {"series": "RNGC1"},
        "unit": "dollars_per_mmbtu",
        "frequency": "monthly",
    },
    {
        "series_id": "NG_RNGWHHD",
        "name": "Natural Gas Spot Price (Henry Hub)",
        "route": "natural-gas/pri/sum",
        "facets": {"process": "PRS", "duoarea": "RGC"},
        "unit": "dollars_per_mmbtu",
        "frequency": "monthly",
    },
    # Petroleum
    {
        "series_id": "PET_RWTC",
        "name": "Crude Oil WTI Spot Price",
        "route": "petroleum/pri/spt",
        "facets": {"series": "RWTC"},
        "unit": "dollars_per_barrel",
        "frequency": "monthly",
    },
    {
        "series_id": "PET_RBRTE",
        "name": "Crude Oil Brent Spot Price",
        "route": "petroleum/pri/spt",
        "facets": {"series": "RBRTE"},
        "unit": "dollars_per_barrel",
        "frequency": "monthly",
    },
    {
        "series_id": "PET_EMM_EPM0U_PTE_NUS_DPG",
        "name": "US Regular Gasoline Retail Price",
        "route": "petroleum/pri/gnd",
        "facets": {"series": "EMM_EPM0U_PTE_NUS_DPG"},
        "unit": "dollars_per_gallon",
        "frequency": "monthly",
    },
    {
        "series_id": "PET_MGFUPUS2",
        "name": "US Total Gasoline Production",
        "route": "petroleum/sum/sndw",
        "facets": {"series": "MGFUPUS2"},
        "unit": "thousand_barrels_per_day",
        "frequency": "weekly",
    },
    # Electricity
    {
        "series_id": "ELEC_GEN_ALL_US",
        "name": "US Total Electricity Net Generation",
        "route": "electricity/electric-power-operational-data",
        "facets": {"sectorid": "99", "fueltypeid": "ALL", "location": "US"},
        "unit": "thousand_mwh",
        "frequency": "monthly",
    },
    {
        "series_id": "ELEC_GEN_SUN_US",
        "name": "US Solar Electricity Generation",
        "route": "electricity/electric-power-operational-data",
        "facets": {"sectorid": "99", "fueltypeid": "SUN", "location": "US"},
        "unit": "thousand_mwh",
        "frequency": "monthly",
    },
    {
        "series_id": "ELEC_GEN_WND_US",
        "name": "US Wind Electricity Generation",
        "route": "electricity/electric-power-operational-data",
        "facets": {"sectorid": "99", "fueltypeid": "WND", "location": "US"},
        "unit": "thousand_mwh",
        "frequency": "monthly",
    },
    {
        "series_id": "ELEC_GEN_NG_US",
        "name": "US Natural Gas Electricity Generation",
        "route": "electricity/electric-power-operational-data",
        "facets": {"sectorid": "99", "fueltypeid": "NG", "location": "US"},
        "unit": "thousand_mwh",
        "frequency": "monthly",
    },
    {
        "series_id": "ELEC_GEN_COL_US",
        "name": "US Coal Electricity Generation",
        "route": "electricity/electric-power-operational-data",
        "facets": {"sectorid": "99", "fueltypeid": "COL", "location": "US"},
        "unit": "thousand_mwh",
        "frequency": "monthly",
    },
    # Coal
    {
        "series_id": "COAL_PROD_US",
        "name": "US Coal Production",
        "route": "coal/mine-production",
        "facets": {"mineStateFips": "US-TOTAL"},
        "unit": "short_tons",
        "frequency": "annual",
    },
    # Total Energy
    {
        "series_id": "TOTAL_ENERGY_CONSUMPTION",
        "name": "US Total Primary Energy Consumption",
        "route": "total-energy/data",
        "facets": {"msn": "TPOBUS"},
        "unit": "quadrillion_btu",
        "frequency": "monthly",
    },
    {
        "series_id": "TOTAL_ENERGY_CO2",
        "name": "US Energy-Related CO2 Emissions",
        "route": "total-energy/data",
        "facets": {"msn": "TETCBUS"},
        "unit": "million_metric_tons",
        "frequency": "monthly",
    },
    # Crude Oil Imports/Exports
    {
        "series_id": "PET_IMPORTS_US",
        "name": "US Crude Oil Imports",
        "route": "petroleum/move/imp",
        "facets": {"series": "MCRIMUS1"},
        "unit": "thousand_barrels",
        "frequency": "monthly",
    },
]


class EIAFetcher(BaseFetcher):
    """Fetcher for EIA (Energy Information Administration) data via API v2.

    API docs: https://www.eia.gov/opendata/documentation.php
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("EIA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "EIA_API_KEY not set. Set via env var or pass api_key."
            )
        self._client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    @property
    def source_name(self) -> str:
        return "eia"

    def fetch_series_list(self, limit: int = 100) -> list[dict]:
        """Return curated list of EIA energy series, up to limit."""
        series = []
        for s in CURATED_EIA_SERIES[:limit]:
            series.append({
                "series_id": s["series_id"],
                "name": s["name"],
                "unit": s["unit"],
                "frequency": s["frequency"],
                "domain": "energy",
                "geo": "US",
            })
        return series

    def fetch_observations(self, series_id: str) -> pd.DataFrame:
        """Fetch observations for an EIA series via v2 API."""
        # Find the series config
        config = None
        for s in CURATED_EIA_SERIES:
            if s["series_id"] == series_id:
                config = s
                break
        if config is None:
            return pd.DataFrame(columns=["ts", "value"])

        return self._fetch_route_data(config)

    def _fetch_route_data(self, config: dict) -> pd.DataFrame:
        """Fetch data from a v2 route with facets."""
        route = config["route"]
        facets = config.get("facets", {})
        frequency = config.get("frequency", "monthly")

        params: dict[str, str | int] = {
            "api_key": self.api_key,
            "frequency": frequency,
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        }

        # Add facet filters
        for key, val in facets.items():
            params[f"facets[{key}][]"] = val

        try:
            resp = self._client.get(f"/{route}/data/", params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            print(f"  WARN: EIA API error for {config['series_id']}: {e}")
            return pd.DataFrame(columns=["ts", "value"])

        response_body = data.get("response", {})
        rows = response_body.get("data", [])

        if not rows:
            return pd.DataFrame(columns=["ts", "value"])

        records = []
        for row in rows:
            period = row.get("period")
            value = row.get("value")
            if period is None or value is None:
                continue
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            # Period can be "2024-01", "2024-01-05", or "2024"
            ts = _parse_period(period)
            if ts is not None:
                records.append({"ts": ts, "value": value})

        return pd.DataFrame(records)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def _parse_period(period: str) -> date | None:
    """Parse EIA period string to date."""
    if not period:
        return None
    try:
        if len(period) == 4:  # "2024"
            return date(int(period), 1, 1)
        elif len(period) == 7:  # "2024-01"
            parts = period.split("-")
            return date(int(parts[0]), int(parts[1]), 1)
        else:  # "2024-01-05"
            ts = pd.Timestamp(period)
            if pd.isna(ts):
                return None
            return ts.date()
    except (ValueError, IndexError):
        return None
