"""NOAA Climate Data Online fetcher (placeholder for post-MVP).

Requires NOAA API token: https://www.ncdc.noaa.gov/cdo-web/token
"""

from __future__ import annotations

import pandas as pd

from src.ingest.base import BaseFetcher


class NOAAFetcher(BaseFetcher):
    """Fetcher for NOAA Climate Data Online. Placeholder for post-MVP."""

    @property
    def source_name(self) -> str:
        return "noaa"

    def fetch_series_list(self, limit: int = 100) -> list[dict]:
        raise NotImplementedError("NOAA fetcher not yet implemented (post-MVP)")

    def fetch_observations(self, series_id: str) -> pd.DataFrame:
        raise NotImplementedError("NOAA fetcher not yet implemented (post-MVP)")
