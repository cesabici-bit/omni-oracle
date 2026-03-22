"""Abstract base class for data source fetchers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.models import TimeSeries
from src.storage.repo import TimeSeriesRepo


class BaseFetcher(ABC):
    """Abstract fetcher for a data source."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier for this source (e.g. 'fred', 'worldbank')."""

    @abstractmethod
    def fetch_series_list(self, limit: int = 100) -> list[dict]:
        """Return a list of series metadata dicts to fetch.

        Each dict must have at least: 'series_id', 'name'.
        """

    @abstractmethod
    def fetch_observations(self, series_id: str) -> pd.DataFrame:
        """Fetch observations for a single series.

        Returns DataFrame with columns ['ts', 'value'] where ts is a date.
        """

    def ingest(self, repo: TimeSeriesRepo, limit: int = 100) -> list[str]:
        """Fetch and store series into the repo.

        Returns list of variable_ids that were successfully ingested.
        """
        series_list = self.fetch_series_list(limit=limit)
        ingested: list[str] = []

        for meta in series_list:
            series_id = meta["series_id"]
            variable_id = f"{self.source_name}:{series_id}"

            try:
                df = self.fetch_observations(series_id)
                if df.empty:
                    continue

                # Build TimeSeries metadata
                ts_meta = TimeSeries(
                    variable_id=variable_id,
                    source=self.source_name,
                    name=meta.get("name", series_id),
                    frequency=meta.get("frequency", "unknown"),
                    unit=meta.get("unit", ""),
                    geo=meta.get("geo", "US"),
                    domain=meta.get("domain", "economics"),
                    observations=len(df),
                    start_date=pd.Timestamp(df["ts"].min()).date(),
                    end_date=pd.Timestamp(df["ts"].max()).date(),
                )

                repo.upsert_series(ts_meta)
                repo.insert_observations_bulk(variable_id, df)
                ingested.append(variable_id)

            except Exception as e:
                print(f"  WARN: Failed to ingest {variable_id}: {e}")
                continue

        return ingested
