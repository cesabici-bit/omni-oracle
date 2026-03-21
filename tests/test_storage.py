"""Tests for DuckDB storage layer (ST-02)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.models import TimeSeries
from src.storage.repo import TimeSeriesRepo


class TestTimeSeriesRepo:
    """L1: CRUD operations on TimeSeriesRepo."""

    def test_upsert_and_get_series(self, repo: TimeSeriesRepo) -> None:
        ts = TimeSeries(
            variable_id="fred:CPIAUCSL",
            source="fred",
            name="Consumer Price Index",
            frequency="monthly",
            unit="index",
            geo="US",
            observations=240,
            start_date=date(2000, 1, 1),
            end_date=date(2019, 12, 1),
        )
        repo.upsert_series(ts)
        result = repo.get_series("fred:CPIAUCSL")
        assert result is not None
        assert result.variable_id == "fred:CPIAUCSL"
        assert result.source == "fred"
        assert result.geo == "US"

    def test_get_nonexistent_series(self, repo: TimeSeriesRepo) -> None:
        result = repo.get_series("nonexistent")
        assert result is None

    def test_list_series_all(
        self, repo: TimeSeriesRepo, sample_series_meta: list[TimeSeries]
    ) -> None:
        for s in sample_series_meta:
            repo.upsert_series(s)
        result = repo.list_series()
        assert len(result) == 10

    def test_list_series_filter_source(
        self, repo: TimeSeriesRepo
    ) -> None:
        ts1 = TimeSeries("fred:A", "fred", "A", "monthly", "idx", "US", 100, date(2000, 1, 1), date(2008, 4, 1))
        ts2 = TimeSeries("wb:B", "worldbank", "B", "annual", "pct", "IT", 50, date(2000, 1, 1), date(2020, 1, 1))
        repo.upsert_series(ts1)
        repo.upsert_series(ts2)
        fred_only = repo.list_series(source="fred")
        assert len(fred_only) == 1
        assert fred_only[0].source == "fred"

    def test_list_series_filter_geo(self, repo: TimeSeriesRepo) -> None:
        ts1 = TimeSeries("a:1", "a", "A", "m", "u", "US", 10, date(2000, 1, 1), date(2000, 10, 1))
        ts2 = TimeSeries("a:2", "a", "B", "m", "u", "IT", 10, date(2000, 1, 1), date(2000, 10, 1))
        repo.upsert_series(ts1)
        repo.upsert_series(ts2)
        it_only = repo.list_series(geo="IT")
        assert len(it_only) == 1
        assert it_only[0].geo == "IT"

    def test_insert_and_get_observations(self, repo: TimeSeriesRepo) -> None:
        ts = TimeSeries("t:1", "t", "T", "m", "u", "US", 5, date(2020, 1, 1), date(2020, 5, 1))
        repo.upsert_series(ts)

        df = pd.DataFrame({
            "ts": pd.date_range("2020-01-01", periods=5, freq="MS").date,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        repo.insert_observations("t:1", df)

        result = repo.get_observations("t:1")
        assert len(result) == 5
        assert list(result["value"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_get_observations_date_range(self, repo: TimeSeriesRepo) -> None:
        ts = TimeSeries("t:2", "t", "T", "m", "u", "US", 12, date(2020, 1, 1), date(2020, 12, 1))
        repo.upsert_series(ts)

        df = pd.DataFrame({
            "ts": pd.date_range("2020-01-01", periods=12, freq="MS").date,
            "value": list(range(1, 13)),
        })
        repo.insert_observations("t:2", df)

        result = repo.get_observations("t:2", start=date(2020, 3, 1), end=date(2020, 6, 1))
        assert len(result) == 4

    def test_get_aligned_pair(self, repo: TimeSeriesRepo) -> None:
        ts1 = TimeSeries("t:x", "t", "X", "m", "u", "US", 5, date(2020, 1, 1), date(2020, 5, 1))
        ts2 = TimeSeries("t:y", "t", "Y", "m", "u", "US", 5, date(2020, 1, 1), date(2020, 5, 1))
        repo.upsert_series(ts1)
        repo.upsert_series(ts2)

        dates = pd.date_range("2020-01-01", periods=5, freq="MS").date
        repo.insert_observations("t:x", pd.DataFrame({"ts": dates, "value": [1, 2, 3, 4, 5]}))
        repo.insert_observations("t:y", pd.DataFrame({"ts": dates, "value": [10, 20, 30, 40, 50]}))

        aligned = repo.get_aligned_pair("t:x", "t:y")
        assert len(aligned) == 5
        assert "x" in aligned.columns
        assert "y" in aligned.columns

    def test_count_series(
        self, repo: TimeSeriesRepo, sample_series_meta: list[TimeSeries]
    ) -> None:
        assert repo.count_series() == 0
        for s in sample_series_meta[:3]:
            repo.upsert_series(s)
        assert repo.count_series() == 3

    def test_upsert_overwrites(self, repo: TimeSeriesRepo) -> None:
        ts1 = TimeSeries("t:u", "t", "Original", "m", "u", "US", 10, date(2020, 1, 1), date(2020, 10, 1))
        repo.upsert_series(ts1)
        ts2 = TimeSeries("t:u", "t", "Updated", "m", "u", "US", 20, date(2020, 1, 1), date(2021, 8, 1))
        repo.upsert_series(ts2)
        result = repo.get_series("t:u")
        assert result is not None
        assert result.name == "Updated"
        assert result.observations == 20
