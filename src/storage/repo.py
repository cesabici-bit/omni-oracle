"""TimeSeriesRepo: CRUD operations on DuckDB for time series data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src.models import TimeSeries
from src.storage.schema import init_db


class TimeSeriesRepo:
    """Repository for storing and querying time series in DuckDB."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
        init_db(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "TimeSeriesRepo":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Series metadata ---

    def upsert_series(self, ts: TimeSeries) -> None:
        """Insert or update series metadata."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO series
                (variable_id, source, name, frequency, unit,
                 geo, observations, start_date, end_date, domain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ts.variable_id,
                ts.source,
                ts.name,
                ts.frequency,
                ts.unit,
                ts.geo,
                ts.observations,
                ts.start_date,
                ts.end_date,
                ts.domain,
            ],
        )

    def get_series(self, variable_id: str) -> TimeSeries | None:
        """Get series metadata by variable_id."""
        result = self.conn.execute(
            "SELECT * FROM series WHERE variable_id = ?", [variable_id]
        ).fetchone()
        if result is None:
            return None
        return TimeSeries(
            variable_id=result[0],
            source=result[1],
            name=result[2],
            frequency=result[3],
            unit=result[4] or "",
            geo=result[5],
            observations=result[6],
            start_date=result[7],
            end_date=result[8],
            domain=result[9] if len(result) > 9 else "economics",
        )

    def list_series(
        self,
        source: str | None = None,
        geo: str | None = None,
    ) -> list[TimeSeries]:
        """List series, optionally filtered by source and/or geo."""
        query = "SELECT * FROM series WHERE 1=1"
        params: list[str] = []
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        if geo is not None:
            query += " AND geo = ?"
            params.append(geo)
        query += " ORDER BY variable_id"
        rows = self.conn.execute(query, params).fetchall()
        return [
            TimeSeries(
                variable_id=r[0],
                source=r[1],
                name=r[2],
                frequency=r[3],
                unit=r[4] or "",
                geo=r[5],
                observations=r[6],
                start_date=r[7],
                end_date=r[8],
                domain=r[9] if len(r) > 9 else "economics",
            )
            for r in rows
        ]

    def count_series(self) -> int:
        """Return total number of series."""
        result = self.conn.execute("SELECT COUNT(*) FROM series").fetchone()
        assert result is not None
        return result[0]

    # --- Observations ---

    def insert_observations(
        self, variable_id: str, df: pd.DataFrame
    ) -> None:
        """Insert observations from a DataFrame with columns ['ts', 'value'].

        Existing observations for the same (variable_id, ts) are replaced.
        """
        assert "ts" in df.columns and "value" in df.columns, (
            "DataFrame must have 'ts' and 'value' columns"
        )
        for _, row in df.iterrows():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO observations (variable_id, ts, value)
                VALUES (?, ?, ?)
                """,
                [variable_id, row["ts"], float(row["value"])],
            )

    def insert_observations_bulk(
        self, variable_id: str, df: pd.DataFrame
    ) -> None:
        """Bulk insert observations — faster for large datasets.

        DataFrame must have columns ['ts', 'value'].
        """
        assert "ts" in df.columns and "value" in df.columns, (
            "DataFrame must have 'ts' and 'value' columns"
        )
        temp = df[["ts", "value"]].copy()
        temp["variable_id"] = variable_id
        temp = temp[["variable_id", "ts", "value"]]
        # Use DuckDB's native DataFrame ingestion
        self.conn.execute(
            """
            INSERT OR REPLACE INTO observations
            SELECT * FROM temp
            """
        )

    def get_observations(
        self,
        variable_id: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Get observations as DataFrame with columns ['ts', 'value']."""
        query = "SELECT ts, value FROM observations WHERE variable_id = ?"
        params: list[object] = [variable_id]
        if start is not None:
            query += " AND ts >= ?"
            params.append(start)
        if end is not None:
            query += " AND ts <= ?"
            params.append(end)
        query += " ORDER BY ts"
        return self.conn.execute(query, params).fetchdf()

    def get_aligned_pair(
        self,
        var_x: str,
        var_y: str,
    ) -> pd.DataFrame:
        """Get two series aligned by date (inner join).

        Returns DataFrame with columns ['ts', 'x', 'y'].
        """
        return self.conn.execute(
            """
            SELECT a.ts, a.value AS x, b.value AS y
            FROM observations a
            INNER JOIN observations b ON a.ts = b.ts
            WHERE a.variable_id = ? AND b.variable_id = ?
            ORDER BY a.ts
            """,
            [var_x, var_y],
        ).fetchdf()

    def get_all_variable_ids(self) -> list[str]:
        """Return all variable_ids that have observations."""
        rows = self.conn.execute(
            "SELECT DISTINCT variable_id FROM observations ORDER BY variable_id"
        ).fetchall()
        return [r[0] for r in rows]
