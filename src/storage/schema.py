"""DuckDB schema definition and initialization for OmniOracle."""

from __future__ import annotations

import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS series (
    variable_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    frequency VARCHAR NOT NULL,
    unit VARCHAR,
    geo VARCHAR NOT NULL,
    observations INTEGER,
    start_date DATE,
    end_date DATE
);

CREATE TABLE IF NOT EXISTS observations (
    variable_id VARCHAR NOT NULL,
    ts DATE NOT NULL,
    value DOUBLE NOT NULL,
    PRIMARY KEY (variable_id, ts)
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY,
    run_date DATE NOT NULL,
    x_id VARCHAR NOT NULL,
    y_id VARCHAR NOT NULL,
    direction VARCHAR NOT NULL,
    lag INTEGER,
    mi DOUBLE,
    granger_pvalue DOUBLE,
    oos_r2 DOUBLE,
    score DOUBLE,
    confidence VARCHAR
);
"""


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they don't exist."""
    conn.execute(SCHEMA_SQL)
