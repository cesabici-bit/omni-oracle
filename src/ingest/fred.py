"""FRED (Federal Reserve Economic Data) fetcher.

Requires FRED_API_KEY environment variable.
Uses fredapi library: https://github.com/mortada/fredapi
"""

from __future__ import annotations

import os

import pandas as pd

from src.ingest.base import BaseFetcher

# Curated list of key US macroeconomic series for MVP
CURATED_FRED_SERIES = [
    {"series_id": "CPIAUCSL", "name": "Consumer Price Index (All Urban)", "unit": "index", "frequency": "monthly"},
    {"series_id": "UNRATE", "name": "Unemployment Rate", "unit": "percent", "frequency": "monthly"},
    {"series_id": "GDP", "name": "Gross Domestic Product", "unit": "billions_dollars", "frequency": "quarterly"},
    {"series_id": "FEDFUNDS", "name": "Federal Funds Effective Rate", "unit": "percent", "frequency": "monthly"},
    {"series_id": "DCOILBRENTEU", "name": "Crude Oil Prices: Brent", "unit": "dollars_per_barrel", "frequency": "daily"},
    {"series_id": "GS10", "name": "10-Year Treasury Constant Maturity Rate", "unit": "percent", "frequency": "monthly"},
    {"series_id": "M2SL", "name": "M2 Money Stock", "unit": "billions_dollars", "frequency": "monthly"},
    {"series_id": "HOUST", "name": "Housing Starts: Total", "unit": "thousands_units", "frequency": "monthly"},
    {"series_id": "RSAFS", "name": "Advance Retail Sales: Total", "unit": "millions_dollars", "frequency": "monthly"},
    {"series_id": "INDPRO", "name": "Industrial Production Index", "unit": "index", "frequency": "monthly"},
    {"series_id": "PAYEMS", "name": "All Employees: Total Nonfarm", "unit": "thousands", "frequency": "monthly"},
    {"series_id": "PPIACO", "name": "Producer Price Index: All Commodities", "unit": "index", "frequency": "monthly"},
    {"series_id": "UMCSENT", "name": "Univ. of Michigan Consumer Sentiment", "unit": "index", "frequency": "monthly"},
    {"series_id": "DEXUSEU", "name": "US/Euro Exchange Rate", "unit": "dollars_per_euro", "frequency": "daily"},
    {"series_id": "TB3MS", "name": "3-Month Treasury Bill Rate", "unit": "percent", "frequency": "monthly"},
    {"series_id": "PCEPILFE", "name": "Core PCE Price Index", "unit": "index", "frequency": "monthly"},
    {"series_id": "CPILFESL", "name": "Core CPI (Less Food and Energy)", "unit": "index", "frequency": "monthly"},
    {"series_id": "JTSJOL", "name": "Job Openings: Total Nonfarm", "unit": "thousands", "frequency": "monthly"},
    {"series_id": "ICSA", "name": "Initial Claims Unemployment Insurance", "unit": "number", "frequency": "weekly"},
    {"series_id": "DGORDER", "name": "Manufacturers New Orders: Durable Goods", "unit": "millions_dollars", "frequency": "monthly"},
    {"series_id": "PERMIT", "name": "New Building Permits", "unit": "thousands_units", "frequency": "monthly"},
    {"series_id": "AWHMAN", "name": "Average Weekly Hours: Manufacturing", "unit": "hours", "frequency": "monthly"},
    {"series_id": "WPSFD49207", "name": "PPI: Finished Goods", "unit": "index", "frequency": "monthly"},
    {"series_id": "MZMSL", "name": "MZM Money Stock", "unit": "billions_dollars", "frequency": "monthly"},
    {"series_id": "USSLIND", "name": "Leading Index for the US", "unit": "index", "frequency": "monthly"},
    {"series_id": "CSUSHPINSA", "name": "S&P/Case-Shiller Home Price Index", "unit": "index", "frequency": "monthly"},
    {"series_id": "VIXCLS", "name": "CBOE Volatility Index: VIX", "unit": "index", "frequency": "daily"},
    {"series_id": "BAMLH0A0HYM2", "name": "ICE BofA US High Yield Spread", "unit": "percent", "frequency": "daily"},
    {"series_id": "T10Y2Y", "name": "10Y-2Y Treasury Spread", "unit": "percent", "frequency": "daily"},
    {"series_id": "DTWEXBGS", "name": "Trade Weighted US Dollar Index", "unit": "index", "frequency": "daily"},
    {"series_id": "BOGMBASE", "name": "Monetary Base", "unit": "millions_dollars", "frequency": "monthly"},
    {"series_id": "TOTALSA", "name": "Total Vehicle Sales", "unit": "millions_units", "frequency": "monthly"},
    {"series_id": "PCE", "name": "Personal Consumption Expenditures", "unit": "billions_dollars", "frequency": "monthly"},
    {"series_id": "PI", "name": "Personal Income", "unit": "billions_dollars", "frequency": "monthly"},
    {"series_id": "CPALTT01USM657N", "name": "CPI: All Items Growth Rate", "unit": "percent", "frequency": "monthly"},
    {"series_id": "GDPC1", "name": "Real GDP", "unit": "billions_chained_2017", "frequency": "quarterly"},
    {"series_id": "PSAVERT", "name": "Personal Saving Rate", "unit": "percent", "frequency": "monthly"},
    {"series_id": "DPCERD3Q086SBEA", "name": "Real PCE per Capita", "unit": "chained_dollars", "frequency": "quarterly"},
    {"series_id": "GPDI", "name": "Gross Private Domestic Investment", "unit": "billions_dollars", "frequency": "quarterly"},
    {"series_id": "IMPGS", "name": "Imports of Goods and Services", "unit": "billions_dollars", "frequency": "quarterly"},
    {"series_id": "EXPGS", "name": "Exports of Goods and Services", "unit": "billions_dollars", "frequency": "quarterly"},
    {"series_id": "GFDEBTN", "name": "Federal Debt: Total Public Debt", "unit": "millions_dollars", "frequency": "quarterly"},
    {"series_id": "FYFSD", "name": "Federal Surplus or Deficit", "unit": "millions_dollars", "frequency": "annual"},
    {"series_id": "RRSFS", "name": "Real Retail Sales", "unit": "millions_dollars", "frequency": "monthly"},
    {"series_id": "NAPM", "name": "ISM Manufacturing: PMI Composite", "unit": "index", "frequency": "monthly"},
    {"series_id": "NEWORDER", "name": "ISM Manufacturing: New Orders", "unit": "index", "frequency": "monthly"},
    {"series_id": "BUSINV", "name": "Total Business Inventories", "unit": "millions_dollars", "frequency": "monthly"},
    {"series_id": "CMRMTSPL", "name": "Real Manufacturing and Trade Sales", "unit": "millions_dollars", "frequency": "monthly"},
    {"series_id": "DSPIC96", "name": "Real Disposable Personal Income", "unit": "billions_chained_2017", "frequency": "monthly"},
    {"series_id": "W875RX1", "name": "Real Personal Income Excl Transfers", "unit": "billions_chained_2017", "frequency": "monthly"},
]


class FREDFetcher(BaseFetcher):
    """Fetcher for Federal Reserve Economic Data (FRED)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not set. Set via env var or pass api_key."
            )
        from fredapi import Fred

        self.fred = Fred(api_key=self.api_key)

    @property
    def source_name(self) -> str:
        return "fred"

    def fetch_series_list(self, limit: int = 100) -> list[dict]:
        """Return curated list of FRED series, up to limit."""
        return CURATED_FRED_SERIES[:limit]

    def fetch_observations(self, series_id: str) -> pd.DataFrame:
        """Fetch observations for a FRED series."""
        raw = self.fred.get_series(series_id)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["ts", "value"])

        # Convert to DataFrame with ts and value columns
        df = pd.DataFrame({"ts": raw.index.date, "value": raw.values})
        df = df.dropna(subset=["value"])
        return df
