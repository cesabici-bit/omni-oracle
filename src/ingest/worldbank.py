"""World Bank data fetcher.

Uses wbgapi library: https://github.com/tgherzog/wbgapi
"""

from __future__ import annotations

import pandas as pd

from src.ingest.base import BaseFetcher

# Key World Bank indicators for MVP
CURATED_WB_INDICATORS = [
    {"series_id": "NY.GDP.MKTP.CD", "name": "GDP (current US$)", "unit": "current_usd", "frequency": "annual"},
    {"series_id": "NY.GDP.MKTP.KD.ZG", "name": "GDP Growth (annual %)", "unit": "percent", "frequency": "annual"},
    {"series_id": "FP.CPI.TOTL.ZG", "name": "Inflation, CPI (annual %)", "unit": "percent", "frequency": "annual"},
    {"series_id": "SL.UEM.TOTL.ZS", "name": "Unemployment (% of labor force)", "unit": "percent", "frequency": "annual"},
    {"series_id": "NE.EXP.GNFS.ZS", "name": "Exports (% of GDP)", "unit": "percent", "frequency": "annual"},
    {"series_id": "NE.IMP.GNFS.ZS", "name": "Imports (% of GDP)", "unit": "percent", "frequency": "annual"},
    {"series_id": "BN.CAB.XOKA.CD", "name": "Current Account Balance", "unit": "current_usd", "frequency": "annual"},
    {"series_id": "GC.DOD.TOTL.GD.ZS", "name": "Central Government Debt (% GDP)", "unit": "percent", "frequency": "annual"},
    {"series_id": "FR.INR.RINR", "name": "Real Interest Rate (%)", "unit": "percent", "frequency": "annual"},
    {"series_id": "PA.NUS.FCRF", "name": "Official Exchange Rate (per US$)", "unit": "lcu_per_usd", "frequency": "annual"},
]

# Expanded World Bank indicators for F5 discovery
EXPANDED_WB_INDICATORS = [
    # Trade & Capital Flows
    {"series_id": "BX.KLT.DINV.CD.WD", "name": "Foreign Direct Investment, net inflows", "unit": "current_usd", "frequency": "annual"},
    {"series_id": "DT.DOD.DECT.CD", "name": "External Debt Stocks, total", "unit": "current_usd", "frequency": "annual"},
    {"series_id": "TM.VAL.MRCH.CD.WT", "name": "Merchandise imports", "unit": "current_usd", "frequency": "annual"},
    {"series_id": "TX.VAL.MRCH.CD.WT", "name": "Merchandise exports", "unit": "current_usd", "frequency": "annual"},
    # Industry & Services
    {"series_id": "NV.IND.TOTL.ZS", "name": "Industry (% of GDP)", "unit": "percent", "frequency": "annual"},
    {"series_id": "NV.SRV.TOTL.ZS", "name": "Services (% of GDP)", "unit": "percent", "frequency": "annual"},
    {"series_id": "NV.AGR.TOTL.ZS", "name": "Agriculture (% of GDP)", "unit": "percent", "frequency": "annual"},
    {"series_id": "NV.IND.MANF.ZS", "name": "Manufacturing (% of GDP)", "unit": "percent", "frequency": "annual"},
    # Demographics
    {"series_id": "SP.POP.TOTL", "name": "Population, total", "unit": "number", "frequency": "annual"},
    {"series_id": "SP.POP.GROW", "name": "Population growth (annual %)", "unit": "percent", "frequency": "annual"},
    {"series_id": "SP.URB.TOTL.IN.ZS", "name": "Urban population (% of total)", "unit": "percent", "frequency": "annual"},
    # Health
    {"series_id": "SP.DYN.LE00.IN", "name": "Life expectancy at birth", "unit": "years", "frequency": "annual"},
    {"series_id": "SH.XPD.CHEX.GD.ZS", "name": "Health expenditure (% of GDP)", "unit": "percent", "frequency": "annual"},
    # Energy & Environment
    {"series_id": "EN.ATM.CO2E.PC", "name": "CO2 emissions (metric tons per capita)", "unit": "metric_tons", "frequency": "annual"},
    {"series_id": "EG.USE.PCAP.KG.OE", "name": "Energy use per capita (kg oil equiv)", "unit": "kg_oil_equiv", "frequency": "annual"},
    {"series_id": "EG.ELC.ACCS.ZS", "name": "Access to electricity (%)", "unit": "percent", "frequency": "annual"},
    # Technology
    {"series_id": "IT.NET.USER.ZS", "name": "Internet users (% of population)", "unit": "percent", "frequency": "annual"},
    {"series_id": "GB.XPD.RSDV.GD.ZS", "name": "R&D expenditure (% of GDP)", "unit": "percent", "frequency": "annual"},
    # Government
    {"series_id": "GC.TAX.TOTL.GD.ZS", "name": "Tax revenue (% of GDP)", "unit": "percent", "frequency": "annual"},
    {"series_id": "MS.MIL.XPND.GD.ZS", "name": "Military expenditure (% of GDP)", "unit": "percent", "frequency": "annual"},
    # Income
    {"series_id": "NY.GNP.PCAP.CD", "name": "GNI per capita (current US$)", "unit": "current_usd", "frequency": "annual"},
    {"series_id": "SI.POV.GINI", "name": "Gini index", "unit": "index", "frequency": "annual"},
]

# Default economies to fetch
DEFAULT_ECONOMIES = ["USA", "GBR", "DEU", "JPN", "CHN", "IND", "BRA", "ITA", "FRA", "CAN"]


class WorldBankFetcher(BaseFetcher):
    """Fetcher for World Bank Open Data."""

    def __init__(
        self,
        economies: list[str] | None = None,
        expanded: bool = False,
    ) -> None:
        self.economies = economies or DEFAULT_ECONOMIES
        self.expanded = expanded

    @property
    def source_name(self) -> str:
        return "worldbank"

    def fetch_series_list(self, limit: int = 100) -> list[dict]:
        """Return indicator x economy combinations."""
        indicators = list(CURATED_WB_INDICATORS)
        if self.expanded:
            indicators = indicators + list(EXPANDED_WB_INDICATORS)
        result: list[dict] = []
        for indicator in indicators:
            for eco in self.economies:
                if len(result) >= limit:
                    return result
                result.append({
                    "series_id": f"{indicator['series_id']}_{eco}",
                    "indicator": indicator["series_id"],
                    "economy": eco,
                    "name": f"{indicator['name']} - {eco}",
                    "unit": indicator["unit"],
                    "frequency": indicator["frequency"],
                    "geo": eco,
                })
        return result

    def fetch_observations(self, series_id: str) -> pd.DataFrame:
        """Fetch observations for a World Bank indicator/economy pair."""
        import wbgapi as wb

        # series_id format: "INDICATOR_ECONOMY" e.g. "NY.GDP.MKTP.CD_USA"
        parts = series_id.rsplit("_", 1)
        if len(parts) != 2:
            return pd.DataFrame(columns=["ts", "value"])

        indicator, economy = parts

        try:
            df = wb.data.DataFrame(indicator, economy, numericTimeKeys=True)
            if df.empty:
                return pd.DataFrame(columns=["ts", "value"])

            # wbgapi returns years as columns, economies as rows
            # Transpose so years become rows
            values = df.iloc[0]  # single economy
            records = []
            for year_col, val in values.items():
                if pd.notna(val):
                    records.append({
                        "ts": pd.Timestamp(f"{int(year_col)}-01-01").date(),
                        "value": float(val),
                    })

            return pd.DataFrame(records)

        except Exception as e:
            print(f"  WARN: Failed to fetch WB {indicator} for {economy}: {e}")
            return pd.DataFrame(columns=["ts", "value"])
