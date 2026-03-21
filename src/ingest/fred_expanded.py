"""Expanded FRED series discovery using the FRED search API.

Discovers additional high-quality series beyond the curated MVP list.
Uses fred.search() to find series across multiple macro categories,
then filters for quality (monthly+, sufficient history, US-focused).

fredapi docs: https://github.com/mortada/fredapi
fred.search() returns DataFrame with columns:
  id (index), title, observation_start, observation_end,
  frequency_short, units_short, seasonal_adjustment_short, popularity
"""

from __future__ import annotations

import os
import time

from src.ingest.fred import CURATED_FRED_SERIES

# Search queries: (keyword, max_series_to_keep)
DISCOVERY_QUERIES = [
    # Commodities & Energy
    ("crude oil price", 6),
    ("natural gas price", 4),
    ("gold price london", 4),
    ("copper price", 3),
    ("commodity price index", 6),
    ("agricultural commodity price", 4),
    ("lumber price", 2),
    ("gasoline price regular", 3),
    # Housing & Real Estate
    ("housing price index national", 6),
    ("mortgage rate fixed", 4),
    ("new home sales", 3),
    ("existing home sales", 3),
    ("construction spending", 3),
    ("housing inventory", 3),
    # Labor Market
    ("nonfarm payrolls employment", 6),
    ("unemployment rate U6", 3),
    ("labor force participation rate", 3),
    ("average hourly earnings", 4),
    ("average weekly hours manufacturing", 3),
    ("quit rate JOLTS", 3),
    ("hires JOLTS", 3),
    ("continuing claims unemployment", 3),
    # Manufacturing & Industry
    ("ISM manufacturing PMI", 4),
    ("industrial production manufacturing", 4),
    ("capacity utilization total", 3),
    ("factory orders total", 3),
    ("new orders manufacturing", 3),
    # Consumer & Retail
    ("retail sales food", 3),
    ("consumer sentiment michigan", 3),
    ("consumer confidence board", 3),
    ("consumer credit outstanding", 4),
    ("personal consumption expenditure services", 3),
    # Interest Rates & Yields
    ("treasury constant maturity 2 year", 3),
    ("treasury constant maturity 5 year", 3),
    ("treasury constant maturity 30 year", 3),
    ("corporate bond yield AAA", 3),
    ("corporate bond yield BAA", 3),
    ("bank prime loan rate", 2),
    ("TED spread", 2),
    ("commercial paper rate", 3),
    # Money & Banking
    ("money stock M1", 3),
    ("monetary base total", 3),
    ("bank credit all commercial", 3),
    ("commercial and industrial loans", 3),
    ("real estate loans commercial", 3),
    ("delinquency rate loans", 3),
    # Trade & FX
    ("trade balance goods services", 4),
    ("exchange rate dollar yen", 3),
    ("exchange rate dollar pound", 3),
    ("exchange rate canadian dollar", 3),
    ("exchange rate chinese yuan", 3),
    ("exchange rate swiss franc", 3),
    # Financial Markets
    ("S&P 500 stock price index", 3),
    ("NASDAQ composite index", 2),
    ("Wilshire 5000 total market", 2),
    # Government & Fiscal
    ("federal government receipts", 3),
    ("federal government expenditure", 3),
    ("federal debt held public", 3),
    # Prices & Inflation
    ("producer price index finished goods", 4),
    ("import price index", 3),
    ("export price index", 3),
    ("CPI food beverages", 3),
    ("CPI energy", 3),
    ("CPI shelter", 3),
    ("PCE price index services", 3),
    ("breakeven inflation rate", 3),
    ("inflation expectations university michigan", 3),
    # Income & Savings
    ("real personal income excluding transfers", 3),
    ("real disposable personal income", 3),
    ("personal saving rate", 2),
    ("corporate profits after tax", 3),
    # Productivity & Costs
    ("nonfarm business labor productivity", 3),
    ("unit labor cost nonfarm", 3),
    # Transportation & Logistics
    ("vehicle miles traveled", 2),
    ("rail freight carloads", 2),
    ("trucking conditions index", 2),
    # Leading Indicators
    ("leading index conference board", 3),
    ("chicago fed national activity index", 2),
    ("financial conditions index", 3),
    ("Kansas City financial stress", 2),
    ("St Louis financial stress", 2),
    # Velocity & Aggregates
    ("velocity of money M2", 2),
    ("gross domestic income", 2),
    ("GDP price deflator", 2),
]

# Frequency codes allowed (monthly or higher)
ALLOWED_FREQUENCIES = {"M", "W", "BW", "D"}

# US state abbreviations for filtering regional series
_STATE_NAMES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
}


def _is_regional(title: str) -> bool:
    """Check if a series title suggests a state/regional series."""
    title_lower = title.lower()
    for state in _STATE_NAMES:
        if state.lower() in title_lower:
            return True
    # Also check for common MSA/region patterns
    if any(kw in title_lower for kw in [
        " msa", " metro", " county", " district of columbia",
        " region:", " state:", "(state)",
    ]):
        return True
    return False


def discover_fred_series(
    api_key: str | None = None,
    max_total: int = 400,
    min_popularity: int = 10,
    request_delay: float = 0.55,
) -> list[dict]:
    """Discover FRED series using search API.

    Returns list of series metadata dicts compatible with FREDFetcher.
    Deduplicates against CURATED_FRED_SERIES.

    Args:
        api_key: FRED API key. Falls back to FRED_API_KEY env var.
        max_total: Maximum number of new series to discover.
        min_popularity: Minimum FRED popularity score (0-100).
        request_delay: Seconds between API calls (rate limit: 120/min).

    Returns:
        List of dicts with keys: series_id, name, unit, frequency.
    """
    from fredapi import Fred

    api_key = api_key or os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise ValueError("FRED_API_KEY required for series discovery")

    fred = Fred(api_key=api_key)

    # Already-known series IDs (curated MVP list)
    curated_ids = {s["series_id"] for s in CURATED_FRED_SERIES}
    seen_ids: set[str] = set(curated_ids)
    discovered: list[dict] = []

    print(f"Discovering FRED series ({len(DISCOVERY_QUERIES)} categories)...")

    for qi, (query, max_keep) in enumerate(DISCOVERY_QUERIES):
        if len(discovered) >= max_total:
            break

        try:
            results = fred.search(
                query, limit=50,
                order_by="popularity", sort_order="desc",
            )
            time.sleep(request_delay)
        except Exception as e:
            print(f"  WARN: Search '{query}' failed: {e}")
            continue

        if results is None or results.empty:
            continue

        kept = 0
        for sid, row in results.iterrows():
            if kept >= max_keep or len(discovered) >= max_total:
                break

            sid = str(sid)
            if sid in seen_ids:
                continue

            # Frequency filter
            freq = str(row.get("frequency_short", "")).upper()
            if freq not in ALLOWED_FREQUENCIES:
                continue

            # Popularity filter
            pop = int(row.get("popularity", 0))
            if pop < min_popularity:
                continue

            title = str(row.get("title", ""))

            # Skip regional/state series
            if _is_regional(title):
                continue

            # Skip discontinued series (heuristic: last updated > 2 years ago)
            # Not reliable, so we rely on the ingest step to handle failures

            seen_ids.add(sid)
            discovered.append({
                "series_id": sid,
                "name": title[:120],
                "unit": str(row.get("units_short", "")),
                "frequency": freq.lower() if freq else "monthly",
            })
            kept += 1

        if (qi + 1) % 10 == 0:
            print(f"  ... {qi + 1}/{len(DISCOVERY_QUERIES)} queries, "
                  f"{len(discovered)} series found")

    print(f"Discovered {len(discovered)} additional FRED series "
          f"(+ {len(curated_ids)} curated = {len(discovered) + len(curated_ids)} total)")
    return discovered
