# OmniOracle

**Automatic discovery of non-trivial statistical truths from public data.**

OmniOracle ingests thousands of public time series across domains (economics, commodities, labor, prices) and automatically discovers causal relationships using a rigorous multi-stage statistical pipeline. No human hypotheses needed -- the engine finds them.

<!-- CI badge placeholder -->
<!-- ![CI](https://github.com/genius-lab/omni-oracle/actions/workflows/ci.yml/badge.svg) -->

## How It Works

```
Public Data APIs (FRED, World Bank, ...)
         |
   [Ingest + Normalize]  -->  551 monthly time series
         |
   [Quality + Stationarity]  -->  241 series (ADF/KPSS tests)
         |
   [MI Screening]  -->  Mutual Information on all pairs, discard 99%
         |
   [Granger Causality]  -->  Direction + optimal lag (BIC)
         |
   [FDR Correction]  -->  Benjamini-Hochberg at alpha=0.05
         |
   [Out-of-Sample Validation]  -->  Temporal train/test split
         |
   [Post-Filters]  -->  Remove identity pairs, seasonality artifacts
         |
   [Cross-Validation]  -->  Sub-period robustness check
         |
   Ranked Hypothesis Cards with score, p-value, lag, confidence
```

### Discovery Pipeline (6 stages)

1. **Mutual Information screening** -- non-parametric dependency measure, catches non-linear relationships that correlation misses
2. **Granger causality** -- tests whether X's past improves prediction of Y beyond Y's own past, identifies direction and optimal lag
3. **Benjamini-Hochberg FDR** -- controls false discovery rate across thousands of simultaneous tests
4. **Out-of-sample validation** -- temporal train/test split, measures incremental R2 on held-out future data
5. **Post-discovery filters** -- removes identity pairs (same series, different name), high-correlation duplicates, seasonality artifacts
6. **Cross-validation on sub-periods** -- splits history in half, requires positive R2 in both halves for "ROBUST" label

## Key Results

From 551 time series (253 FRED + 298 World Bank), the engine discovered:

- **4,297 raw hypotheses** -> **3,484 after filters** -> **1,114 OOS-validated**
- **8/8 known economic relationships rediscovered** without being told to look for them:
  - Okun's Law (unemployment <-> GDP)
  - Oil prices -> CPI (3-6 month lag)
  - Fed Funds Rate <-> Treasury yields
  - M2 money supply -> inflation
  - and 4 more
- **2 ROBUST trading signals** (positive R2 in both sub-periods):

| Signal | Target | Lag | OOS R2 | Backtest Sharpe |
|--------|--------|-----|--------|-----------------|
| PCE Price Index | Price Pressures Measure | 1 month | 0.294 | **+2.19** |
| Brent Crude | Price Pressures Measure | 2 months | 0.215 | **+1.28** |

Both signals beat random strategies at >2 sigma significance.

## Installation

```bash
# Clone
git clone https://github.com/genius-lab/omni-oracle.git
cd omni-oracle

# Setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Set FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
echo "FRED_API_KEY=your_key_here" > .env
```

## Usage

```bash
# Run tests
pytest tests/ -v

# Full check (lint + test + verify)
make check-all

# Ingest data (requires FRED_API_KEY)
python -m src.ingest.fred --limit 500
python -m src.ingest.worldbank --limit 300

# Run discovery pipeline
python -m src.pipeline --source fred worldbank --db data/f5_discovery.duckdb

# View filtered results (fast, from cache)
python -m src.run_f5_filter

# Re-run filters from scratch
python -m src.run_f5_filter --recompute

# Backtest ROBUST trading signals
python -m src.backtest
```

## Output Format

Each discovery is a **Hypothesis Card**:

```
+-----------------------------------------------------------+
| #9  Score: 7.5/10  [HIGH]
+-----------------------------------------------------------+
|  Personal Consumption Expenditures: Chain-type Price Index
|    x->y  (lag: 1 periods)
|  Price Pressures Measure
|
|  MI: 0.0650  |  Granger p: 7.57e-29  |  OOS R2: 0.2941
+-----------------------------------------------------------+
```

Fields: mutual information, Granger p-value, out-of-sample R2, direction, lag, confidence level, caveats.

## Project Structure

```
omni-oracle/
  src/
    ingest/        # Data fetchers (FRED, World Bank)
    storage/       # DuckDB repository layer
    preprocess/    # Quality checks, stationarity transforms
    discovery/     # MI screening, Granger causality
    validation/    # FDR correction, OOS validation
    scoring/       # Composite ranking
    output/        # Hypothesis cards, trading reports, filters
    backtest.py    # Trading signal backtesting
    pipeline.py    # End-to-end orchestrator
  tests/           # 63+ tests (unit, L2 domain, L3 property-based, L5 real data)
  verify/          # M4 cross-tool verification (alternative implementations)
```

## Verification

OmniOracle uses a 5-level verification framework:

- **L1**: Unit tests on every module
- **L2**: Domain sanity tests with values from external sources (Granger 1969, Toda-Yamamoto 1995, FRED documentation)
- **L3**: Property-based tests (Hypothesis library) checking statistical invariants
- **L4**: Golden snapshot -- smoke test output reviewed by human
- **L5**: Real-data validation -- system must rediscover known economic relationships from literature

Cross-tool verification (M4): alternative MI and Granger implementations in `verify/` confirm main pipeline results.

## Tech Stack

Python 3.12+ | Pandas | SciPy | Scikit-learn | Statsmodels | DuckDB | FRED API | World Bank API

## License

MIT

## Disclaimer

All results are **statistical associations**, not proof of causation. Trading signals are historical backtests -- past performance does not guarantee future results. This is a research tool, not financial advice.
