# OmniOracle

**Automatic discovery of non-trivial statistical truths from heterogeneous public data.**

OmniOracle ingests hundreds of public time series across domains (economics, commodities, labor, prices, demographics) and automatically discovers statistically significant lagged relationships using a rigorous multi-stage pipeline. No human hypotheses needed -- the engine finds them, validates them, and filters out the noise.

[![CI](https://github.com/cesabici-bit/omni-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/cesabici-bit/omni-oracle/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## The Problem

Public data contains thousands of latent relationships. Economists know some (Oil -> CPI, Fed Funds -> Yields), but manually screening 500+ time series (125,000+ pairwise combinations) is intractable. Most automated approaches drown in false positives from multiple testing.

OmniOracle tackles this with a 6-stage statistical pipeline that goes from raw data to validated, ranked hypotheses -- automatically.

## Pipeline

```
Public Data APIs (FRED, World Bank, EIA, NOAA)
         |
   [Ingest + Normalize]       551 monthly time series
         |
   [Quality + Stationarity]   ADF/KPSS tests, differencing
         |
   [MI Screening]             Mutual Information: discard 99% of pairs
         |
   [Lagged MI Direction]      Non-linear directional test + optimal lag
         |
   [FDR Correction]           Benjamini-Hochberg at alpha=0.05
         |
   [OOS Validation]           Ridge/RF walk-forward, incremental R2
         |
   [Post-Filters]             Blacklist derived series, remove identity
         |                    pairs, high-correlation duplicates
   [Walk-Forward CV]          Multi-window robustness check
         |
   Ranked Hypothesis Cards
```

### Why This Order Matters

Each stage is more expensive than the previous one. MI screening (fast, non-parametric) eliminates 99% of pairs before the expensive directional test runs. FDR correction prevents the multiple-testing explosion. OOS validation catches overfitting. Post-filters catch tautologies (see [Lessons Learned](#lessons-learned)). Walk-forward CV catches regime-dependent relationships.

## Key Results

### Discovery: It Works

From 551 time series (253 FRED + 298 World Bank), pipeline v2 (Lagged MI + Ridge/RF walk-forward):

| Metric | Value |
|--------|-------|
| Clean hypotheses | 6,882 |
| Known relationships rediscovered | **8/8** (100%) |
| Walk-forward ROBUST signals | 5 (4 adjusted-robust) |

The engine finds known economic relationships **without being told to look for them**:

- Okun's Law (unemployment <-> GDP growth)
- Oil prices -> CPI (3-6 month lag)
- Fed Funds Rate <-> Treasury yields
- M2 money supply -> inflation
- Corporate credit spreads -> economic activity
- Manufacturing hours -> manufacturing employment
- Consumer confidence -> retail spending
- Housing starts -> construction employment

### Trading: It Doesn't Work

The 5 ROBUST signals were backtested with a simple directional strategy (Ridge, 60/40 train/test split). **None beat the random benchmark** (no Sharpe ratio > 2 sigma above random shuffles):

| Signal | Lag | OOS R2 | Backtest Sharpe | vs Random |
|--------|-----|--------|-----------------|-----------|
| Imports -> Gas Price | 8 | 0.57 | -0.10 | NO |
| Imports -> Gas Price | 3 | 0.53 | -0.57 | NO |
| Imports -> Trade Balance | 11 | 0.52 | -0.15 | NO |
| USD/EUR -> Semiconductor | 8 | 0.22 | -0.15 | NO |
| Fed Collateral -> Exports | 4 | 0.21 | +0.14 | NO |

**Why high R2 but no trading edge?** Walk-forward R2 measures variance explained -- the model captures the *shape* of the relationship. But directional trading needs consistent *sign* prediction, and with near-zero coefficients or regime-shifting relationships, the direction is essentially a coin flip. Additionally, Imports -> Trade Balance is near-tautological (imports are an accounting component of trade balance).

### Conclusions

1. **The discovery engine works**: it reliably finds genuine statistical relationships, including all known benchmarks
2. **Public monthly macro data has no tradable edge**: if a signal in FRED data were actionable, it would have been arbitraged away long ago
3. **Statistical significance != economic significance**: a relationship can be statistically robust but have zero practical value
4. **Honest negative results are valuable**: knowing that automated discovery from public data doesn't produce alpha is useful information for anyone considering this path

## Installation

```bash
git clone https://github.com/cesabici-bit/omni-oracle.git
cd omni-oracle

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Set FRED API key (free: https://fred.stlouisfed.org/docs/api/api_key.html)
echo "FRED_API_KEY=your_key_here" > .env
```

## Usage

```bash
# Run all checks (lint + test + cross-tool verification)
make check-all

# Run tests only
pytest tests/ -v

# Ingest data (requires FRED_API_KEY)
python -m src.ingest.fred --limit 500
python -m src.ingest.worldbank --limit 300

# Run discovery pipeline
python -m src.run_f5

# Apply filters + cross-validation (fast, from cache)
python -m src.run_f5_filter --refilter

# Re-run full pipeline from scratch (~50 min)
python -m src.run_f5_filter --recompute

# Backtest ROBUST signals
python -m src.backtest
```

## Output Format

Each discovery is a **Hypothesis Card**:

```
+-----------------------------------------------------------+
| #6  Score: 7.1/10  [HIGH]
+-----------------------------------------------------------+
|  ICE BofA US Corporate Index Option-Adjusted Spread
|    x->y  (lag: 2 periods)
|  Chicago Fed National Activity Index
|
|  MI: 0.1335  |  Direction p: 9.90e-03  |  OOS R2: 0.2581
+-----------------------------------------------------------+
```

## Verification

5-level verification framework designed to catch errors at every stage:

| Level | What | How |
|-------|------|-----|
| **L1** Unit | Each function does what it claims | 46 unit tests |
| **L2** Domain | Results are plausible in the domain | 11 tests with values from published sources (Granger 1969, Toda-Yamamoto 1995, FRED docs) |
| **L3** Property | Statistical invariants hold for any valid input | 6 property-based tests (Hypothesis library) |
| **L4** Golden | Pipeline output is stable and human-reviewed | Smoke test snapshot, approved once |
| **L5** Real data | System rediscovers known truths from literature | 10 tests against documented economic relationships |

**Cross-tool verification (M4)**: Alternative MI (histogram-based) and Granger (manual OLS) implementations in `verify/` confirm main pipeline results.

Total: **118 tests**, all passing.

## Lessons Learned

### EC-003: Derived Series Create Tautological Discoveries

The St. Louis Fed Price Pressures Measure (STLPPM) is a FAVAR model that takes 104 input series (including PCE and commodity prices) and outputs a 12-month forward inflation probability. When we included STLPPM as a discoverable variable, the engine correctly found that PCE and Brent Crude "predict" it -- but this is circular (input predicts output of model), not a genuine causal discovery.

**Fix**: Blacklist derived/model-based series. Before including any series in the discovery pool, verify: (1) is it forward-looking? (2) are its inputs already in the pool?

**Reference**: Jackson, Kliesen, Owyang (2015) "A Measure of Price Pressures", Federal Reserve Bank of St. Louis Review, 97(1), pp.25-52.

### High Walk-Forward R2 Does Not Imply Tradability

Walk-forward cross-validation measures whether a model consistently explains variance across time windows. A signal can have R2 = 0.57 (strong) but produce Sharpe = -0.10 (useless for trading) because:
- The regression coefficient can be near-zero (direction prediction is noise)
- The relationship can be near-tautological (accounting identity, not causal)
- Regime shifts can invert the coefficient sign between windows

**Takeaway**: OOS R2 validates *statistical* relationships. *Economic* significance requires separate testing (backtest, position sizing, transaction costs).

## Project Structure

```
omni-oracle/
  src/
    ingest/        # Data fetchers (FRED, World Bank, EIA, NOAA)
    storage/       # DuckDB repository layer
    preprocess/    # Quality checks, stationarity transforms
    discovery/     # MI screening, lagged MI directional test
    validation/    # FDR correction, OOS temporal validation (Ridge/RF)
    scoring/       # Composite ranking
    output/        # Hypothesis cards, trading reports, walk-forward CV
    pipeline.py    # End-to-end orchestrator
    backtest.py    # Trading signal backtester
  tests/           # 118 tests (L1-L5 verification levels)
  verify/          # M4 cross-tool verification
```

## Tech Stack

Python 3.12+ | Pandas | SciPy | Scikit-learn | Statsmodels | DuckDB | FRED API | World Bank API

## License

MIT

## Acknowledgments

Development assisted by [Claude Code](https://claude.ai/claude-code) (Anthropic).

## Disclaimer

All results are **statistical associations**, not proof of causation. This is a research tool, not financial advice. Past statistical relationships do not guarantee future persistence.
