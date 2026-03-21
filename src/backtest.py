"""Backtesting module for OmniOracle trading signals.

Tests whether lagged statistical relationships identified by the discovery
engine have predictive value in a simple directional trading strategy.

Strategy: for each (signal, target, lag) tuple:
  - At time t, observe signal change at t
  - Predict target direction at t+lag based on OLS coefficient sign from training set
  - Go long if predicted up, short if predicted down
  - Compute returns on test set only

Usage:
    python -m src.backtest
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess.stationarity import check_and_transform
from src.storage.repo import TimeSeriesRepo

F5_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "f5_discovery.duckdb"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# The two ROBUST relationships from cross-validation
ROBUST_SIGNALS = [
    {
        "name": "PCE_Price_to_Price_Pressures",
        "signal_id": "fred:PCEPI",
        "signal_name": "Personal Consumption Expenditures: Chain-type Price Index",
        "target_id": "fred:STLPPM",
        "target_name": "Price Pressures Measure",
        "lag": 1,
    },
    {
        "name": "Brent_to_Price_Pressures",
        "signal_id": "fred:POILBREUSDM",
        "signal_name": "Global price of Brent Crude",
        "target_id": "fred:STLPPM",
        "target_name": "Price Pressures Measure",
        "lag": 2,
    },
]


@dataclass
class BacktestResult:
    """Result of a single backtest."""

    name: str
    signal_name: str
    target_name: str
    lag: int
    train_size: int
    test_size: int
    # Strategy metrics
    sharpe_ratio: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    # Benchmarks
    buyhold_sharpe: float
    buyhold_return: float
    buyhold_max_drawdown: float
    random_sharpe_mean: float
    random_sharpe_std: float
    # Raw equity curves for plotting
    strategy_equity: list[float]
    buyhold_equity: list[float]
    test_dates: list[str]


def _get_series(repo: TimeSeriesRepo, var_id: str) -> pd.Series:
    """Fetch and resample a series to monthly."""
    obs = repo.get_observations(var_id)
    if obs.empty:
        raise ValueError(f"No data for {var_id}")
    s = pd.Series(obs["value"].values, index=pd.to_datetime(obs["ts"]))
    s = s.sort_index().resample("MS").mean().dropna()
    return s


def _max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown from equity curve."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown))


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio (assuming 0 risk-free rate)."""
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def run_backtest(
    signal: pd.Series,
    target: pd.Series,
    lag: int,
    train_ratio: float = 0.60,
    n_random: int = 1000,
) -> dict:
    """Run backtest for a signal-target pair.

    Strategy logic:
    1. Train OLS: target_change(t) = alpha + beta * signal_change(t - lag)
    2. On test set: if beta * signal_change(t - lag) > 0 -> long, else short
    3. Position is +1 or -1, held for 1 period, return = position * target_change(t)

    Args:
        signal: stationary signal series (monthly, DatetimeIndex)
        target: stationary target series (monthly, DatetimeIndex)
        lag: number of periods signal leads target
        train_ratio: fraction of data for training
        n_random: number of random strategies for benchmark

    Returns:
        dict with all backtest metrics
    """
    # Align series
    common = signal.index.intersection(target.index)
    if len(common) < lag + 40:
        raise ValueError(f"Too few common obs ({len(common)}) for lag={lag}")

    signal_al = signal.loc[common].values.astype(float)
    target_al = target.loc[common].values.astype(float)
    dates = common

    # Build lagged arrays: at time t, signal_lagged[t] = signal[t - lag]
    # target_returns[t] = target[t] (already stationary = changes)
    n = len(signal_al)
    signal_lagged = signal_al[: n - lag]  # signal at t-lag
    target_current = target_al[lag:]  # target at t
    dates_aligned = dates[lag:]

    # Train/test split
    n_aligned = len(signal_lagged)
    n_train = int(n_aligned * train_ratio)
    n_test = n_aligned - n_train

    if n_test < 10:
        raise ValueError(f"Test set too small ({n_test})")

    x_train = signal_lagged[:n_train]
    y_train = target_current[:n_train]
    x_test = signal_lagged[n_train:]
    y_test = target_current[n_train:]
    test_dates = dates_aligned[n_train:]

    # Train: simple OLS to get beta sign
    x_train_aug = np.column_stack([np.ones(n_train), x_train])
    try:
        beta, _, _, _ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("OLS failed")

    beta_signal = beta[1]  # coefficient of signal

    # Generate positions on test set: +1 (long) or -1 (short)
    predictions = beta_signal * x_test
    positions = np.where(predictions > 0, 1.0, -1.0)

    # Strategy returns
    strategy_returns = positions * y_test

    # Buy-and-hold benchmark: always long
    buyhold_returns = y_test

    # Since target is stationary (differenced), returns are additive
    # Use cumulative sum for equity curve
    strategy_cumret = np.cumsum(strategy_returns)
    buyhold_cumret = np.cumsum(buyhold_returns)

    # Normalize: equity = 100 + cumulative returns
    strategy_eq = 100 + strategy_cumret
    buyhold_eq = 100 + buyhold_cumret

    # Metrics
    sharpe = _sharpe_ratio(strategy_returns)
    ann_return = float(np.mean(strategy_returns) * 12)
    mdd = _max_drawdown(strategy_eq)
    win_rate = float(np.mean(strategy_returns > 0))

    bh_sharpe = _sharpe_ratio(buyhold_returns)
    bh_return = float(np.mean(buyhold_returns) * 12)
    bh_mdd = _max_drawdown(buyhold_eq)

    # Random benchmark: shuffle positions n_random times
    rng = np.random.default_rng(42)
    random_sharpes = []
    for _ in range(n_random):
        random_pos = rng.choice([-1.0, 1.0], size=n_test)
        random_ret = random_pos * y_test
        random_sharpes.append(_sharpe_ratio(random_ret))

    random_sharpes_arr = np.array(random_sharpes)

    return {
        "train_size": n_train,
        "test_size": n_test,
        "beta_signal": float(beta_signal),
        "sharpe_ratio": sharpe,
        "annualized_return": ann_return,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "total_trades": n_test,
        "buyhold_sharpe": bh_sharpe,
        "buyhold_return": bh_return,
        "buyhold_max_drawdown": bh_mdd,
        "random_sharpe_mean": float(np.mean(random_sharpes_arr)),
        "random_sharpe_std": float(np.std(random_sharpes_arr)),
        "strategy_equity": strategy_eq.tolist(),
        "buyhold_equity": buyhold_eq.tolist(),
        "test_dates": [d.strftime("%Y-%m-%d") for d in test_dates],
    }


def render_backtest_report(name: str, signal_name: str, target_name: str,
                           lag: int, result: dict) -> str:
    """Render a human-readable backtest report."""
    lines = [
        "=" * 70,
        f"  BACKTEST: {name}",
        "=" * 70,
        f"  Signal: {signal_name}",
        f"  Target: {target_name}",
        f"  Lag: {lag} months",
        f"  Train: {result['train_size']} obs | Test: {result['test_size']} obs",
        f"  Beta (signal coeff): {result['beta_signal']:.6f}",
        "",
        "  --- Strategy Performance (test set) ---",
        f"  Sharpe Ratio:      {result['sharpe_ratio']:+.3f}",
        f"  Annualized Return: {result['annualized_return']:+.4f}",
        f"  Max Drawdown:      {result['max_drawdown']:.2%}",
        f"  Win Rate:          {result['win_rate']:.1%}",
        f"  Total Trades:      {result['total_trades']}",
        "",
        "  --- Buy-and-Hold Benchmark ---",
        f"  Sharpe Ratio:      {result['buyhold_sharpe']:+.3f}",
        f"  Annualized Return: {result['buyhold_return']:+.4f}",
        f"  Max Drawdown:      {result['buyhold_max_drawdown']:.2%}",
        "",
        "  --- Random Benchmark (1000 shuffles) ---",
        f"  Mean Sharpe:       {result['random_sharpe_mean']:+.3f}",
        f"  Std Sharpe:        {result['random_sharpe_std']:.3f}",
        "",
    ]

    # Verdict
    beats_random = result["sharpe_ratio"] > (
        result["random_sharpe_mean"] + 2 * result["random_sharpe_std"]
    )
    beats_buyhold = result["sharpe_ratio"] > result["buyhold_sharpe"]

    verdict_parts = []
    if beats_random:
        verdict_parts.append("beats random (>2 sigma)")
    else:
        verdict_parts.append("does NOT beat random")
    if beats_buyhold:
        verdict_parts.append("beats buy-and-hold")
    else:
        verdict_parts.append("does NOT beat buy-and-hold")

    lines.append(f"  VERDICT: {' | '.join(verdict_parts)}")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("  OmniOracle -- Backtest: ROBUST Trading Signals")
    print("=" * 70)
    print()

    repo = TimeSeriesRepo(F5_DB_PATH)
    results_all = []

    for sig in ROBUST_SIGNALS:
        print(f"Processing: {sig['name']}...")

        # Get raw series
        signal_raw = _get_series(repo, sig["signal_id"])
        target_raw = _get_series(repo, sig["target_id"])

        # Make stationary
        signal_stat, signal_info = check_and_transform(signal_raw)
        target_stat, target_info = check_and_transform(target_raw)

        print(f"  Signal: {len(signal_stat)} obs "
              f"(transforms: {signal_info.transformations})")
        print(f"  Target: {len(target_stat)} obs "
              f"(transforms: {target_info.transformations})")

        # Run backtest
        try:
            result = run_backtest(
                signal_stat, target_stat,
                lag=sig["lag"],
                train_ratio=0.60,
            )
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            print()
            continue

        report = render_backtest_report(
            sig["name"], sig["signal_name"], sig["target_name"],
            sig["lag"], result,
        )
        print()
        print(report)
        print()

        results_all.append({
            "name": sig["name"],
            "signal_id": sig["signal_id"],
            "signal_name": sig["signal_name"],
            "target_id": sig["target_id"],
            "target_name": sig["target_name"],
            "lag": sig["lag"],
            **{k: v for k, v in result.items()
               if k not in ("strategy_equity", "buyhold_equity", "test_dates")},
            "equity_curve_length": len(result["strategy_equity"]),
        })

    repo.close()

    # Export
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "f5_backtest.json"
    output_data = {
        "disclaimer": (
            "Backtest on historical data. "
            "Past performance does NOT guarantee future results."
        ),
        "methodology": (
            "Simple directional strategy: OLS trained on first 60%, "
            "tested on last 40%. Position = +1 (long) or -1 (short) "
            "based on predicted direction from lagged signal."
        ),
        "results": results_all,
    }
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"Results exported to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
