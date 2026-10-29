"""Rule-based benchmarks, traded with the GP's backtest (docs/RESEARCH_PLAN.md, non-GP benchmarks).

Every benchmark except buy-and-hold produces a daily buy signal and goes through
``regime_fusion.engine.backtest`` with the stock-year's exit settings, t+1
execution and costs, exactly like the GP models.

* Buy-and-hold: buy at the first open, hold to the last close; one cost per side.
* TS(d, r) (ActTrade's trading-strategy benchmark): buy whenever flat, so the
  exit rule alone decides the trades.
* Random entry: buy signals drawn at random at a rate that matches a target
  number of trades (the model's), many times; the model's Sharpe ratio is then
  compared with the distribution, which separates entry skill from the exit rule.
* Moving-average crossover: buy while the fast average of total-return closes
  is above the slow one.
* Time-series momentum: buy while the return over the look-back is positive.

Signals use data up to the close of each day only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_fusion.engine import metrics
from regime_fusion.engine.backtest import _simulate, backtest
from regime_fusion.engine.fitness import sharpe as fast_sharpe


def buy_and_hold(prices: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Daily returns of buying at the first open and holding to the last close."""
    close, cost = prices["adj_close"], cost_bps / 1e4
    daily = close.pct_change(fill_method=None).fillna(0.0)
    daily.iloc[0] = close.iloc[0] / (prices["adj_open"].iloc[0] * (1 + cost)) - 1
    daily.iloc[-1] = (1 + daily.iloc[-1]) * (1 - cost) - 1
    return daily


def always(prices: pd.DataFrame) -> np.ndarray:
    """TS(d, r): a buy signal every day; the exit rule decides."""
    return np.ones(len(prices), dtype=bool)


def moving_average_crossover(prices: pd.DataFrame, fast: int, slow: int) -> np.ndarray:
    close = prices["adj_close"]
    signal = close.rolling(fast).mean() > close.rolling(slow).mean()
    return signal.to_numpy(dtype=bool)


def time_series_momentum(prices: pd.DataFrame, lookback: int) -> np.ndarray:
    close = prices["adj_close"]
    return (close / close.shift(lookback) > 1.0).to_numpy(dtype=bool)


def random_entry(prices: pd.DataFrame, days: int, rate: float, cost_bps: float, target_trades: int,
                 rng: np.random.Generator, draws: int = 1000) -> dict[str, float]:
    """Sharpe ratios of random entries at the rate that gives ``target_trades`` on average."""
    arrays = [prices[c].to_numpy(dtype=np.float64) for c in ("adj_open", "adj_high", "adj_low", "adj_close")]
    cost = cost_bps / 1e4

    def run(probability: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        sharpes, trades = np.empty(n), np.empty(n)
        for k in range(n):
            signal = rng.random(len(prices)) < probability
            out = _simulate(signal, *arrays, days, rate, cost)
            sharpes[k], trades[k] = fast_sharpe(out[0]), out[-1]
        return sharpes, trades

    if target_trades <= 0:
        return {"probability": 0.0, "mean_trades": 0.0, "mean_sharpe": 0.0, "sharpes": np.zeros(draws)}
    low, high = 0.0, 1.0
    for _ in range(20):  # bisection on the daily entry probability
        middle = (low + high) / 2
        if run(middle, 40)[1].mean() < target_trades:
            low = middle
        else:
            high = middle
    sharpes, trades = run((low + high) / 2, draws)
    return {"probability": (low + high) / 2, "mean_trades": float(trades.mean()),
            "mean_sharpe": float(sharpes.mean()), "sharpes": sharpes}


def percentile_of(value: float, draws: np.ndarray) -> float:
    """Share of random draws the value beats (ties count half)."""
    draws = np.asarray(draws)
    return float(((draws < value).sum() + 0.5 * (draws == value).sum()) / draws.size)


def evaluate_signal(signal: np.ndarray, prices: pd.DataFrame, days: int, rate: float,
                    costs: tuple[float, ...]) -> dict[str, dict[str, float]]:
    """Metrics of a benchmark signal at every cost level."""
    return {f"{cost:g}bps": metrics.summary(backtest(signal, prices, days, rate, cost)) for cost in costs}
