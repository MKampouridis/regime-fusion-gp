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

``run_rule`` runs a rule like a GP rung (``configs/benchmarks.yaml``): the
parameters of a rule with a grid are chosen on the validation year by Sharpe
ratio (fewer than ``min_trades`` trades score zero), and stage 2 trades the test
year with them. Results are written like GP runs, so campaigns and the analysis
treat rules, machine-learning benchmarks and rungs alike.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import StudyConfig
from regime_fusion.engine import metrics
from regime_fusion.engine.backtest import Result, _simulate, backtest
from regime_fusion.engine.fitness import sharpe as fast_sharpe
from regime_fusion.engine.panel import price_entries, stock_prices
from regime_fusion.engine.run import RunSpec, fingerprint, pack_signal, result_path
from regime_fusion.windows import walk_forward


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

    def run(probability: float, uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sharpes, trades = np.empty(len(uniforms)), np.empty(len(uniforms))
        for k, row in enumerate(uniforms):
            out = _simulate(row < probability, *arrays, days, rate, cost)
            sharpes[k], trades[k] = fast_sharpe(out[0]), out[-1]
        return sharpes, trades

    if target_trades <= 0:
        return {"probability": 0.0, "mean_trades": 0.0, "mean_sharpe": 0.0, "sharpes": np.zeros(draws)}
    fixed = rng.random((40, len(prices)))  # the same draws at every step, so the search is not misled by noise
    low, high = 0.0, 1.0
    for _ in range(20):  # bisection on the daily entry probability
        middle = (low + high) / 2
        if run(middle, fixed)[1].mean() < target_trades:
            low = middle
        else:
            high = middle
    sharpes, trades = run((low + high) / 2, rng.random((draws, len(prices))))
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


def buy_and_hold_result(prices: pd.DataFrame, cost_bps: float) -> Result:
    """Buy-and-hold as a backtest result with one trade, for the common measures."""
    daily = buy_and_hold(prices, cost_bps)
    held = pd.Series(True, index=prices.index)
    held.iloc[-1] = False
    trades = pd.DataFrame({"entry": [prices.index[0]], "exit": [prices.index[-1]],
                           "entry_price": [prices["adj_open"].iloc[0]], "exit_price": [prices["adj_close"].iloc[-1]],
                           "reason": ["end"], "net_return": [float((1 + daily).prod() - 1)]})
    return Result(daily, held, trades)


def rule_signal(kind: str, prices: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """A rule's buy signal on every day of ``prices`` (computed on the whole history, causally)."""
    if kind == "always":
        values = always(prices)
    elif kind == "moving_average":
        values = moving_average_crossover(prices, int(params["fast"]), int(params["slow"]))
    elif kind == "momentum":
        values = time_series_momentum(prices, int(params["lookback"]))
    else:
        raise ValueError(f"no signal for rule kind {kind!r}")
    return pd.Series(values, index=prices.index)


def run_rule(spec: RunSpec, rule, settings, config: StudyConfig, *, campaign: str = "dev",
             write: bool = True) -> dict[str, Any]:
    """Choose a rule's parameters on the validation year and trade the evaluation period."""
    started = time.perf_counter()
    window = next(w for w in walk_forward(config) if w.test_year == spec.test_year)
    prices = stock_prices(spec.stock)
    validation = prices.loc[str(window.validation.start):str(window.validation.end)]
    evaluated_on = window.validation if spec.stage == 1 else window.test
    later = prices.loc[str(evaluated_on.start):str(evaluated_on.end)]
    primary = config.primary_cost_bps_per_side
    table, best = [], None
    if rule.kind == "buy_and_hold":
        best = {"params": {}, "sharpe": metrics.sharpe(buy_and_hold(validation, primary))}
    else:
        for params in rule.candidates():
            result = backtest(rule_signal(rule.kind, prices, params).loc[validation.index], validation,
                              spec.exit_days, spec.exit_rate, primary)
            trades = len(result.trades)
            score = metrics.sharpe(result.daily) if trades >= settings.min_trades else 0.0
            table.append({**params, "sharpe": score, "trades": trades})
            if best is None or score > best["sharpe"]:
                best = {"params": params, "sharpe": score}
    evaluation, signal = {}, None
    if rule.kind != "buy_and_hold":
        signal = rule_signal(rule.kind, prices, best["params"]).loc[later.index].to_numpy(dtype=bool)
    for cost in config.costs_bps_per_side:
        if rule.kind == "buy_and_hold":
            result = buy_and_hold_result(later, cost)
        else:
            result = backtest(signal, later, spec.exit_days, spec.exit_rate, cost)
        evaluation[f"{cost:g}bps"] = metrics.summary(result)
    record = {
        **spec.__dict__,
        "rung": rule.name,
        "campaign": campaign,
        "scheme": config.windows.scheme,
        "sources": [],
        "train": [str(window.validation.start), str(window.validation.end)],  # where parameters were chosen
        "evaluate": [str(evaluated_on.start), str(evaluated_on.end)],
        "model": f"{rule.name} {best['params']}",
        "fitness": best["sharpe"],
        "validation_sharpe": best["sharpe"],
        "candidates": table,
        "evaluation": evaluation,
        "signal": None if signal is None else pack_signal(signal),  # buy-and-hold has no signal
        "branches": {},
        "fingerprint": fingerprint(config, settings.method_digest(rule.name),
                                   price_entries(spec.stock, prices, prices.loc[:str(evaluated_on.end)].index), later),
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, benchmarks_digest=settings.digest(), rule=rule.__dict__)
        provenance.write(result_path(campaign, rule.name, spec), record)
    return record

