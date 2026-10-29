"""Daily returns of a campaign's methods, and the tests the plan runs on them.

Every run stores its buy signal over the evaluation period (one bit per session),
so its daily returns can be rebuilt exactly with the backtest, at any cost.

* ``run_returns``: one run's daily returns (buy-and-hold has no signal and is
  recomputed directly).
* ``method_returns``: one daily series per method. Within a stock-year the
  daily returns of its runs are averaged (equal capital in every run, so no run
  is picked after the fact); across stocks they are averaged day by day (an
  equal-weight portfolio of the stocks); test years follow one another.
* ``tests``: Hansen's SPA test of the methods against a benchmark, the Deflated
  Sharpe Ratio of each method given every configuration tried, and the
  probability of backtest overfitting across the methods
  (``regime_fusion.analysis.stats``).
* ``by_regime``: each portfolio's results on the days of each regime state.
* ``random_entry_check``: for each stock-year, where a method's median Sharpe
  ratio falls among random entries with the same exit rule and number of trades.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from regime_fusion.analysis import stats
from regime_fusion.data import store
from regime_fusion.engine.backtest import backtest
from regime_fusion.engine import benchmarks
from regime_fusion.engine.benchmarks import PRICE_COLUMNS, buy_and_hold
from regime_fusion.engine.run import unpack_signal
from regime_fusion.ml.models import load_settings as load_benchmarks


def stock_prices(stock: str) -> pd.DataFrame:
    """A stock's prices on the sessions the engine uses (those with a close)."""
    prices = pd.read_parquet(store.layer("interim", "prices", f"{stock}.parquet"))
    return prices.loc[prices["adj_close"].notna(), PRICE_COLUMNS]


def run_returns(row: pd.Series, prices: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Daily returns of one run over its evaluation period."""
    days = prices.loc[str(row["evaluate_start"]):str(row["evaluate_end"])]
    if not isinstance(row.get("signal"), str):  # None, or missing after a round trip through a table
        if row["rung"] not in buy_and_hold_methods():
            raise ValueError(f"{row['rung']} {row['stock']} {row['test_year']} run {row['run']}: no stored signal")
        return buy_and_hold(days, cost_bps)
    signal = unpack_signal(row["signal"])
    if len(signal) != len(days):
        raise ValueError(f"{row['rung']} {row['stock']} {row['test_year']} run {row['run']}: "
                         f"{len(signal)} signal days for {len(days)} sessions")
    return backtest(signal, days, int(row["exit_days"]), float(row["exit_rate"]), cost_bps).daily


def buy_and_hold_methods() -> set[str]:
    rules = load_benchmarks().rules
    return {name for name, rule in rules.items() if rule.kind == "buy_and_hold"}


def method_returns(runs: pd.DataFrame, cost_bps: float,
                   prices_of: Callable[[str], pd.DataFrame] = stock_prices) -> pd.DataFrame:
    """Daily returns of each method's equal-weight portfolio; columns are methods."""
    cache: dict[str, pd.DataFrame] = {}
    columns = {}
    for method, group in runs.groupby("rung"):
        per_stock_year = []
        for (stock, _), cell in group.groupby(["stock", "test_year"]):
            if stock not in cache:
                cache[stock] = prices_of(stock)
            daily = pd.concat([run_returns(row, cache[stock], cost_bps) for _, row in cell.iterrows()], axis=1)
            per_stock_year.append(daily.mean(axis=1).rename(stock))
        by_stock = pd.concat(per_stock_year, axis=1).T.groupby(level=0).sum(min_count=1).T
        columns[method] = by_stock.mean(axis=1)
    return pd.DataFrame(columns).sort_index()


def tests(returns: pd.DataFrame, benchmark: str, candidates: list[str], n_trials: int,
          rng: np.random.Generator, bootstraps: int = 1000, blocks: int = 16) -> dict[str, object]:
    """SPA of ``candidates`` against ``benchmark``; Deflated Sharpe of every method; PBO of all methods."""
    complete = returns.dropna()
    differences = complete[candidates].sub(complete[benchmark], axis=0)
    sharpes = complete.mean() / complete.std(ddof=1)
    trial_variance = float(sharpes.var(ddof=1)) if len(sharpes) > 1 else 0.0
    out: dict[str, object] = {
        "days": len(complete),
        "annualised_sharpe": (sharpes * np.sqrt(252)).round(4).to_dict(),
        "spa": {"benchmark": benchmark, "candidates": candidates,
                **stats.spa(differences.to_numpy(), rng, bootstraps=bootstraps)},
        "deflated_sharpe": {method: stats.deflated_sharpe(complete[method].to_numpy(), n_trials, trial_variance)
                            for method in complete.columns},
        "n_trials": n_trials,
    }
    if complete.shape[1] >= 2 and len(complete) >= 2 * blocks:
        out["pbo"] = stats.pbo(complete.to_numpy(), blocks=blocks)
    return out


def regime_states(name: str, years: list[int], root=None, scheme: str = "rolling") -> pd.Series:
    """Most probable regime state of each test-year day (stage-2 probabilities of shared regimes,
    fitted on the campaign's walk-forward scheme). Regimes per stock have no common state."""
    from regime_fusion.regimes import market

    if market.per_stock(name, market.load_settings()):
        raise FileNotFoundError(f"{name} regimes are per stock; a portfolio has no common regime state")
    parts = []
    for year in years:
        path = market.probabilities_path(name, year, 2, root=root, scheme=scheme)
        probabilities = pd.read_parquet(path).loc[str(year)]
        parts.append(probabilities.idxmax(axis=1).str.removeprefix("p_state").astype(int))
    return pd.concat(parts).rename("state")


def by_regime(returns: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    """Rows: (method, state): days, annualised Sharpe ratio and annualised mean return."""
    rows = []
    aligned = states.reindex(returns.index)
    for method in returns.columns:
        for state, days in returns[method].groupby(aligned):
            spread = days.std(ddof=1)
            rows.append({"method": method, "state": int(state), "days": len(days),
                         "sharpe": float(days.mean() / spread * np.sqrt(252)) if spread > 0 else 0.0,
                         "annual_return": float(days.mean() * 252)})
    return pd.DataFrame(rows).set_index(["method", "state"])


def random_entry_check(runs: pd.DataFrame, method: str, cost_bps: float, rng: np.random.Generator,
                       draws: int = 1000, prices_of: Callable[[str], pd.DataFrame] = stock_prices) -> pd.DataFrame:
    """Per stock-year: the method's median Sharpe ratio and its percentile among random entries."""
    rows = []
    cost = f"{cost_bps:g}bps"
    for (stock, year), cell in runs[runs["rung"] == method].groupby(["stock", "test_year"]):
        first = cell.iloc[0]
        days = prices_of(stock).loc[str(first["evaluate_start"]):str(first["evaluate_end"])]
        trades = int(round(cell[f"trades_{cost}"].median()))
        sharpe = float(cell[f"sharpe_{cost}"].median())
        draw = benchmarks.random_entry(days, int(first["exit_days"]), float(first["exit_rate"]), cost_bps,
                                       trades, rng, draws)
        rows.append({"stock": stock, "test_year": year, "trades": trades, "sharpe": sharpe,
                     "random_mean_sharpe": draw["mean_sharpe"],
                     "percentile": benchmarks.percentile_of(sharpe, draw["sharpes"])})
    return pd.DataFrame(rows)

