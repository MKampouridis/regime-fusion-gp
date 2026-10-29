"""Fitness of individuals, from the backtester.

Each individual gives four signals: the whole tree ("C") and each branch on its
own (FA, SA, TA). Each signal is backtested with the exit rule of the stock-year
(regime_fusion.engine.backtest) and yields daily returns, a number of trades and
a Sharpe ratio (annualised, from daily returns, as the study's primary metric).

* ActTrade's fitness is the weighted sum of the four Sharpe ratios. As in
  ActTrade, a signal with fewer than ``min_trades`` trades scores zero. ActTrade's
  code computes Sharpe ratios from per-trade returns; here they come from daily
  returns, like every metric of the study.
* Lexicase-type selection uses *cases*: the annualised mean daily return of each
  signal in each of ``blocks`` consecutive time blocks of the training period.

Results are cached by tree, so identical branches are backtested once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numba import njit

from regime_fusion.engine.backtest import _simulate
from regime_fusion.engine.tree import SOURCES, Individual, boolean, fused

COMPONENTS = ("C", *SOURCES)
DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Market:
    """Prices of one stock over one period, as arrays, and the exit rule."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    days: int
    rate: float
    cost: float  # fraction per side

    def daily(self, signal: np.ndarray) -> tuple[np.ndarray, int]:
        out = _simulate(signal, self.open, self.high, self.low, self.close, self.days, self.rate, self.cost)
        return out[0], int(out[-1])


@dataclass(frozen=True)
class Settings:
    weights: Mapping[str, float]   # C, FA, SA, TA
    min_trades: int = 3
    blocks: int = 8


@njit(cache=True)
def sharpe(daily: np.ndarray) -> float:
    n = daily.size
    if n < 2:
        return 0.0
    mean = daily.mean()
    spread = np.sqrt(((daily - mean) ** 2).sum() / (n - 1))
    return 0.0 if spread == 0.0 else mean / spread * np.sqrt(252.0)


def block_means(daily: np.ndarray, blocks: int) -> np.ndarray:
    """Annualised mean daily return in each of ``blocks`` consecutive blocks."""
    return np.array([part.mean() * DAYS_PER_YEAR for part in np.array_split(daily, blocks)])


class Evaluator:
    """Computes and caches the statistics of trees on one training period.

    The cache keeps, per tree, the signal, the number of trades, the Sharpe ratio
    and the block means (not the daily returns), and ``retain`` prunes it to the
    trees of the current population, so memory stays bounded.
    """

    def __init__(self, data: Mapping[str, np.ndarray], market: Market, settings: Settings):
        self.data, self.market, self.settings = data, market, settings
        self._cache: dict[tuple, tuple] = {}

    def component(self, key: tuple, signal_of) -> tuple:
        """(signal, trades, Sharpe, block means) of a tree, computed once."""
        found = self._cache.get(key)
        if found is None:
            signal = signal_of()
            daily, trades = self.market.daily(signal)
            found = (signal, trades, sharpe(daily), block_means(daily, self.settings.blocks))
            self._cache[key] = found
        return found

    def evaluate(self, individual: Individual) -> None:
        stats, branch_signals = {}, {}
        for s, tree in individual.branches.items():
            stats[s] = self.component(_branch_key(s, tree), lambda t=tree: boolean(t, self.data)[0])
            branch_signals[s] = stats[s][0]
        stats["C"] = self.component(
            ("C", *individual.key()), lambda: fused(individual.fusion, branch_signals, self.data)[0]
        )
        weights, floor = self.settings.weights, self.settings.min_trades
        components = ("C", *individual.branches)
        individual.fitness = float(sum(
            weights.get(c, 0.0) * (stats[c][2] if stats[c][1] >= floor else 0.0) for c in components
        ))
        individual.stats = {c: {"sharpe": float(stats[c][2]), "trades": int(stats[c][1])} for c in components}
        individual.stats["cases"] = np.concatenate([stats[c][3] for c in components])

    def retain(self, population) -> None:
        keep = set()
        for individual in population:
            keep.add(("C", *individual.key()))
            keep.update(_branch_key(s, tree) for s, tree in individual.branches.items())
        self._cache = {key: value for key, value in self._cache.items() if key in keep}


def _branch_key(source: str, tree) -> tuple:
    return ("B", source, tuple(tree))
