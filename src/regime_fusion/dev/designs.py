"""Trade designs for the development experiments (E3), and markets that score signals with them.

A design turns a daily Boolean signal (known at the close of day t) into daily returns, trading at
the open of t+1, so nothing is used before it is known:

* ``acttrade``: ActTrade's exit (``engine/backtest.py``): a buy signal opens a position, closed at a
  take-profit of r or at the close of the d-th day. Fixed d and r here (no exit selection, which
  would need a validation year of its own).
* ``hold``: long while the signal is on. A signal at the close of t opens a position at the open of
  t+1; the position is closed at the open after the first close at which the signal is off, or at
  which the close is ``stop`` below the highest close since entry (a trailing stop). After a stop,
  the signal must turn off before a new entry.
* ``long``: long by default; the signal says when to step aside (``hold`` on the negated signal,
  without a stop). A signal that never fires is buy-and-hold.

Costs are a fraction of the traded value per side. Daily returns are marked to market at each close
with the costs on the entry and exit days, as in ``engine/backtest.py``; a position still open on
the last day is closed at its close.

``StockMarket`` and ``PooledMarket`` give ``daily(signal) -> (daily returns, trades)``, which is all
``engine.fitness.Evaluator`` needs, and ``simulate(signal)``, which adds each trade's net return
(for the PSR lower bound of ``dev.fitness``, whose evidence is the number of trades). A pooled market holds several stocks' rows one after another
(the trees act day by day, so one signal over the stacked rows is each stock's signal); it scores
each stock separately and returns the equal-weight portfolio of the stocks present on each day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from regime_fusion.engine.backtest import _simulate

DESIGNS = ("acttrade", "hold", "long")
ACTTRADE_EXIT = (10, 0.03)   # (d, r): the middle of the main campaign's exit grid
TRAILING_STOP = 0.10


@njit(cache=True)
def simulate_hold(signal, open_, close, stop, cost):
    """Daily returns, number of trades and each trade's net return, holding while ``signal`` is on."""
    n = len(signal)
    daily = np.zeros(n)
    returns = np.zeros(n)
    trades = 0
    entered = 0.0     # entry price with its cost
    holding = False
    armed = True      # False after a stop, until the signal turns off
    last = 0.0        # value at the last close, per share, net of the entry cost
    peak = 0.0
    for day in range(1, n):
        if not signal[day - 1]:
            armed = True
        if holding:
            stopped = close[day - 1] <= peak * (1.0 - stop)
            if not signal[day - 1] or stopped:
                daily[day] = open_[day] * (1.0 - cost) / last - 1.0
                returns[trades] = open_[day] * (1.0 - cost) / entered - 1.0
                holding = False
                trades += 1
                if stopped:
                    armed = False
                continue
            daily[day] = close[day] / last - 1.0
            last = close[day]
            peak = max(peak, close[day])
            if day == n - 1:
                daily[day] = (1.0 + daily[day]) * (1.0 - cost) - 1.0
                returns[trades] = close[day] * (1.0 - cost) / entered - 1.0
                holding = False
                trades += 1
        elif signal[day - 1] and armed:
            holding = True
            entered = open_[day] * (1.0 + cost)
            last = entered
            daily[day] = close[day] / last - 1.0
            last = close[day]
            peak = close[day]
            if day == n - 1:
                daily[day] = (1.0 + daily[day]) * (1.0 - cost) - 1.0
                returns[trades] = close[day] * (1.0 - cost) / entered - 1.0
                holding = False
                trades += 1
    return daily, trades, returns[:trades]


@dataclass(frozen=True)
class Design:
    name: str
    cost: float   # fraction per side

    def __post_init__(self) -> None:
        if self.name not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}")

    def simulate(self, signal: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """Daily returns, number of trades and each trade's net return."""
        signal = np.asarray(signal, dtype=np.bool_)
        if self.name == "acttrade":
            days, rate = ACTTRADE_EXIT
            daily, _, _, _, p_in, p_out, _, n = _simulate(signal, open_, high, low, close, days, rate, self.cost)
            n = int(n)
            return daily, n, p_out[:n] * (1.0 - self.cost) / (p_in[:n] * (1.0 + self.cost)) - 1.0
        if self.name == "hold":
            daily, n, returns = simulate_hold(signal, open_, close, TRAILING_STOP, self.cost)
        else:
            daily, n, returns = simulate_hold(~signal, open_, close, np.inf, self.cost)
        return daily, int(n), returns

    def daily(self, signal: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
              close: np.ndarray) -> tuple[np.ndarray, int]:
        daily, n, _ = self.simulate(signal, open_, high, low, close)
        return daily, n


@dataclass(frozen=True)
class StockMarket:
    """One stock's prices over a period, scored with a design."""
    design: Design
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def simulate(self, signal: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        return self.design.simulate(signal, self.open, self.high, self.low, self.close)

    def daily(self, signal: np.ndarray) -> tuple[np.ndarray, int]:
        daily, n, _ = self.simulate(signal)
        return daily, n


class PooledMarket:
    """Several stocks' rows stacked; ``daily`` gives the equal-weight portfolio and all trades."""

    def __init__(self, markets: list[StockMarket], positions: list[np.ndarray], days: int):
        if len(markets) != len(positions):
            raise ValueError("one array of day positions per stock")
        self.markets, self.positions, self.days = markets, positions, days
        self.bounds = np.cumsum([0, *(len(m.close) for m in markets)])
        self.count = np.zeros(days)
        for p in positions:
            np.add.at(self.count, p, 1.0)

    def simulate(self, signal: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """The equal-weight portfolio's daily returns, all trades, and every trade's net return."""
        total, trades, returns = np.zeros(self.days), 0, []
        for m, a, b, p in zip(self.markets, self.bounds[:-1], self.bounds[1:], self.positions):
            daily, n, r = m.simulate(signal[a:b])
            np.add.at(total, p, daily)
            trades += n
            returns.append(r)
        portfolio = np.divide(total, self.count, out=np.zeros(self.days), where=self.count > 0)
        return portfolio, trades, np.concatenate(returns) if returns else np.zeros(0)

    def daily(self, signal: np.ndarray) -> tuple[np.ndarray, int]:
        daily, n, _ = self.simulate(signal)
        return daily, n
