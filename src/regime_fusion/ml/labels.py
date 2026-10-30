"""Labels for the machine-learning benchmarks: the outcome of the trade a buy signal would open.

The plan's fairness rules ask for labels that match the exit rule. For a signal
at the close of day t, the trade is the one ``regime_fusion.engine.backtest``
would make: buy at the open of t+1, sell at the take-profit target or at the
close of the d-th day, net of costs on both sides. The label is whether that
trade gains.

A trade that would end after the last day given has no outcome, so labels are
always computed on the training rows alone: the last trades of a training
window, which would end in the validation or test period, are left out
(purged) rather than peeking at it. A trade is labelled only if its time exit
falls within the rows given, even when it reaches its target earlier: keeping
the early winners among the last trades would leave only gains there, since
their losers end at the time exit, after the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from regime_fusion.engine.backtest import GAP, TARGET, TIME


def trade_returns(prices: pd.DataFrame, days: int, rate: float, cost_bps: float) -> pd.Series:
    """Net return of the trade a buy signal at each day's close opens (missing when it cannot finish)."""
    arrays = [prices[c].to_numpy(dtype=np.float64) for c in ("adj_open", "adj_high", "adj_low", "adj_close")]
    out = _trade_returns(*arrays, int(days), float(rate), float(cost_bps) / 1e4)
    return pd.Series(out, index=prices.index, name="trade_return")


@njit(cache=True)
def _trade_returns(open_, high, low, close, days, rate, cost):
    n = len(open_)
    out = np.full(n, np.nan)
    for t in range(n - 1):
        start = t + 1
        if np.isnan(open_[start]) or start + days - 1 > n - 1:
            continue  # no bar to buy at, or a time exit after the last row: no outcome either way
        target = open_[start] * (1.0 + rate)
        for day in range(start, n):
            if np.isnan(close[day]):
                continue  # no bar: nothing happens (as in the backtest)
            kind, sell = 0, 0.0
            if day > start and open_[day] >= target:
                kind, sell = GAP, open_[day]
            elif high[day] >= target:
                kind, sell = TARGET, target
            elif day - start + 1 >= days:
                kind, sell = TIME, close[day]
            if kind:
                out[t] = sell * (1.0 - cost) / (open_[start] * (1.0 + cost)) - 1.0
                break
    return out
