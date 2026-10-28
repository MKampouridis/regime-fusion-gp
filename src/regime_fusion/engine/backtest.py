"""Backtest of a daily buy signal with ActTrade's exit rule (docs/DECISIONS.md).

* One long position at a time. A buy signal at the close of day t buys at the
  open of t+1; signals while a position is held are ignored.
* Take-profit: a limit order at the entry price times (1 + r), filled on the
  first day whose high reaches it, the entry day included, or at the open when
  a day opens above the target.
* Time exit: at the close of the d-th day of the position (the entry day is the
  first), as an order placed in advance, so no price is used before it is known.
* A position still open on the last day of the period is closed at its close.
* Costs are a fraction of the traded value per side, paid on entry and exit.

Prices are total-return (adjusted) open, high, low and close. Daily returns are
marked to market at each close, with costs on the entry and exit days, so the
product of a trade's daily returns equals its net return; days without a
position return zero. A session without a bar returns zero; a position cannot
be entered on it, and a time exit due on it happens at the next close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

TARGET, GAP, TIME, END = 1, 2, 3, 4
REASONS = {TARGET: "target", GAP: "gap", TIME: "time", END: "end"}


@dataclass(frozen=True)
class Result:
    daily: pd.Series      # daily returns, net of costs
    held: pd.Series       # whether a position is held at each close
    trades: pd.DataFrame  # entry, exit, prices, reason and net return of each trade


def backtest(signal, prices: pd.DataFrame, days: int, rate: float, cost_bps: float) -> Result:
    """Run the backtest on aligned prices (``adj_open``, ``adj_high``, ``adj_low``, ``adj_close``)."""
    if days < 1 or rate <= 0 or cost_bps < 0:
        raise ValueError("need days >= 1, rate > 0 and cost_bps >= 0")
    signal = np.asarray(signal, dtype=np.bool_)
    if len(signal) != len(prices):
        raise ValueError("one signal per session is needed")
    arrays = [prices[c].to_numpy(dtype=np.float64) for c in ("adj_open", "adj_high", "adj_low", "adj_close")]
    daily, held, entry, exit_, p_in, p_out, reason, n = _simulate(
        signal, *arrays, int(days), float(rate), float(cost_bps) / 1e4
    )
    index = prices.index
    trades = pd.DataFrame({
        "entry": index[entry[:n]], "exit": index[exit_[:n]],
        "entry_price": p_in[:n], "exit_price": p_out[:n],
        "reason": [REASONS[int(k)] for k in reason[:n]],
    })
    cost = cost_bps / 1e4
    trades["net_return"] = trades["exit_price"] * (1 - cost) / (trades["entry_price"] * (1 + cost)) - 1
    return Result(pd.Series(daily, index=index), pd.Series(held, index=index), trades)


@njit(cache=True)
def _simulate(signal, open_, high, low, close, days, rate, cost):
    n = len(signal)
    daily = np.zeros(n)
    held = np.zeros(n, dtype=np.bool_)
    entry = np.zeros(n, dtype=np.int64)
    exit_ = np.zeros(n, dtype=np.int64)
    p_in = np.zeros(n)
    p_out = np.zeros(n)
    reason = np.zeros(n, dtype=np.int64)
    trades = 0
    holding = False
    start = 0
    target = 0.0
    last = 0.0  # value of the position at the last close, per share and net of the entry cost
    for day in range(n):
        if not holding:
            if day == 0 or not signal[day - 1] or np.isnan(open_[day]):
                continue
            holding, start = True, day
            p_in[trades] = open_[day]
            entry[trades] = day
            target = open_[day] * (1.0 + rate)
            last = open_[day] * (1.0 + cost)
        # From here a position is held during this day.
        if np.isnan(close[day]):
            held[day] = True  # no bar: nothing happens, the return is zero
            continue
        sell, kind = 0.0, 0
        if day > start and open_[day] >= target:
            sell, kind = open_[day], GAP
        elif high[day] >= target:
            sell, kind = target, TARGET
        elif day - start + 1 >= days:
            sell, kind = close[day], TIME
        elif day == n - 1:
            sell, kind = close[day], END
        if kind:
            daily[day] = sell * (1.0 - cost) / last - 1.0
            exit_[trades] = day
            p_out[trades] = sell
            reason[trades] = kind
            trades += 1
            holding = False
        else:
            daily[day] = close[day] / last - 1.0
            last = close[day]
            held[day] = True
    if holding:  # the last session had no bar: close at the last known price
        exit_[trades] = n - 1
        p_out[trades] = last
        reason[trades] = END
        daily[n - 1] = (1.0 + daily[n - 1]) * (1.0 - cost) - 1.0
        held[n - 1] = False
        trades += 1
    return daily, held, entry, exit_, p_in, p_out, reason, trades
