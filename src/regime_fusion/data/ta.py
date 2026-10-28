"""ActTrade's twelve technical indicators, daily.

From the total-return prices of ``regime_fusion.data.prices``, for look-back
windows of 5 and 10 sessions (ActTrade paper, equations 8-13):

==================  ==========================================================
moving_average_n    mean close over the last n sessions
momentum_n          close minus the close n sessions earlier
roc_n               100 x (close / close n sessions earlier - 1)
volatility_n        sample standard deviation of p(t-i) / p(t-n) - 1, i = 0..n-1
williams_r_n        -100 x (highest high - close) / (highest high - lowest low)
midprice_n          (highest high + lowest low) / 2 over the last n sessions
==================  ==========================================================

Moving average, momentum and midprice are price levels. Yahoo's adjusted prices
are scaled by every dividend and split up to the download date, so a level on
day t would depend on later events. Levels are therefore expressed in the prices
of day t: the window's adjusted prices times traded close / adjusted close of
day t, which adjusts past prices for the events up to t only. Ratios are
unaffected. Highs and lows are adjusted by the same factor as closes.

Where the paper is ambiguous: the moving average averages n sessions
(equation 8 sums n + 1 prices but divides by n), and the midprice is the
midpoint (high + low) / 2, as the text says, rather than (high - low) / 2 as
equation 13 prints. Values are raw; ActTrade's scaling to [-1, 1] is fitted on
training windows later. A window that includes a session without a bar gives a
missing value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10)
NAMES = ("moving_average", "momentum", "roc", "volatility", "williams_r", "midprice")


def columns(windows=WINDOWS) -> list[str]:
    return [f"{name}_{n}" for n in windows for name in NAMES]


def indicators(prices: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """The indicators for each session of an aligned price table."""
    close, high, low = prices["adj_close"], prices["adj_high"], prices["adj_low"]
    to_today = prices["traded_close"] / close  # past adjusted prices in day-t prices
    out = {}
    for n in windows:
        past = close.shift(n)
        highest, lowest = high.rolling(n).max(), low.rolling(n).min()
        ratios = pd.concat([close.shift(i) / past - 1.0 for i in range(n)], axis=1)
        out[f"moving_average_{n}"] = close.rolling(n).mean() * to_today
        out[f"momentum_{n}"] = (close - past) * to_today
        out[f"roc_{n}"] = (close / past - 1.0) * 100.0
        out[f"volatility_{n}"] = ratios.std(axis=1, ddof=1, skipna=False)
        out[f"williams_r_{n}"] = -100.0 * (highest - close) / (highest - lowest)
        out[f"midprice_{n}"] = (highest + lowest) / 2.0 * to_today
    frame = pd.DataFrame(out, index=prices.index).loc[:, columns(windows)]
    return frame.replace([np.inf, -np.inf], np.nan)
