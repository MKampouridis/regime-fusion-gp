"""Stationary inputs for the development experiments (E5).

The campaigns scale each indicator to the minimum and maximum of its training window, as ActTrade
did. Many indicators are levels that trend (moving averages of the price, revenue, EPS), so a later
year's values fall outside the training range, and a rule evolved on a threshold stops firing, or
fires every day, once training ends (in 2018, 72-91% of AAPL's values of such indicators lay outside
its 2015-2017 range).

Here every indicator enters instead as its trailing percentile rank among the last ``WINDOW``
sessions (the current one included), mapped to [-1, 1]. Tied values share the average of their ranks:
fundamentals change only when a report is filed, so most of a window's values are tied, and giving
ties the highest rank (as a first version did, until 4 November 2026) put a firm's R&D at the top
rank on every day and its revenue on two days in three. "High compared with the last year" means the same in every year, and the rank of day t uses
only data known by the close of t. A rank needs at least ``MIN_PERIODS`` known values in its window;
the processed indicators start in January 2014, a year before the first training day.
"""

from __future__ import annotations

import pandas as pd

from regime_fusion.data import store
from regime_fusion.engine.panel import MIN_COVERAGE, PROCESSED, Panel, stock_prices
from regime_fusion.windows import DateRange

WINDOW = 252
MIN_PERIODS = 126
DROPPED = ("market_value", "diluted_shares")   # left out by the campaigns too (engine/panel.py)


def trailing_rank(raw: pd.DataFrame) -> pd.DataFrame:
    """Each column's trailing percentile rank over ``WINDOW`` rows, mapped to [-1, 1]."""
    ranks = raw.rolling(WINDOW, min_periods=MIN_PERIODS).rank(pct=True, method="average")
    return (2.0 * ranks - 1.0).where(raw.notna())


def load_stationary_panel(stock: str, train: DateRange, evaluate: DateRange, sources: tuple[str, ...],
                          root=None) -> Panel:
    """A ``Panel`` (engine/panel.py) whose indicators are trailing ranks, not scaled levels."""
    prices = stock_prices(stock, root)
    sessions = prices.index[(prices.index >= pd.Timestamp(train.start)) & (prices.index <= pd.Timestamp(evaluate.end))]
    train_days = sessions[sessions <= pd.Timestamp(train.end)]
    evaluate_days = sessions[sessions >= pd.Timestamp(evaluate.start)]
    frames = []
    for source in sources:
        table = pd.read_parquet(store.layer("processed", PROCESSED[source], f"{stock}.parquet", root=root))
        table = table.loc[table.index <= pd.Timestamp(evaluate.end), [c for c in table.columns if c not in DROPPED]]
        frames.append(trailing_rank(table).add_prefix(f"{source.lower()}:"))
    ranked = pd.concat(frames, axis=1).reindex(sessions)
    fit = ranked.loc[train_days]
    keep = [c for c in ranked.columns if fit[c].notna().mean() >= MIN_COVERAGE and fit[c].nunique() > 1]
    return Panel(ranked[keep], prices.loc[sessions], train_days, evaluate_days, {})
