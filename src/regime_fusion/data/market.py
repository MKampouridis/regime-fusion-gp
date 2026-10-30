"""Market-wide series: the S&P 500 index, the SPY fund and the VIX.

* S&P 500 index (Yahoo ``^GSPC``): a price index, for market regimes.
* SPY (Yahoo): the S&P 500 fund; its adjusted close is a total-return benchmark.
* VIX: from Cboe's own history file (open, high, low and close of each day).
  Since mid-2022 the file also has values on US exchange holidays, from Cboe's
  extended trading hours; alignment to NYSE sessions drops them.

Index and fund bars go through ``regime_fusion.data.prices`` like any stock.

The VIX is calculated until 16:15 New York time, a quarter of an hour after the
stock market closes, so the VIX close of day t is not strictly available by the
close of t. The interim table keeps each value on its own day; a feature built
from it must use the close of t-1 (decided on 28 October 2026, ``docs/DECISIONS.md``).
No feature uses the VIX yet.
"""

from __future__ import annotations

import io

import pandas as pd

from regime_fusion.data.calendar import TradingCalendar

MARKET_SYMBOLS = {"GSPC": "^GSPC", "SPY": "SPY"}
VIX_COLUMNS = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]


def fetch_vix(url: str, user_agent: str) -> bytes:
    """Cboe's VIX history file, as published."""
    import requests

    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    return response.content


def parse_vix(content: bytes) -> pd.DataFrame:
    """Daily VIX values indexed by date (tz-naive)."""
    frame = pd.read_csv(io.BytesIO(content))
    if list(frame.columns) != VIX_COLUMNS:
        raise ValueError(f"unexpected VIX columns {list(frame.columns)}; expected {VIX_COLUMNS}")
    dates = pd.to_datetime(frame.pop("DATE"), format="%m/%d/%Y")
    frame.index = pd.DatetimeIndex(dates, name="date")
    frame.columns = [column.lower() for column in frame.columns]
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("VIX dates must be unique and increasing")
    return frame.astype(float)


def align_vix(vix: pd.DataFrame, calendar: TradingCalendar) -> tuple[pd.DataFrame, dict[str, int]]:
    """VIX on every session of the calendar, and counts of gaps and off-calendar rows."""
    inside = vix.loc[calendar.sessions[0]:calendar.sessions[-1]]
    off_calendar = int((~inside.index.isin(calendar.sessions)).sum())
    aligned = inside.reindex(calendar.sessions)
    aligned.index.name = "session"
    counts = {
        "off_calendar_rows": off_calendar,
        "missing_sessions": int(aligned["close"].isna().sum()),
        "non_positive": int((aligned[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
    }
    return aligned, counts
