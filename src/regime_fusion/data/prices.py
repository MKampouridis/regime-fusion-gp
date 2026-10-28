"""Daily prices from Yahoo Finance: download, alignment to the trading calendar, checks.

Raw files keep what ``yfinance`` returns with ``auto_adjust=False, actions=True``:

=====================  ==========================================================
Open, High, Low, Close prices adjusted for every split up to the download date
Adj Close              close adjusted for splits and dividends
Volume                 split-adjusted volume
Dividends              cash dividend per (split-adjusted) share, on its ex-date
Stock Splits           split ratio on its ex-date, 0 otherwise. Yahoo also records
                       some spin-offs this way (eBay 2.376 in 2015, HP 2.202 in 2015)
=====================  ==========================================================

Downloads run to the present day even though the study ends in 2025: Yahoo's
prices are adjusted for every later split, so the splits after 2025 are needed
to recover traded prices. ``align`` turns a raw file into one row per session of
the calendar, from the first to the last trading day:

* ``open, high, low, close`` as Yahoo gives them (split-adjusted);
* ``adj_open, adj_high, adj_low, adj_close`` also dividend-adjusted, i.e.
  total-return prices; returns are computed from ``adj_close``;
* ``volume``, ``dividend``, and ``split`` (1 when there is none);
* ``traded_close``: the price actually paid that day, the close times every
  later split ratio. Fundamental ratios such as price/earnings need it, because
  filed per-share values are not split-adjusted.

Yahoo scales the whole history by splits and dividends that happen later. Ratios
of prices within a look-back window are unaffected, so technical indicators must
be scale-invariant (returns, price ratios, relative DC thresholds) and must never
use price levels. Sessions without a bar are kept as missing rows, not filled.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regime_fusion.data.calendar import TradingCalendar

log = logging.getLogger(__name__)

RAW_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits")
BIG_MOVE = 0.15  # daily moves larger than this (in absolute value) are listed for checking
STALE_RUN = 5    # this many sessions in a row with the same close is suspicious


class MissingPrices(RuntimeError):
    """Yahoo has no prices for the symbol over the requested range."""


def download(
    symbol: str, start: dt.date, *, max_retries: int = 4, pause: float = 1.0
) -> pd.DataFrame:
    """Daily bars from ``start`` to the present day, as Yahoo returns them."""
    import yfinance as yf

    for attempt in range(1, max_retries + 1):
        try:
            frame = yf.Ticker(symbol).history(
                start=start.isoformat(), interval="1d", auto_adjust=False, actions=True
            )
        except Exception as problem:  # rate limits and network errors: wait and retry
            if attempt == max_retries:
                raise
            wait = pause * 2 ** attempt
            log.warning("%s: %s; retrying in %.0f s", symbol, problem, wait)
            time.sleep(wait)
            continue
        if frame.empty:
            raise MissingPrices(f"Yahoo has no prices for {symbol} since {start}")
        missing = [c for c in RAW_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{symbol}: Yahoo response lacks columns {missing}")
        return frame.loc[:, list(RAW_COLUMNS)]
    raise AssertionError("unreachable")


def align(raw: pd.DataFrame, calendar: TradingCalendar) -> pd.DataFrame:
    """One row per session from the first to the last bar; see the module notes."""
    frame = raw.copy()
    frame.index = _new_york_dates(frame.index)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    # Every split in the file counts, including those after the calendar ends.
    split = frame["Stock Splits"].replace(0.0, 1.0).fillna(1.0)
    later_splits = split[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)

    on_sessions = frame.index.isin(calendar.sessions)
    frame, split, later_splits = frame[on_sessions], split[on_sessions], later_splits[on_sessions]
    if frame.empty:
        raise ValueError("no bars fall on trading sessions")
    factor = frame["Adj Close"] / frame["Close"]
    out = pd.DataFrame(
        {
            "open": frame["Open"], "high": frame["High"],
            "low": frame["Low"], "close": frame["Close"],
            "adj_open": frame["Open"] * factor, "adj_high": frame["High"] * factor,
            "adj_low": frame["Low"] * factor, "adj_close": frame["Adj Close"],
            "volume": frame["Volume"].astype(float),
            "dividend": frame["Dividends"].fillna(0.0), "split": split,
            "traded_close": frame["Close"] * later_splits,
        }
    )
    sessions = calendar.between(out.index[0], out.index[-1])
    out = out.reindex(sessions)
    out.index.name = "session"
    return out


@dataclass
class PriceCheck:
    """Coverage and plausibility of one aligned price series."""

    first: pd.Timestamp
    last: pd.Timestamp
    sessions: int
    missing_sessions: int
    off_calendar_rows: int
    bad_rows: int            # non-positive prices or a close outside the day's range
    zero_volume: int
    longest_stale_run: int
    big_moves: list[tuple[str, float]] = field(default_factory=list)  # to check by hand
    flags: list[str] = field(default_factory=list)  # problems with coverage or quality


def check(raw: pd.DataFrame, aligned: pd.DataFrame, study_start: dt.date, study_end: dt.date,
          calendar: TradingCalendar) -> PriceCheck:
    """Coverage and plausibility checks.

    ``flags`` lists problems. Daily moves above 15% are listed separately: most are
    genuine (earnings, March 2020), but a spin-off that Yahoo does not adjust for
    also shows up there, so each is checked by hand once.
    """
    raw_days = _new_york_dates(raw.index)
    inside = (raw_days >= calendar.sessions[0]) & (raw_days <= calendar.sessions[-1])
    off_calendar = int((inside & ~raw_days.isin(calendar.sessions)).sum())

    present = aligned.dropna(subset=["close"])
    prices = present[["open", "high", "low", "close"]]
    bad = (
        prices.isna().any(axis=1)
        | (prices <= 0).any(axis=1)
        | (present["low"] > present[["open", "close"]].min(axis=1) * (1 + 1e-6))
        | (present["high"] < present[["open", "close"]].max(axis=1) * (1 - 1e-6))
    )
    moves = aligned["adj_close"].pct_change(fill_method=None)
    big = moves[moves.abs() > BIG_MOVE]
    result = PriceCheck(
        first=aligned.index[0],
        last=aligned.index[-1],
        sessions=len(aligned),
        missing_sessions=int(aligned["close"].isna().sum()),
        off_calendar_rows=off_calendar,
        bad_rows=int(bad.sum()),
        zero_volume=int((present["volume"] == 0).sum()),
        longest_stale_run=_longest_run(present["close"].to_numpy()),
        big_moves=[(day.date().isoformat(), round(float(r), 4)) for day, r in big.items()],
    )
    first_needed = calendar.between(study_start, study_end)[0]
    last_needed = calendar.between(study_start, study_end)[-1]
    if result.first > first_needed:
        result.flags.append(f"starts {result.first.date()}, after the study start {first_needed.date()}")
    if result.last < last_needed:
        result.flags.append(f"ends {result.last.date()}, before the study end {last_needed.date()}")
    for name in ("missing_sessions", "off_calendar_rows", "bad_rows", "zero_volume"):
        if getattr(result, name):
            result.flags.append(f"{name.replace('_', ' ')}: {getattr(result, name)}")
    if result.longest_stale_run >= STALE_RUN:
        result.flags.append(f"close unchanged for {result.longest_stale_run} sessions")
    return result


def _new_york_dates(index) -> pd.DatetimeIndex:
    """Bar dates as tz-naive New York dates (Yahoo stamps daily bars at midnight there)."""
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    return dates.normalize()


def _longest_run(values: np.ndarray) -> int:
    """Longest number of consecutive sessions with an unchanged value."""
    if values.size == 0:
        return 0
    longest = current = 1
    for previous, value in zip(values[:-1], values[1:]):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest
