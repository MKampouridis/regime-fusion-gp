"""News timing: which session each article counts for, and daily news aggregates.

An article counts for the first session whose close is strictly later than the
time it became available (``regime_fusion.data.calendar``):

* GDELT: the time of the 15-minute update that first carried the record (the GKG
  ``DATE`` field, UTC, as YYYYMMDDHHMMSS). GDELT sees an article no earlier than
  its publication, so this time never looks ahead.
* FNSPID: the publication time in UTC. Items stamped exactly at midnight UTC carry
  only a date (their time is unknown), so they count for the first session after
  that date, in case they appeared after the close.

Sums for news are kept per hour (UTC) before they are assigned to sessions. The
NYSE closes on the hour (16:00, or 13:00 on early-close days), so every article
in an hour belongs to the same session as the hour's start, and hourly sums lose
nothing for the cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_fusion.data.calendar import TradingCalendar


def gdelt_time(values) -> pd.DatetimeIndex:
    """GDELT ``DATE`` values (YYYYMMDDHHMMSS, UTC) as UTC timestamps."""
    text = pd.Series(values).astype("int64").astype(str)
    return pd.DatetimeIndex(pd.to_datetime(text, format="%Y%m%d%H%M%S", utc=True))


def fnspid_time(values) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """FNSPID ``Date`` strings ("2020-05-22 07:23:25 UTC") as UTC timestamps, and a
    mask of the items that carry only a date (stamped exactly at midnight)."""
    text = pd.Series(values).astype(str).str.removesuffix(" UTC")
    stamps = pd.DatetimeIndex(pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", utc=True))
    return stamps, np.asarray(stamps == stamps.normalize())


def assign_sessions(calendar: TradingCalendar, stamps: pd.DatetimeIndex,
                    date_only: np.ndarray | None = None) -> pd.DatetimeIndex:
    """Session each item counts for; date-only items go to the first session after their date."""
    sessions = calendar.session_of_timestamp(stamps).to_numpy().copy()
    if date_only is not None and np.any(date_only):
        dates = stamps[date_only].tz_convert("UTC").tz_localize(None).normalize()
        sessions[date_only] = calendar.session_after_date(dates).to_numpy()
    return pd.DatetimeIndex(sessions)


def hourly_buckets(stamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Start of the UTC hour of each timestamp."""
    return stamps.tz_convert("UTC").floor("h")


def daily_sums(buckets: pd.DataFrame, calendar: TradingCalendar,
               sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Sum hourly news buckets per stock and session.

    ``buckets`` has a ``stock`` column, an ``hour`` column (UTC start of the hour)
    and numeric columns to add up, such as article counts and tone sums. The result
    has one row per (session, stock) in ``sessions`` x the stocks present, with
    zeros on sessions without news.
    """
    closes = calendar.closes
    if not ((closes.minute == 0) & (closes.second == 0)).all():
        raise ValueError("hourly buckets need closing times on the hour")
    hours = pd.DatetimeIndex(buckets["hour"])
    if hours.tz is None or not (hours == hours.floor("h")).all():
        raise ValueError("hour must hold timezone-aware starts of hours")
    values = buckets.drop(columns=["stock", "hour"])
    keyed = values.assign(stock=buckets["stock"].to_numpy(),
                          session=calendar.session_of_timestamp(hours))
    totals = keyed.dropna(subset=["session"]).groupby(["session", "stock"]).sum()
    stocks = sorted(buckets["stock"].unique())
    full = pd.MultiIndex.from_product([sessions, stocks], names=["session", "stock"])
    return totals.reindex(full, fill_value=0)
