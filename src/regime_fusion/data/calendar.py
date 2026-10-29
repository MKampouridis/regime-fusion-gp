"""NYSE trading calendar, and the rules that decide on which day information may be used.

The study runs on New York Stock Exchange sessions. A value used on day t must
be available by the close of t; trades execute on t+1. Two rules turn the time
at which an item became public into the first session on which it may be used:

* An item with a timestamp (a news article) belongs to the first session whose
  close is strictly later than the timestamp. An article at 15:59 New York time
  counts for that day; one at 16:00 or later counts for the next session. On
  early-close days the cut-off is the early close (13:00).
* An item with a date but no time of day (an SEC filing date, a news item stamped
  only with its date) belongs to the first session strictly after that date,
  because it may have appeared after the close.

Sessions and closing times, including holidays, unscheduled closures (such as
5 December 2018 and 9 January 2025) and early closes, come from the
``exchange_calendars`` package (calendar "XNYS").
"""

from __future__ import annotations

from dataclasses import dataclass

import exchange_calendars as xc
import numpy as np
import pandas as pd

from regime_fusion.data.settings import DataSettings, load_settings

EXCHANGE = "XNYS"


@dataclass(frozen=True)
class TradingCalendar:
    """Trading sessions (tz-naive dates) and their closing times (UTC)."""

    sessions: pd.DatetimeIndex
    closes: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if len(self.sessions) == 0 or len(self.sessions) != len(self.closes):
            raise ValueError("need at least one session and one closing time per session")
        if self.sessions.tz is not None or (self.sessions != self.sessions.normalize()).any():
            raise ValueError("sessions must be tz-naive dates")
        if self.closes.tz is None or str(self.closes.tz) != "UTC":
            raise ValueError("closing times must be in UTC")
        if not (self.sessions.is_monotonic_increasing and self.sessions.is_unique):
            raise ValueError("sessions must be strictly increasing")
        if not self.closes.is_monotonic_increasing:
            raise ValueError("closing times must be increasing")
        # Positions are found with integer searches, so both indexes use one unit.
        object.__setattr__(self, "sessions", self.sessions.as_unit("ns"))
        object.__setattr__(self, "closes", self.closes.as_unit("ns"))

    @classmethod
    def nyse(cls, start, end) -> TradingCalendar:
        """NYSE sessions from ``start`` to ``end``, both inclusive."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        closes = xc.get_calendar(EXCHANGE, start=start, end=end).closes.loc[start:end]
        return cls(pd.DatetimeIndex(closes.index), pd.DatetimeIndex(closes))

    def between(self, start, end) -> pd.DatetimeIndex:
        """Sessions from ``start`` to ``end``, both inclusive."""
        inside = (self.sessions >= pd.Timestamp(start)) & (self.sessions <= pd.Timestamp(end))
        return self.sessions[inside]

    def session_of_timestamp(self, timestamps) -> pd.DatetimeIndex:
        """First session whose close is strictly later than each timezone-aware timestamp.

        Timestamps after the last close in the calendar map to NaT, as do missing ones.
        """
        stamps = _utc_index(timestamps)
        positions = np.searchsorted(self.closes.asi8, stamps.asi8, side="right")
        return self._sessions_at(positions, stamps.isna())

    def session_after_date(self, dates) -> pd.DatetimeIndex:
        """First session strictly after each calendar date (NaT beyond the calendar)."""
        days = _date_index(dates)
        positions = np.searchsorted(self.sessions.asi8, days.asi8, side="right")
        return self._sessions_at(positions, days.isna())

    def session_on_or_after_date(self, dates) -> pd.DatetimeIndex:
        """The session on each calendar date, or the next one if the date has none (NaT beyond the calendar).

        Not for trading: an item with only a date may come after that day's close
        (use ``session_after_date``). It suits comparing sources by the day they report.
        """
        days = _date_index(dates)
        positions = np.searchsorted(self.sessions.asi8, days.asi8, side="left")
        return self._sessions_at(positions, days.isna())

    def _sessions_at(self, positions: np.ndarray, missing: np.ndarray) -> pd.DatetimeIndex:
        found = (positions < len(self.sessions)) & ~np.asarray(missing)
        out = np.full(len(positions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        out[found] = self.sessions.to_numpy()[positions[found]]
        return pd.DatetimeIndex(out)


def study_calendar(settings: DataSettings | None = None) -> TradingCalendar:
    """Sessions over the whole download range (warm-up year included)."""
    settings = settings or load_settings()
    return TradingCalendar.nyse(settings.download_start, settings.download_end)


def _utc_index(timestamps) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if index.tz is None:
        raise ValueError(
            "timestamps must be timezone-aware; localise them first (GDELT and FNSPID times are UTC)"
        )
    return index.tz_convert("UTC").as_unit("ns")


def _date_index(dates) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if index.tz is not None:
        raise ValueError("dates must not carry a timezone")
    known = index[~index.isna()]
    if (known != known.normalize()).any():
        raise ValueError("dates must not have a time of day; use session_of_timestamp for timestamps")
    return index.as_unit("ns")
