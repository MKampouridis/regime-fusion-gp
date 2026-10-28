import datetime as dt

import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import prices
from regime_fusion.data.calendar import TradingCalendar

CAL = TradingCalendar.nyse("2015-03-02", "2015-03-13")  # ten sessions
START, END = dt.date(2015, 3, 2), dt.date(2015, 3, 13)


def yahoo_bars(days, close, **columns):
    """Bars shaped like yfinance's output: tz-aware New York midnights."""
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="America/New_York") for day in days], name="Date")
    close = np.asarray(close, dtype=float)
    frame = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
         "Adj Close": close, "Volume": 1000.0, "Dividends": 0.0, "Stock Splits": 0.0},
        index=index,
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def sessions(n=10):
    return [day.date().isoformat() for day in CAL.sessions[:n]]


def test_traded_close_undoes_every_later_split_including_after_the_calendar():
    days = sessions() + ["2015-03-16"]  # one bar after the calendar ends
    splits = np.zeros(11)
    splits[5] = 4.0    # 4-for-1 on 9 March
    splits[10] = 2.0   # 2-for-1 on 16 March, outside the calendar
    aligned = prices.align(yahoo_bars(days, [10.0] * 11, **{"Stock Splits": splits}), CAL)
    assert len(aligned) == 10
    assert aligned["traded_close"].tolist() == [80.0] * 5 + [20.0] * 5
    assert aligned["split"].tolist() == [1.0] * 5 + [4.0] + [1.0] * 4


def test_dividend_adjustment_scales_the_whole_bar():
    adj = [9.0] * 4 + [10.0] * 6  # an ex-dividend date on the fifth session
    dividends = [0.0] * 4 + [1.0] + [0.0] * 5
    aligned = prices.align(yahoo_bars(sessions(), [10.0] * 10, **{"Adj Close": adj, "Dividends": dividends}), CAL)
    first = aligned.iloc[0]
    assert first["adj_close"] == 9.0
    assert first["adj_high"] == pytest.approx(10.1 * 0.9)
    assert aligned["dividend"].sum() == 1.0


def test_gaps_are_kept_and_off_calendar_bars_dropped():
    days = sessions()
    del days[3]                       # a session without a bar
    days.insert(4, "2015-03-07")      # a stray Saturday bar
    raw = yahoo_bars(days, np.arange(10.0, 20.0))
    aligned = prices.align(raw, CAL)
    assert aligned.index.equals(CAL.sessions)
    assert aligned["close"].isna().sum() == 1 and np.isnan(aligned["close"].iloc[3])
    result = prices.check(raw, aligned, START, END, CAL)
    assert result.missing_sessions == 1 and result.off_calendar_rows == 1


def test_naive_dates_are_accepted():
    raw = yahoo_bars(sessions(), [10.0] * 10)
    raw.index = raw.index.tz_localize(None)
    assert prices.align(raw, CAL)["close"].notna().all()


def test_checks_flag_late_start_big_moves_bad_bars_and_stale_prices():
    close = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 13.5, 13.0, 13.2]
    low = np.asarray(close) * 0.99
    low[8] = 13.5  # low above the close
    volume = [1000.0] * 9 + [0.0]
    raw = yahoo_bars(sessions()[1:], close[1:], Low=low[1:], Volume=volume[1:])
    aligned = prices.align(raw, CAL)
    result = prices.check(raw, aligned, START, END, CAL)
    assert result.first == pd.Timestamp("2015-03-03")
    assert result.big_moves == [("2015-03-10", 0.3)]
    assert result.bad_rows == 1 and result.zero_volume == 1
    assert result.longest_stale_run == 5
    text = " | ".join(result.flags)
    for expected in ("after the study start", "bad rows: 1", "zero volume: 1",
                     "unchanged for 5 sessions"):
        assert expected in text


def test_clean_series_has_no_flags():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 10)))
    raw = yahoo_bars(sessions(), close)
    result = prices.check(raw, prices.align(raw, CAL), START, END, CAL)
    assert result.flags == []


def test_spin_off_correction_recomputes_the_day_and_keeps_other_returns():
    days = sessions()
    raw = yahoo_bars(days, [10.0, 10.5, 10.2, 10.4, 10.3, 6.0, 6.1, 6.3, 6.2, 6.4])
    parent = prices.align(raw, CAL)   # Yahoo missed a spin-off on the sixth session
    child = prices.align(yahoo_bars(days, [20.0] * 10), CAL)
    fix = prices.Correction("PARENT", CAL.sessions[5], "child spun off, 1 for 5", 1.0, {"CHILD": 0.2})
    true = prices.corrected_return(parent, {"CHILD": child}, fix)
    assert true == pytest.approx((6.0 + 20.0 * 0.2) / 10.3 - 1)
    fixed = prices.apply_correction(parent, fix.date, true)
    before, after = parent["adj_close"].pct_change(), fixed["adj_close"].pct_change()
    assert after.iloc[5] == pytest.approx(true)
    np.testing.assert_allclose(after.drop(after.index[5]).dropna(), before.drop(before.index[5]).dropna())
    np.testing.assert_allclose(fixed["traded_close"], parent["traded_close"])


def test_repository_corrections_point_at_main_stocks_on_sessions():
    from regime_fusion.data.calendar import study_calendar
    from regime_fusion.data.universe import load_universe

    universe, calendar = load_universe(), study_calendar()
    corrections = prices.load_corrections()
    assert {c.stock for c in corrections} == {"XRX", "DD", "UTX"}
    for correction in corrections:
        assert universe.get(correction.stock).role == "main"
        assert correction.date in calendar.sessions
        assert correction.kept > 0 and all(v > 0 for v in correction.received.values())
