import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import news
from regime_fusion.data.calendar import TradingCalendar
from regime_fusion.testing import check_point_in_time

CAL = TradingCalendar.nyse("2015-11-02", "2015-12-31")


def test_gdelt_update_times_are_utc():
    stamps = news.gdelt_time([20151124201500, "20151124204500"])
    assert stamps[0] == pd.Timestamp("2015-11-24 20:15", tz="UTC")
    assert stamps[1].tz is not None


@pytest.mark.parametrize(
    "stamp, session",
    [
        (20151124204500, "2015-11-24"),  # 15:45 in New York
        (20151124210000, "2015-11-25"),  # 16:00: next session
        (20151127175900, "2015-11-27"),  # 12:59 on the early-close day after Thanksgiving
        (20151127180000, "2015-11-30"),  # 13:00 on that day
        (20151126150000, "2015-11-27"),  # Thanksgiving itself
    ],
)
def test_gdelt_records_count_for_the_session_closing_after_their_update(stamp, session):
    assert news.assign_sessions(CAL, news.gdelt_time([stamp]))[0] == pd.Timestamp(session)


def test_fnspid_items_without_a_time_wait_for_the_next_session():
    stamps, date_only = news.fnspid_time(
        ["2015-11-24 14:30:00 UTC", "2015-11-24 00:00:00 UTC", "2015-11-27 00:00:00 UTC"]
    )
    assert date_only.tolist() == [False, True, True]
    sessions = news.assign_sessions(CAL, stamps, date_only)
    # 14:30 UTC is 09:30 in New York: same day. A bare date counts the next session.
    assert sessions.tolist() == [pd.Timestamp("2015-11-24"), pd.Timestamp("2015-11-25"),
                                 pd.Timestamp("2015-11-30")]


def test_hourly_buckets_start_on_the_hour():
    stamps = news.gdelt_time([20151124204500, 20151124210000])
    assert news.hourly_buckets(stamps).tolist() == [
        pd.Timestamp("2015-11-24 20:00", tz="UTC"), pd.Timestamp("2015-11-24 21:00", tz="UTC"),
    ]


def make_buckets():
    hours = pd.DatetimeIndex([
        "2015-11-24 19:00", "2015-11-24 20:00", "2015-11-24 21:00",   # 14:00, 15:00, 16:00 New York
        "2015-11-27 17:00", "2015-11-27 18:00", "2015-11-28 15:00",   # 12:00, 13:00 early close; Saturday
        "2015-11-24 20:00",
    ], tz="UTC")
    return pd.DataFrame({
        "stock": ["AAPL"] * 6 + ["MSFT"],
        "hour": hours,
        "articles": [1, 2, 4, 8, 16, 32, 5],
        "tone_sum": [0.5, -1.0, 2.0, 0.0, 1.0, -3.0, 1.5],
    })


def test_daily_sums_follow_the_close_cut_off():
    sessions = CAL.between("2015-11-24", "2015-11-30")
    daily = news.daily_sums(make_buckets(), CAL, sessions)
    apple = daily.xs("AAPL", level="stock")["articles"]
    assert apple.to_dict() == {
        pd.Timestamp("2015-11-24"): 1 + 2,       # hours before 16:00
        pd.Timestamp("2015-11-25"): 4,           # the 16:00 hour
        pd.Timestamp("2015-11-27"): 8,           # before the 13:00 early close
        pd.Timestamp("2015-11-30"): 16 + 32,     # after it, and Saturday
    }
    assert daily.loc[(pd.Timestamp("2015-11-25"), "MSFT"), "articles"] == 0
    assert daily.loc[(pd.Timestamp("2015-11-24"), "MSFT"), "tone_sum"] == 1.5


def test_daily_sums_are_point_in_time():
    buckets = make_buckets()
    buckets["available"] = news.assign_sessions(CAL, pd.DatetimeIndex(buckets["hour"]))
    sessions = CAL.between("2015-11-24", "2015-11-30")

    def build(items):
        if items.empty:
            return pd.DataFrame(0, index=sessions, columns=["articles"])
        daily = news.daily_sums(items.drop(columns="available"), CAL, sessions)
        return daily["articles"].groupby(level="session").sum().to_frame()

    check_point_in_time(build, buckets, "available", sessions)


def test_bad_buckets_are_rejected():
    buckets = make_buckets()
    buckets["hour"] = buckets["hour"] + pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="starts of hours"):
        news.daily_sums(buckets, CAL, CAL.sessions)
