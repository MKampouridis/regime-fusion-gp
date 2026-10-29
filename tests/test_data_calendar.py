import pandas as pd
import pytest

from regime_fusion.data.calendar import TradingCalendar, study_calendar

CAL = TradingCalendar.nyse("2014-01-01", "2025-12-31")


def ny(text):
    return pd.Timestamp(text, tz="America/New_York")


def only(index):
    assert len(index) == 1
    return index[0]


def test_session_counts_match_the_nyse_year_by_year():
    # 2018 and 2025 include the closures for the funerals of Presidents Bush and
    # Carter. The total, 2,766, equals the number of daily S&P 500 index bars on
    # Yahoo Finance over 2015-2025.
    expected = {2015: 252, 2016: 252, 2017: 251, 2018: 251, 2019: 252, 2020: 253,
                2021: 252, 2022: 251, 2023: 250, 2024: 252, 2025: 250}
    counts = pd.Series(CAL.sessions.year).value_counts()
    assert {year: int(counts[year]) for year in expected} == expected
    assert sum(expected.values()) == 2766


def test_closures_early_closes_and_clock_changes():
    assert pd.Timestamp("2018-12-05") not in CAL.sessions
    assert pd.Timestamp("2025-01-09") not in CAL.sessions
    closes = pd.Series(CAL.closes, index=CAL.sessions)
    assert closes[pd.Timestamp("2015-11-27")] == ny("2015-11-27 13:00")  # day after Thanksgiving
    assert closes[pd.Timestamp("2019-07-03")] == ny("2019-07-03 13:00")
    assert closes[pd.Timestamp("2015-03-06")] == ny("2015-03-06 16:00")  # clocks change on 8 March
    assert closes[pd.Timestamp("2015-03-09")] == ny("2015-03-09 16:00")


@pytest.mark.parametrize(
    "stamp, session",
    [
        ("2015-03-09 09:00:00", "2015-03-09"),  # before the open counts for the day
        ("2015-03-09 15:59:59", "2015-03-09"),
        ("2015-03-09 16:00:00", "2015-03-10"),  # at the close: next session
        ("2015-03-06 16:30:00", "2015-03-09"),  # Friday evening: Monday
        ("2015-03-07 12:00:00", "2015-03-09"),  # Saturday
        ("2015-11-26 10:00:00", "2015-11-27"),  # Thanksgiving
        ("2015-11-27 12:59:59", "2015-11-27"),  # early close at 13:00
        ("2015-11-27 13:00:00", "2015-11-30"),
        ("2018-12-04 17:00:00", "2018-12-06"),  # 5 December 2018 closed
    ],
)
def test_timestamps_belong_to_the_first_session_closing_after_them(stamp, session):
    assert only(CAL.session_of_timestamp([ny(stamp)])) == pd.Timestamp(session)


def test_utc_timestamps_respect_daylight_saving():
    # 20:59 UTC is 15:59 in New York in winter but 16:59 in summer.
    winter = pd.Timestamp("2015-01-05 20:59", tz="UTC")
    summer = pd.Timestamp("2015-07-06 20:59", tz="UTC")
    assert only(CAL.session_of_timestamp([winter])) == pd.Timestamp("2015-01-05")
    assert only(CAL.session_of_timestamp([summer])) == pd.Timestamp("2015-07-07")


@pytest.mark.parametrize(
    "day, session",
    [
        ("2015-03-09", "2015-03-10"),  # filed on a Monday: usable on Tuesday
        ("2015-03-06", "2015-03-09"),  # Friday: Monday
        ("2015-03-07", "2015-03-09"),  # Saturday: Monday
        ("2015-11-25", "2015-11-27"),  # the day before Thanksgiving
        ("2013-06-28", "2014-01-02"),  # before the calendar starts: its first session
    ],
)
def test_dates_belong_to_the_first_session_after_them(day, session):
    assert only(CAL.session_after_date([day])) == pd.Timestamp(session)


@pytest.mark.parametrize(
    "day, session",
    [("2015-03-09", "2015-03-09"), ("2015-03-06", "2015-03-06"), ("2015-03-07", "2015-03-09"),
     ("2015-11-26", "2015-11-27")],  # Monday, Friday, Saturday, Thanksgiving
)
def test_a_date_has_its_own_session_or_the_next(day, session):
    assert only(CAL.session_on_or_after_date([day])) == pd.Timestamp(session)


def test_missing_and_out_of_range_values_give_nat():
    assert CAL.session_of_timestamp([pd.NaT, ny("2025-12-31 16:00")]).isna().all()
    assert CAL.session_after_date([None, "2025-12-31"]).isna().all()


def test_ambiguous_inputs_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        CAL.session_of_timestamp([pd.Timestamp("2015-03-09 15:00")])
    with pytest.raises(ValueError, match="time of day"):
        CAL.session_after_date([pd.Timestamp("2015-03-09 15:00")])
    with pytest.raises(ValueError, match="timezone"):
        CAL.session_after_date([ny("2015-03-09")])


def test_study_calendar_spans_the_download_range():
    calendar = study_calendar()
    assert calendar.sessions[0] == pd.Timestamp("2014-01-02")
    assert calendar.sessions[-1] == pd.Timestamp("2025-12-31")
    assert len(calendar.between("2015-01-01", "2015-12-31")) == 252
