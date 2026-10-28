import numpy as np
import pandas as pd
import pytest
import yaml

from regime_fusion.config import ConfigError
from regime_fusion.data import fundamentals
from regime_fusion.data.calendar import TradingCalendar
from regime_fusion.data.fundamentals import Concept
from regime_fusion.testing import check_point_in_time

CAL = TradingCalendar.nyse("2014-01-01", "2016-12-31")
REVENUE = Concept("revenue", "flow", ("us-gaap:Revenues", "us-gaap:SalesRevenueNet"))
ASSETS = Concept("assets", "instant", ("us-gaap:Assets",))


def facts(rows):
    """Filed facts: (tag, start or None, end, value, filed[, form])."""
    records = []
    for i, row in enumerate(rows):
        tag, start, end, value, filed, *form = row
        taxonomy, concept = tag.split(":")
        records.append(dict(cik=1, taxonomy=taxonomy, concept=concept, unit="USD",
                            start=pd.Timestamp(start) if start else pd.NaT, end=pd.Timestamp(end),
                            value=float(value), accn=f"0000000001-{filed}-{i}",
                            form=form[0] if form else "10-Q", filed=pd.Timestamp(filed), frame=None))
    frame = pd.DataFrame.from_records(records)
    frame["available"] = CAL.session_after_date(frame["filed"])
    return frame


# A calendar-year firm. The 10-K of February 2016 restates fiscal 2014 from 100 to 101.
CALENDAR_YEAR = facts([
    ("us-gaap:Revenues", "2014-01-01", "2014-12-31", 100, "2015-02-20", "10-K"),
    ("us-gaap:Revenues", "2015-01-01", "2015-03-31", 30, "2015-05-01"),
    ("us-gaap:Revenues", "2014-01-01", "2014-03-31", 25, "2015-05-01"),
    ("us-gaap:Revenues", "2015-04-01", "2015-06-30", 32, "2015-08-03"),
    ("us-gaap:Revenues", "2015-01-01", "2015-06-30", 62, "2015-08-03"),
    ("us-gaap:Revenues", "2014-01-01", "2014-06-30", 50, "2015-08-03"),
    ("us-gaap:Revenues", "2015-01-01", "2015-09-30", 95, "2015-11-02"),
    ("us-gaap:Revenues", "2014-01-01", "2014-09-30", 75, "2015-11-02"),
    ("us-gaap:Revenues", "2015-01-01", "2015-12-31", 130, "2016-02-19", "10-K"),
    ("us-gaap:Revenues", "2014-01-01", "2014-12-31", 101, "2016-02-19", "10-K"),
])


def row(panel, day):
    return panel.loc[pd.Timestamp(day)]


def test_ttm_follows_each_filing_from_the_next_session():
    panel = fundamentals.asof(CALENDAR_YEAR, REVENUE, CAL.sessions)
    assert np.isnan(row(panel, "2015-02-20")["revenue_ttm"])         # the 10-K's filing day
    expected = {
        "2015-02-23": (100, "2014-12-31"),             # fiscal 2014
        "2015-05-04": (100 + 30 - 25, "2015-03-31"),   # Friday's 10-Q, first used on Monday
        "2015-08-04": (100 + 62 - 50, "2015-06-30"),
        "2015-11-03": (100 + 95 - 75, "2015-09-30"),
        "2016-02-22": (130, "2015-12-31"),
    }
    for day, (value, end) in expected.items():
        assert row(panel, day)["revenue_ttm"] == value
        assert row(panel, day)["revenue_end"] == pd.Timestamp(end)
    assert row(panel, "2015-08-03")["revenue_ttm"] == 105  # the filing day still shows the old value


def test_restatements_count_only_once_filed():
    panel = fundamentals.asof(CALENDAR_YEAR, REVENUE, CAL.sessions)
    # Before the 2016 10-K the TTM a year before the latest end is unknown here;
    # afterwards it is the restated fiscal 2014 value.
    assert np.isnan(row(panel, "2015-11-03")["revenue_ttm_prev"])
    assert row(panel, "2016-02-22")["revenue_ttm_prev"] == 101


def test_daily_values_are_point_in_time():
    build = lambda items: fundamentals.asof(items, REVENUE, CAL.sessions)  # noqa: E731
    check_point_in_time(build, CALENDAR_YEAR, "available", CAL.between("2015-01-01", "2016-06-30"))


def test_52_53_week_year_uses_apples_reported_figures():
    # Apple, fiscal 2015 (364 days) and the first quarters of fiscal 2015 and 2016, in $m.
    apple = facts([
        ("us-gaap:SalesRevenueNet", "2014-09-28", "2015-09-26", 233715, "2015-10-28", "10-K"),
        ("us-gaap:SalesRevenueNet", "2015-09-27", "2015-12-26", 75872, "2016-01-27"),
        ("us-gaap:SalesRevenueNet", "2014-09-28", "2014-12-27", 74599, "2016-01-27"),
    ])
    panel = fundamentals.asof(apple, REVENUE, CAL.sessions)
    assert row(panel, "2016-01-28")["revenue_ttm"] == 234988


def test_twelve_week_quarters_are_recognised():
    # A PepsiCo-style year: the first quarter has 12 weeks (84 days).
    pepsi = facts([
        ("us-gaap:Revenues", "2014-12-28", "2015-12-26", 1000, "2016-02-11", "10-K"),
        ("us-gaap:Revenues", "2015-12-27", "2016-03-19", 210, "2016-04-21"),
        ("us-gaap:Revenues", "2014-12-28", "2015-03-21", 200, "2016-04-21"),
    ])
    panel = fundamentals.asof(pepsi, REVENUE, CAL.sessions)
    assert row(panel, "2016-04-22")["revenue_ttm"] == 1010


def test_four_quarters_are_summed_when_nothing_else_is_filed():
    quarterly = facts([
        ("us-gaap:Revenues", "2015-01-01", "2015-03-31", 10, "2015-05-01"),
        ("us-gaap:Revenues", "2015-04-01", "2015-06-30", 11, "2015-08-03"),
        ("us-gaap:Revenues", "2015-07-01", "2015-09-30", 12, "2015-11-02"),
        ("us-gaap:Revenues", "2015-10-01", "2015-12-31", 13, "2016-02-19"),
    ])
    panel = fundamentals.asof(quarterly, REVENUE, CAL.sessions)
    assert np.isnan(row(panel, "2015-11-03")["revenue_ttm"])
    assert row(panel, "2016-02-22")["revenue_ttm"] == 46


def test_preferred_tag_wins_within_a_filing_and_tag_changes_are_followed():
    switch = facts([
        ("us-gaap:SalesRevenueNet", "2014-01-01", "2014-12-31", 100, "2015-02-20", "10-K"),
        ("us-gaap:Revenues", "2015-01-01", "2015-12-31", 120, "2016-02-19", "10-K"),
        ("us-gaap:SalesRevenueNet", "2015-01-01", "2015-12-31", 999, "2016-02-19", "10-K"),
    ])
    panel = fundamentals.asof(switch, REVENUE, CAL.sessions)
    assert row(panel, "2015-02-23")["revenue_ttm"] == 100   # only the old tag exists
    assert row(panel, "2016-02-22")["revenue_ttm"] == 120   # Revenues preferred in the same filing
    assert row(panel, "2016-02-22")["revenue_ttm_prev"] == 100


def test_instants_take_the_latest_period_end_and_later_restatements():
    balance = facts([
        ("us-gaap:Assets", None, "2014-12-31", 500, "2015-02-20", "10-K"),
        ("us-gaap:Assets", None, "2015-03-31", 520, "2015-05-01"),
        ("us-gaap:Assets", None, "2015-03-31", 525, "2015-08-03", "10-Q/A"),
        ("us-gaap:Assets", None, "2014-12-31", 505, "2015-08-03", "10-Q/A"),  # older end: ignored
    ])
    panel = fundamentals.asof(balance, ASSETS, CAL.sessions)
    assert np.isnan(row(panel, "2015-02-20")["assets"])
    assert row(panel, "2015-02-23")["assets"] == 500
    assert row(panel, "2015-05-04")["assets"] == 520
    assert row(panel, "2015-08-04")["assets"] == 525
    assert row(panel, "2015-08-04")["assets_end"] == pd.Timestamp("2015-03-31")
    check_point_in_time(lambda items: fundamentals.asof(items, ASSETS, CAL.sessions),
                        balance, "available", CAL.between("2015-01-01", "2015-12-31"))


def test_panel_without_facts_is_empty_but_indexed():
    panel = fundamentals.daily_panel(CALENDAR_YEAR.iloc[:0], [REVENUE, ASSETS], CAL.sessions)
    assert panel.index.equals(CAL.sessions) and panel.isna().all().all()
    assert list(panel.columns) == ["revenue_ttm", "revenue_ttm_prev", "revenue_end",
                                   "assets", "assets_end", "assets_filed"]


def test_latest_kind_takes_the_quarter_and_reports_its_filing():
    shares = Concept("diluted_shares", "latest", ("us-gaap:Shares",))
    frame = facts([
        ("us-gaap:Shares", "2015-01-01", "2015-12-31", 1000, "2016-02-19", "10-K"),  # year
        ("us-gaap:Shares", "2015-10-01", "2015-12-31", 990, "2016-02-19", "10-K"),   # fourth quarter
        ("us-gaap:Shares", "2015-07-01", "2015-09-30", 1010, "2015-11-02"),
    ])
    panel = fundamentals.asof(frame, shares, CAL.sessions)
    assert row(panel, "2015-11-03")["diluted_shares"] == 1010
    assert row(panel, "2016-02-22")["diluted_shares"] == 990
    assert row(panel, "2016-02-22")["diluted_shares_filed"] == pd.Timestamp("2016-02-19")
    check_point_in_time(lambda items: fundamentals.asof(items, shares, CAL.sessions),
                        frame, "available", CAL.between("2015-10-01", "2016-06-30"))


def test_repository_concepts_load(tmp_path):
    concepts = {c.name: c for c in fundamentals.load_concepts()}
    assert concepts["revenue"].kind == "flow" and concepts["assets"].kind == "instant"
    assert concepts["shares_outstanding"].unit == "shares"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"concepts": {"x": {"kind": "stock", "tags": ["us-gaap:X"]}}}))
    with pytest.raises(ConfigError):
        fundamentals.load_concepts(bad)


def test_coverage_reports_thin_and_stale_concepts():
    panel = fundamentals.daily_panel(CALENDAR_YEAR, [REVENUE], CAL.sessions)
    sessions = CAL.between("2015-01-01", "2016-06-30")
    summary, flags = fundamentals.coverage(panel, [REVENUE], sessions)
    assert summary["revenue"]["share"] == round(float((sessions >= "2015-02-23").mean()), 4)
    assert summary["revenue"]["max_age_days"] == 182  # 30 June 2016, fiscal 2015 the latest
    assert flags == []
    summary, flags = fundamentals.coverage(panel, [REVENUE], sessions, min_share=0.95)
    assert flags == [f"revenue: values on {summary['revenue']['share']:.0%} of days"]
    _, flags = fundamentals.coverage(panel, [REVENUE], sessions[40:], max_age_days=100)
    assert flags == ["revenue: latest period up to 182 days old"]
