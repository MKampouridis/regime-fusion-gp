import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_fusion.data import cli, gdelt, sa, store
from regime_fusion.data.calendar import TradingCalendar
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.data.universe import load_universe
from regime_fusion.testing import check_no_lookahead, check_point_in_time


def hourly_rows(rows):
    """Hourly GDELT sums from (hour, stock, variants, records, tone_sum, negative_share_sum)."""
    frame = pd.DataFrame(0.0, index=range(len(rows)), columns=list(gdelt.SUM_COLUMNS))
    frame.insert(0, "hour", pd.DatetimeIndex([pd.Timestamp(r[0], tz="UTC") for r in rows]))
    frame.insert(1, "stock", [r[1] for r in rows])
    frame.insert(2, "variants", [r[2] for r in rows])
    frame["records"] = [float(r[3]) for r in rows]
    frame["gcam_records"] = frame["records"]
    frame["tone_sum"] = [float(r[4]) for r in rows]
    frame["lm_negative_share_sum"] = [float(r[5]) for r in rows]
    return frame


@pytest.fixture
def rules(tmp_path):
    path = tmp_path / "names.yaml"
    path.write_text(yaml.safe_dump({
        "prefixes": ["access investor kit for", "the"],
        "stocks": {
            "AAPL": {"accept": ["apple{S}", "apple (iphone|store)( .*)?"]},
            "ABBV": "all",
            "F": {"accept": ["(.* )?ford motor( .*)?"], "reject": ["(.* )?ford motor credit( .*)?"]},
            "UTX": {"accept": ["(.* )?united technologies( .*)?",
                               {"pattern": "(.* )?raytheon( .*)?", "since": "2020-04-03"}]},
            "KODK": {"accept": [".*"], "reject": ["(.* )?kodak black( .*)?"], "veto": ["(.* )?kodak black( .*)?"]},
        }}))
    return sa.load_rules(path)


@pytest.fixture
def calendar():
    return TradingCalendar.nyse("2015-01-01", "2015-12-31")


def test_names_count_by_whole_name_with_prefixes_rejects_and_dates(rules):
    day = pd.Timestamp("2016-05-02")
    assert rules.counts("AAPL", "apple inc", day)
    assert rules.counts("AAPL", "apple, inc", day)
    assert rules.counts("AAPL", "access investor kit for apple inc", day)
    assert rules.counts("AAPL", "apple store", day)
    assert not rules.counts("AAPL", "apple daily", day)
    assert rules.counts("AAPL", "apple daily|apple inc", day)  # one name that counts is enough
    assert rules.counts("ABBV", "drugmaker abbvie inc", day)
    assert rules.counts("F", "the ford motor co", day)
    assert not rules.counts("F", "ford motor credit co", day)
    assert not rules.counts("F", "ford foundation", day)
    # Dated patterns: Raytheon is UTX only from the merger on, and the cache keeps days apart.
    assert not rules.counts("UTX", "raytheon co", "2019-06-03")
    assert rules.counts("UTX", "raytheon co", "2020-06-01")
    assert not rules.counts("UTX", "raytheon co", "2019-06-03")
    assert rules.counts("UTX", "united technologies corp", "2019-06-03")
    # A veto rules out the whole record: the rapper's articles also call him plain "kodak".
    assert rules.counts("KODK", "eastman kodak|kodak", day)
    assert not rules.counts("KODK", "kodak black|kodak", day)
    assert not rules.counts("KODK", "the kodak black", day)          # after the prefixes are removed


def test_rules_are_required_for_every_stock(tmp_path):
    path = tmp_path / "names.yaml"
    path.write_text(yaml.safe_dump({"stocks": {"AAPL": "all"}}))
    with pytest.raises(ValueError, match="no name rules"):
        sa.load_rules(path, stocks=["AAPL", "F"])
    path.write_text(yaml.safe_dump({"stocks": {"AAPL": {"reject": ["apple daily"]}}}))
    with pytest.raises(ValueError, match="accept"):
        sa.load_rules(path)


def test_repository_rules_cover_the_universe_and_known_cases():
    universe = load_universe(cli.UNIVERSE_FILE)
    stocks = [s.id for s in universe.select("main", "holdout", include_holdout=True)]
    rules = sa.load_rules(stocks=stocks)
    aliases = yaml.safe_load(gdelt.DEFAULT_ALIASES.read_text())["stocks"]
    assert set(rules.stocks) == set(aliases)
    day = pd.Timestamp("2017-03-01")
    cases = {
        ("AAPL", "apple inc"): True, ("AAPL", "apple daily"): False,
        ("AAPL", "apple federal credit union"): False,
        ("F", "ford motor co"): True, ("F", "ford foundation"): False,
        ("ABT", "abbott laboratories"): True, ("ABT", "tony abbott department"): False,
        ("ANTM", "anthem inc"): True, ("ANTM", "national anthem"): False,
        ("D", "dominion energy inc"): True, ("D", "old dominion university"): False,
        ("CL", "colgate palmolive"): True, ("CL", "colgate university"): False,
        ("CTAS", "cintas corporation"): True, ("CTAS", "cintas center"): False,
        ("GE", "ge capital"): True, ("GM", "general motors co"): True,
        ("KO", "coca-cola co"): True, ("KO", "coca-cola european partners"): False,
        ("BMY", "keytruda to bristol-myers squibb co"): True,
        ("NYT", "new york times"): False, ("NYT", "new york times co"): True,
        ("SO", "atlanta-based southern company"): True, ("SO", "norfolk southern co"): False,
        ("SO", "kansas city southern co"): False, ("SO", "southern co-operative"): False,
        ("MCD", "mcdonald corp"): True, ("MCD", "laquan mcdonalds"): False, ("MCD", "robert mcdonald"): False,
        ("GE", "portland general electric"): False, ("PM", "philip morris united states"): False,
        ("WFC", "national association of home builders wells fargo"): False, ("KO", "diet coke"): True,
        ("TSLA", "tesla model"): True, ("TSLA", "nikola tesla"): False, ("AAPL", "apple carplay"): True,
        ("AMZN", "amazon india"): True, ("AMZN", "peruvian amazon"): False, ("HAL", "benjamin halliburton"): False,
        ("HAL", "houston-based halliburton"): True, ("VZ", "verizon wireless arena"): False,
        ("KODK", "kodak black|kodak"): False, ("KODK", "eastman kodak|kodak"): True,
        ("INTC", "flynn intel group|intel"): False, ("INTC", "intel corp|intel"): True,
        ("INTC", "senate intel committee|intel"): False, ("KO", "coca-cola scholars foundation|coca-cola"): False,
        ("MET", "metlife investment advisors llc"): True,   # its own unit, though in notices of stakes
        ("DD", "dupont fire chief larry creekmore|dupont"): False, ("DD", "dupont|dowdupont inc"): True,
        ("DD", "nisqually national wildlife refuge in dupont|dupont"): False,
    }
    for (stock, name), expected in cases.items():
        assert rules.counts(stock, name, day) is expected, (stock, name)
    assert not rules.counts("UTX", "raytheon co", "2019-01-02")
    assert rules.counts("UTX", "raytheon technologies corp", "2021-01-04")
    assert rules.counts("SYMC", "symantec corp", "2018-01-02")
    assert not rules.counts("SYMC", "symantec", "2020-01-02")  # Broadcom's division by then


def test_rows_are_kept_by_the_rules(rules):
    hourly = hourly_rows([
        ("2015-03-02 14:00", "AAPL", "apple inc", 3, 0, 0),
        ("2015-03-02 14:00", "AAPL", "apple daily", 2, 0, 0),
        ("2015-03-02 15:00", "ABBV", "abbvie inc", 1, 0, 0),
        ("2015-03-02 15:00", "MSFT", "microsoft corp", 1, 0, 0),  # no rules: left out
    ])
    assert sa.keep(hourly, rules, "2015-03-02").tolist() == [True, False, True, False]
    names = Counter()
    sa.tally(names, hourly, sa.keep(hourly, rules, "2015-03-02"))
    assert names[("AAPL", True, "apple inc")] == 3 and names[("AAPL", False, "apple daily")] == 2


def test_news_after_the_close_counts_for_the_next_session(calendar):
    # 2 March 2015: the close is 16:00 in New York, 21:00 UTC.
    hourly = hourly_rows([
        ("2015-03-02 20:00", "AAPL", "apple inc", 2, 1.0, 0.0),
        ("2015-03-02 21:00", "AAPL", "apple inc", 5, 2.0, 0.0),
        ("2015-03-03 02:00", "AAPL", "apple inc", 1, 3.0, 0.0),
    ])
    sums = sa.session_sums(hourly, calendar)
    assert sums.loc[(pd.Timestamp("2015-03-02"), "AAPL"), "records"] == 2
    assert sums.loc[(pd.Timestamp("2015-03-03"), "AAPL"), "records"] == 6
    assert sums.loc[(pd.Timestamp("2015-03-03"), "AAPL"), "tone_sum"] == 5.0
    assert sa.session_sums(hourly.iloc[:0], calendar).empty


def test_coverage_is_the_share_of_a_sessions_files_read(calendar):
    slots = pd.date_range("2015-02-27 00:00", "2015-03-04 23:45", freq="15min", tz="UTC")
    # Drop 72 of the 288 files of Monday's session (Friday 21:00 to Monday 21:00 UTC).
    gap = (slots >= "2015-02-28 00:00") & (slots < "2015-02-28 18:00")
    cover = sa.coverage(slots[~gap], calendar)
    assert cover[pd.Timestamp("2015-03-02")] == pytest.approx(216 / 288)
    assert cover[pd.Timestamp("2015-03-03")] == 1.0
    assert cover[pd.Timestamp("2015-03-06")] == 0.0  # no files read yet
    assert np.isnan(cover.iloc[0])
    # On 9 March 2015 clocks went forward: Monday's session ran from 21:00 to 20:00 UTC.
    spring = pd.date_range("2015-03-06 21:00", "2015-03-09 19:45", freq="15min", tz="UTC")
    assert sa.coverage(spring, calendar)[pd.Timestamp("2015-03-09")] == 1.0


def test_failed_files_are_not_counted_as_read():
    day_files = pd.DataFrame({"stamp": ["20150301000000", "20150301001500", "20150301003000"]})
    failed = ["https://data.gdeltproject.org/gdeltv2/20150301001500.gkg.csv.zip: not on GDELT's server (404)"]
    stamps = sa.fetched_stamps(day_files, failed)
    assert list(stamps) == [pd.Timestamp("2015-03-01 00:00", tz="UTC"), pd.Timestamp("2015-03-01 00:30", tz="UTC")]


def test_indicators_by_hand(monkeypatch):
    monkeypatch.setattr(sa, "BASELINE", 3)
    monkeypatch.setattr(sa, "MIN_BASELINE", 2)
    sessions = pd.bdate_range("2015-03-02", periods=8)
    cover = pd.Series([1.0, 1.0, 0.8, 0.4, 1.0, 1.0, 1.0, 1.0], index=sessions)
    sums = pd.DataFrame(0.0, index=sessions, columns=list(gdelt.SUM_COLUMNS))
    sums["records"] = [2, 0, 4, 9, 1, 3, 0, 5]
    sums["gcam_records"] = sums["records"]
    sums["tone_sum"] = [-2.0, 0, 4.0, 9.0, -1.0, 6.0, 0, 5.0]
    sums["lm_negative_share_sum"] = [0.04, 0, 0.04, 0.09, 0.01, 0.06, 0, 0.05]
    table = sa.indicators(sums, cover)
    assert list(table.columns) == sa.columns() and len(sa.columns()) == 12
    a1 = table["attention_1"]
    assert a1.iloc[0] == pytest.approx(np.log1p(2)) and a1.iloc[1] == 0.0
    assert a1.iloc[2] == pytest.approx(np.log1p(4 / 0.8))  # scaled up for missing files
    assert np.isnan(a1.iloc[3])  # less than half the files: no indicators
    assert table["tone_1"].iloc[0] == -1.0 and np.isnan(table["tone_1"].iloc[1])  # no article, no tone
    assert np.isnan(table["tone_1"].iloc[3])
    assert table["lm_negative_1"].iloc[2] == pytest.approx(1.0)  # percent of words
    # Five sessions: at least three with coverage; the invalid session is left out and the count scaled.
    assert np.isnan(table["attention_5"].iloc[1]) and not np.isnan(table["attention_5"].iloc[2])
    assert table["attention_5"].iloc[4] == pytest.approx(np.log1p((2 + 0 + 5 + 1) * 5 / 4))
    assert table["tone_5"].iloc[4] == pytest.approx((-2 + 4 - 1) / 7)
    # Abnormal attention: today's attention minus the mean of the previous three (at least two).
    assert np.isnan(table["abnormal_attention_1"].iloc[1])
    expected = np.log1p(3) - np.mean([np.log1p(4 / 0.8), np.log1p(1)])  # session 3 has no value
    assert table["abnormal_attention_1"].iloc[5] == pytest.approx(expected)


def test_indicators_never_look_ahead():
    rng = np.random.default_rng(3)
    n = 220
    sessions = pd.bdate_range("2015-03-02", periods=n)
    frame = pd.DataFrame(0.0, index=sessions, columns=list(gdelt.SUM_COLUMNS))
    frame["records"] = rng.poisson(3, n).astype(float)
    frame["gcam_records"] = frame["records"]
    frame["tone_sum"] = frame["records"] * rng.normal(-1, 2, n)
    frame["lm_negative_share_sum"] = frame["records"] * rng.uniform(0, 0.03, n)
    frame["coverage"] = np.where(rng.random(n) < 0.05, 0.3, 1.0)

    def compute(data):
        return sa.indicators(data[list(gdelt.SUM_COLUMNS)], data["coverage"])

    check_no_lookahead(compute, frame, points=[0, 4, 30, 90, 150, n - 1])


def test_indicators_use_only_news_and_files_public_by_the_close():
    rng = np.random.default_rng(5)
    cal = TradingCalendar.nyse("2015-02-02", "2015-07-31")
    slots = pd.date_range("2015-02-02 00:00", "2015-07-31 23:45", freq="15min", tz="UTC")
    slots = slots[(rng.random(len(slots)) > 0.03) & ~((slots >= "2015-04-14") & (slots < "2015-04-16"))]
    hours = pd.date_range("2015-02-02 00:00", "2015-07-31 23:00", freq="h", tz="UTC")
    hours = hours[rng.random(len(hours)) < 0.3]
    news = hourly_rows([(h.tz_localize(None), "AAPL", "apple inc", rng.integers(1, 5), rng.normal(), 0.02)
                        for h in hours])
    items = pd.concat([
        pd.DataFrame({"kind": "file", "time": slots}),
        news.assign(kind="news", time=news["hour"]),
    ], ignore_index=True)
    items["available"] = cal.session_of_timestamp(pd.DatetimeIndex(items["time"]))
    items = items.dropna(subset=["available"])

    def build(chosen):
        files = pd.DatetimeIndex(chosen.loc[chosen["kind"] == "file", "time"])
        rows = chosen[chosen["kind"] == "news"]
        sums = sa.session_sums(rows, cal)
        own = sums.xs("AAPL", level="stock") if len(sums) else sums.droplevel("stock")
        return sa.indicators(own, sa.coverage(files, cal))

    points = cal.between("2015-03-02", "2015-07-30")[::9]
    check_point_in_time(build, items, "available", points)


def test_sa_command_builds_indicators_from_extracted_days(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    days = ["20150302", "20150303", "20150304"]
    lines = []
    for day in days:
        for stamp in pd.date_range(day, periods=96, freq="15min"):
            url = f"https://data.gdeltproject.org/gdeltv2/{stamp:%Y%m%d%H%M%S}.gkg.csv.zip"
            lines.append(f"100 abc {url}")
    monkeypatch.setattr(cli, "gdelt_master_text", lambda: "\n".join(lines))
    lost = "https://data.gdeltproject.org/gdeltv2/20150302220000.gkg.csv.zip: not on GDELT's server (404)"
    for day, failed in (("20150302", [lost]), ("20150303", [])):  # 4 March is not extracted yet
        stamp = pd.Timestamp(day)
        hourly = hourly_rows([
            (stamp + pd.Timedelta(hours=14), "AAPL", "apple inc", 3, -3.0, 0.03),
            (stamp + pd.Timedelta(hours=15), "AAPL", "apple daily", 4, 8.0, 0.0),
            (stamp + pd.Timedelta(hours=15), "F", "ford foundation", 2, 1.0, 0.0),
        ])
        summary = {"day": day, "failed": failed, "extraction": gdelt.EXTRACTION}
        store.write_parquet(cli.gdelt_path(day, "hourly"), hourly, **summary)
    assert cli.main(["sa", "--stocks", "AAPL", "F"]) == 0
    table = pd.read_parquet(store.layer("processed", "sa", "AAPL.parquet"))
    assert list(table.columns) == sa.columns()
    assert table.loc["2015-03-03", "attention_1"] == pytest.approx(np.log1p(3 * 96 / 95))
    assert table.loc["2015-03-03", "tone_1"] == -1.0
    assert np.isnan(table.loc["2015-03-05", "attention_1"])  # its news day is not extracted yet
    interim = pd.read_parquet(store.layer("interim", "news", "AAPL.parquet"))
    assert interim.loc["2015-03-03", "coverage"] == pytest.approx(95 / 96)  # 21:00 to 21:00 UTC; 22:00 lost
    assert interim.loc["2015-03-03", "articles"] == pytest.approx(3 * 96 / 95)
    assert pd.read_parquet(store.layer("processed", "sa", "F.parquet"))["attention_1"].max() == 0.0
    report = json.loads(store.layer("reports", "sa_coverage.json").read_text())
    assert report["AAPL"]["top_names_left_out"] == {"apple daily": 8}
    assert report["F"]["top_names_left_out"] == {"ford foundation": 4}
    assert report["_coverage"]["gdelt_days_pending"] == 1


def test_agreement_with_fnspid_is_the_rank_correlation_of_weekly_counts():
    days = pd.bdate_range("2016-01-04", periods=150)
    weekly = np.repeat(np.arange(30) % 7, 5).astype(float)          # news volume that rises and falls
    agreed = sa.fnspid_agreement(pd.Series(weekly * 10, days), pd.Series(weekly, days))
    assert agreed == {"weeks": 30, "weekly_rank_correlation": 1.0, "fnspid_days_with_gdelt_news": 1.0}
    reverse = sa.fnspid_agreement(pd.Series(6 - weekly, days), pd.Series(weekly, days))
    assert reverse["weekly_rank_correlation"] == -1.0 and reverse["fnspid_days_with_gdelt_news"] < 1
    short = sa.fnspid_agreement(pd.Series(weekly[:50], days[:50]), pd.Series(weekly[:50], days[:50]))
    assert short["weeks"] == 10 and short["weekly_rank_correlation"] is None   # too few weeks to say
