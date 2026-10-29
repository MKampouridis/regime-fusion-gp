import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import cli, finbert, gdelt, store
from regime_fusion.data.calendar import TradingCalendar, study_calendar
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.testing import check_point_in_time


def fnspid_rows(rows):
    return pd.DataFrame(rows, columns=["Date", "Article_title", "stock"])


def gdelt_rows(rows):
    return pd.DataFrame(rows, columns=["date", "stock", "variants", "title"])


def fake_scorer(texts):
    """Positive for "beats", negative for "misses", neutral otherwise."""
    out = np.tile([0.1, 0.1, 0.8], (len(texts), 1))
    for i, text in enumerate(texts):
        if "beats" in text:
            out[i] = [0.9, 0.05, 0.05]
        elif "misses" in text:
            out[i] = [0.05, 0.9, 0.05]
    return out


@pytest.fixture
def calendar():
    return TradingCalendar.nyse("2019-01-01", "2019-12-31")


def test_headlines_count_for_the_session_after_their_date_once(calendar):
    raw = fnspid_rows([
        ("2019-03-04 15:00:00 UTC", "Apple beats estimates", "AAPL"),   # a time of day is not trusted:
        ("2019-03-04 15:00:00 UTC", "Apple beats estimates", "AAPL"),   # the same item twice
        ("2019-03-04 22:00:00 UTC", "Apple misses on iPhone", "AAPL"),
        ("2019-03-04 00:00:00 UTC", "Apple news", "AAPL"),
        ("2018-12-28 00:00:00 UTC", "Apple old news", "AAPL"),          # before the calendar: dropped
    ])
    items = finbert.fnspid_headlines([raw], calendar)
    assert len(items) == 3 and "Apple old news" not in set(items["title"])
    assert (items["session"] == pd.Timestamp("2019-03-05")).all()     # every one counts the next session


def test_gdelt_headlines_count_from_the_update_that_carried_them_once_a_session(calendar):
    raw = gdelt_rows([
        (20190304203000, "AAPL", "apple inc", "Apple beats estimates"),      # 15:30 in New York: that session
        (20190304211500, "AAPL", "apple inc", "Apple beats estimates"),      # 16:15, after the close: the next
        (20190304214500, "AAPL", "apple inc", "Apple &amp; Google miss"),    # HTML entities are decoded
        (20190305150000, "AAPL", "apple inc", "Apple beats estimates"),      # a copy on another site, same
        (20190305160000, "F", "ford", "Apple beats estimates"),              # session: once; another stock counts
    ])
    items = finbert.gdelt_headlines([raw], calendar)
    assert list(zip(items["stock"], items["session"].dt.strftime("%m-%d"), items["title"])) == [
        ("AAPL", "03-04", "Apple beats estimates"), ("AAPL", "03-05", "Apple beats estimates"),
        ("AAPL", "03-05", "Apple & Google miss"), ("F", "03-05", "Apple beats estimates")]
    assert items["time"].iloc[1] == pd.Timestamp("2019-03-04 21:15", tz="UTC")   # the first copy's time


def test_a_source_covers_the_years_with_at_most_two_months_without_a_headline():
    sessions = TradingCalendar.nyse("2018-01-01", "2020-12-31").sessions
    dates = ([f"2018-{m:02d}-10" for m in range(1, 13)] + [f"2019-{m:02d}-10" for m in range(1, 11)]
             + [f"2020-{m:02d}-10" for m in range(1, 10)])
    items = pd.DataFrame({"session": pd.to_datetime(dates)})
    assert finbert.covered_years(items, sessions) == [2018, 2019]       # 2020 lacks three months
    assert finbert.covered_years(items.iloc[:0], sessions) == []

    class Window:
        def __init__(self, first, test_year):
            self.train, self.test_year = type("Range", (), {"start": pd.Timestamp(f"{first}-02-01")}), test_year

    assert finbert.covered_test_years([2015, 2016, 2017, 2018, 2019, 2020], [Window(2015, 2019), Window(2016, 2020),
                                      Window(2017, 2021)]) == [2019, 2020]


def test_scores_are_kept_and_indicators_built(calendar):
    raw = fnspid_rows([("2019-03-01 00:00:00 UTC", "Apple beats estimates", "AAPL"),
                       ("2019-03-04 00:00:00 UTC", "Apple misses on iPhone", "AAPL"),
                       ("2019-03-04 00:00:00 UTC", "Apple beats estimates", "AAPL"),
                       ("2019-03-11 00:00:00 UTC", "Apple holds", "AAPL")])
    items = finbert.fnspid_headlines([raw], calendar)
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_scorer(texts)

    scores = finbert.score_texts(items, pd.DataFrame(columns=["key", "positive", "negative", "neutral"]), counting)
    assert calls == [3] and len(scores) == 3                     # each distinct headline once
    again = finbert.score_texts(items, scores, counting)
    assert calls == [3] and len(again) == 3                       # nothing new to score
    table = finbert.indicators(items, scores, calendar.sessions, [2019])
    assert list(table.columns) == finbert.COLUMNS
    assert table.loc["2019-03-04", "finbert_sentiment_1"] == pytest.approx(0.85)
    assert table.loc["2019-03-05", "finbert_sentiment_1"] == pytest.approx(0.0)
    assert table.loc["2019-03-05", "finbert_attention_1"] == pytest.approx(np.log1p(2))
    assert table.loc["2019-03-05", "finbert_sentiment_5"] == pytest.approx((0.85 - 0.85 + 0.85) / 3)
    assert np.isnan(table.loc["2019-03-06", "finbert_sentiment_1"]) and table.loc["2019-03-06", "finbert_attention_1"] == 0
    assert (table.loc["2019-03-13":, "finbert_attention_1"] == 0).all()   # a covered year: no headline, no news
    assert finbert.indicators(items, scores, calendar.sessions, []).isna().all().all()   # not covered: unknown
    unread = finbert.indicators(items, scores, calendar.sessions, [2019], pd.DatetimeIndex(["2019-03-05"]))
    assert unread.loc["2019-03-05"].isna().all() and unread.loc["2019-03-04", "finbert_sentiment_1"] == pytest.approx(0.85)
    # An unread session is not a session without news: the 5-session window counts the read ones only.
    assert unread.loc["2019-03-06", "finbert_attention_5"] == pytest.approx(np.log1p(1 * 5 / 4))
    assert unread.loc["2019-03-06", "finbert_sentiment_5"] == pytest.approx(0.85)
    gaps = finbert.indicators(items, scores, calendar.sessions, [2019],
                              pd.DatetimeIndex(["2019-03-04", "2019-03-05", "2019-03-06"]))
    assert gaps.loc["2019-03-07", "finbert_attention_1"] == 0                      # read, no headline
    assert gaps.loc["2019-03-07":"2019-03-08", "finbert_attention_5"].isna().all()   # two of five sessions read
    assert gaps.loc["2019-03-11", "finbert_attention_5"] == 0                      # three read, no headline


def test_indicators_use_only_headlines_public_by_the_close(calendar):
    rng = np.random.default_rng(0)
    times = pd.to_datetime("2019-01-02") + pd.to_timedelta(rng.integers(0, 360 * 24, 400), unit="h")
    titles = rng.choice(["Apple beats", "Apple misses", "Apple holds"], 400)
    raw = fnspid_rows([(f"{t:%Y-%m-%d %H:%M:%S} UTC", s, "AAPL") for t, s in zip(times, titles)])
    items = finbert.fnspid_headlines([raw], calendar)
    scores = finbert.score_texts(items, pd.DataFrame(columns=["key", "positive", "negative", "neutral"]),
                                 fake_scorer)
    years = finbert.covered_years(items, calendar.sessions)  # dataset metadata, fixed once
    assert years == [2019]

    def build(chosen):
        return finbert.indicators(chosen, scores, calendar.sessions, years)

    check_point_in_time(build, items, "session", calendar.sessions[20::25])


def test_finbert_command_writes_each_sources_indicators(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    folder = store.layer("raw", "fnspid")
    folder.mkdir(parents=True)
    rows = [(f"2019-{m:02d}-04 00:00:00 UTC", f"Ford news {m}", "F") for m in range(1, 13)]
    fnspid_rows(rows + [("2019-03-04 15:00:00 UTC", "Ford misses estimates", "F"),
                        ("2019-03-04 15:00:00 UTC", "Apple beats estimates", "AAPL")]).to_parquet(folder / "a.parquet")
    for month in range(1, 13):                                  # GDELT titles: one day a month in 2024
        day = f"2024{month:02d}05"
        titles = gdelt_rows([(int(day + "150000"), "AAPL", "apple inc", "Apple beats estimates"),
                             (int(day + "150000"), "AAPL", "apple daily", "Apple Daily misses")])  # not the firm
        store.write_parquet(cli.gdelt_path(day, "titles"), titles, extraction=gdelt.EXTRACTION)
        store.write_parquet(cli.gdelt_path(day, "hourly"), titles.iloc[:0], extraction=gdelt.EXTRACTION)
    sessions = study_calendar().sessions
    cover = pd.Series(1.0, index=sessions).mask(sessions == pd.Timestamp("2024-03-05"), 0.2)
    store.write_parquet(store.layer("interim", "news", "AAPL.parquet"), cover.to_frame("coverage"))
    monkeypatch.setattr(finbert, "finbert_scorer", lambda: (fake_scorer, "abc123"))
    assert cli.main(["finbert", "--stocks", "AAPL", "F"]) == 0

    fnspid = pd.read_parquet(store.layer("processed", "finbert", "fnspid", "F.parquet"))
    assert fnspid.loc["2019-03-05", "finbert_sentiment_1"] == pytest.approx(-0.85 / 2)  # with "Ford news 3"
    assert fnspid.loc[:"2018-12-31"].isna().all().all()          # years FNSPID does not cover: unknown
    assert pd.read_parquet(store.layer("processed", "finbert", "fnspid", "AAPL.parquet")).isna().all().all()
    sidecar = json.loads(store.sidecar(store.layer("processed", "finbert", "fnspid", "F.parquet")).read_text())
    assert sidecar["covered_years"] == [2019] and sidecar["source"] == "fnspid"

    titled = pd.read_parquet(store.layer("processed", "finbert", "gdelt", "AAPL.parquet"))
    assert titled.loc["2024-02-05", "finbert_sentiment_1"] == pytest.approx(0.85)   # "Apple Daily" is left out
    assert titled.loc["2024-02-05", "finbert_attention_1"] == pytest.approx(np.log1p(1))
    assert titled.loc["2024-03-05"].isna().all()                 # GDELT read too few files that session
    assert titled.loc[:"2023-12-31"].isna().all().all()
    report = json.loads(store.layer("reports", "finbert_coverage.json").read_text())
    assert report["gdelt"]["AAPL"]["covered_years"] == [2024] and report["fnspid"]["F"]["covered_test_years"] == []
    assert len(pd.read_parquet(store.layer("interim", "finbert", "scores.parquet"))) == 14


def test_finbert_command_scores_fnspid_without_gdelt_and_rescores_for_a_new_revision(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    folder = store.layer("raw", "fnspid")
    folder.mkdir(parents=True)
    fnspid_rows([(f"2019-{m:02d}-04 00:00:00 UTC", f"Ford news {m}", "F") for m in range(1, 13)]).to_parquet(
        folder / "a.parquet")
    calls = []

    def scorer_at(revision):
        def score(texts):
            calls.append(len(texts))
            return fake_scorer(texts)
        return lambda: (score, revision)

    monkeypatch.setattr(finbert, "finbert_scorer", scorer_at("abc123"))
    assert cli.main(["finbert", "--stocks", "F"]) == 0           # no GDELT titles yet: FNSPID all the same
    assert store.layer("processed", "finbert", "fnspid", "F.parquet").exists() and calls == [12]
    assert cli.main(["finbert", "--stocks", "F"]) == 0
    assert calls == [12]                                          # same revision: the kept scores are reused
    monkeypatch.setattr(finbert, "finbert_scorer", scorer_at("def456"))
    assert cli.main(["finbert", "--stocks", "F"]) == 0
    assert calls == [12, 12]                                      # another revision: every headline again
    sidecar = json.loads(store.sidecar(store.layer("interim", "finbert", "scores.parquet")).read_text())
    assert sidecar["revision"] == "def456"
    assert cli.main(["finbert", "--stocks", "F", "--sources", "gdelt"]) == 1


def test_the_cross_check_with_gdelt_counts_fnspid_headlines_on_their_own_date(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    folder = store.layer("raw", "fnspid")
    folder.mkdir(parents=True)
    fnspid_rows([("2019-03-04 00:00:00 UTC", "Ford on Monday", "F"), ("2019-03-08 00:00:00 UTC", "Ford on Friday", "F"),
                 ("2019-03-09 00:00:00 UTC", "Ford on Saturday", "F")]).to_parquet(folder / "a.parquet")
    calendar = TradingCalendar.nyse("2019-01-01", "2019-12-31")
    counts, _ = cli.fnspid_counts(calendar, ["F"])["F"]
    assert counts.to_dict() == {pd.Timestamp("2019-03-04"): 1, pd.Timestamp("2019-03-08"): 1,
                                pd.Timestamp("2019-03-11"): 1}
