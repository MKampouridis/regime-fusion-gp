import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import cli, finbert, store
from regime_fusion.data.calendar import TradingCalendar
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.testing import check_point_in_time


def fnspid_rows(rows):
    return pd.DataFrame(rows, columns=["Date", "Article_title", "stock"])


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
    items = finbert.headlines([raw], calendar)
    assert len(items) == 3 and "Apple old news" not in set(items["title"])
    assert (items["session"] == pd.Timestamp("2019-03-05")).all()     # every one counts the next session


def test_scores_are_kept_and_indicators_built(calendar):
    raw = fnspid_rows([("2019-03-01 00:00:00 UTC", "Apple beats estimates", "AAPL"),
                       ("2019-03-04 00:00:00 UTC", "Apple misses on iPhone", "AAPL"),
                       ("2019-03-04 00:00:00 UTC", "Apple beats estimates", "AAPL"),
                       ("2019-03-11 00:00:00 UTC", "Apple holds", "AAPL")])
    items = finbert.headlines([raw], calendar)
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_scorer(texts)

    scores = finbert.score_texts(items, pd.DataFrame(columns=["key", "positive", "negative", "neutral"]), counting)
    assert calls == [3] and len(scores) == 3                     # each distinct headline once
    again = finbert.score_texts(items, scores, counting)
    assert calls == [3] and len(again) == 3                       # nothing new to score
    window = finbert.coverage(items)
    assert window == (pd.Timestamp("2019-03-04"), pd.Timestamp("2019-03-12"))
    table = finbert.indicators(items, scores, calendar.sessions, window)
    assert list(table.columns) == finbert.COLUMNS
    assert table.loc["2019-03-04", "finbert_sentiment_1"] == pytest.approx(0.85)
    assert table.loc["2019-03-05", "finbert_sentiment_1"] == pytest.approx(0.0)
    assert table.loc["2019-03-05", "finbert_attention_1"] == pytest.approx(np.log1p(2))
    assert table.loc["2019-03-05", "finbert_sentiment_5"] == pytest.approx((0.85 - 0.85 + 0.85) / 3)
    assert np.isnan(table.loc["2019-03-06", "finbert_sentiment_1"]) and table.loc["2019-03-06", "finbert_attention_1"] == 0
    assert table.loc[:"2019-03-01"].isna().all().all()             # before FNSPID covers the stock: unknown
    assert table.loc["2019-03-13":].isna().all().all()             # after its last headline: unknown, not zero
    assert finbert.indicators(items.iloc[:0], scores, calendar.sessions, None).isna().all().all()


def test_indicators_use_only_headlines_public_by_the_close(calendar):
    rng = np.random.default_rng(0)
    times = pd.to_datetime("2019-01-02") + pd.to_timedelta(rng.integers(0, 360 * 24, 400), unit="h")
    titles = rng.choice(["Apple beats", "Apple misses", "Apple holds"], 400)
    raw = fnspid_rows([(f"{t:%Y-%m-%d %H:%M:%S} UTC", s, "AAPL") for t, s in zip(times, titles)])
    items = finbert.headlines([raw], calendar)
    scores = finbert.score_texts(items, pd.DataFrame(columns=["key", "positive", "negative", "neutral"]),
                                 fake_scorer)
    window = finbert.coverage(items)  # dataset metadata, fixed once

    def build(chosen):
        return finbert.indicators(chosen, scores, calendar.sessions, window)

    check_point_in_time(build, items, "session", calendar.sessions[20::25])


def test_finbert_command_writes_indicators(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    folder = store.layer("raw", "fnspid")
    folder.mkdir(parents=True)
    fnspid_rows([("2019-03-04 15:00:00 UTC", "Apple beats estimates", "AAPL"),
                 ("2019-03-04 15:00:00 UTC", "Ford misses estimates", "F")]).to_parquet(folder / "a.parquet")
    monkeypatch.setattr(finbert, "finbert_scorer", lambda: (fake_scorer, "abc123"))
    assert cli.main(["finbert", "--stocks", "AAPL", "F"]) == 0
    table = pd.read_parquet(store.layer("processed", "finbert", "F.parquet"))
    assert table.loc["2019-03-05", "finbert_sentiment_1"] == pytest.approx(-0.85)  # the session after its date
    assert len(pd.read_parquet(store.layer("interim", "finbert", "scores.parquet"))) == 2
