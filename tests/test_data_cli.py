"""The download commands, with the network replaced by fakes."""

import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import cli, market, prices
from regime_fusion.data.store import DATA_DIR_ENV, sidecar


def fake_bars(symbol, start, **_):
    if symbol == "NOPE":
        raise prices.MissingPrices("no prices")
    days = pd.bdate_range("2014-01-02", "2014-01-31", tz="America/New_York")
    close = 50 + np.arange(len(days), dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
         "Adj Close": close, "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0},
        index=days.rename("Date"),
    )


@pytest.fixture
def offline(monkeypatch, tmp_path):
    calls = []

    def download(symbol, start, **kwargs):
        calls.append(symbol)
        return fake_bars(symbol, start, **kwargs)

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(prices, "download", download)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    vix = b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2014,14.32,14.59,14.00,14.23\n"
    monkeypatch.setattr(market, "fetch_vix", lambda url, agent: vix)
    return tmp_path, calls


def test_prices_are_downloaded_once_then_read_from_the_cache(offline):
    root, calls = offline
    assert cli.main(["prices", "--stocks", "AAPL", "BRK.B"]) == 0
    assert calls == ["AAPL", "BRK-B"]
    for stock in ("AAPL", "BRK.B"):
        assert (root / "raw" / "yahoo" / "stocks" / f"{stock}.parquet").exists()
        aligned = pd.read_parquet(root / "interim" / "prices" / f"{stock}.parquet")
        assert aligned.index[0] == pd.Timestamp("2014-01-02")
    record = json.loads(sidecar(root / "raw" / "yahoo" / "stocks" / "AAPL.parquet").read_text())
    assert record["symbol"] == "AAPL" and record["source"].startswith("Yahoo")
    report = json.loads((root / "reports" / "price_checks.json").read_text())
    assert any("before the study end" in flag for flag in report["AAPL"]["flags"])

    assert cli.main(["prices", "--stocks", "AAPL"]) == 0
    assert calls == ["AAPL", "BRK-B"]  # cached
    assert cli.main(["prices", "--stocks", "AAPL", "--refresh"]) == 0
    assert calls == ["AAPL", "BRK-B", "AAPL"]


def test_excluded_stocks_are_skipped(offline):
    _, calls = offline
    assert cli.main(["prices", "--stocks", "TWX"]) == 0
    assert calls == []


def test_missing_prices_make_the_command_fail(offline, monkeypatch):
    universe = cli.load_universe()
    stock = universe.get("AAPL")
    patched = type(universe)(tuple(
        s if s.id != "AAPL" else type(stock)(**{**stock.__dict__, "yahoo": "NOPE"})
        for s in universe.stocks
    ))
    monkeypatch.setattr(cli, "load_universe", lambda path=None: patched)
    assert cli.main(["prices", "--stocks", "AAPL", "MSFT"]) == 1


def test_market_series_are_aligned(offline):
    root, calls = offline
    assert cli.main(["market"]) == 0
    assert calls == ["^GSPC", "SPY"]
    for name in ("GSPC", "SPY", "VIX"):
        assert (root / "interim" / "market" / f"{name}.parquet").exists()
    vix = pd.read_parquet(root / "interim" / "market" / "VIX.parquet")
    assert vix.loc[pd.Timestamp("2014-01-02"), "close"] == 14.23


FACTS = {
    "cik": 320193,
    "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2013-01-01", "end": "2013-12-31", "val": 100.0, "accn": "k1", "form": "10-K",
         "filed": "2014-02-20"},
        {"start": "2014-01-01", "end": "2014-03-31", "val": 30.0, "accn": "q1", "form": "10-Q",
         "filed": "2014-05-01"},
        {"start": "2013-01-01", "end": "2013-03-31", "val": 25.0, "accn": "q1", "form": "10-Q",
         "filed": "2014-05-01"},
        {"start": "2014-01-01", "end": "2014-03-31", "val": 30.0, "accn": "e1", "form": "8-K",
         "filed": "2014-04-25"},
    ]}}}},
}


class FakeEdgar:
    urls = []

    def __init__(self, user_agent, max_per_second):
        assert "@" in user_agent

    def get(self, url):
        FakeEdgar.urls.append(url)
        if "companyfacts" in url:
            return json.dumps(FACTS).encode()
        return json.dumps({"name": "Apple Inc.", "formerNames": []}).encode()


def test_sec_and_fundamentals_commands(offline, monkeypatch):
    root, _ = offline
    FakeEdgar.urls = []
    monkeypatch.setattr(cli.sec, "Client", FakeEdgar)
    assert cli.main(["sec", "--stocks", "AAPL"]) == 0
    assert FakeEdgar.urls == [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/submissions/CIK0000320193.json",
    ]
    facts = pd.read_parquet(root / "interim" / "sec" / "AAPL.parquet")
    assert sorted(facts["form"].unique()) == ["10-K", "10-Q"]  # the 8-K is dropped
    assert facts["available"].min() == pd.Timestamp("2014-02-21")
    report = json.loads((root / "reports" / "sec_checks.json").read_text())
    assert report["AAPL"]["reports_per_year"]["2014"] == 2

    assert cli.main(["sec", "--stocks", "AAPL"]) == 0
    assert len(FakeEdgar.urls) == 2  # cached

    assert cli.main(["fundamentals", "--stocks", "AAPL"]) == 0
    panel = pd.read_parquet(root / "interim" / "fundamentals" / "AAPL.parquet")
    assert panel.loc[pd.Timestamp("2014-05-02"), "revenue_ttm"] == 105.0


def test_fundamentals_need_the_sec_step_first(offline):
    assert cli.main(["fundamentals", "--stocks", "MSFT"]) == 1


def test_unknown_stock_ids_are_reported(offline):
    with pytest.raises(SystemExit, match="META"):
        cli.main(["prices", "--stocks", "META"])  # Facebook's id is its 2015 ticker, FB


def test_reports_from_partial_runs_are_merged(offline):
    root, _ = offline
    assert cli.main(["prices", "--stocks", "AAPL"]) == 0
    assert cli.main(["prices", "--stocks", "MSFT"]) == 0
    report = json.loads((root / "reports" / "price_checks.json").read_text())
    assert set(report) == {"AAPL", "MSFT"}
