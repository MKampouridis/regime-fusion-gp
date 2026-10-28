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
    monkeypatch.setattr(cli, "load_universe", lambda: patched)
    assert cli.main(["prices", "--stocks", "AAPL", "MSFT"]) == 1


def test_market_series_are_aligned(offline):
    root, calls = offline
    assert cli.main(["market"]) == 0
    assert calls == ["^GSPC", "SPY"]
    for name in ("GSPC", "SPY", "VIX"):
        assert (root / "interim" / "market" / f"{name}.parquet").exists()
    vix = pd.read_parquet(root / "interim" / "market" / "VIX.parquet")
    assert vix.loc[pd.Timestamp("2014-01-02"), "close"] == 14.23
