import json
import shutil

import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import cli, draw, market, prices
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.data.universe import load_universe

SP100_SHARES = {"Information Technology": 0.16, "Financials": 0.15, "Industrials": 0.14,
                "Health Care": 0.13, "Consumer Discretionary": 0.13, "Consumer Staples": 0.11,
                "Energy": 0.10, "Materials": 0.04, "Utilities": 0.02,
                "Telecommunications Services": 0.02}


def rng(seed=1):
    return np.random.default_rng(seed)


def test_shuffles_depend_only_on_the_seed_and_the_set_of_ids():
    ids = ["C", "A", "B", "D"]
    assert draw.shuffled(ids, rng(3)) == draw.shuffled(reversed(ids), rng(3))
    assert sorted(draw.shuffled(ids, rng(3))) == sorted(ids)
    orders = {tuple(draw.shuffled(ids, rng(seed))) for seed in range(20)}
    assert len(orders) > 1


def test_allocation_tops_the_sector_mix_up_to_the_index():
    # ActTrade's 17 usable stocks: 8 in technology, 6 consumer discretionary, 3 staples.
    fixed = {"Information Technology": 8, "Consumer Discretionary": 6, "Consumer Staples": 3}
    available = {sector: 10 for sector in SP100_SHARES}
    counts = draw.allocate(SP100_SHARES, fixed, 50, available, rng())
    assert sum(counts.values()) == 33
    assert counts["Information Technology"] == 0          # already at its 8 of 50
    assert counts["Financials"] in (7, 8) and counts["Industrials"] == 7
    assert counts["Energy"] == 5 and counts["Materials"] == 2
    assert counts["Utilities"] == 1 and counts["Telecommunications Services"] == 1
    total = {s: counts[s] + fixed.get(s, 0) for s in SP100_SHARES}
    assert all(abs(total[s] - 50 * SP100_SHARES[s]) <= 1 for s in SP100_SHARES)


def test_allocation_respects_what_each_sector_has():
    counts = draw.allocate({"A": 0.5, "B": 0.5}, {}, 10, {"A": 2, "B": 20}, rng())
    assert counts == {"A": 2, "B": 8}
    with pytest.raises(ValueError):
        draw.allocate({"A": 0.5, "B": 0.5}, {}, 10, {"A": 2, "B": 3}, rng())


def test_unusable_stocks_are_skipped_in_order():
    taken, skipped = draw.take_usable(["A", "B", "C", "D"], 2, lambda i: i != "B")
    assert taken == ["A", "C"] and skipped == ["B"]
    with pytest.raises(ValueError):
        draw.take_usable(["A", "B"], 2, lambda i: i == "A")


def test_main_draw_is_reproducible_and_follows_the_allocation():
    candidates = {f"S{chr(65 + i)}": sector for i, sector in enumerate(
        ["Financials"] * 5 + ["Energy"] * 5 + ["Information Technology"] * 5)}
    fixed = {"AAPL": "Information Technology"}
    shares = {"Financials": 0.4, "Energy": 0.4, "Information Technology": 0.2}
    first = draw.draw_main(candidates, fixed, shares, 6, rng(7), lambda i: True)
    again = draw.draw_main(candidates, fixed, shares, 6, rng(7), lambda i: True)
    assert first == again
    drawn, skipped = first
    # Targets for 6 stocks: 2.4, 2.4 and 1.2; AAPL leaves technology 0.2 short, so the
    # five places go 2 + 2 to financials and energy, and the fifth to one of them at random.
    sectors = pd.Series([candidates[i] for i in drawn]).value_counts().to_dict()
    assert sorted(sectors.values()) == [2, 3] and "Information Technology" not in sectors
    assert skipped == []
    unusable = draw.draw_main(candidates, fixed, shares, 6, rng(7), lambda i: i != drawn[0])
    assert drawn[0] in unusable[1] and drawn[0] not in unusable[0]


def test_ticker_candidates_prefer_the_old_ticker_then_common_shares():
    assert draw.ticker_candidates("BF.B", ["BF-A", "BF-B"]) == ["BF-B", "BF-A"]
    assert draw.ticker_candidates("XRX", ["XRXDW", "XRX"]) == ["XRX", "XRXDW"]
    assert draw.ticker_candidates("OLD", ["NEWCO", "NEW"]) == ["NEW", "NEWCO"]
    assert draw.ticker_candidates("DD", ["CTA-PA", "CTA-PB"]) == []  # preferred shares only


def test_an_old_ticker_taken_over_by_another_2015_firm_is_rejected(monkeypatch):
    row = pd.Series({"cik": 20171, "name": "Chubb Corp."})
    monkeypatch.setattr(prices, "yahoo_name", lambda symbol: "Chubb Limited")
    symbol, ciks, why = cli.resolve_holdout("CB", row, {}, {"CB": 896159}, lambda s: True,
                                            {20171: "CB", 896159: "ACE"})
    assert symbol is None and "ACE" in why
    # A reorganised firm (BlackRock's new holding company in 2024) keeps its ticker.
    row = pd.Series({"cik": 1364742, "name": "BlackRock"})
    monkeypatch.setattr(prices, "yahoo_name", lambda symbol: "BlackRock, Inc.")
    symbol, ciks, why = cli.resolve_holdout("BLK", row, {}, {"BLK": 2012383}, lambda s: True,
                                            {1364742: "BLK"})
    assert symbol == "BLK" and ciks == (1364742, 2012383)


def full_bars(symbol, start, **_):
    if symbol.startswith("ZZ"):
        raise prices.MissingPrices(symbol)
    days = pd.bdate_range("2014-01-02", "2015-03-31", tz="America/New_York")
    first = 15 if symbol.startswith("LATE") else 0  # a series that starts too late
    close = np.linspace(20.0, 30.0, len(days))[first:]
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Adj Close": close,
         "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0},
        index=days[first:].rename("Date"),
    )


class FakeEdgar:
    """SEC's current tickers: every January 2015 CIK keeps its ticker, except a few."""

    def __init__(self, user_agent, max_per_second):
        table = pd.read_csv(cli.CONSTITUENTS_FILE).drop_duplicates("cik")
        self.tickers = {str(i): {"cik_str": int(row.cik), "ticker": row.ticker.replace(".", "-"),
                                 "title": row.name} for i, row in enumerate(table.itertuples())}

    def get(self, url):
        return json.dumps(self.tickers).encode()


def test_draw_command_offline(monkeypatch, tmp_path):
    universe_file = tmp_path / "universe.yaml"
    shutil.copy(cli.UNIVERSE_FILE, universe_file)
    monkeypatch.setattr(cli, "UNIVERSE_FILE", universe_file)
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(prices, "download", full_bars)
    monkeypatch.setattr(prices, "yahoo_name", lambda symbol: None)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(cli.sec, "Client", FakeEdgar)
    monkeypatch.setattr(market, "fetch_vix", lambda url, agent: b"")

    with pytest.raises(SystemExit, match="drawn already"):  # the repository's file is drawn
        cli.main(["draw"])
    assert cli.main(["draw", "--redraw"]) == 0
    universe = load_universe(universe_file)
    main = universe.select("main")
    holdout = universe.select("holdout", include_holdout=True)
    assert len(main) == 50 and len(holdout) == 50
    acttrade = [s for s in main if s.acttrade]
    assert len(acttrade) == 17 and universe.get("ATVI").role == "excluded"
    assert universe.get("KO").acttrade and universe.get("GOOGL").role == "main"
    assert "GOOG" not in {s.id for s in universe.stocks}  # the same firm as GOOGL
    sp100 = set(pd.read_csv(cli.CONSTITUENTS_FILE).query("sp100")["ticker"])
    assert not {s.id for s in holdout} & sp100
    assert not {c for s in holdout for c in s.ciks} & {c for s in main for c in s.ciks}

    first_draw = universe_file.read_text()
    assert cli.main(["draw", "--redraw"]) == 0
    assert universe_file.read_text() == first_draw  # the same seeds give the same draw
