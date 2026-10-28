import pandas as pd
import pytest
import yaml

from regime_fusion.config import ConfigError
from regime_fusion.data.cli import CONSTITUENTS_FILE
from regime_fusion.data.universe import DEFAULT_UNIVERSE, load_universe


def write_universe(tmp_path, entries):
    path = tmp_path / "universe.yaml"
    path.write_text(yaml.safe_dump({"stocks": entries}))
    return path


def entry(**changes):
    base = dict(id="AAPL", name="Apple Inc.", sector="Information Technology",
                ciks=[320193], yahoo="AAPL", role="candidate", acttrade=None)
    return {**base, **changes}


def test_repository_universe_holds_the_january_2015_sp100():
    universe = load_universe()
    assert len(universe.stocks) == 100
    google = universe.get("GOOG")
    assert google.ciks == (1288776, 1652044) and google.yahoo == "GOOG"
    assert universe.get("BRK.B").yahoo == "BRK-B"
    assert universe.get("FB").yahoo == "META"
    for stock in universe.select("excluded"):
        assert stock.yahoo is None and stock.notes


def test_universe_agrees_with_the_constituents_list():
    table = pd.read_csv(CONSTITUENTS_FILE).set_index("ticker")
    universe = load_universe()
    assert {s.id for s in universe.stocks} == set(table.index[table["sp100"]])
    for stock in universe.stocks:
        row = table.loc[stock.id]
        assert stock.name == row["name"] and stock.sector == row["sector"]
        assert int(row["cik"]) in stock.ciks  # the January 2015 CIK is always kept


def test_holdout_stocks_need_an_explicit_request(tmp_path):
    universe = load_universe(write_universe(tmp_path, [entry(role="holdout")]))
    with pytest.raises(ValueError, match="untouched"):
        universe.select("holdout")
    assert [s.id for s in universe.select("holdout", include_holdout=True)] == ["AAPL"]
    assert universe.select("main") == []


@pytest.mark.parametrize(
    "entries, message",
    [
        ([entry(), entry()], "repeated stock id"),
        ([entry(), entry(id="APPL", yahoo="AAPL")], "repeated Yahoo symbol"),
        ([entry(role="core")], "role"),
        ([entry(role="main", yahoo=None)], "Yahoo symbol"),
        ([entry(role="excluded")], "notes"),
        ([entry(ciks=[])], "CIK"),
        ([entry(id="aapl")], "January 2015 ticker"),
        ([entry(acttrade="yes")], "acttrade"),
        ([entry(id=f"S{c}", yahoo=f"S{c}", acttrade=True) for c in "ABCDEFGHIJKLMNOPQRS"], "18"),
    ],
)
def test_inconsistent_universes_are_rejected(tmp_path, entries, message):
    with pytest.raises(ConfigError, match=message):
        load_universe(write_universe(tmp_path, entries))


def test_entries_missing_a_field_are_reported(tmp_path):
    incomplete = entry()
    del incomplete["sector"]
    with pytest.raises(ConfigError, match="bad stock entry"):
        load_universe(write_universe(tmp_path, [incomplete]))


def test_default_path_points_at_the_repository_file():
    assert DEFAULT_UNIVERSE.name == "universe.yaml" and DEFAULT_UNIVERSE.exists()
