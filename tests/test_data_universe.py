import pandas as pd
import pytest
import yaml

from regime_fusion.config import ConfigError, load_config
from regime_fusion.data import draw
from regime_fusion.data.cli import CONSTITUENTS_FILE
from regime_fusion.data.universe import DEFAULT_UNIVERSE, load_universe
from regime_fusion.seeds import make_rng


def write_universe(tmp_path, entries):
    path = tmp_path / "universe.yaml"
    path.write_text(yaml.safe_dump({"stocks": entries}))
    return path


def entry(**changes):
    base = dict(id="AAPL", name="Apple Inc.", sector="Information Technology",
                ciks=[320193], yahoo="AAPL", role="candidate", acttrade=None)
    return {**base, **changes}


def test_repository_universe_is_drawn():
    universe = load_universe()
    main = universe.select("main")
    holdout = universe.select("holdout", include_holdout=True)
    assert len(main) == 50 and len(holdout) == 50
    assert sum(bool(s.acttrade) for s in main) == 17
    assert universe.get("ATVI").role == "excluded" and universe.get("ATVI").acttrade
    google = universe.get("GOOGL")
    assert google.role == "main" and google.ciks == (1288776, 1652044)
    assert universe.get("BRK.B").yahoo == "BRK-B" and universe.get("FB").yahoo == "META"
    assert not {c for s in main for c in s.ciks} & {c for s in holdout for c in s.ciks}
    for stock in universe.select("excluded"):
        assert stock.yahoo is None and stock.notes


def test_main_universe_mirrors_the_sector_mix_of_the_sp100():
    table = pd.read_csv(CONSTITUENTS_FILE)
    members = table[table["sp100"]]
    shares = members["sector"].value_counts() / len(members)
    counts = pd.Series([s.sector for s in load_universe().select("main")]).value_counts()
    for sector, share in shares.items():
        assert abs(counts.get(sector, 0) - 50 * share) <= 1, sector


def test_main_draw_is_reproduced_by_its_seed():
    universe = load_universe()
    table = pd.read_csv(CONSTITUENTS_FILE)
    members = table[table["sp100"]]
    shares = (members["sector"].value_counts() / len(members)).to_dict()
    fixed = {s.id: s.sector for s in universe.select("main") if s.acttrade}
    candidates = {s.id: s.sector for s in universe.select("main", "unused") if not s.acttrade}
    drawn, _ = draw.draw_main(candidates, fixed, shares, 50,
                              make_rng(load_config().base_seed, "universe", "main"), lambda i: True)
    assert sorted(drawn) == sorted(s.id for s in universe.select("main") if not s.acttrade)


def test_universe_agrees_with_the_constituents_list():
    table = pd.read_csv(CONSTITUENTS_FILE).set_index("ticker")
    universe = load_universe()
    acttrade_extras = {"ATVI", "KODK", "NYT", "TSLA"}  # not in the S&P 500 in January 2015
    ids = {s.id for s in universe.stocks}
    assert set(table.index[table["sp100"]]) - {"GOOG"} <= ids  # GOOG: the same firm as GOOGL
    for stock in universe.stocks:
        if stock.id in acttrade_extras:
            assert stock.acttrade
            continue
        row = table.loc[stock.id]
        assert stock.name == row["name"] and stock.sector == row["sector"]
        assert int(row["cik"]) in stock.ciks  # the January 2015 CIK is always kept
    for stock in universe.select("holdout", include_holdout=True):
        assert not table.loc[stock.id, "sp100"]


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
