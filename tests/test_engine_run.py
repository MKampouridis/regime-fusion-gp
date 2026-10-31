import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.config import load_config
from regime_fusion.engine import cli
from regime_fusion.engine.panel import frame_digest, load_panel, scale
from regime_fusion.engine.rungs import finbert_source, load_rungs
from regime_fusion.engine.run import (FINGERPRINT_FORMAT, RunSpec, choose_exits, exits_path, result_path, run,
                                      unpack_signal)
from regime_fusion.engine.tree import from_record, signals
from regime_fusion.windows import walk_forward
from regime_fusion.data import store
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.windows import DateRange

SMALL = {"population": 30, "generations": 3}


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """Prices, indicators and regime probabilities of one stock, 2014-2019."""
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    days = pd.bdate_range("2014-01-02", "2019-12-31", name="session")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(days))))
    prices = pd.DataFrame({"adj_open": close, "adj_high": close * 1.01, "adj_low": close * 0.99,
                           "adj_close": close}, index=days)
    for folder in ("interim/prices",):
        (tmp_path / folder).mkdir(parents=True)
    prices.to_parquet(tmp_path / "interim/prices/XYZ.parquet")
    for kind, n in (("fa", 3), ("sa", 2), ("ta", 3)):
        (tmp_path / "processed" / kind).mkdir(parents=True)
        table = pd.DataFrame({f"x{i}": rng.normal(size=len(days)) for i in range(n)}, index=days)
        table.to_parquet(tmp_path / "processed" / kind / "XYZ.parquet")
    folder = tmp_path / "processed" / "regimes" / "volatility"
    folder.mkdir(parents=True)
    p1 = rng.uniform(0, 1, len(days))
    for stage in (1, 2):
        pd.DataFrame({"p_state0": 1 - p1, "p_state1": p1}, index=days).to_parquet(folder / f"2019.stage{stage}.parquet")
    return tmp_path


def test_scaling_is_fitted_on_training_rows_only():
    days = pd.bdate_range("2018-01-01", periods=10)
    raw = pd.DataFrame({"fa:x": np.arange(10.0), "fa:thin": [1.0] + [np.nan] * 9}, index=days)
    train = days[:5]
    scaled = scale(raw, train, "minmax")
    assert list(scaled.columns) == ["fa:x"]  # the thin column is dropped
    np.testing.assert_allclose(scaled["fa:x"].iloc[:5], [-1, -0.5, 0, 0.5, 1])
    assert scaled["fa:x"].iloc[9] == 3.5      # 2 * 9 / 4 - 1: later values can leave [-1, 1]
    changed = raw.copy()
    changed.iloc[5:] *= 100
    pd.testing.assert_frame_equal(scale(changed, train, "rank").iloc[:5], scale(raw, train, "rank").iloc[:5])
    assert scale(raw, train, "rank")["fa:x"].iloc[9] == 1.0


def test_panel_brings_indicators_prices_and_regimes_together(data_dir):
    regimes = data_dir / "processed/regimes/volatility/2019.stage2.parquet"
    panel = load_panel("XYZ", DateRange(pd.Timestamp("2015-02-01").date(), pd.Timestamp("2018-12-31").date()),
                       DateRange(pd.Timestamp("2019-01-01").date(), pd.Timestamp("2019-12-31").date()),
                       ("FA", "SA", "TA"), regimes)
    names = panel.names()
    assert len(names["FA"]) == 3 and len(names["SA"]) == 2 and len(names["RG"]) == 2
    assert panel.train[-1] < panel.evaluate[0] and panel.evaluate[-1] == pd.Timestamp("2019-12-31")
    probability = pd.read_parquet(regimes)["p_state1"].loc[panel.evaluate[0]]
    assert panel.indicators.loc[panel.evaluate[0], "rg:p_state1"] == pytest.approx(2 * probability - 1)


def test_runs_are_reproducible_and_written_with_provenance(data_dir):
    config, rungs = load_config(), load_rungs()
    spec = RunSpec("XYZ", 2019, 3, 2, 10, 0.02)
    first = run(spec, rungs["A0"], config, campaign="test", overrides=SMALL)
    again = run(spec, rungs["A0"], config, campaign="test", overrides=SMALL, write=False)
    assert first["model"] == again["model"] and first["evaluation"] == again["evaluation"]
    assert set(first["evaluation"]) == {"0bps", "5bps", "10bps", "25bps"}
    assert first["evaluate"] == ["2019-01-01", "2019-12-31"] and first["train"][1] == "2018-12-31"
    saved = json.loads(result_path("test", "A0", spec).read_text())
    assert saved["provenance"]["rung_digest"] == rungs["A0"].digest()
    mark = saved["fingerprint"]                                  # what the result depends on
    assert mark["config"] == config.digest() and mark["method"] == rungs["A0"].digest(None, SMALL)
    assert mark["method"] != rungs["A0"].digest()                # overridden settings are other settings
    window = next(w for w in walk_forward(config) if w.test_year == 2019)
    first_day = pd.read_parquet(data_dir / "interim/prices/XYZ.parquet").loc[str(window.final_train.start):].index[0]
    years = range(first_day.year, 2020)                          # every year the run reads, whole
    assert set(mark["inputs"]) == {f"{name}@{year}" for year in years for name in (
        "interim/prices/XYZ.parquet", *(f"processed/{k}/XYZ.parquet" for k in ("fa", "sa", "ta")))}
    ta = pd.read_parquet(data_dir / "processed/ta/XYZ.parquet")
    assert mark["inputs"]["processed/ta/XYZ.parquet@2017"] == frame_digest(ta.loc["2017"])
    assert mark["format"] == FINGERPRINT_FORMAT
    panel = load_panel("XYZ", window.final_train, window.test, ("FA", "SA", "TA"))
    assert mark["evaluation_prices"] == frame_digest(panel.prices.loc[panel.evaluate])
    assert panel.inputs == mark["inputs"]
    exact = signals(from_record(saved["tree"]), panel.arrays(panel.evaluate))["C"]   # the model, kept exactly
    np.testing.assert_array_equal(exact, unpack_signal(saved["signal"]))
    other = run(RunSpec("XYZ", 2019, 4, 2, 10, 0.02), rungs["A0"], config, overrides=SMALL, write=False)
    assert other["model"] != first["model"]  # another run number, another seed


def test_regime_rung_and_single_type_baseline(data_dir):
    config, rungs = load_config(), load_rungs()
    spec = RunSpec("XYZ", 2019, 0, 2, 10, 0.02)
    gated = run(spec, rungs["A2"], config, overrides=SMALL, write=False)
    assert gated["indicators"]["RG"] == 2
    baseline = run(spec, rungs["GP-TA"], config, overrides=SMALL, write=False)
    assert baseline["sources"] == ["TA"] and baseline["model"].startswith("fusion: TA")


def test_exit_settings_are_chosen_on_the_validation_year(data_dir):
    config, rungs = load_config(), load_rungs()
    grid = {"days": [5, 10], "rates": [0.02], "runs": 2}
    choice = choose_exits("XYZ", 2019, rungs["A0"], config, grid, campaign="test", overrides=SMALL)
    assert (choice["days"], choice["rate"]) in {(5, 0.02), (10, 0.02)}
    assert len(choice["candidates"]) == 2
    made = json.loads(exits_path("test", "XYZ", 2019).read_text())["provenance"]
    assert made["overrides"] == SMALL and made["method_digest"] == rungs["A0"].digest(None, SMALL)
    assert cli.main(["run", "--rung", "A0", "--stock", "XYZ", "--test-year", "2019", "--campaign", "test",
                     "--population", "20", "--generations", "2"]) == 0


def test_a_source_without_usable_indicators_is_reported(data_dir):
    days = pd.read_parquet(data_dir / "processed/sa/XYZ.parquet").index
    pd.DataFrame({"x0": np.nan}, index=days).to_parquet(data_dir / "processed/sa/XYZ.parquet")
    train, test = DateRange(pd.Timestamp("2015-01-01").date(), pd.Timestamp("2017-12-31").date()), \
        DateRange(pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-12-31").date())
    with pytest.raises(ValueError, match="no SA indicator"):
        load_panel("XYZ", train, test, ("FA", "SA", "TA"))
    assert load_panel("XYZ", train, test, ("FA", "TA")).names()["SA"] == []


def test_robustness_rungs_add_finbert_to_the_sentiment_branch(data_dir):
    days = pd.read_parquet(data_dir / "processed/sa/XYZ.parquet").index
    table = pd.DataFrame({"finbert_sentiment_1": np.linspace(-1, 1, len(days))}, index=days)
    store.write_parquet(store.layer("processed", "finbert", "fnspid", "XYZ.parquet"), table,
                        covered_years=[2014, 2015, 2016, 2017, 2018])
    rungs = load_rungs()
    assert rungs["A0-FB"].finbert and not rungs["A0"].finbert and rungs["A1-FB"].fusion == rungs["A1"].fusion
    assert [finbert_source(year) for year in (2019, 2023, 2024, 2025)] == ["fnspid", "fnspid", "gdelt", "gdelt"]
    with pytest.raises(ValueError, match="no FinBERT headline source"):
        finbert_source(2030)
    train, test = DateRange(pd.Timestamp("2015-01-01").date(), pd.Timestamp("2017-12-31").date()), \
        DateRange(pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-12-31").date())
    panel = load_panel("XYZ", train, test, ("FA", "SA", "TA"), finbert="fnspid")
    assert "sa:finbert_sentiment_1" in panel.names()["SA"] and len(panel.names()["SA"]) == 3
    record = run(RunSpec("XYZ", 2019, 0, 1, 10, 0.02), rungs["A0-FB"], load_config(), write=False,
                 overrides=SMALL)
    assert record["indicators"]["SA"] == 3 and record["finbert"] == "fnspid"
    with pytest.raises(ValueError, match=r"do not cover \[2019\]"):   # the test year is not covered
        run(RunSpec("XYZ", 2019, 0, 2, 10, 0.02), rungs["A0-FB"], load_config(), write=False, overrides=SMALL)


def test_pilot_alternatives_are_rungs_of_their_own():
    from regime_fusion.regimes import market

    rungs = load_rungs()
    assert rungs["A1-rank"].scaling == "rank" and rungs["A1"].scaling == "minmax"
    assert rungs["A3-fine"].regimes == "dc_fine" and rungs["A4-wide"].regimes == "dc_news_wide"
    assert rungs["A1-rank"].digest() != rungs["A1"].digest()
    settings = market.load_settings()
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 600))),
                      index=pd.bdate_range("2016-01-01", periods=600))
    assert settings["inputs"]["dc"]["threshold"] == settings["inputs"]["dc_fine"]["threshold"]  # A3 took 0.5%
    fine = market.inputs(close, "dc_fine", settings)
    pilot_a3 = {**settings["inputs"]["dc"], "threshold": 0.01}  # the pilot's A3
    coarse = market.inputs(close, "dc", {**settings, "inputs": {**settings["inputs"], "dc": pilot_a3}})
    assert list(fine.columns) == list(coarse.columns)
    assert fine["log_events"].mean() > coarse["log_events"].mean()  # more events at a smaller threshold


def test_gp_pilot_candidates_share_one_budget():
    rungs = load_rungs()
    for name, population, generations, tournament in (("A1-p250-t2", 250, 400, 2), ("A3-p2000-t4", 2000, 50, 4),
                                                     ("A1-p1000-t4", 1000, 100, 4)):
        gp = rungs[name].gp
        assert (gp.population, gp.generations, gp.tournament) == (population, generations, tournament)
    assert rungs["A3-p500-t2"].regimes == "dc" and rungs["A1-p500-t2"].fusion == rungs["A1"].fusion
    assert rungs["A1-p500-t2"].spec["gp"]["population"] == 500  # recorded in the provenance too
    assert rungs["A1"].gp.population == 2000 and rungs["A1-p500-t2"].digest() != rungs["A1-p500-t4"].digest()


def test_validation_checkpoints_are_kept_in_stage_one_only(data_dir):
    rung, config = load_rungs()["A1"], load_config()
    small = {"population": 30, "generations": 25}
    record = run(RunSpec("XYZ", 2019, 0, 1, 10, 0.02), rung, config, write=False, overrides=small)
    assert [c["generation"] for c in record["checkpoints"]] == [0, 10, 20, 25]
    last = record["checkpoints"][-1]
    assert last["best_fitness"] == record["fitness"]
    assert last["sharpe"] == pytest.approx(record["evaluation"]["10bps"]["sharpe"])  # the model itself
    test_year = run(RunSpec("XYZ", 2019, 0, 2, 10, 0.02), rung, config, write=False, overrides=small)
    assert test_year["checkpoints"] == []


def test_a_session_without_a_bar_is_refused_wherever_prices_are_read(data_dir):
    from regime_fusion.analysis import portfolio
    from regime_fusion.engine import benchmarks
    from regime_fusion.ml.models import load_settings

    path = data_dir / "interim/prices/XYZ.parquet"
    prices = pd.read_parquet(path)
    prices.loc["2018-03-07", "adj_close"] = np.nan          # dropping it would shift every count of days
    prices.to_parquet(path)
    train, test = DateRange(pd.Timestamp("2015-01-01").date(), pd.Timestamp("2017-12-31").date()), \
        DateRange(pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-12-31").date())
    with pytest.raises(ValueError, match="without a full bar"):
        load_panel("XYZ", train, test, ("FA", "TA"))
    settings = load_settings()
    with pytest.raises(ValueError, match="without a full bar"):
        benchmarks.run_rule(RunSpec("XYZ", 2019, 0, 1, 10, 0.03), settings.rules["RB-MA"], settings,
                            load_config(), write=False)
    with pytest.raises(ValueError, match="without a full bar"):
        portfolio.stock_prices("XYZ")
