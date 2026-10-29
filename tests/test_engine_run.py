import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.config import load_config
from regime_fusion.engine import cli
from regime_fusion.engine.panel import load_panel, scale
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.run import RunSpec, choose_exits, result_path, run
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
    assert cli.main(["run", "--rung", "A0", "--stock", "XYZ", "--test-year", "2019", "--campaign", "test",
                     "--population", "20", "--generations", "2"]) == 0
