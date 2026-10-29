import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")

from regime_fusion.config import load_config  # noqa: E402
from regime_fusion.data.store import DATA_DIR_ENV  # noqa: E402
from regime_fusion.engine import cli  # noqa: E402
from regime_fusion.engine.backtest import backtest  # noqa: E402
from regime_fusion.engine.run import RunSpec, result_path  # noqa: E402
from regime_fusion.ml import models  # noqa: E402
from regime_fusion.ml.labels import trade_returns  # noqa: E402
from regime_fusion.ml.run import run_ml  # noqa: E402


def ohlc(rows):
    frame = pd.DataFrame(rows, columns=["adj_open", "adj_high", "adj_low", "adj_close"], dtype=float)
    frame.index = pd.bdate_range("2018-01-01", periods=len(frame))
    return frame


def test_labels_are_the_trades_the_backtest_would_make():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    open_ = close * np.exp(rng.normal(0, 0.01, 300))
    prices = ohlc(np.column_stack([open_, np.maximum(open_, close) * 1.01, np.minimum(open_, close) * 0.99, close]))
    prices.iloc[50, :] = np.nan  # a session without a bar
    outcome = trade_returns(prices, 5, 0.03, 10)
    for t in (0, 10, 45, 48, 120, 250):
        signal = np.zeros(len(prices), dtype=bool)
        signal[t] = True
        trades = backtest(signal, prices, 5, 0.03, 10).trades
        assert trades["net_return"].iloc[0] == pytest.approx(outcome.iloc[t])
    for t in range(len(prices) - 6, len(prices) - 1):  # near the end, some trades cannot finish
        signal = np.zeros(len(prices), dtype=bool)
        signal[t] = True
        trade = backtest(signal, prices, 5, 0.03, 10).trades.iloc[0]
        if trade["reason"] == "end":  # the backtest closes it on the last day; it has no label
            assert np.isnan(outcome.iloc[t])
        else:
            assert trade["net_return"] == pytest.approx(outcome.iloc[t])
    assert np.isnan(outcome.iloc[-1])
    assert np.isnan(outcome.iloc[49])  # the next session has no bar to buy at


def test_labels_by_hand():
    prices = ohlc([[10, 10, 10, 10], [10, 10.4, 9.9, 10.1], [10.2, 10.6, 10.1, 10.5], [10, 10, 9, 9.5],
                   [9.5, 9.6, 9.4, 9.5]])
    outcome = trade_returns(prices, 2, 0.05, 0)
    assert outcome.iloc[0] == pytest.approx(10.5 / 10 - 1)     # target 10.5 reached on day 2
    assert outcome.iloc[1] == pytest.approx(9.5 / 10.2 - 1)    # time exit at the close of day 2
    assert outcome.iloc[2] == pytest.approx(9.5 / 10 - 1)
    assert np.isnan(outcome.iloc[3]) and np.isnan(outcome.iloc[4])


def test_models_learn_a_clear_pattern_and_regimes_split_the_data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 3))
    states = (rng.random(600) < 0.5).astype(int)
    y = np.where(states == 0, x[:, 0] > 0, x[:, 1] > 0).astype(int)  # the useful feature depends on the regime
    settings = models.load_settings()
    for name in ("ML-LR", "ML-LGBM", "ML-RS-LGBM"):
        spec = settings.models[name]
        learner = models.build(spec, spec.candidates()[-1], seed=3).fit(x, y, states)
        accuracy = ((learner.predict(x, states) > 0.5) == y).mean()
        assert accuracy > (0.9 if name == "ML-RS-LGBM" else 0.6), name
    constant = models.build(settings.models["ML-LGBM"], {"num_leaves": 7}, seed=3).fit(x, np.ones(600, int))
    assert (constant.predict(x) == 1.0).all()


def test_candidates_cover_the_grid_in_a_fixed_order():
    spec = models.ModelSpec("m", "lightgbm", 1, {"b": [1, 2], "a": [3]})
    assert spec.candidates() == [{"a": 3, "b": 1}, {"a": 3, "b": 2}]


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """A stock whose first indicator foretells good trades in the training and later years."""
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    days = pd.bdate_range("2014-01-02", "2019-12-31", name="session")
    rng = np.random.default_rng(4)
    signal = rng.normal(size=len(days))
    drift = np.where(np.roll(signal, 1) > 0.5, 0.01, -0.002)  # a high value today: a rise tomorrow
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, len(days))))
    open_ = np.r_[close[0], close[:-1]]  # each day opens at the previous close: the move comes after the open
    prices = pd.DataFrame({"adj_open": open_, "adj_high": np.maximum(open_, close) * 1.004,
                           "adj_low": np.minimum(open_, close) * 0.996, "adj_close": close}, index=days)
    (tmp_path / "interim/prices").mkdir(parents=True)
    prices.to_parquet(tmp_path / "interim/prices/XYZ.parquet")
    for kind in ("fa", "sa", "ta"):
        (tmp_path / "processed" / kind).mkdir(parents=True)
        table = pd.DataFrame({"x0": signal if kind == "ta" else rng.normal(size=len(days)),
                              "x1": rng.normal(size=len(days))}, index=days)
        table.to_parquet(tmp_path / "processed" / kind / "XYZ.parquet")
    folder = tmp_path / "processed" / "regimes" / "dc"
    folder.mkdir(parents=True)
    p1 = (np.arange(len(days)) // 120 % 2).astype(float)
    for stage in (1, 2):
        pd.DataFrame({"p_state0": 1 - p1, "p_state1": p1}, index=days).to_parquet(folder / f"2019.stage{stage}.parquet")
    return tmp_path


def test_a_benchmark_is_tuned_on_validation_and_tested_on_the_test_year(data_dir):
    config, settings = load_config(), models.load_settings()
    spec = RunSpec("XYZ", 2019, 0, 2, 1, 0.5)  # sell at the next close: the label is the next day's move
    record = run_ml(spec, settings.models["ML-LGBM"], settings, config, campaign="t")
    assert record["evaluate"] == ["2019-01-01", "2019-12-31"] and record["train"][1] == "2018-12-31"
    assert record["indicators"]["RG"] == 2 and record["indicators"]["TA"] == 2
    assert len(record["candidates"]) == len(settings.models["ML-LGBM"].candidates()) * len(settings.quantiles)
    assert record["validation_sharpe"] == max(c["sharpe"] for c in record["candidates"])
    assert record["evaluation"]["10bps"]["sharpe"] > 1.0  # it found the pattern out of sample
    saved = json.loads(result_path("t", "ML-LGBM", spec).read_text())
    assert saved["provenance"]["ml_settings_digest"] == settings.digest()
    again = run_ml(spec, settings.models["ML-LGBM"], settings, config, write=False)
    assert again["evaluation"] == record["evaluation"]  # reproducible
    stage1 = run_ml(RunSpec("XYZ", 2019, 0, 1, 1, 0.5), settings.models["ML-LR"], settings, config, write=False)
    assert stage1["evaluate"] == ["2018-01-01", "2018-12-31"]
    assert stage1["evaluation"]["10bps"]["sharpe"] == pytest.approx(stage1["validation_sharpe"])


def test_benchmarks_run_through_the_engine_commands(data_dir):
    assert cli.main(["run", "--rung", "ML-RS-LGBM", "--stock", "XYZ", "--test-year", "2019", "--stage", "1",
                     "--exit", "1", "0.5", "--campaign", "t"]) == 0
    assert result_path("t", "ML-RS-LGBM", RunSpec("XYZ", 2019, 0, 1, 1, 0.5)).exists()
    with pytest.raises(SystemExit, match="unknown"):
        cli.main(["run", "--rung", "ML-NONE", "--stock", "XYZ", "--test-year", "2019", "--exit", "1", "0.5"])
    assert cli.main(["tasks", "--campaign", "t", "--kind", "runs", "--rungs", "A0", "ML-LR", "--stocks", "AAPL",
                     "--years", "2019", "--stages", "1", "--development"]) == 0
    plan = json.loads((data_dir / "results/t/tasks.runs.json").read_text())
    assert [len(t["runs"]) for t in plan["tasks"]] == [30, 1]  # the GP's 30 runs; one for a deterministic model


def test_gated_network_learns_which_source_to_trust_in_each_regime():
    pytest.importorskip("torch")
    from regime_fusion.ml.neural import Gated, groups

    rng = np.random.default_rng(5)
    n = 2000
    columns = ["fa:a", "fa:b", "sa:a", "ta:a", "ta:b", "rg:p_state0", "rg:p_state1"]
    x = rng.uniform(-1, 1, size=(n, len(columns)))
    state = (rng.random(n) < 0.5).astype(int)
    x[:, 5], x[:, 6] = np.where(state == 0, 1.0, -1.0), np.where(state == 1, 1.0, -1.0)  # 2p - 1
    y = np.where(state == 0, x[:, 0] > 0, x[:, 3] > 0).astype(int)  # FA matters in state 0, TA in state 1
    assert groups(columns) == ({"FA": [0, 1], "SA": [2], "TA": [3, 4]}, [5, 6])
    learner = Gated(seed=1, columns=columns, hidden=16, epochs=60).fit(x, y)
    assert ((learner.predict(x) > 0.5) == y).mean() > 0.85
    weights = learner.gate_weights(x)  # columns FA, SA, TA
    assert weights[state == 0, 0].mean() > weights[state == 0, 2].mean()
    assert weights[state == 1, 2].mean() > weights[state == 1, 0].mean()
    again = Gated(seed=1, columns=columns, hidden=16, epochs=60).fit(x, y)
    np.testing.assert_array_equal(again.predict(x), learner.predict(x))  # reproducible


def test_gated_benchmark_runs_with_the_common_protocol(data_dir):
    pytest.importorskip("torch")
    config, settings = load_config(), models.load_settings()
    spec = RunSpec("XYZ", 2019, 3, 2, 1, 0.5)
    record = run_ml(spec, settings.models["ML-GATED"], settings, config, write=False)
    assert record["rung"] == "ML-GATED" and settings.models["ML-GATED"].runs == 10
    assert record["evaluation"]["10bps"]["sharpe"] > 1.0  # it found the pattern out of sample


def test_a_model_that_saw_one_class_does_not_buy_every_day():
    from regime_fusion.ml.run import decide

    nothing = decide(np.zeros(50), np.zeros(10), 0.7)          # never saw a gaining trade
    assert not nothing.any()
    assert decide(np.ones(50), np.ones(10), 0.7).all()          # only gaining trades: buys
    ranked = decide(np.arange(101.0), np.array([69.0, 70.0, 71.0]), 0.7)  # the 0.7 quantile is 70
    assert ranked.tolist() == [False, True, True]
    constant = models.Constant(np.zeros(30))
    assert not decide(constant.predict(np.zeros((30, 2))), constant.predict(np.zeros((5, 2))), 0.5).any()
