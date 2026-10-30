import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from regime_fusion import provenance  # noqa: E402
from regime_fusion.config import load_config  # noqa: E402
from regime_fusion.data.store import DATA_DIR_ENV  # noqa: E402
from regime_fusion.engine import cli, tasks  # noqa: E402
from regime_fusion.engine.run import RunSpec, exits_path, result_path  # noqa: E402
from regime_fusion.ml import models, tft  # noqa: E402
from regime_fusion.windows import DateRange  # noqa: E402

SMALL = models.ModelSpec("ML-TFT", "tft", 1, {"d_model": [8], "dropout": [0.1], "epochs": [2], "sequence": [5],
                                               "heads": [2], "batch": [128]})


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """Two stocks with prices, indicators and market regimes, 2014-2019."""
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    days = pd.bdate_range("2014-01-02", "2019-12-31", name="session")
    rng = np.random.default_rng(0)
    for stock in ("XYZ", "ABC"):
        signal = rng.normal(size=len(days))
        drift = np.where(np.roll(signal, 1) > 0.5, 0.01, -0.002)
        close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, len(days))))
        opening = np.r_[close[0], close[:-1]]
        prices = pd.DataFrame({"adj_open": opening, "adj_high": np.maximum(opening, close) * 1.004,
                               "adj_low": np.minimum(opening, close) * 0.996, "adj_close": close}, index=days)
        (tmp_path / "interim/prices").mkdir(parents=True, exist_ok=True)
        prices.to_parquet(tmp_path / f"interim/prices/{stock}.parquet")
        for kind in ("fa", "sa", "ta"):
            (tmp_path / "processed" / kind).mkdir(parents=True, exist_ok=True)
            table = pd.DataFrame({"x0": signal if kind == "ta" else rng.normal(size=len(days))}, index=days)
            table.to_parquet(tmp_path / "processed" / kind / f"{stock}.parquet")
    folder = tmp_path / "processed" / "regimes" / "dc"
    folder.mkdir(parents=True)
    p1 = (np.arange(len(days)) // 120 % 2).astype(float)
    for stage in (1, 2):
        pd.DataFrame({"p_state0": 1 - p1, "p_state1": p1}, index=days).to_parquet(folder / f"2019.stage{stage}.parquet")
    return tmp_path


def test_sequences_stop_at_each_stocks_first_day_and_never_reach_later_days(data_dir):
    settings = models.load_settings()
    train = DateRange(pd.Timestamp("2015-02-02").date(), pd.Timestamp("2017-12-29").date())
    later = DateRange(pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-12-31").date())
    pooled = tft.assemble(["XYZ", "ABC"], train, later, settings, 2019, 1, {"XYZ": (1, 0.5), "ABC": (1, 0.5)})
    first_abc = pooled.offsets["ABC"]
    seq = tft.windows(pooled, np.array([first_abc + 1]), np.array([first_abc]), 4)
    assert (seq[0, :2] == 0).all()                                   # before ABC's first day: zeros, not XYZ's rows
    np.testing.assert_array_equal(seq[0, 2:], pooled.features[first_abc:first_abc + 2])
    changed = pooled.features.copy()
    pooled.features[first_abc + 2:] += 5.0                            # later days change...
    np.testing.assert_array_equal(tft.windows(pooled, np.array([first_abc + 1]), np.array([first_abc]), 4), seq)
    pooled.features = changed                                         # ...but not the sequence up to the day


def test_the_pooled_model_runs_once_and_writes_a_record_per_stock(data_dir):
    config, settings = load_config(), models.load_settings()
    exits = {"XYZ": (1, 0.5), "ABC": (1, 0.5)}
    records = tft.run_tft(2019, 2, 0, ["XYZ", "ABC"], exits, SMALL, settings, config, campaign="t")
    assert set(records["XYZ"]["indicators"]) == {"FA", "SA", "TA", "RG"}       # counted by type, as elsewhere
    assert set(records) == {"XYZ", "ABC"}
    for stock, record in records.items():
        assert record["evaluate"] == ["2019-01-01", "2019-12-31"] and record["rung"] == "ML-TFT"
        assert set(record["selection_weights"]) == {"FA", "SA", "TA", "RG"}
        assert sum(record["selection_weights"].values()) == pytest.approx(1.0, abs=1e-4)
        assert result_path("t", "ML-TFT", RunSpec(stock, 2019, 0, 2, 1, 0.5)).exists()
    again = tft.run_tft(2019, 2, 0, ["XYZ", "ABC"], exits, SMALL, settings, config, write=False)
    assert again["XYZ"]["evaluation"] == records["XYZ"]["evaluation"]  # reproducible


def test_campaigns_give_the_pooled_model_one_task_for_all_stocks(data_dir, monkeypatch):
    config = load_config()
    assert cli.main(["tasks", "--campaign", "c", "--kind", "runs", "--rungs", "ML-TFT", "--stocks", "AAPL", "F",
                     "--years", "2019", "--stages", "1", "--development"]) == 0
    _, planned = tasks.load_plan("c", "runs")
    assert [(t.stock, len(t.runs)) for t in planned] == [("*", 10)]
    assert tasks.plan_stocks("c", "runs") == ["AAPL", "F"]
    with pytest.raises(FileNotFoundError, match="exit settings"):
        tasks.execute(planned[0], "c", config, {}, development_only=True)
    calls = []

    def fake_tft(year, stage, run, stocks, exits, model, settings, config, *, campaign):
        calls.append(run)
        for stock in stocks:
            spec = RunSpec(stock, year, run, stage, *exits[stock])
            mark = {"config": config.digest(), "method": settings.method_digest(model.name), "inputs": {},
                    "evaluation_prices": ""}
            provenance.write(result_path(campaign, model.name, spec), {
                **spec.__dict__, "rung": model.name, "fingerprint": mark,
                "evaluation": {f"{c:g}bps": {} for c in config.costs_bps_per_side}})

    monkeypatch.setattr("regime_fusion.ml.tft.run_tft", fake_tft)
    for stock in ("AAPL", "F"):
        provenance.write(exits_path("c", stock, 2019), {"days": 10, "rate": 0.02})
    assert tasks.execute(planned[0], "c", config, {}, development_only=True) == {"done": 10, "skipped": 0}
    assert tasks.finished(planned[0], "c") and calls == list(range(10))
    assert tasks.execute(planned[0], "c", config, {}, development_only=True) == {"done": 0, "skipped": 10}
    with pytest.raises(ValueError, match="across all stocks"):
        tasks.run_method("ML-TFT", RunSpec("AAPL", 2019, 0, 1, 10, 0.02), {}, config, "c")
    plan = json.loads((data_dir / "results/c/tasks.runs.json").read_text())
    assert plan["stocks"] == ["AAPL", "F"]
