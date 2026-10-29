import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.analysis import cli, pilot, results
from regime_fusion.data.store import DATA_DIR_ENV


def fake_runs(n_runs=5, seed=0):
    """Runs of three methods on 12 stock-years; B beats A by 0.5 and C by 1.0 in every stock-year."""
    rng = np.random.default_rng(seed)
    rows = []
    for stock in ("S1", "S2", "S3", "S4"):
        for year in (2019, 2020, 2021):
            base = rng.normal()
            for rung, shift in (("A", 0.0), ("B", 0.5), ("C", 1.0)):
                for run in range(n_runs):
                    sharpe = base + shift + rng.normal(0, 0.1)
                    rows.append({"rung": rung, "stock": stock, "test_year": year, "stage": 2, "run": run,
                                 "fitness": rng.normal(), "seconds": 10.0 + shift, "sharpe_10bps": sharpe,
                                 "annual_return_10bps": 0.1, "max_drawdown_10bps": -0.2, "cvar_95_10bps": -0.03,
                                 "trades_per_year_10bps": 12.0, "trades_10bps": 0.0 if run == 0 else 12.0,
                                 "fa_sharpe_10bps": sharpe - 0.3, "ta_sharpe_10bps": sharpe + 0.2})
    return pd.DataFrame(rows)


def test_stock_year_medians_and_best_fitness_runs():
    runs = fake_runs()
    medians = results.stock_year_medians(runs, "sharpe_10bps")
    assert medians.shape == (12, 3)
    cell = runs[(runs.stock == "S1") & (runs.test_year == 2019) & (runs.rung == "B")]
    assert medians.loc[("S1", 2019), "B"] == cell["sharpe_10bps"].median()
    best = results.best_fitness(runs, "sharpe_10bps")
    assert best.loc[("S1", 2019), "B"] == cell.loc[cell["fitness"].idxmax(), "sharpe_10bps"]


def test_integration_is_the_gain_over_the_best_branch():
    runs = fake_runs()
    assert results.integration(runs).to_numpy() == pytest.approx(-0.2)
    assert results.integration(runs[["sharpe_10bps"]]).isna().all()


def test_summary_and_comparisons():
    runs = fake_runs()
    table = results.summary(runs)
    assert table.loc["B", "stock_years"] == 12 and table.loc["B", "never_trades"] == pytest.approx(0.2)
    assert table.loc["C", "median_sharpe_10bps"] > table.loc["A", "median_sharpe_10bps"]
    medians = results.stock_year_medians(runs, "sharpe_10bps")
    tests = results.compare(medians, results.ladder(["A", "B", "C"]))
    pairs = tests["pairs"].set_index(["first", "second"])
    assert pairs.loc[("B", "A"), "median_difference"] == pytest.approx(0.5, abs=0.1)
    assert (pairs["p_holm"] < 0.01).all()
    assert tests["friedman"]["p"] < 0.001 and list(tests["mean_ranks"].index) == ["C", "B", "A"]
    assert results.by_year(runs, "sharpe_10bps").shape == (3, 3)


def test_timing_convergence_and_run_counts():
    runs = fake_runs(n_runs=60)
    speed = pilot.timing(runs, stocks=50, years=7, runs_per=30)
    assert speed.loc["A", "median_seconds"] == 10.0
    assert speed.loc["A", "campaign_cpu_hours"] == pytest.approx(10.0 * 50 * 7 * 30 / 3600)
    history = pd.DataFrame([{"rung": rung, "stock": "S1", "test_year": 2019, "stage": 1, "run": run,
                             "generation": g, "best_fitness": (g if rung == "late" else min(g, 10)) + 1.0}
                            for rung in ("early", "late") for run in range(3) for g in range(51)])
    converged = pilot.convergence(history)
    assert converged["still_improving"].loc["early", "still_improving"] == 0.0
    assert converged["still_improving"].loc["late", "still_improving"] == 1.0
    assert converged["curve"].loc[50, "late"] == 1.0 and converged["curve"].loc[0, "late"] == pytest.approx(1 / 51)
    counts = pilot.run_count_check(runs, "sharpe_10bps", np.random.default_rng(1), sizes=(10, 30, 50), draws=200)
    assert list(counts.index) == [10, 30, 50] and counts["cells"].eq(36).all()
    assert counts["sd"].is_monotonic_decreasing


def test_commands_write_the_tables(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    folder = tmp_path / "results" / "pilot"
    folder.mkdir(parents=True)
    fake_runs().to_parquet(folder / "runs.parquet")
    run_file = folder / "A" / "S1" / "2019.stage2.run00.json"
    run_file.parent.mkdir(parents=True)
    run_file.write_text(json.dumps({"rung": "A", "stock": "S1", "test_year": 2019, "stage": 2, "run": 0,
                                    "history": [{"generation": g, "best_fitness": 1.0 + g} for g in range(20)]}))
    assert cli.main(["compare", "--campaign", "pilot", "--rungs", "A", "B", "C", "--against", "A"]) == 0
    analysis = folder / "analysis"
    assert (analysis / "medians.sharpe_10bps.stage2.csv").exists()
    assert (analysis / "conover.sharpe_10bps.stage2.csv").exists()
    wilcoxon = pd.read_csv(analysis / "wilcoxon.sharpe_10bps.stage2.csv")
    assert len(wilcoxon) == 4  # the ladder (B-A, C-B) and every method against A (B-A again, C-A)
    assert cli.main(["pilot", "--campaign", "pilot"]) == 0
    assert (analysis / "timing.csv").exists() and (analysis / "still_improving.csv").exists()
    with pytest.raises(SystemExit, match="no stage-1 runs"):
        cli.main(["compare", "--campaign", "pilot", "--stage", "1", "--rungs", "A"])
    with pytest.raises(SystemExit, match="collect"):
        cli.main(["compare", "--campaign", "other"])
