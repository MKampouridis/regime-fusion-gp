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
    with_ci = results.compare(medians, [("B", "A")], np.random.default_rng(0))["pairs"].iloc[0]
    assert with_ci["ci_low"] < 0.5 < with_ci["ci_high"] and with_ci["ci_high"] - with_ci["ci_low"] < 0.3
    assert tests["friedman"]["p"] < 0.001 and list(tests["mean_ranks"].index) == ["C", "B", "A"]
    assert results.by_year(runs, "sharpe_10bps").shape == (3, 3)


def test_timing_convergence_and_run_counts():
    runs = fake_runs(n_runs=60)
    speed = pilot.timing(runs, stocks=50, years=7, runs_for={"A": 30, "B": 1, "C": 10}, pooled=frozenset({"C"}))
    assert speed.loc["A", "median_seconds"] == 10.0
    assert speed.loc["A", "campaign_cpu_hours"] == pytest.approx(10.0 * 50 * 7 * 30 / 3600)
    assert speed.loc["B", "campaign_cpu_hours"] == pytest.approx(10.5 * 50 * 7 * 1 / 3600)  # one run each
    assert speed.loc["C", "campaign_cpu_hours"] == pytest.approx(11.0 * 7 * 10 / 3600)      # pooled: no stocks
    assert speed.loc["C", "runs"] == 3 * 60  # one per test year and run, not per stock
    history = pd.DataFrame([{"rung": rung, "stock": "S1", "test_year": 2019, "stage": 1, "run": run,
                             "generation": g, "best_fitness": (g if rung == "late" else min(g, 10)) - 30.0}
                            for rung in ("early", "late") for run in range(3) for g in range(51)])
    converged = pilot.convergence(history)  # fitness negative throughout: progress must still run 0 to 1
    assert converged["still_improving"].loc["early", "still_improving"] == 0.0
    assert converged["still_improving"].loc["late", "still_improving"] == 1.0
    assert converged["curve"].loc[50, "late"] == 1.0 and converged["curve"].loc[0, "late"] == 0.0
    assert converged["curve"].loc[5, "early"] == pytest.approx(0.5)
    generations = pilot.convergence_generation(history)
    assert generations.xs("early", level="rung").eq(10).all() and generations.xs("late", level="rung").eq(50).all()
    flat = history.assign(best_fitness=1.0)
    assert pilot.convergence_generation(flat).eq(0).all()
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
    with pytest.raises(SystemExit, match="confirmatory comparisons need"):
        cli.main(["compare", "--campaign", "pilot", "--rungs", "A", "B", "C", "--confirmatory"])
    assert (analysis / "timing.csv").exists() and (analysis / "still_improving.csv").exists()
    with pytest.raises(SystemExit, match="no stage-1 runs"):
        cli.main(["compare", "--campaign", "pilot", "--stage", "1", "--rungs", "A"])
    with pytest.raises(SystemExit, match="collect"):
        cli.main(["compare", "--campaign", "other"])


def test_portfolio_returns_are_rebuilt_from_the_stored_signals(monkeypatch, tmp_path):
    from regime_fusion.analysis import portfolio
    from regime_fusion.engine.backtest import backtest
    from regime_fusion.engine.run import pack_signal, unpack_signal

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2018-01-01", "2020-12-31")
    rows = []
    (tmp_path / "interim" / "prices").mkdir(parents=True)
    for stock in ("S1", "S2"):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(days))))
        prices = pd.DataFrame({"adj_open": close, "adj_high": close * 1.01, "adj_low": close * 0.99,
                               "adj_close": close}, index=days)
        prices.to_parquet(tmp_path / "interim" / "prices" / f"{stock}.parquet")
        for year in (2019, 2020):
            n = int((days.year == year).sum())
            for rung in ("A0", "A1", "RB-BH"):
                for run in range(2):
                    signal = rng.random(n) < 0.2
                    rows.append({"rung": rung, "stock": stock, "test_year": year, "stage": 2, "run": run,
                                 "exit_days": 5, "exit_rate": 0.02, "evaluate_start": f"{year}-01-01",
                                 "evaluate_end": f"{year}-12-31", "fitness": 0.0, "seconds": 1.0,
                                 "signal": None if rung == "RB-BH" else pack_signal(signal)})
    runs = pd.DataFrame(rows)
    assert (unpack_signal(rows[0]["signal"]) == unpack_signal(pack_signal(unpack_signal(rows[0]["signal"])))).all()
    first = runs.iloc[0]
    prices = portfolio.stock_prices("S1")
    expected = backtest(unpack_signal(first["signal"]), prices.loc["2019"], 5, 0.02, 10).daily
    pd.testing.assert_series_equal(portfolio.run_returns(first, prices, 10), expected)
    returns = portfolio.method_returns(runs, 10)
    assert list(returns.columns) == ["A0", "A1", "RB-BH"] and len(returns) == int((days.year >= 2019).sum())
    cell = runs[(runs.rung == "A1") & (runs.stock == "S2") & (runs.test_year == 2020)]
    s2 = pd.concat([portfolio.run_returns(r, portfolio.stock_prices("S2"), 10) for _, r in cell.iterrows()],
                   axis=1).mean(axis=1)
    s1 = pd.concat([portfolio.run_returns(r, prices, 10) for _, r in
                    runs[(runs.rung == "A1") & (runs.stock == "S1") & (runs.test_year == 2020)].iterrows()],
                   axis=1).mean(axis=1)
    assert returns.loc["2020", "A1"].to_numpy() == pytest.approx(((s1 + s2) / 2).to_numpy())
    outcome = portfolio.tests(returns, "RB-BH", ["A0", "A1"], n_trials=3, rng=np.random.default_rng(1),
                              bootstraps=200)
    assert 0 <= outcome["spa"]["p"] <= 1 and set(outcome["deflated_sharpe"]) == {"A0", "A1", "RB-BH"}
    assert "pbo" in outcome
    with pytest.raises(ValueError, match="no stored signal"):
        portfolio.run_returns(first.copy().replace({first["signal"]: None}), prices, 10)

    folder = tmp_path / "results" / "main"
    folder.mkdir(parents=True)
    runs.to_parquet(folder / "runs.parquet")
    assert cli.main(["portfolio", "--campaign", "main"]) == 0
    saved = json.loads((folder / "analysis" / "portfolio_tests.10bps.stage2.json").read_text())
    assert saved["spa"]["benchmark"] == "RB-BH" and saved["n_trials"] == 3


def test_trust_map_averages_listening_by_regime():
    runs = pd.DataFrame({"rung": ["A3", "A3", "A1"], "listen_fa_state0": [1.0, 0.5, np.nan],
                         "listen_ta_state0": [0.0, 0.5, np.nan], "listen_fa_state1": [0.2, 0.4, np.nan],
                         "listen_fa": [0.6, 0.45, 1.0]})
    trust = results.trust_map(runs)
    assert trust.loc[("A3", "0"), "fa"] == 0.75 and trust.loc[("A3", "1"), "fa"] == pytest.approx(0.3)
    assert trust.loc[("A3", "0"), "ta"] == 0.25 and "A1" not in trust.index.get_level_values(0)
    assert results.trust_map(runs[["rung", "listen_fa"]]).empty


def test_regime_breakdown_and_random_entry(monkeypatch, tmp_path):
    from regime_fusion.analysis import portfolio
    from regime_fusion.engine.run import pack_signal

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    days = pd.bdate_range("2018-01-01", "2020-12-31")
    folder = tmp_path / "processed" / "regimes" / "dc"
    folder.mkdir(parents=True)
    p1 = (np.arange(len(days)) % 10 < 3).astype(float)
    for year in (2019, 2020):
        pd.DataFrame({"p_state0": 1 - p1, "p_state1": p1}, index=days).to_parquet(folder / f"{year}.stage2.parquet")
    states = portfolio.regime_states("dc", [2019, 2020])
    assert states.index.min() == pd.Timestamp("2019-01-01") and set(states.unique()) == {0, 1}
    returns = pd.DataFrame({"A3": np.where(p1 == 1, 0.01, -0.001)}, index=days).loc["2019":]
    table = portfolio.by_regime(returns, states)
    assert table.loc[("A3", 1), "annual_return"] == pytest.approx(2.52)
    assert table.loc[("A3", 0), "days"] + table.loc[("A3", 1), "days"] == len(returns)

    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    prices = pd.DataFrame({"adj_open": close, "adj_high": close * 1.01, "adj_low": close * 0.99,
                           "adj_close": close}, index=days)
    runs = pd.DataFrame([{"rung": "A3", "stock": "S1", "test_year": 2019, "stage": 2, "run": r, "exit_days": 5,
                          "exit_rate": 0.02, "evaluate_start": "2019-01-01", "evaluate_end": "2019-12-31",
                          "sharpe_10bps": 3.0 + r, "trades_10bps": 20.0, "signal": pack_signal([True])}
                         for r in range(3)])
    check = portfolio.random_entry_check(runs, "A3", 10, np.random.default_rng(3), draws=50,
                                         prices_of=lambda stock: prices)
    assert check.loc[0, "trades"] == 20 and check.loc[0, "sharpe"] == 4.0
    assert check.loc[0, "percentile"] > 0.9  # a Sharpe ratio of 4 beats almost every random entry


def test_gp_parameter_pilot_picks_the_cheapest_converged_setting():
    def candidate(population, tournament, converge_at, final):
        cap = 100_000 // population
        rows, marks = [], []
        for base in ("A1", "A3"):
            for stock in ("S1", "S2", "S3"):
                for run in range(3):
                    shift = 0.1 * run + (0.2 if stock == "S2" else 0.0)  # the same for every candidate
                    for g in range(cap + 1):
                        fitness = final * min(g / converge_at, 1.0) + shift
                        rows.append({"rung": f"{base}-p{population}-t{tournament}", "stock": stock,
                                     "test_year": 2019, "stage": 1, "run": run, "generation": g,
                                     "best_fitness": fitness})
                        if g % 10 == 0 or g == cap:
                            marks.append({**rows[-1], "sharpe": 0.5 if population == 500 else 0.6})
        return rows, marks

    rows, marks = [], []
    for args in ((250, 4, 40, 0.8),     # converges early but lower: premature
                 (500, 2, 100, 1.0),    # 50,000 evaluations to converge
                 (1000, 2, 60, 1.0),    # 60,000
                 (2000, 4, 50, 1.02)):  # the highest fitness, but 100,000 evaluations
        r, m = candidate(*args)
        rows += r
        marks += m
    table, choice = pilot.gp_parameters(pd.DataFrame(rows), pd.DataFrame(marks))
    assert table.loc[(250, 4), "generations"] == 40 and table.loc[(500, 2), "generations"] == 100
    assert table.loc[(250, 4), "fitness_vs_best"] == pytest.approx(-0.22)
    assert (choice["population"], choice["tournament"], choice["generations"]) == (500, 2, 100)
    assert choice["validation_sharpe"] == 0.5 and choice["flag"] is False  # 0.1 below the best is no flag yet
    with pytest.raises(ValueError, match="candidates"):
        pilot.gp_parameters(pd.DataFrame(rows).assign(rung="A1"), pd.DataFrame(marks))
