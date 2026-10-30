import dataclasses
import json

import pandas as pd
import pytest

from regime_fusion import provenance
from regime_fusion.config import load_config
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.engine import cli, tasks
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.run import FINGERPRINT_FORMAT, exits_path, result_path


@pytest.fixture
def config():
    return load_config()


def fake_result(spec, name, config, method, data="v1"):
    """A small result as a run writes it, fingerprint included (``data`` names the data's version)."""
    evaluation = {f"{c:g}bps": {"sharpe": 0.5 + spec.run, "annual_return": 0.1, "max_drawdown": -0.1,
                                "cvar_95": -0.02, "trades": 4.0, "trades_per_year": 4.0}
                  for c in config.costs_bps_per_side}
    inputs = {f"interim/prices/{spec.stock}.parquet@{spec.test_year}": data}
    return {**spec.__dict__, "rung": name, "fitness": 1.0, "seconds": 0.1, "model": "GT(x, 0.1)",
            "evaluation": evaluation, "signal": "AA==", "branches": {"TA": {"sharpe": 0.2, "trades": 3.0}},
            "selection_weights": {"FA": 0.2, "SA": 0.3, "TA": 0.5},     # as ML-TFT keeps them
            "fingerprint": {"format": FINGERPRINT_FORMAT, "config": config.digest(), "method": method,
                            "inputs": inputs, "evaluation_prices": data}}


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    """Stand-ins for evolution: they write small result files and count calls."""
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    calls = {"runs": [], "exits": []}

    def fake_run(spec, rung, config, *, campaign, **_):
        calls["runs"].append((rung.name, spec.stock, spec.test_year, spec.stage, spec.run))
        record = fake_result(spec, rung.name, config, rung.digest())
        provenance.write(result_path(campaign, rung.name, spec), record)
        return record

    def fake_exits(stock, test_year, rung, config, grid, *, campaign, **_):
        calls["exits"].append((stock, test_year))
        provenance.write(exits_path(campaign, stock, test_year), {"days": 10, "rate": 0.02})
        return {"days": 10, "rate": 0.02}

    monkeypatch.setattr(tasks, "run", fake_run)
    monkeypatch.setattr(tasks, "choose_exits", fake_exits)
    return calls


def test_plans_number_tasks_in_blocks(config):
    exits = tasks.plan("exits", ["AAPL", "F"], [2019, 2020], config)
    assert [(t.stock, t.test_year, t.stage) for t in exits] == [("AAPL", 2019, 1), ("F", 2019, 1),
                                                                ("AAPL", 2020, 1), ("F", 2020, 1)]
    runs = tasks.plan("runs", ["AAPL"], [2019], config, rungs=["A0", "A1"], runs=30, block=12)
    assert [t.runs for t in runs[:3]] == [tuple(range(12)), tuple(range(12, 24)), tuple(range(24, 30))]
    assert {t.rung for t in runs} == {"A0", "A1"} and len(runs) == 6
    with pytest.raises(ValueError, match="rung"):
        tasks.plan("runs", ["AAPL"], [2019], config)


def test_development_campaigns_never_reach_later_data(config):
    assert tasks.plan("exits", ["AAPL"], [2019], config, development_only=True)  # validation year 2018
    assert tasks.plan("runs", ["AAPL"], [2019], config, rungs=["A0"], stages=(1,), development_only=True)
    with pytest.raises(ValueError, match="development"):
        tasks.plan("runs", ["AAPL"], [2019], config, rungs=["A0"], stages=(2,), development_only=True)
    with pytest.raises(ValueError, match="development"):
        tasks.plan("exits", ["AAPL"], [2020], config, development_only=True)


def test_tasks_are_saved_once_and_resume_where_they_stopped(fake_engine, config):
    rungs = load_rungs()
    planned = tasks.plan("runs", ["AAPL"], [2019], config, rungs=["A0"], runs=3)
    tasks.save_plan("c", "runs", planned, development_only=False)
    with pytest.raises(FileExistsError):
        tasks.save_plan("c", "runs", planned, development_only=False)
    development, loaded = tasks.load_plan("c", "runs")
    assert loaded == [dataclasses.replace(t, plan="runs") for t in planned] and development is False
    with pytest.raises(FileNotFoundError, match="exits tasks first"):
        tasks.execute(loaded[0], "c", config, rungs, development_only=False)
    exit_task = tasks.plan("exits", ["AAPL"], [2019], config)[0]
    assert tasks.execute(exit_task, "c", config, rungs, development_only=False) == {"done": 1, "skipped": 0}
    assert tasks.execute(exit_task, "c", config, rungs, development_only=False) == {"done": 0, "skipped": 1}
    one = tasks.RunSpec("AAPL", 2019, 1, 2, 10, 0.02)
    provenance.write(result_path("c", "A0", one), {"done": True})               # not a result: refused
    with pytest.raises(ValueError, match="not a whole result"):
        tasks.execute(loaded[0], "c", config, rungs, development_only=False)
    other = fake_result(tasks.RunSpec("AAPL", 2019, 2, 2, 10, 0.02), "A0", config, rungs["A0"].digest())
    provenance.write(result_path("c", "A0", one), other)                        # another run's result: refused
    with pytest.raises(ValueError, match="another run"):
        tasks.execute(loaded[0], "c", config, rungs, development_only=False)
    provenance.write(result_path("c", "A0", one), fake_result(one, "A0", config, rungs["A0"].digest()))
    assert tasks.execute(loaded[0], "c", config, rungs, development_only=False) == {"done": 1, "skipped": 2}
    assert [c[-1] for c in fake_engine["runs"]] == [0, 2]  # run 0 was done before the stop, run 1 was there
    assert tasks.finished(loaded[0], "c")


def test_a_campaign_refuses_results_made_under_other_settings_or_data(fake_engine, config, monkeypatch):
    rungs = load_rungs()
    assert cli.main(["tasks", "--campaign", "c", "--kind", "exits", "--stocks", "AAPL", "--years", "2019"]) == 0
    assert cli.main(["tasks", "--campaign", "c", "--kind", "runs", "--rungs", "A0", "A1", "--stocks", "AAPL",
                     "--years", "2019", "--runs", "2"]) == 0
    assert cli.main(["run-task", "--campaign", "c", "--kind", "exits", "--index", "0"]) == 0
    assert cli.main(["run-task", "--campaign", "c", "--kind", "runs", "--index", "0"]) == 0   # A0, runs 0 and 1
    spec = tasks.RunSpec("AAPL", 2019, 1, 2, 10, 0.02)
    other = load_rungs()
    changed = {**other, "A0": dataclasses.replace(other["A0"], spec={**other["A0"].spec, "scaling": "rank"})}
    _, planned = tasks.load_plan("c", "runs")
    with pytest.raises(ValueError, match="the settings of A0 changed since plan"):   # the plan froze them
        tasks.execute(planned[0], "c", config, changed, development_only=False)
    with pytest.raises(ValueError, match="other settings of A0"):                   # and so did the result
        tasks.finished_result(result_path("c", "A0", spec), "A0", spec, config, changed["A0"].digest())
    assert cli.main(["run-task", "--campaign", "c", "--kind", "runs", "--index", "1"]) == 0   # A1 is unchanged
    assert cli.main(["collect", "--campaign", "c"]) == 0
    provenance.write(result_path("c", "A1", spec), fake_result(spec, "A1", config, rungs["A1"].digest(), data="v2"))
    with pytest.raises(SystemExit, match="2 versions of these data"):              # A1 run 1 saw other prices
        cli.main(["collect", "--campaign", "c"])
    provenance.write(result_path("c", "A1", spec), fake_result(spec, "A1", config, "other"))
    with pytest.raises(SystemExit, match="A1: runs with different settings"):
        cli.main(["collect", "--campaign", "c"])
    for broken in ({}, [], {**fake_result(spec, "A1", config, rungs["A1"].digest()), "fingerprint": {"config": "x"}}):
        provenance.write(result_path("c", "A1", spec), broken)
        with pytest.raises(SystemExit, match="not a (whole )?result"):
            cli.main(["collect", "--campaign", "c"])


def test_commands_plan_run_and_collect_a_campaign(fake_engine, tmp_path, config):
    assert cli.main(["tasks", "--campaign", "pilot", "--kind", "exits", "--stocks", "AAPL", "F",
                     "--years", "2019", "--development"]) == 0
    assert cli.main(["tasks", "--campaign", "pilot", "--kind", "runs", "--rungs", "A0", "A1", "--stocks",
                     "AAPL", "F", "--years", "2019", "--stages", "1", "--runs", "2", "--development"]) == 0
    with pytest.raises(SystemExit, match="development"):
        cli.main(["tasks", "--campaign", "pilot2", "--kind", "runs", "--rungs", "A0", "--stocks", "AAPL",
                  "--years", "2019", "--development"])
    with pytest.raises(SystemExit, match="hold-out"):
        cli.main(["tasks", "--campaign", "x", "--kind", "exits", "--stocks", "AMG"])
    assert cli.main(["run-task", "--campaign", "pilot", "--kind", "exits", "--index", "0", "--count", "2"]) == 0
    for index in range(4):
        assert cli.main(["run-task", "--campaign", "pilot", "--kind", "runs", "--index", str(index)]) == 0
    with pytest.raises(SystemExit, match="no task 4"):
        cli.main(["run-task", "--campaign", "pilot", "--kind", "runs", "--index", "4"])
    assert len(fake_engine["runs"]) == 8 and fake_engine["exits"] == [("AAPL", 2019), ("F", 2019)]
    assert cli.main(["status", "--campaign", "pilot"]) == 0
    assert cli.main(["collect", "--campaign", "pilot"]) == 0
    table = pd.read_parquet(tmp_path / "results" / "pilot" / "runs.parquet")
    assert len(table) == 8 and set(table["rung"]) == {"A0", "A1"}
    assert {"sharpe_10bps", "trades_0bps", "ta_sharpe_10bps", "model", "select_ta"} <= set(table.columns)
    assert table.loc[table["run"] == 1, "sharpe_10bps"].eq(1.5).all()
    plan = json.loads((tmp_path / "results" / "pilot" / "tasks.runs.json").read_text())
    assert plan["development_only"] is True and len(plan["tasks"]) == 4


def test_result_files_are_written_whole(tmp_path):
    path = tmp_path / "a" / "b.json"
    provenance.write(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}
    assert list(path.parent.iterdir()) == [path]  # no temporary file left behind


def test_expanding_campaigns_use_expanding_windows_throughout(fake_engine, tmp_path, monkeypatch):
    from regime_fusion.config import ConfigError, with_scheme
    from regime_fusion.regimes import market
    from regime_fusion.windows import walk_forward

    config = load_config()
    expanding = with_scheme(config, "expanding")
    window = next(w for w in walk_forward(expanding) if w.test_year == 2021)
    assert str(window.train.start) == "2015-02-01" and str(window.final_train.end) == "2020-12-31"
    with pytest.raises(ConfigError):
        with_scheme(config, "sideways")
    assert market.probabilities_path("dc", 2021, 2, root=tmp_path, scheme="expanding") == \
        tmp_path / "processed" / "regimes" / "dc" / "expanding" / "2021.stage2.parquet"
    seen = []
    real = tasks.run_method

    def spy(name, spec, rungs, config, campaign):
        seen.append(config.windows.scheme)
        return real(name, spec, rungs, config, campaign)

    monkeypatch.setattr(tasks, "run_method", spy)
    assert cli.main(["tasks", "--campaign", "wide", "--kind", "exits", "--stocks", "AAPL", "--years", "2021",
                     "--scheme", "expanding"]) == 0
    assert cli.main(["tasks", "--campaign", "wide", "--kind", "runs", "--rungs", "A0", "--stocks", "AAPL",
                     "--years", "2021", "--runs", "1", "--scheme", "expanding"]) == 0
    assert tasks.plan_scheme("wide", "runs") == "expanding"
    assert cli.main(["run-task", "--campaign", "wide", "--kind", "exits", "--index", "0"]) == 0
    assert cli.main(["run-task", "--campaign", "wide", "--kind", "runs", "--index", "0"]) == 0
    assert seen == ["expanding"]


def test_a_campaign_can_hold_several_plans_of_runs(fake_engine, tmp_path):
    assert cli.main(["tasks", "--campaign", "p", "--kind", "exits", "--stocks", "AAPL", "F", "--years", "2019",
                     "--development"]) == 0
    assert cli.main(["tasks", "--campaign", "p", "--kind", "runs", "--rungs", "A0", "--stocks", "AAPL", "F",
                     "--years", "2019", "--stages", "1", "--runs", "2", "--development"]) == 0
    assert cli.main(["tasks", "--campaign", "p", "--kind", "runs", "--plan", "check100", "--rungs", "A0",
                     "--stocks", "AAPL", "--years", "2019", "--stages", "1", "--runs", "5", "--development"]) == 0
    assert tasks.plan_names("p") == ["check100", "exits", "runs"]
    assert cli.main(["run-task", "--campaign", "p", "--kind", "exits", "--index", "0", "--count", "2"]) == 0
    assert cli.main(["run-task", "--campaign", "p", "--plan", "check100", "--index", "0"]) == 0
    assert sorted(c[-1] for c in fake_engine["runs"]) == [0, 1, 2, 3, 4]
    assert cli.main(["run-task", "--campaign", "p", "--kind", "runs", "--index", "0"]) == 0
    assert len(fake_engine["runs"]) == 5  # runs 0 and 1 of AAPL were already done by the other plan
    _, check = tasks.load_plan("p", "check100")
    assert check[0].plan == "check100" and tasks.finished(check[0], "p")
    assert cli.main(["status", "--campaign", "p"]) == 0


def test_finished_tasks_hold_results_made_with_the_current_settings(fake_engine, config):
    rungs = load_rungs()
    assert cli.main(["tasks", "--campaign", "c", "--kind", "exits", "--stocks", "AAPL", "--years", "2019"]) == 0
    assert cli.main(["tasks", "--campaign", "c", "--kind", "runs", "--rungs", "A0", "--stocks", "AAPL",
                     "--years", "2019", "--stages", "2", "--runs", "2"]) == 0
    assert cli.main(["run-all", "--campaign", "c", "--kind", "exits"]) == 0
    assert cli.main(["run-all", "--campaign", "c", "--kind", "runs"]) == 0
    _, [task] = tasks.load_plan("c", "runs")
    assert tasks.finished(task, "c", config, rungs)
    spec = tasks.RunSpec("AAPL", 2019, 1, 2, 10, 0.02)
    provenance.write(result_path("c", "A0", spec), fake_result(spec, "A0", config, "other"))   # made otherwise
    assert tasks.finished(task, "c")                                  # the file is there,
    assert not tasks.finished(task, "c", config, rungs)               # but not as the campaign's settings make it
    with pytest.raises(ValueError, match="other settings of A0"):    # so run-all runs the task, which says why
        cli.main(["run-all", "--campaign", "c", "--kind", "runs"])
    _, [choose] = tasks.load_plan("c", "exits")
    provenance.write(exits_path("c", "AAPL", 2019), {"days": 10, "rate": 0.02,
                                                     "provenance": {"config_digest": "other"}})
    assert not tasks.finished(choose, "c", config, rungs)
    with pytest.raises(ValueError, match="another study configuration"):
        tasks.execute(choose, "c", config, rungs, development_only=False)


def test_a_changed_exit_grid_leaves_exit_choices_and_runs_unfinished(fake_engine, config, monkeypatch):
    rungs = load_rungs()
    assert cli.main(["tasks", "--campaign", "c", "--kind", "exits", "--stocks", "AAPL", "F", "--years", "2019"]) == 0
    assert cli.main(["tasks", "--campaign", "c", "--kind", "runs", "--rungs", "A0", "A1", "--stocks", "AAPL", "F",
                     "--years", "2019", "--stages", "2", "--runs", "2"]) == 0
    assert cli.main(["run-all", "--campaign", "c", "--kind", "exits"]) == 0
    assert cli.main(["run-all", "--campaign", "c", "--kind", "runs"]) == 0
    calls = []
    real = tasks.exit_settings
    monkeypatch.setattr(tasks, "exit_settings", lambda *args: calls.append(args[1:3]) or real(*args))
    assert tasks.unfinished("c", "runs", config, rungs) == []
    assert sorted(calls) == [("AAPL", 2019), ("F", 2019)]            # once per stock-year, not per task
    grid = {**tasks.exit_grid(), "days": [5]}
    monkeypatch.setattr(tasks, "exit_grid", lambda: grid)            # the grid edited after the plans
    assert tasks.unfinished("c", "exits", config, rungs) == [0, 1]   # as execute sees them
    assert tasks.unfinished("c", "runs", config, rungs) == [0, 1, 2, 3]
    _, planned = tasks.load_plan("c", "runs")
    with pytest.raises(ValueError, match="the exit grid changed"):
        tasks.execute(planned[0], "c", config, rungs, development_only=False)


def test_a_pooled_run_checks_every_stocks_result_before_it_runs(fake_engine, config, monkeypatch):
    calls = []
    monkeypatch.setattr("regime_fusion.ml.tft.run_tft", lambda *args, **kwargs: calls.append(args))
    assert cli.main(["tasks", "--campaign", "c", "--kind", "runs", "--rungs", "ML-TFT", "--stocks", "AAPL", "F",
                     "--years", "2019", "--stages", "1", "--development"]) == 0
    for stock in ("AAPL", "F"):
        provenance.write(exits_path("c", stock, 2019), {"days": 10, "rate": 0.02})
    spec = tasks.RunSpec("F", 2019, 0, 1, 10, 0.02)                  # AAPL's run 0 is missing, F's made otherwise
    provenance.write(result_path("c", "ML-TFT", spec), fake_result(spec, "ML-TFT", config, "other"))
    _, [task] = tasks.load_plan("c", "runs")
    with pytest.raises(ValueError, match="other settings of ML-TFT"):
        tasks.execute(task, "c", config, {}, development_only=True)
    assert calls == []                                                # nothing run, so nothing overwritten


def test_fingerprints_of_overlapping_windows_compare_the_years_they_share(tmp_path):
    from regime_fusion.engine.panel import input_entries

    days = pd.bdate_range("2015-01-01", "2019-12-31")
    table = pd.DataFrame({"x": range(len(days))}, index=days, dtype=float)
    path = tmp_path / "processed" / "ta" / "S.parquet"
    early = input_entries(path, table, days[days.year <= 2018], tmp_path)      # one test year's window
    late = input_entries(path, table, days[days.year >= 2016], tmp_path)       # the next year's
    shared = sorted(set(early) & set(late))
    assert shared == [f"processed/ta/S.parquet@{y}" for y in (2016, 2017, 2018)]
    assert all(early[k] == late[k] for k in shared)
    rebuilt = table.copy()
    rebuilt.loc["2017-06-01", "x"] = -1.0                                      # the file changes in between
    again = input_entries(path, rebuilt, days[days.year >= 2016], tmp_path)
    assert [k for k in shared if again[k] != early[k]] == ["processed/ta/S.parquet@2017"]   # so collect sees it


def test_a_whole_result_is_the_same_for_resuming_and_collecting(config):
    spec = tasks.RunSpec("AAPL", 2019, 0, 2, 10, 0.02)
    whole = fake_result(spec, "A0", config, "m")
    costs = config.costs_bps_per_side
    assert tasks.record_problem(whole, costs) is None
    empty = {**whole, "evaluation": {f"{c:g}bps": {} for c in costs}}
    assert "lacks the evaluation" in tasks.record_problem(empty, costs)
    assert tasks.record_problem({**whole, "signal": None}, costs) == "it has no stored signal"
    assert tasks.record_problem({**whole, "rung": "RB-BH", "signal": None}, costs) is None   # holds, no signal
    older = {**whole, "fingerprint": {k: v for k, v in whole["fingerprint"].items() if k != "format"}}
    assert "older version of the code (fingerprint format 1" in tasks.record_problem(older, costs)
    assert "not a whole result" in tasks.record_problem({**whole, "fingerprint": {"config": "x"}}, costs)


def test_run_all_does_every_unfinished_task(fake_engine):
    assert cli.main(["tasks", "--campaign", "all", "--kind", "exits", "--stocks", "AAPL", "F", "--years", "2019",
                     "--development"]) == 0
    assert cli.main(["run-all", "--campaign", "all", "--kind", "exits"]) == 0
    assert fake_engine["exits"] == [("AAPL", 2019), ("F", 2019)]
    assert cli.main(["run-all", "--campaign", "all", "--kind", "exits"]) == 0
    assert len(fake_engine["exits"]) == 2  # nothing left


def test_run_all_in_parallel_reports_failed_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))  # no data at all: every task fails in its worker
    assert cli.main(["tasks", "--campaign", "x", "--kind", "exits", "--stocks", "AAPL", "F", "--years", "2019",
                     "--development"]) == 0
    assert cli.main(["run-all", "--campaign", "x", "--kind", "exits", "--workers", "2"]) == 1


def test_a_campaign_keeps_to_one_scheme_and_pilots_keep_off_the_hold_out(fake_engine, tmp_path):
    config = load_config()
    assert cli.main(["tasks", "--campaign", "main", "--kind", "exits", "--stocks", "AAPL", "--years", "2019"]) == 0
    assert tasks.campaign_scheme("main") == "rolling"
    with pytest.raises(SystemExit, match="need a campaign of their own"):
        cli.main(["tasks", "--campaign", "main", "--kind", "runs", "--plan", "robust", "--rungs", "A0",
                  "--stocks", "AAPL", "--years", "2019", "--scheme", "expanding"])
    with pytest.raises(SystemExit, match="need a campaign of their own"):
        cli.main(["run", "--rung", "A0", "--stock", "AAPL", "--test-year", "2019", "--exit", "10", "0.02",
                  "--campaign", "main", "--scheme", "expanding"])
    with pytest.raises(SystemExit, match="hold-out"):
        cli.main(["tasks", "--campaign", "dev", "--kind", "exits", "--include-holdout", "--years", "2019",
                  "--development"])
    planned = tasks.plan("exits", ["AMG"], [2019], config, development_only=True)
    with pytest.raises(ValueError, match="hold-out"):
        tasks.save_plan("dev", "exits", planned, development_only=True)
    with pytest.raises(ValueError, match="hold-out"):
        tasks.execute(planned[0], "dev", config, {}, development_only=True)


def test_fixed_exits_for_the_gp_parameter_pilot(fake_engine, tmp_path):
    assert cli.main(["tasks", "--campaign", "gpp", "--kind", "runs", "--rungs", "A1-p250-t2", "A1-p2000-t4",
                     "--stocks", "AAPL", "F", "--years", "2019", "--stages", "1", "--runs", "2", "--development",
                     "--exit", "20", "0.03"]) == 0
    saved = json.loads(exits_path("gpp", "F", 2019).read_text())
    assert (saved["days"], saved["rate"], saved["fixed"]) == (20, 0.03, True)
    assert cli.main(["run-all", "--campaign", "gpp", "--kind", "runs"]) == 0
    assert sorted({c[0] for c in fake_engine["runs"]}) == ["A1-p2000-t4", "A1-p250-t2"]
    assert saved["provenance"]["git"] and "config_digest" in saved["provenance"]   # with its provenance
    with pytest.raises(SystemExit, match="campaign of its own"):
        cli.main(["tasks", "--campaign", "gpp", "--kind", "runs", "--plan", "other", "--rungs", "A1",
                  "--stocks", "F", "--years", "2019", "--stages", "1", "--development", "--exit", "10", "0.02"])
    assert not tasks.plan_path("gpp", "other").exists()                     # refused before the plan is saved
    with pytest.raises(SystemExit, match="exit setting of runs"):
        cli.main(["tasks", "--campaign", "gpp2", "--kind", "exits", "--rungs", "A1", "--stocks", "F",
                  "--years", "2019", "--development", "--exit", "20", "0.03"])
    assert not tasks.plan_path("gpp2", "exits").exists()


def test_a_fixed_exit_never_joins_a_chosen_one(fake_engine):
    provenance.write(exits_path("mixed", "F", 2019), {"stock": "F", "test_year": 2019, "days": 20, "rate": 0.03})
    with pytest.raises(ValueError, match="a chosen exit"):  # even when the values agree
        tasks.fix_exits("mixed", [("AAPL", 2019), ("F", 2019)], 20, 0.03, "rolling")
    assert not exits_path("mixed", "AAPL", 2019).exists()                  # nothing written


def test_finbert_rungs_run_only_on_stock_years_their_headlines_cover(fake_engine, config):
    from regime_fusion.data import store

    for stock, years in (("AAPL", range(2015, 2021)), ("ABBV", range(2016, 2021))):
        store.write_parquet(store.layer("processed", "finbert", "fnspid", f"{stock}.parquet"), pd.DataFrame(),
                            covered_years=list(years))
    planned = tasks.plan("runs", ["AAPL", "ABBV"], [2019, 2020], config, rungs=["A0", "A0-FB"], stages=(1, 2), runs=2)
    kept = tasks.drop_uncovered(planned, load_rungs(), config)
    finbert = {(t.stock, t.test_year) for t in kept if t.rung == "A0-FB"}
    assert finbert == {("AAPL", 2019), ("AAPL", 2020), ("ABBV", 2020)}   # ABBV's 2019 window starts in 2015
    assert [t for t in kept if t.rung == "A0"] == [t for t in planned if t.rung == "A0"]
    with pytest.raises(SystemExit, match="no FinBERT indicators"):           # F has none: a clear message
        cli.main(["tasks", "--campaign", "fb", "--kind", "runs", "--rungs", "A0-FB", "--stocks", "F",
                  "--years", "2019", "--stages", "1"])



def test_direct_commands_keep_to_a_development_only_campaign(fake_engine, monkeypatch):
    seen = []
    record = {"fitness": 0.0, "evaluation": {"10bps": {"sharpe": 0.0, "trades": 0}}, "seconds": 0.0, "model": ""}
    monkeypatch.setattr(cli, "run", lambda spec, *args, **kwargs: seen.append(spec) or record)
    monkeypatch.setattr(cli, "choose_exits", lambda *args, **kwargs: seen.append(args[:2]) or {"days": 10, "rate": 0.02})
    run = ["run", "--rung", "A0", "--stock", "AAPL", "--test-year", "2019", "--exit", "10", "0.02"]
    assert cli.main(["tasks", "--campaign", "pilot", "--kind", "exits", "--stocks", "AAPL", "--years", "2019",
                     "--development"]) == 0
    with pytest.raises(SystemExit, match="after the development period"):
        cli.main([*run, "--campaign", "pilot", "--stage", "2"])                  # 2019 is a test year
    with pytest.raises(SystemExit, match="after the development period"):
        cli.main(["exits", "--campaign", "pilot", "--stock", "AAPL", "--test-year", "2020"])   # validates on 2019
    with pytest.raises(SystemExit, match="hold-out"):
        cli.main(["run", "--campaign", "pilot", "--rung", "A0", "--stock", "AMG", "--test-year", "2019",
                  "--exit", "10", "0.02"])
    with pytest.raises(SystemExit, match="development-only"):
        cli.main([*run, "--campaign", "pilot", "--stage", "2", "--after-development"])
    assert seen == []
    assert cli.main([*run, "--campaign", "pilot"]) == 0 and seen[0].stage == 1   # stage 1 by default: 2018
    with pytest.raises(SystemExit, match="after the development period"):       # a new campaign: development-only
        cli.main([*run, "--campaign", "scratch", "--stage", "2"])
    assert tasks.campaign_record("scratch")["development_only"] is True
    assert cli.main([*run, "--campaign", "later", "--stage", "2", "--after-development"]) == 0
    with pytest.raises(SystemExit, match="for development-only campaigns"):
        cli.main([*run, "--campaign", "later", "--population", "20"])
    with pytest.raises(SystemExit, match="not development-only"):
        cli.main(["tasks", "--campaign", "later", "--kind", "exits", "--stocks", "AAPL", "--years", "2019",
                  "--development"])


def test_a_campaign_recorded_before_campaigns_kept_their_status_takes_it_from_its_plans(fake_engine, monkeypatch):
    from regime_fusion.data import store

    record = {"fitness": 0.0, "evaluation": {"10bps": {"sharpe": 0.0, "trades": 0}}, "seconds": 0.0, "model": ""}
    monkeypatch.setattr(cli, "run", lambda spec, *args, **kwargs: record)
    for campaign, flags in (("main", []), ("pilot", ["--development"])):
        assert cli.main(["tasks", "--campaign", campaign, "--kind", "runs", "--rungs", "A0", "--stocks", "AAPL",
                         "--years", "2019", "--stages", "1", *flags]) == 0
        path = store.data_dir() / "results" / campaign / "campaign.json"
        old = {k: v for k, v in json.loads(path.read_text()).items() if k != "development_only"}
        path.write_text(json.dumps(old))                              # as recorded before statuses were kept
    run = ["run", "--rung", "A0", "--stock", "AAPL", "--test-year", "2019", "--exit", "10", "0.02", "--stage", "2"]
    assert cli.main([*run, "--campaign", "main"]) == 0                # without --after-development
    assert tasks.campaign_record("main")["development_only"] is False # it is still the main campaign
    with pytest.raises(SystemExit, match="after the development period"):
        cli.main([*run, "--campaign", "pilot"])
    assert tasks.campaign_record("pilot")["development_only"] is True


def test_a_direct_run_keeps_to_the_campaigns_exits_and_never_overwrites_a_result(fake_engine, config, monkeypatch):
    calls = []

    def fake_run(spec, rung, config, *, campaign, **_):
        calls.append(spec)
        record = fake_result(spec, rung.name, config, rung.digest())
        provenance.write(result_path(campaign, rung.name, spec), record)
        return record

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main(["tasks", "--campaign", "dev", "--kind", "exits", "--stocks", "AAPL", "F", "--years", "2019",
                     "--development"]) == 0
    assert cli.main(["run-all", "--campaign", "dev", "--kind", "exits"]) == 0         # 10 days or +2%
    run = ["run", "--campaign", "dev", "--rung", "A0", "--stock", "AAPL", "--test-year", "2019"]
    with pytest.raises(SystemExit, match="other exit settings need a campaign of their own"):
        cli.main([*run, "--exit", "5", "0.05"])
    assert cli.main([*run, "--exit", "10", "0.02"]) == 0 and len(calls) == 1       # the campaign's own exits
    assert cli.main(run) == 0 and len(calls) == 1                                   # already there: kept
    with pytest.raises(SystemExit, match="other settings of A0"):                   # an override never replaces it
        cli.main([*run, "--population", "20"])
    provenance.write(exits_path("dev", "F", 2019), {"days": 10, "rate": 0.02, "provenance": {"config_digest": "x"}})
    with pytest.raises(SystemExit, match="another study configuration"):
        cli.main(["run", "--campaign", "dev", "--rung", "A0", "--stock", "F", "--test-year", "2019"])
    assert len(calls) == 1
    chosen = []
    monkeypatch.setattr(cli, "choose_exits", lambda *args, **kwargs: chosen.append(args[:2]))
    assert cli.main(["exits", "--campaign", "dev", "--stock", "AAPL", "--test-year", "2019"]) == 0
    assert chosen == []                                            # already set: never replaced
    with pytest.raises(SystemExit, match="another study configuration"):
        cli.main(["exits", "--campaign", "dev", "--stock", "F", "--test-year", "2019"])
