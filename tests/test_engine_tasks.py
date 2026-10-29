import dataclasses
import json

import pandas as pd
import pytest

from regime_fusion import provenance
from regime_fusion.config import load_config
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.engine import cli, tasks
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.run import exits_path, result_path


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    """Stand-ins for evolution: they write small result files and count calls."""
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    calls = {"runs": [], "exits": []}

    def fake_run(spec, rung, config, *, campaign, **_):
        calls["runs"].append((rung.name, spec.stock, spec.test_year, spec.stage, spec.run))
        evaluation = {f"{c:g}bps": {"sharpe": 0.5 + spec.run, "trades": 4.0} for c in config.costs_bps_per_side}
        record = {**spec.__dict__, "rung": rung.name, "fitness": 1.0, "seconds": 0.1, "model": "GT(x, 0.1)",
                  "evaluation": evaluation, "branches": {"TA": {"sharpe": 0.2, "trades": 3.0}}}
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
    result_path("c", "A0", tasks.RunSpec("AAPL", 2019, 1, 2, 10, 0.02)).parent.mkdir(parents=True, exist_ok=True)
    provenance.write(result_path("c", "A0", tasks.RunSpec("AAPL", 2019, 1, 2, 10, 0.02)), {"done": True})
    assert tasks.execute(loaded[0], "c", config, rungs, development_only=False) == {"done": 2, "skipped": 1}
    assert [c[-1] for c in fake_engine["runs"]] == [0, 2]  # run 1 was already there
    assert tasks.finished(loaded[0], "c")


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
    assert {"sharpe_10bps", "trades_0bps", "ta_sharpe_10bps", "model"} <= set(table.columns)
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
