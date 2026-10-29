"""Campaigns as numbered tasks, for job arrays on CERES or a loop on one machine.

A campaign's tasks are planned once and saved to
``results/<campaign>/tasks.<kind>.json``; a job-array element then runs task
number i (``python -m regime_fusion.engine run-task``). There are two kinds:

* ``exits``: choose one stock-year's exit settings (d, r) with rung A0 on the
  validation year (every candidate of the grid, 5 runs each);
* ``runs``: evolve and evaluate one rung on one stock, test year and stage, for
  a block of run numbers (all 30 by default, about 8 minutes). A rule-based or
  machine-learning benchmark (``configs/benchmarks.yaml``) takes a rung's place,
  with its own number of runs. A benchmark trained across all stocks (the
  Temporal Fusion Transformer) has one task per test year, stage and run block
  for all the campaign's stocks (stock "*").

Exit settings must exist before the runs of a stock-year can start. Tasks can be
repeated safely: finished runs and exit choices are skipped, and result files
are written whole or not at all, so a task stopped by a time limit is simply
submitted again. ``collect`` gathers a campaign's runs into one table.

A campaign planned as development-only (the pilot) may evaluate on data up to
``period.development_end`` only: planning refuses any task that would look at
later data, and running checks again.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import StudyConfig
from regime_fusion.data import store
from regime_fusion.engine.rungs import Rung, exit_grid
from regime_fusion.engine.run import RunSpec, choose_exits, exits_path, result_path, run
from regime_fusion.ml.models import load_settings as load_benchmarks
from regime_fusion.windows import DateRange, walk_forward

KINDS = ("exits", "runs")
log = logging.getLogger("regime_fusion.engine")


@dataclass(frozen=True)
class Task:
    kind: str
    stock: str
    test_year: int
    rung: str
    stage: int
    runs: tuple[int, ...] = ()


def evaluation_range(config: StudyConfig, test_year: int, stage: int) -> DateRange:
    """The data a task is evaluated on: the validation year (stage 1) or the test year (stage 2)."""
    window = next((w for w in walk_forward(config) if w.test_year == test_year), None)
    if window is None:
        raise ValueError(f"{test_year} is not a test year of the study")
    return window.validation if stage == 1 else window.test


def plan(kind: str, stocks: list[str], years: list[int], config: StudyConfig, *, rungs: list[str] = (),
         stages: tuple[int, ...] = (2,), runs: int | dict[str, int] = 30, block: int | None = None,
         development_only: bool = False, pooled: frozenset[str] = frozenset()) -> list[Task]:
    """The tasks of one kind, ordered by rung, test year, stage, stock and run block.

    ``runs`` is the number of runs per configuration, or a number for each rung;
    methods in ``pooled`` get one task for all stocks (stock "*").
    """
    if kind == "exits":
        tasks = [Task("exits", stock, year, "A0", 1) for year in years for stock in stocks]
    elif kind == "runs":
        if not rungs:
            raise ValueError("runs need at least one rung")
        count = runs if isinstance(runs, dict) else dict.fromkeys(rungs, runs)
        tasks = [Task("runs", stock, year, rung, stage, tuple(range(first, min(first + (block or count[rung]),
                                                                                count[rung]))))
                 for rung in rungs for year in years for stage in stages
                 for stock in (["*"] if rung in pooled else stocks)
                 for first in range(0, count[rung], block or count[rung])]
    else:
        raise ValueError(f"kind must be one of {KINDS}")
    if development_only:
        for task in tasks:
            check_development(task, config)
    return tasks


def check_development(task: Task, config: StudyConfig) -> None:
    end = evaluation_range(config, task.test_year, task.stage).end
    if end > config.period.development_end:
        raise ValueError(f"{task.kind} for {task.test_year} stage {task.stage} evaluates up to {end}, "
                         f"after the development period (to {config.period.development_end})")


def plan_path(campaign: str, kind: str) -> Path:
    return store.data_dir() / "results" / campaign / f"tasks.{kind}.json"


def save_plan(campaign: str, kind: str, tasks: list[Task], *, development_only: bool,
              replace: bool = False, stocks: list[str] | None = None) -> Path:
    path = plan_path(campaign, kind)
    if path.exists() and not replace:
        raise FileExistsError(f"{path} exists; running jobs use its numbering (pass replace to plan again)")
    record = {"campaign": campaign, "kind": kind, "development_only": development_only,
              "stocks": sorted(stocks if stocks is not None else {t.stock for t in tasks} - {"*"}),
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "git": provenance.git_state(), "tasks": [dataclasses.asdict(t) for t in tasks]}
    provenance.write(path, record)
    return path


def load_plan(campaign: str, kind: str) -> tuple[bool, list[Task]]:
    """Whether the campaign is development-only, and its tasks."""
    path = plan_path(campaign, kind)
    if not path.exists():
        raise FileNotFoundError(f"no {kind} tasks planned for campaign {campaign!r} ({path})")
    record = json.loads(path.read_text())
    tasks = [Task(**{**t, "runs": tuple(t["runs"])}) for t in record["tasks"]]
    return bool(record["development_only"]), tasks


def plan_stocks(campaign: str, kind: str) -> list[str]:
    """The stocks of a campaign's plan (those a pooled task covers)."""
    return list(json.loads(plan_path(campaign, kind).read_text())["stocks"])


def execute(task: Task, campaign: str, config: StudyConfig, rungs: dict[str, Rung], *,
            development_only: bool) -> dict[str, int]:
    """Do what is left of one task; returns how many items were done and skipped."""
    if development_only:
        check_development(task, config)
    if task.stock == "*":
        return execute_pooled(task, campaign, config)
    if task.kind == "exits":
        if exits_path(campaign, task.stock, task.test_year).exists():
            return {"done": 0, "skipped": 1}
        choose_exits(task.stock, task.test_year, rungs["A0"], config, exit_grid(), campaign=campaign)
        return {"done": 1, "skipped": 0}
    exits = exits_path(campaign, task.stock, task.test_year)
    if not exits.exists():
        raise FileNotFoundError(f"{task.stock} {task.test_year}: no exit settings at {exits}; "
                                "run the exits tasks first")
    chosen = json.loads(exits.read_text())
    done = skipped = 0
    for number in task.runs:
        spec = RunSpec(task.stock, task.test_year, number, task.stage, chosen["days"], chosen["rate"])
        if result_path(campaign, task.rung, spec).exists():
            skipped += 1
            continue
        run_method(task.rung, spec, rungs, config, campaign)
        done += 1
    return {"done": done, "skipped": skipped}


def execute_pooled(task: Task, campaign: str, config: StudyConfig) -> dict[str, int]:
    """Runs of a benchmark trained across all the campaign's stocks; each writes a result per stock."""
    from regime_fusion.ml.tft import run_tft

    stocks = plan_stocks(campaign, task.kind)
    exits = {}
    for stock in stocks:
        path = exits_path(campaign, stock, task.test_year)
        if not path.exists():
            raise FileNotFoundError(f"{stock} {task.test_year}: no exit settings at {path}; run the exits tasks first")
        chosen = json.loads(path.read_text())
        exits[stock] = (int(chosen["days"]), float(chosen["rate"]))
    settings = load_benchmarks()
    done = skipped = 0
    for number in task.runs:
        if all(result_path(campaign, task.rung, RunSpec(s, task.test_year, number, task.stage, *exits[s])).exists()
               for s in stocks):
            skipped += 1
            continue
        run_tft(task.test_year, task.stage, number, stocks, exits, settings.models[task.rung], settings, config,
                campaign=campaign)
        done += 1
    return {"done": done, "skipped": skipped}


def run_method(name: str, spec: RunSpec, rungs: dict[str, Rung], config: StudyConfig,
               campaign: str) -> dict[str, Any]:
    """One run of a GP rung, a rule-based benchmark or a machine-learning benchmark."""
    if name in rungs:
        return run(spec, rungs[name], config, campaign=campaign)
    settings = load_benchmarks()
    if name in settings.rules:
        from regime_fusion.engine.benchmarks import run_rule

        return run_rule(spec, settings.rules[name], settings, config, campaign=campaign)
    if name in settings.models and settings.models[name].pooled:
        raise ValueError(f"{name} is trained across all stocks; run it as a campaign's task")
    if name in settings.models:
        from regime_fusion.ml.run import run_ml

        return run_ml(spec, settings.models[name], settings, config, campaign=campaign)
    raise ValueError(f"unknown rung or benchmark {name!r}")


def finished(task: Task, campaign: str) -> bool:
    if task.kind == "exits":
        return exits_path(campaign, task.stock, task.test_year).exists()
    stocks = plan_stocks(campaign, task.kind) if task.stock == "*" else [task.stock]
    root = store.data_dir() / "results" / campaign / task.rung
    return all((root / stock / f"{task.test_year}.stage{task.stage}.run{n:02d}.json").exists()
               for stock in stocks for n in task.runs)


def collect(campaign: str, costs: list[float], primary: float) -> pd.DataFrame:
    """One row per run of the campaign: its settings, fitness and evaluation measures."""
    root = store.data_dir() / "results" / campaign
    rows = []
    for path in sorted(root.glob("*/*/*.stage*.run*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("collect: %s is not readable; skipped", path)
            continue
        row = {key: record.get(key) for key in ("rung", "stock", "test_year", "stage", "run", "exit_days",
                                                "exit_rate", "fitness", "seconds")}
        row["evaluate_start"], row["evaluate_end"] = record.get("evaluate", [None, None])
        for cost in costs:
            for measure, value in record["evaluation"][f"{cost:g}bps"].items():
                row[f"{measure}_{cost:g}bps"] = value
        for source, measures in record.get("branches", {}).items():
            row[f"{source.lower()}_sharpe_{primary:g}bps"] = measures["sharpe"]
            row[f"{source.lower()}_trades_{primary:g}bps"] = measures["trades"]
        listening = record.get("listening") or {}
        for source, share in listening.get("all", {}).items():
            row[f"listen_{source.lower()}"] = share
        for state, shares in listening.get("by_state", {}).items():
            for source, share in shares.items():
                row[f"listen_{source.lower()}_state{state}"] = share
        row["model"] = record.get("model")
        row["signal"] = record.get("signal")
        rows.append(row)
    return pd.DataFrame(rows)
