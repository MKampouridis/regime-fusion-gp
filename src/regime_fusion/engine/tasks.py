"""Campaigns as numbered tasks, for job arrays on CERES or a loop on one machine.

A campaign's tasks are planned once and saved to
``results/<campaign>/tasks.<plan>.json`` (the plan is named after its kind
unless given a name, so that a campaign can hold several plans of runs, such as
the pilot's 100-run check); a job-array element then runs task number i
(``python -m regime_fusion.engine run-task``). There are two kinds:

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

A campaign is one experiment, so its results must all come from the same
settings and data:

* a plan freezes the digests of the study configuration, of each method's
  settings and of the exit grid (``manifest``), and every task checks them
  before it runs;
* a result already there is skipped only if it is whole and was made under the
  current settings (``finished_result``); anything else stops the task;
* ``collect`` refuses a campaign whose results disagree on the configuration, on
  a method's settings or on the data of any file and sessions they read (their
  fingerprints, ``engine/run.py``).

A changed setting or new data therefore need a campaign of their own.

A campaign planned as development-only (the pilot) may evaluate on data up to
``period.development_end`` only: planning refuses any task that would look at
later data, and running checks again; it may not include hold-out stocks.

A campaign uses one walk-forward scheme, rolling (the main results) or
expanding (the robustness check). The scheme, and whether the campaign is
development-only, are recorded in ``results/<campaign>/campaign.json`` when it
is first planned or run, and every command that works on the campaign, the
direct ``run`` and ``exits`` included, keeps to them. Results and exit settings
carry no scheme in their paths, so expanding windows need a campaign of their
own.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import StudyConfig
from regime_fusion.data import store
from regime_fusion.engine.panel import finbert_years
from regime_fusion.engine.rungs import Rung, exit_grid, finbert_source
from regime_fusion.engine.run import RunSpec, choose_exits, exits_path, result_path, run
from regime_fusion.ml.models import load_settings as load_benchmarks
from regime_fusion.windows import DateRange, walk_forward

KINDS = ("exits", "runs")
FINGERPRINT = {"config", "method", "inputs", "evaluation_prices"}  # what every result's fingerprint holds
log = logging.getLogger("regime_fusion.engine")


@dataclass(frozen=True)
class Task:
    kind: str
    stock: str
    test_year: int
    rung: str
    stage: int
    runs: tuple[int, ...] = ()
    plan: str = ""      # the name of the plan the task belongs to


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


def drop_uncovered(tasks: list[Task], rungs: dict[str, Rung], config: StudyConfig) -> list[Task]:
    """The tasks without runs of FinBERT rungs on stock-years whose window (first training year to the
    test year) their headline source does not cover: those stock-years are not part of the check."""
    first_year = {w.test_year: w.train.start.year for w in walk_forward(config)}
    years: dict[tuple[str, str], set[int]] = {}
    kept = []
    for task in tasks:
        rung = rungs.get(task.rung)
        if task.kind == "runs" and rung is not None and rung.finbert:
            key = (finbert_source(task.test_year), task.stock)
            if key not in years:
                years[key] = finbert_years(*key)
            if not set(range(first_year[task.test_year], task.test_year + 1)) <= years[key]:
                continue
        kept.append(task)
    return kept


def check_development(task: Task, config: StudyConfig) -> None:
    end = evaluation_range(config, task.test_year, task.stage).end
    if end > config.period.development_end:
        raise ValueError(f"{task.kind} for {task.test_year} stage {task.stage} evaluates up to {end}, "
                         f"after the development period (to {config.period.development_end})")


def campaign_record(campaign: str) -> dict[str, Any] | None:
    """What ``results/<campaign>/campaign.json`` records (None for a new campaign)."""
    path = store.data_dir() / "results" / campaign / "campaign.json"
    return json.loads(path.read_text()) if path.exists() else None


def campaign_scheme(campaign: str) -> str | None:
    """The walk-forward scheme a campaign was started with (None for a new campaign)."""
    record = campaign_record(campaign)
    return None if record is None else str(record["scheme"])


def development_campaign(campaign: str) -> bool:
    """Whether a campaign is development-only (as recorded, or as one of its plans says)."""
    if (campaign_record(campaign) or {}).get("development_only"):
        return True
    return any(load_plan(campaign, name)[0] for name in plan_names(campaign))


def claim_campaign(campaign: str, scheme: str, development_only: bool | None = None) -> bool:
    """Record a new campaign's scheme and whether it is development-only, or refuse work that does not
    keep to them; returns whether the campaign is development-only (``None`` takes its own status)."""
    path = store.data_dir() / "results" / campaign / "campaign.json"
    record = campaign_record(campaign)
    if record is None:
        provenance.write(path, {"campaign": campaign, "scheme": scheme, "development_only": bool(development_only),
                                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        return bool(development_only)
    if record["scheme"] != scheme:
        raise ValueError(f"campaign {campaign!r} uses {record['scheme']} windows; "
                         f"{scheme} windows need a campaign of their own")
    if "development_only" not in record:  # recorded before campaigns kept their status: its plans tell
        names = plan_names(campaign)
        status = any(load_plan(campaign, name)[0] for name in names) if names else development_only
        if status is None:
            return False
        record = {**record, "development_only": bool(status)}
        provenance.write(path, record)
    if development_only is not None and bool(development_only) != bool(record["development_only"]):
        status = "development-only" if record["development_only"] else "not development-only"
        raise ValueError(f"campaign {campaign!r} is {status}; the other kind of work needs a campaign of its own")
    return bool(record["development_only"])


def method_digest(name: str, rungs: dict[str, Rung]) -> str:
    """Digest of a GP rung's or a benchmark's settings."""
    return rungs[name].digest() if name in rungs else load_benchmarks().method_digest(name)


def manifest(tasks: list[Task], config: StudyConfig, rungs: dict[str, Rung]) -> dict[str, Any]:
    """The settings a plan's tasks run with, as digests: the study configuration, each method and,
    for exit choices, the exit grid."""
    frozen = {"config": config.digest(), "methods": {m: method_digest(m, rungs) for m in sorted({t.rung for t in tasks})}}
    if any(t.kind == "exits" for t in tasks):
        frozen["exit_grid"] = provenance.digest(exit_grid())
    return frozen


def check_manifest(task: Task, campaign: str, config: StudyConfig, rungs: dict[str, Rung]) -> None:
    """Refuse a task whose settings changed since its plan was made."""
    name = task.plan or task.kind
    path = plan_path(campaign, name)
    frozen = json.loads(path.read_text()).get("manifest") if path.exists() else None
    if not frozen:
        return
    now = manifest([task], config, rungs)
    changed = [what for what, then, current in (
        ("the study configuration", frozen["config"], now["config"]),
        (f"the settings of {task.rung}", frozen["methods"].get(task.rung), now["methods"][task.rung]),
        ("the exit grid", frozen.get("exit_grid"), now.get("exit_grid"))) if then != current]
    if changed:
        raise ValueError(f"{' and '.join(changed)} changed since plan {name!r} of campaign {campaign!r} was "
                         "made; results under new settings need a campaign of their own")


def finished_result(path: Path, name: str, spec: RunSpec, config: StudyConfig, method: str) -> bool:
    """Whether a run's result is there. A result that is there must be whole and made with these
    settings; anything else stops the task rather than being skipped or overwritten."""
    if not path.exists():
        return False
    problem = result_problem(path, name, spec, config, method)
    if problem:
        raise ValueError(f"{path}: {problem}. A campaign's results must all come from the same settings: "
                         "delete the file to run it again, or start a new campaign for new settings")
    return True


def result_problem(path: Path, name: str, spec: RunSpec, config: StudyConfig, method: str) -> str | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "not a readable result"
    if not isinstance(record, dict):
        return "not a result"
    wrong = sorted(key for key, value in {"rung": name, **spec.__dict__}.items() if record.get(key) != value)
    if wrong:
        return f"it is the result of another run ({', '.join(wrong)} differ)"
    if not {f"{c:g}bps" for c in config.costs_bps_per_side} <= set(record.get("evaluation") or {}):
        return "it lacks the evaluation at some cost"
    mark = record.get("fingerprint") or {}
    if mark.get("config") != config.digest():
        return "it was made under another study configuration, or before results carried one"
    if mark.get("method") != method:
        return f"it was made with other settings of {name}"
    return None


def check_roles(stocks: list[str]) -> None:
    """Development-only work may not touch the hold-out stocks."""
    from regime_fusion.data.universe import DEFAULT_UNIVERSE, load_universe

    universe = load_universe(DEFAULT_UNIVERSE)
    known = {stock.id: stock.role for stock in universe.stocks}
    held_out = sorted(s for s in stocks if known.get(s) == "holdout")
    if held_out:
        raise ValueError(f"development-only work may not use hold-out stocks: {held_out}")


def fix_exits(campaign: str, stock_years: list[tuple[str, int]], days: int, rate: float, scheme: str,
              config: StudyConfig | None = None, *, write: bool = True) -> int:
    """Give stock-years one fixed exit setting instead of choosing it (the GP-parameter pilot).

    Existing fixed settings are kept when equal; any other existing setting is
    refused before anything is written, so a campaign never mixes fixed and
    chosen exits. With ``write=False`` it only checks. Returns how many it wrote
    (or would write).
    """
    todo = []
    for stock, year in stock_years:
        path = exits_path(campaign, stock, year)
        if path.exists():
            current = json.loads(path.read_text())
            if not current.get("fixed") or (int(current["days"]), float(current["rate"])) != (int(days), float(rate)):
                kind = "a fixed" if current.get("fixed") else "a chosen"
                raise ValueError(f"{stock} {year} already has {kind} exit after {current['days']} days or at "
                                 f"{current['rate']:+.1%}; a fixed exit needs a campaign of its own")
            continue
        todo.append((path, stock, year))
    if write:
        for path, stock, year in todo:
            provenance.write(path, {"stock": stock, "test_year": year, "days": int(days), "rate": float(rate),
                                    "fixed": True, "scheme": scheme, "provenance": provenance.collect(config)})
    return len(todo)


def plan_path(campaign: str, name: str) -> Path:
    return store.data_dir() / "results" / campaign / f"tasks.{name}.json"


def plan_names(campaign: str) -> list[str]:
    """The plans a campaign holds."""
    folder = store.data_dir() / "results" / campaign
    return sorted(path.name.removeprefix("tasks.").removesuffix(".json") for path in folder.glob("tasks.*.json"))


def save_plan(campaign: str, kind: str, tasks: list[Task], *, development_only: bool,
              replace: bool = False, stocks: list[str] | None = None, scheme: str = "rolling",
              name: str | None = None, frozen: dict[str, Any] | None = None) -> Path:
    """Save a plan; ``frozen`` is its ``manifest``, which its tasks check before they run."""
    name = name or kind
    path = plan_path(campaign, name)
    if path.exists() and not replace:
        raise FileExistsError(f"{path} exists; running jobs use its numbering (pass replace to plan again)")
    if development_only:
        check_roles(sorted(stocks if stocks is not None else {t.stock for t in tasks} - {"*"}))
    claim_campaign(campaign, scheme, development_only)
    named = [dataclasses.replace(t, plan=name) for t in tasks]
    record = {"campaign": campaign, "name": name, "kind": kind, "development_only": development_only,
              "scheme": scheme, "stocks": sorted(stocks if stocks is not None else {t.stock for t in tasks} - {"*"}),
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "git": provenance.git_state(), "manifest": frozen, "tasks": [dataclasses.asdict(t) for t in named]}
    provenance.write(path, record)
    return path


def load_plan(campaign: str, name: str) -> tuple[bool, list[Task]]:
    """Whether the plan is development-only, and its tasks."""
    path = plan_path(campaign, name)
    if not path.exists():
        raise FileNotFoundError(f"no plan {name!r} for campaign {campaign!r} ({path})")
    record = json.loads(path.read_text())
    tasks = [Task(**{**t, "runs": tuple(t["runs"]), "plan": t.get("plan") or name}) for t in record["tasks"]]
    return bool(record["development_only"]), tasks


def plan_scheme(campaign: str, name: str) -> str:
    """The walk-forward scheme of a plan (rolling unless planned otherwise)."""
    return str(json.loads(plan_path(campaign, name).read_text()).get("scheme", "rolling"))


def plan_stocks(campaign: str, name: str) -> list[str]:
    """The stocks of a plan (those a pooled task covers)."""
    return list(json.loads(plan_path(campaign, name).read_text())["stocks"])


def planned_runs(campaign: str, name: str | None = None) -> pd.DataFrame:
    """The runs a campaign's plans of runs list (only plan ``name``'s if given): rung, stock, test
    year, stage and run, once each."""
    rows = []
    for plan in [name] if name else plan_names(campaign):
        path = plan_path(campaign, plan)
        if not path.exists():
            raise FileNotFoundError(f"no plan {plan!r} for campaign {campaign!r} ({path})")
        record = json.loads(path.read_text())
        if record["kind"] != "runs":
            continue
        for task in record["tasks"]:
            for stock in record["stocks"] if task["stock"] == "*" else [task["stock"]]:
                rows += [{"rung": task["rung"], "stock": stock, "test_year": int(task["test_year"]),
                          "stage": int(task["stage"]), "run": int(n)} for n in task["runs"]]
    return pd.DataFrame(rows, columns=["rung", "stock", "test_year", "stage", "run"]).drop_duplicates(
        ignore_index=True)


def execute(task: Task, campaign: str, config: StudyConfig, rungs: dict[str, Rung], *,
            development_only: bool) -> dict[str, int]:
    """Do what is left of one task; returns how many items were done and skipped."""
    development_only = claim_campaign(campaign, config.windows.scheme) or development_only
    if development_only:
        check_development(task, config)
        check_roles(plan_stocks(campaign, task.plan or task.kind) if task.stock == "*" else [task.stock])
    check_manifest(task, campaign, config, rungs)
    if task.stock == "*":
        return execute_pooled(task, campaign, config)
    if task.kind == "exits":
        if exits_path(campaign, task.stock, task.test_year).exists():
            exit_settings(campaign, task.stock, task.test_year, config)  # refuses another configuration's
            return {"done": 0, "skipped": 1}
        choose_exits(task.stock, task.test_year, rungs["A0"], config, exit_grid(), campaign=campaign)
        return {"done": 1, "skipped": 0}
    days, rate = exit_settings(campaign, task.stock, task.test_year, config)
    method = method_digest(task.rung, rungs)
    done = skipped = 0
    for number in task.runs:
        spec = RunSpec(task.stock, task.test_year, number, task.stage, days, rate)
        if finished_result(result_path(campaign, task.rung, spec), task.rung, spec, config, method):
            skipped += 1
            continue
        run_method(task.rung, spec, rungs, config, campaign)
        done += 1
    return {"done": done, "skipped": skipped}


def exit_settings(campaign: str, stock: str, test_year: int, config: StudyConfig) -> tuple[int, float]:
    """A stock-year's exit settings (d, r), which must have been set under the current configuration."""
    path = exits_path(campaign, stock, test_year)
    if not path.exists():
        raise FileNotFoundError(f"{stock} {test_year}: no exit settings at {path}; run the exits tasks first")
    chosen = json.loads(path.read_text())
    made_under = (chosen.get("provenance") or {}).get("config_digest")
    if made_under is not None and made_under != config.digest():
        raise ValueError(f"{stock} {test_year}: the exit settings at {path} were set under another study "
                         "configuration; new settings need a campaign of their own")
    return int(chosen["days"]), float(chosen["rate"])


def execute_pooled(task: Task, campaign: str, config: StudyConfig) -> dict[str, int]:
    """Runs of a benchmark trained across all the campaign's stocks; each writes a result per stock."""
    from regime_fusion.ml.tft import run_tft

    stocks = plan_stocks(campaign, task.plan or task.kind)
    exits = {stock: exit_settings(campaign, stock, task.test_year, config) for stock in stocks}
    settings = load_benchmarks()
    method = settings.method_digest(task.rung)
    done = skipped = 0
    for number in task.runs:
        specs = [RunSpec(s, task.test_year, number, task.stage, *exits[s]) for s in stocks]
        there = [finished_result(result_path(campaign, task.rung, spec), task.rung, spec, config, method)
                 for spec in specs]  # every stock's, so that a result made otherwise stops the task
        if all(there):
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


def finished(task: Task, campaign: str, config: StudyConfig | None = None,
             rungs: dict[str, Rung] | None = None) -> bool:
    """Whether a task's results are all there. Given the configuration and the rungs, each must also be
    whole and made with the current settings, as ``execute`` requires; one that is not leaves the task
    unfinished, so that running it stops on that result and says why."""
    if task.kind == "exits":
        if not exits_path(campaign, task.stock, task.test_year).exists():
            return False
        if config is not None:
            try:
                exit_settings(campaign, task.stock, task.test_year, config)
            except ValueError:
                return False
        return True
    stocks = plan_stocks(campaign, task.plan or task.kind) if task.stock == "*" else [task.stock]
    if config is None or rungs is None:
        root = store.data_dir() / "results" / campaign / task.rung
        return all((root / stock / f"{task.test_year}.stage{task.stage}.run{n:02d}.json").exists()
                   for stock in stocks for n in task.runs)
    try:
        check_manifest(task, campaign, config, rungs)
        method = method_digest(task.rung, rungs)
        for stock in stocks:
            days, rate = exit_settings(campaign, stock, task.test_year, config)
            for n in task.runs:
                spec = RunSpec(stock, task.test_year, n, task.stage, days, rate)
                path = result_path(campaign, task.rung, spec)
                if not path.exists() or result_problem(path, task.rung, spec, config, method):
                    return False
    except (FileNotFoundError, ValueError):
        return False
    return True


def collect(campaign: str, costs: list[float], primary: float) -> pd.DataFrame:
    """One row per run of the campaign: its settings, fitness and evaluation measures.

    Refuses a campaign whose results are not all whole and made under one study configuration, one
    version of each method's settings and one version of the data of each file and sessions read.
    """
    root = store.data_dir() / "results" / campaign
    rows, problems, commits = [], [], set()
    configs, methods, inputs = defaultdict(set), defaultdict(set), defaultdict(set)
    for path in sorted(root.glob("*/*/*.stage*.run*.json")):
        try:
            record = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            problems.append(f"{path}: not a readable result")
            continue
        mark = record.get("fingerprint") if isinstance(record, dict) else None
        if not isinstance(mark, dict) or not FINGERPRINT <= set(mark) or "evaluation" not in record:
            problems.append(f"{path}: not a whole result, or written before results carried fingerprints")
            continue
        configs[mark["config"]].add(str(record.get("rung")))
        methods[record.get("rung")].add(mark["method"])
        for key, digest in mark["inputs"].items():
            inputs[key].add(digest)
        commits.add(((record.get("provenance") or {}).get("git") or {}).get("commit"))
        row = {key: record.get(key) for key in ("rung", "stock", "test_year", "stage", "run", "exit_days",
                                                "exit_rate", "fitness", "seconds", "scheme")}
        row["prices_digest"] = mark["evaluation_prices"]
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
        for source, weight in (record.get("selection_weights") or {}).items():  # ML-TFT's counterpart
            row[f"select_{source.lower()}"] = weight
        row["model"] = record.get("model")
        row["signal"] = record.get("signal")
        rows.append(row)
    if len(configs) > 1:
        problems.append("runs under different study configurations: "
                        + "; ".join(f"{digest} ({', '.join(sorted(m))})" for digest, m in configs.items()))
    problems += [f"{m}: runs with different settings ({', '.join(sorted(d))})" for m, d in methods.items() if len(d) > 1]
    problems += [f"{key}: runs read {len(d)} versions of these data" for key, d in inputs.items() if len(d) > 1]
    if problems:
        shown = "\n  ".join(problems[:20]) + (f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else "")
        raise ValueError(f"campaign {campaign!r}: its results do not all come from the same settings and data; "
                         f"new settings or data need a campaign of their own:\n  {shown}")
    commits.discard(None)
    if len(commits) > 1:
        log.warning("collect: %s's runs come from %d commits of the code: %s", campaign, len(commits),
                    ", ".join(sorted(c[:7] for c in commits)))
    return pd.DataFrame(rows)
