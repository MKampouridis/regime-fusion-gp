"""Command line for GP runs: ``python -m regime_fusion.engine <command>``.

Commands (each has --help):

  run       evolve one model (rung, stock, test year, run number, stage) and evaluate it;
            a rule-based or machine-learning benchmark (configs/benchmarks.yaml) can take the rung's place
  exits     choose a stock-year's exit settings (d, r) on its validation year with rung A0
  tasks     plan a campaign as numbered tasks, for job arrays (engine/tasks.py)
  run-task  do task number i of a campaign (what a job-array element runs)
  status    how many of a campaign's tasks are finished
  collect   gather a campaign's runs into one table, results/<campaign>/runs.parquet

Results go to $RFGP_DATA_DIR/results/<campaign>/ with a provenance record each.
"""

from __future__ import annotations

import argparse
import json
import logging

from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.data.universe import DEFAULT_UNIVERSE, load_universe
from regime_fusion.engine import tasks
from regime_fusion.engine.rungs import exit_grid, load_rungs
from regime_fusion.engine.run import RunSpec, choose_exits, exits_path, run
from regime_fusion.ml.models import load_settings as load_benchmarks

log = logging.getLogger("regime_fusion.engine")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m regime_fusion.engine", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("run", help="evolve and evaluate one model")
    sub.add_argument("--rung", required=True)
    sub.add_argument("--stock", required=True)
    sub.add_argument("--test-year", type=int, required=True)
    sub.add_argument("--run", type=int, default=0)
    sub.add_argument("--stage", type=int, choices=(1, 2), default=2)
    sub.add_argument("--exit", nargs=2, metavar=("DAYS", "RATE"), help="exit settings; default: chosen ones")
    add_common(sub)
    sub.set_defaults(handler=run_command)

    sub = commands.add_parser("exits", help="choose a stock-year's exit settings with rung A0")
    sub.add_argument("--stock", required=True)
    sub.add_argument("--test-year", type=int, required=True)
    add_common(sub)
    sub.set_defaults(handler=exits_command)

    sub = commands.add_parser("tasks", help="plan a campaign as numbered tasks")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--kind", choices=tasks.KINDS, required=True)
    sub.add_argument("--rungs", nargs="+", default=[], help="rungs (runs only)")
    sub.add_argument("--stocks", nargs="+", help="stock ids (default: the main universe)")
    sub.add_argument("--include-holdout", action="store_true", help="also the hold-out stocks")
    sub.add_argument("--years", nargs="+", type=int, help="test years (default: all)")
    sub.add_argument("--stages", nargs="+", type=int, choices=(1, 2), default=[2], help="(runs only; default 2)")
    sub.add_argument("--runs", type=int, help="runs per configuration (default: the study's)")
    sub.add_argument("--block", type=int, help="runs per task (default: all of them)")
    sub.add_argument("--development", action="store_true",
                     help="development-only campaign (the pilot): no data after the development period")
    sub.add_argument("--replace", action="store_true", help="plan again although tasks exist")
    sub.set_defaults(handler=tasks_command)

    sub = commands.add_parser("run-task", help="do task number i of a campaign")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--kind", choices=tasks.KINDS, required=True)
    sub.add_argument("--index", type=int, required=True, help="task number, from 0 (a job array's index)")
    sub.add_argument("--count", type=int, default=1, help="do this many consecutive tasks (default 1)")
    sub.set_defaults(handler=run_task_command)

    sub = commands.add_parser("status", help="how many of a campaign's tasks are finished")
    sub.add_argument("--campaign", required=True)
    sub.set_defaults(handler=status_command)

    sub = commands.add_parser("collect", help="gather a campaign's runs into one table")
    sub.add_argument("--campaign", required=True)
    sub.set_defaults(handler=collect_command)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.handler(args) or 0


def add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--campaign", default="dev", help="results folder (default: dev)")
    sub.add_argument("--sources", nargs="+", help="override the rung's sources (development only)")
    sub.add_argument("--population", type=int, help="override the population size (development only)")
    sub.add_argument("--generations", type=int, help="override the generations (development only)")


def overrides(args: argparse.Namespace) -> dict:
    out = {}
    if args.population:
        out["population"] = args.population
    if args.generations:
        out["generations"] = args.generations
    return out


def run_command(args: argparse.Namespace) -> int:
    config, rungs, benchmarks = load_config(), load_rungs(), load_benchmarks()
    if args.rung not in rungs and args.rung not in benchmarks.methods():
        raise SystemExit(f"unknown rung or benchmark {args.rung!r}")
    if args.exit:
        days, rate = int(args.exit[0]), float(args.exit[1])
    else:
        path = exits_path(args.campaign, args.stock, args.test_year)
        if not path.exists():
            raise SystemExit(f"no exit settings at {path}; run the exits command or pass --exit")
        chosen = json.loads(path.read_text())
        days, rate = chosen["days"], chosen["rate"]
    spec = RunSpec(args.stock, args.test_year, args.run, args.stage, days, rate)
    if args.rung in rungs:
        record = run(spec, rungs[args.rung], config, campaign=args.campaign,
                     sources=tuple(args.sources) if args.sources else None, overrides=overrides(args))
    else:
        record = tasks.run_method(args.rung, spec, rungs, config, args.campaign)
    primary = f"{config.primary_cost_bps_per_side:g}bps"
    log.info("%s %s %d run %d stage %d: train fitness %.3f; evaluation Sharpe %.3f with %d trades; %.1f s",
             args.rung, args.stock, args.test_year, args.run, args.stage, record["fitness"],
             record["evaluation"][primary]["sharpe"], record["evaluation"][primary]["trades"],
             record["seconds"])
    log.info("model:\n%s", record["model"])
    return 0


def exits_command(args: argparse.Namespace) -> int:
    config, rungs = load_config(), load_rungs()
    choice = choose_exits(args.stock, args.test_year, rungs["A0"], config, exit_grid(), campaign=args.campaign,
                          sources=tuple(args.sources) if args.sources else None, overrides=overrides(args))
    log.info("%s %d: exit after %d days or at +%.1f%%", args.stock, args.test_year, choice["days"],
             100 * choice["rate"])
    return 0


def tasks_command(args: argparse.Namespace) -> int:
    config = load_config()
    universe = load_universe(DEFAULT_UNIVERSE)
    if args.stocks:
        chosen = [universe.get(stock) for stock in args.stocks]
        if any(stock.role == "holdout" for stock in chosen) and not args.include_holdout:
            raise SystemExit("hold-out stocks need --include-holdout")
        stocks = [stock.id for stock in chosen]
    else:
        roles = ("main", "holdout") if args.include_holdout else ("main",)
        stocks = [stock.id for stock in universe.select(*roles, include_holdout=args.include_holdout)]
    gp_rungs, benchmarks = load_rungs(), load_benchmarks().methods()
    unknown = sorted(set(args.rungs) - set(gp_rungs) - set(benchmarks))
    if unknown:
        raise SystemExit(f"unknown rungs or benchmarks {unknown}")
    years = args.years or list(config.period.test_years)
    runs = {rung: (benchmarks[rung].runs if rung in benchmarks else args.runs or config.runs_per_configuration)
            for rung in args.rungs}
    try:
        planned = tasks.plan(args.kind, stocks, years, config, rungs=args.rungs, stages=tuple(args.stages),
                             runs=runs, block=args.block, development_only=args.development)
        path = tasks.save_plan(args.campaign, args.kind, planned, development_only=args.development,
                               replace=args.replace)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from None
    log.info("%s: %d %s tasks (indices 0 to %d) in %s", args.campaign, len(planned), args.kind,
             len(planned) - 1, path)
    return 0


def run_task_command(args: argparse.Namespace) -> int:
    config, rungs = load_config(), load_rungs()
    development_only, planned = tasks.load_plan(args.campaign, args.kind)
    chosen = planned[args.index:args.index + args.count]
    if not chosen:
        raise SystemExit(f"no task {args.index}: the {args.kind} plan has {len(planned)} tasks")
    for number, task in enumerate(chosen, start=args.index):
        outcome = tasks.execute(task, args.campaign, config, rungs, development_only=development_only)
        log.info("task %d (%s %s %s %d stage %d): %d done, %d already done", number, task.kind, task.rung,
                 task.stock, task.test_year, task.stage, outcome["done"], outcome["skipped"])
    return 0


def status_command(args: argparse.Namespace) -> int:
    for kind in tasks.KINDS:
        try:
            _, planned = tasks.load_plan(args.campaign, kind)
        except FileNotFoundError:
            continue
        left = [i for i, task in enumerate(planned) if not tasks.finished(task, args.campaign)]
        log.info("%s %s: %d of %d tasks finished%s", args.campaign, kind, len(planned) - len(left), len(planned),
                 f"; first unfinished: {left[:10]}" if left else "")
    return 0


def collect_command(args: argparse.Namespace) -> int:
    config = load_config()
    table = tasks.collect(args.campaign, list(config.costs_bps_per_side), config.primary_cost_bps_per_side)
    path = store.data_dir() / "results" / args.campaign / "runs.parquet"
    store.write_parquet(path, table, campaign=args.campaign)
    log.info("%s: %d runs collected in %s", args.campaign, len(table), path)
    return 0

