"""Command line for GP runs: ``python -m regime_fusion.engine <command>``.

Commands (each has --help):

  run    evolve one model (rung, stock, test year, run number, stage) and evaluate it
  exits  choose a stock-year's exit settings (d, r) on its validation year with rung A0

Results go to $RFGP_DATA_DIR/results/<campaign>/ with a provenance record each.
"""

from __future__ import annotations

import argparse
import json
import logging

from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.engine.rungs import exit_grid, load_rungs
from regime_fusion.engine.run import RunSpec, choose_exits, run

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
    config, rung = load_config(), load_rungs()[args.rung]
    if args.exit:
        days, rate = int(args.exit[0]), float(args.exit[1])
    else:
        path = store.data_dir() / "results" / args.campaign / "exits" / f"{args.stock}.{args.test_year}.json"
        if not path.exists():
            raise SystemExit(f"no exit settings at {path}; run the exits command or pass --exit")
        chosen = json.loads(path.read_text())
        days, rate = chosen["days"], chosen["rate"]
    spec = RunSpec(args.stock, args.test_year, args.run, args.stage, days, rate)
    record = run(spec, rung, config, campaign=args.campaign,
                 sources=tuple(args.sources) if args.sources else None, overrides=overrides(args))
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
