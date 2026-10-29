"""Command line for the analysis: ``python -m regime_fusion.analysis <command>``.

Commands (each has --help):

  compare  per-stock-year medians of a campaign's methods and the paired tests between them
  pilot    timing, convergence and the 100-run check of the pilot

Both read results/<campaign>/runs.parquet (``python -m regime_fusion.engine collect``)
and write tables to results/<campaign>/analysis/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from regime_fusion.analysis import pilot, results
from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.seeds import make_rng

log = logging.getLogger("regime_fusion.analysis")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m regime_fusion.analysis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("compare", help="per-stock-year medians and paired tests")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--measure", default="sharpe_10bps", help="a column of runs.parquet (default sharpe_10bps)")
    sub.add_argument("--rungs", nargs="+", help="methods in ladder order (default: all, sorted)")
    sub.add_argument("--against", help="also compare every method with this one")
    sub.add_argument("--stage", type=int, choices=(1, 2), default=2, help="1 for the pilot (validation year)")
    sub.set_defaults(handler=compare_command)

    sub = commands.add_parser("pilot", help="timing, convergence and the 100-run check")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--measure", default="sharpe_10bps")
    sub.set_defaults(handler=pilot_command)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.handler(args) or 0


def load_runs(campaign: str) -> pd.DataFrame:
    path = store.data_dir() / "results" / campaign / "runs.parquet"
    if not path.exists():
        raise SystemExit(f"no {path}; run `python -m regime_fusion.engine collect --campaign {campaign}`")
    return pd.read_parquet(path)


def output(campaign: str) -> Path:
    folder = store.data_dir() / "results" / campaign / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def compare_command(args: argparse.Namespace) -> int:
    runs = load_runs(args.campaign)
    runs = runs[runs["stage"] == args.stage]
    rungs = args.rungs or sorted(runs["rung"].unique())
    missing = sorted(set(rungs) - set(runs["rung"]))
    if missing:
        raise SystemExit(f"no stage-{args.stage} runs for {missing}")
    runs = runs[runs["rung"].isin(rungs)]
    folder, name = output(args.campaign), f"{args.measure}.stage{args.stage}"
    medians = results.stock_year_medians(runs, args.measure)[rungs]
    pairs = results.ladder(rungs)
    if args.against:
        pairs += [(rung, args.against) for rung in rungs if rung != args.against]
    tests = results.compare(medians, pairs)
    medians.to_csv(folder / f"medians.{name}.csv")
    results.best_fitness(runs, args.measure)[rungs].to_csv(folder / f"best_fitness.{name}.csv")
    results.by_year(runs, args.measure)[rungs].to_csv(folder / f"by_year.{name}.csv")
    cost = args.measure.rsplit("_", 1)[-1] if args.measure.endswith("bps") else "10bps"
    results.summary(runs, cost).loc[rungs].to_csv(folder / f"summary.stage{args.stage}.{cost}.csv")
    tests["pairs"].to_csv(folder / f"wilcoxon.{name}.csv", index=False)
    if "conover" in tests:
        tests["conover"].to_csv(folder / f"conover.{name}.csv", index=False)
        (folder / f"friedman.{name}.json").write_text(json.dumps(
            {**tests["friedman"], "stock_years": tests["stock_years"],
             "mean_ranks": tests["mean_ranks"].round(3).to_dict()}, indent=2) + "\n")
    log.info("%s, stage %d, %s: median over stock-years\n%s", args.campaign, args.stage, args.measure,
             medians.median().round(3).to_string())
    log.info("Wilcoxon-Holm:\n%s", tests["pairs"].round(4).to_string(index=False))
    log.info("tables in %s", folder)
    return 0


def pilot_command(args: argparse.Namespace) -> int:
    config = load_config()
    runs = load_runs(args.campaign)
    folder = output(args.campaign)
    speed = pilot.timing(runs, stocks=50, years=len(config.period.test_years),
                         runs_per=config.runs_per_configuration)
    speed.to_csv(folder / "timing.csv")
    history = pilot.histories(args.campaign)
    if len(history):
        converged = pilot.convergence(history)
        converged["curve"].to_csv(folder / "convergence_curve.csv")
        converged["still_improving"].to_csv(folder / "still_improving.csv")
        log.info("runs still improving over their last 10 generations:\n%s",
                 converged["still_improving"].round(3).to_string())
    counts = pilot.run_count_check(runs, args.measure, make_rng(config.base_seed, "pilot", "run-count"))
    counts.to_csv(folder / f"run_count.{args.measure}.csv")
    log.info("seconds per run and campaign CPU hours:\n%s", speed.round(2).to_string())
    log.info("spread of the median of n runs (%s):\n%s", args.measure, counts.round(4).to_string())
    return 0
