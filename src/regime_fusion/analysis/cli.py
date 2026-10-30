"""Command line for the analysis: ``python -m regime_fusion.analysis <command>``.

Commands (each has --help):

  compare    per-stock-year medians of a campaign's methods and the per-stock paired tests between them
  portfolio  daily returns of each method's equal-weight portfolio; the confirmatory paired tests, SPA,
             Deflated Sharpe, PBO, by regime
  random     each method's percentile among random entries with the same exit rule and trades
  pilot      timing, convergence and the 100-run check of the pilot
  gp-params  the GP-parameter pilot: population, generations and tournament size

Both read results/<campaign>/runs.parquet (``python -m regime_fusion.engine collect``)
and write tables to results/<campaign>/analysis/.

compare, portfolio and random check the runs against the campaign's plans:
every planned run of the methods analysed must have a result (--incomplete
instead leaves out the stock-years with runs missing, and says which), runs
outside the plans are left out, and --plan keeps one plan's runs only (the
pilot's 10 runs, say, without the 100-run check's extra ones). The portfolios
use the stock-years that every method analysed covers.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from regime_fusion.analysis import pilot, portfolio, results, stats
from regime_fusion.ml.models import load_settings as load_benchmarks
from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.engine import metrics, tasks
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
    sub.add_argument("--confirmatory", action="store_true",
                     help="test exactly the six comparisons of docs/PROTOCOL.md (their per-stock view; "
                          "their confirmatory tests are portfolio's)")
    sub.add_argument("--stage", type=int, choices=(1, 2), default=2, help="1 for the pilot (validation year)")
    add_selection(sub)
    sub.set_defaults(handler=compare_command)

    sub = commands.add_parser("portfolio", help="portfolio returns of each method; the confirmatory tests, SPA, "
                                                "Deflated Sharpe, PBO")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--methods", nargs="+", help="methods to compare (default: all in the campaign)")
    sub.add_argument("--cost", type=float, help="basis points per side (default: the primary cost)")
    sub.add_argument("--stage", type=int, choices=(1, 2), default=2)
    sub.add_argument("--benchmark", help="SPA benchmark (default: the best rule-based or ML benchmark)")
    sub.add_argument("--trials", type=int, help="configurations tried, for the Deflated Sharpe "
                                                "(default: counted from the plans of --trials-from)")
    sub.add_argument("--trials-from", nargs="+", metavar="CAMPAIGN",
                     help="campaigns whose plans list the configurations tried (default: this one; the "
                          "protocol counts the pilot's too)")
    sub.add_argument("--pairs", nargs="+", metavar="FIRST:SECOND",
                     help="comparisons for the paired Sharpe-ratio tests (default: the six confirmatory ones of "
                          "docs/PROTOCOL.md whose methods are analysed)")
    sub.add_argument("--regimes", default="dc", help="market regimes for the breakdown (default dc, A3's)")
    add_selection(sub)
    sub.set_defaults(handler=portfolio_command)

    sub = commands.add_parser("random", help="percentiles among random entries (1,000 draws)")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--methods", nargs="+", required=True)
    sub.add_argument("--cost", type=float, help="basis points per side (default: the primary cost)")
    sub.add_argument("--stage", type=int, choices=(1, 2), default=2)
    sub.add_argument("--draws", type=int, default=1000)
    add_selection(sub)
    sub.set_defaults(handler=random_command)

    sub = commands.add_parser("gp-params", help="the GP-parameter pilot's table and choice")
    sub.add_argument("--campaign", required=True)
    sub.set_defaults(handler=gp_params_command)

    sub = commands.add_parser("pilot", help="timing, convergence and the 100-run check")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--measure", default="sharpe_10bps")
    sub.set_defaults(handler=pilot_command)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.handler(args) or 0


def add_selection(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--plan", help="only the runs of this plan (default: those of every plan of runs)")
    sub.add_argument("--incomplete", action="store_true",
                     help="leave out the stock-years with planned runs missing instead of stopping")


def select(args: argparse.Namespace, runs: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """The runs of ``methods`` and the stage to analyse, checked against the campaign's plans (see the
    module notes)."""
    runs = runs[(runs["stage"] == args.stage) & runs["rung"].isin(methods)]
    try:
        planned = tasks.planned_runs(args.campaign, args.plan)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    planned = planned[(planned["stage"] == args.stage) & planned["rung"].isin(methods)]
    if planned.empty:
        if args.plan:
            raise SystemExit(f"plan {args.plan!r} has no stage-{args.stage} runs of {methods}")
        log.warning("%s: no plan lists these runs, so whether all of them are there is not checked", args.campaign)
        return runs
    inside = runs.merge(planned, on=results.RUN_KEYS)
    if len(inside) < len(runs):
        log.warning("%s: %d runs that no plan%s lists are left out", args.campaign, len(runs) - len(inside),
                    f" but {args.plan!r}" if args.plan else "")
    missing = results.missing_runs(inside, planned)
    if missing.empty:
        return inside
    counts = missing.groupby("rung").size().to_dict()
    if not args.incomplete:
        raise SystemExit(f"{len(missing)} planned runs have no result yet ({counts}); finish them "
                         "(engine status lists them), or pass --incomplete to leave out their stock-years")
    lacking = missing[results.KEYS].drop_duplicates()
    log.warning("--incomplete: %d stock-years with planned runs missing (%s) are left out for every method",
                len(lacking), counts)
    return results.keep_stock_years(inside, lacking, keep=False)


def trial_count(args: argparse.Namespace, methods: list[str]) -> dict[str, object]:
    """How many configurations count as trials for the Deflated Sharpe Ratio, and where from: the
    distinct methods the plans of ``--trials-from`` list (docs/PROTOCOL.md, section 5)."""
    if args.trials:
        return {"count": args.trials, "from": "given (--trials)"}
    campaigns = args.trials_from or [args.campaign]
    unplanned = [c for c in args.trials_from or [] if not tasks.plan_names(c)]
    if unplanned:
        raise SystemExit(f"--trials-from: no plans for {unplanned}")
    configurations = sorted({name for campaign in campaigns for name in tasks.planned_runs(campaign)["rung"]})
    if not configurations:
        log.warning("Deflated Sharpe: no plans list the configurations tried; counting the %d methods analysed",
                    len(methods))
        return {"count": len(methods), "from": "the methods analysed (no plans)", "configurations": methods}
    if not any(tasks.development_campaign(c) for c in campaigns):
        log.warning("Deflated Sharpe: %d configurations from %s only; the protocol also counts the pilot's "
                    "(pass --trials-from with the pilot campaigns too)", len(configurations), campaigns)
    return {"count": len(configurations), "from": f"the plans of {campaigns}", "configurations": configurations}


def paired_comparisons(given: list[str] | None, methods: list[str]) -> list[tuple[str, str]]:
    """The comparisons of the paired Sharpe-ratio tests: ``--pairs`` as given, or else the protocol's
    six confirmatory ones whose methods are both analysed."""
    if not given:
        pairs = [pair for pair in results.CONFIRMATORY if set(pair) <= set(methods)]
        if 0 < len(pairs) < len(results.CONFIRMATORY):
            log.warning("%d of the protocol's %d confirmatory comparisons have both methods here; Holm's "
                        "adjustment covers those only", len(pairs), len(results.CONFIRMATORY))
        return pairs
    pairs = []
    for item in given:
        first, colon, second = item.partition(":")
        if not (first and colon and second):
            raise SystemExit(f"--pairs: {item!r} is not FIRST:SECOND")
        pairs.append((first, second))
    absent = sorted({method for pair in pairs for method in pair} - set(methods))
    if absent:
        raise SystemExit(f"--pairs: no portfolio for {absent}")
    return pairs


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
    rungs = args.rungs or sorted(runs.loc[runs["stage"] == args.stage, "rung"].unique())
    runs = select(args, runs, rungs)
    missing = sorted(set(rungs) - set(runs["rung"]))
    if missing:
        raise SystemExit(f"no stage-{args.stage} runs for {missing}")
    runs = runs[runs["rung"].isin(rungs)]
    folder, name = output(args.campaign), f"{args.measure}.stage{args.stage}"
    medians = results.stock_year_medians(runs, args.measure)[rungs]
    if args.confirmatory:
        pairs = results.CONFIRMATORY
        absent = sorted({m for pair in pairs for m in pair} - set(rungs))
        if absent:
            raise SystemExit(f"the confirmatory comparisons need {absent}")
    else:
        pairs = results.ladder(rungs)
        if args.against:
            pairs += [(rung, args.against) for rung in rungs if rung != args.against]
    rng = make_rng(load_config().base_seed, "analysis", "compare", args.campaign, args.measure)
    tests = results.compare(medians, pairs, rng)
    medians.to_csv(folder / f"medians.{name}.csv")
    results.best_fitness(runs, args.measure)[rungs].to_csv(folder / f"best_fitness.{name}.csv")
    results.by_year(runs, args.measure)[rungs].to_csv(folder / f"by_year.{name}.csv")
    trust = results.trust_map(runs)
    if not trust.empty:
        trust.to_csv(folder / f"trust_map.stage{args.stage}.csv")
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
    log.info("Wilcoxon-Holm on stock-years, with the check clustered by year:\n%s",
             tests["pairs"].round(4).to_string(index=False))
    log.info("tables in %s", folder)
    return 0


def portfolio_command(args: argparse.Namespace) -> int:
    config = load_config()
    cost = config.primary_cost_bps_per_side if args.cost is None else args.cost
    runs = load_runs(args.campaign)
    methods = args.methods or sorted(runs.loc[runs["stage"] == args.stage, "rung"].unique())
    runs = select(args, runs, methods)
    absent = sorted(set(methods) - set(runs["rung"]))
    if absent:
        raise SystemExit(f"no stage-{args.stage} runs for {absent}")
    pairs = paired_comparisons(args.pairs, methods)
    shared = results.common_stock_years(runs)
    every = runs[results.KEYS].drop_duplicates()
    if len(shared) < len(every):
        fewest = runs.groupby("rung")[results.KEYS].apply(lambda g: len(g.drop_duplicates())).idxmin()
        log.warning("the methods cover different stock-years: the portfolios use the %d that all of them "
                    "cover, of %d (%s covers the fewest)", len(shared), len(every), fewest)
        runs = results.keep_stock_years(runs, shared)
    returns = portfolio.method_returns(runs, cost)
    benchmarks = [m for m in returns.columns if m.startswith(("RB-", "ML-"))]
    benchmark = args.benchmark or (max(benchmarks, key=lambda m: metrics.sharpe(returns[m].dropna()))
                                   if benchmarks else None)  # a flat portfolio (never trades) counts as zero
    if benchmark is None:
        raise SystemExit("no benchmark in the campaign; pass --benchmark")
    candidates = [m for m in returns.columns if m not in benchmarks]
    if not candidates:
        raise SystemExit("no GP rung in the campaign to test against the benchmark")
    rng = make_rng(config.base_seed, "analysis", "spa", args.campaign)
    trials = trial_count(args, list(returns.columns))
    outcome = portfolio.tests(returns, benchmark, candidates, int(trials["count"]), rng)
    outcome.update(trials=trials, stock_years=len(shared), stock_years_left_out=len(every) - len(shared))
    folder, name = output(args.campaign), f"{cost:g}bps.stage{args.stage}"
    if pairs:  # the same resampled days for every comparison and cost
        paired = stats.sharpe_differences(returns, pairs, make_rng(config.base_seed, "analysis",
                                                                   "sharpe-difference", args.campaign))
        paired.to_csv(folder / f"sharpe_difference.{name}.csv", index=False)
        outcome["sharpe_difference"] = paired.to_dict(orient="records")
        log.info("difference of annualised Sharpe ratios (stationary bootstrap, Holm):\n%s",
                 paired.round(4).to_string(index=False))
    store.write_parquet(folder / f"portfolio.{name}.parquet", returns, campaign=args.campaign, cost_bps=cost)
    (folder / f"portfolio_tests.{name}.json").write_text(json.dumps(outcome, indent=2, default=float) + "\n")
    log.info("%s at %g bps: annualised Sharpe of each portfolio %s", args.campaign, cost,
             outcome["annualised_sharpe"])
    log.info("SPA against %s: p = %.4f; Deflated Sharpe: %s", benchmark, outcome["spa"]["p"],
             {k: round(v, 3) for k, v in outcome["deflated_sharpe"].items()})
    if args.stage == 2:
        years = sorted(int(y) for y in runs["test_year"].unique())
        schemes = set(runs["scheme"].dropna()) if "scheme" in runs else set()
        scheme = schemes.pop() if len(schemes) == 1 else "rolling"
        try:
            states = portfolio.regime_states(args.regimes, years, scheme=scheme)
        except FileNotFoundError as error:
            log.warning("no breakdown by regime: %s", error)
        else:
            portfolio.by_regime(returns, states).to_csv(folder / f"by_regime.{args.regimes}.{name}.csv")
    return 0


def random_command(args: argparse.Namespace) -> int:
    config = load_config()
    cost = config.primary_cost_bps_per_side if args.cost is None else args.cost
    runs = select(args, load_runs(args.campaign), args.methods)
    folder = output(args.campaign)
    def rng_of(stock: str, year: int):  # seeds by stock-year, never by method (CLAUDE.md)
        return make_rng(config.base_seed, "analysis", "random", stock, year)

    for method in args.methods:
        table = portfolio.random_entry_check(runs, method, cost, rng_of, draws=args.draws)
        if table.empty:
            raise SystemExit(f"no stage-{args.stage} runs of {method}")
        table.to_csv(folder / f"random_entry.{method}.{cost:g}bps.stage{args.stage}.csv", index=False)
        log.info("%s: above the 95th percentile of random entry in %.1f%% of %d stock-years "
                 "(median percentile %.2f)", method, 100 * (table["percentile"] > 0.95).mean(), len(table),
                 table["percentile"].median())
    return 0


def gp_params_command(args: argparse.Namespace) -> int:
    history = pilot.histories(args.campaign)
    if history.empty:
        raise SystemExit(f"no GP runs in campaign {args.campaign!r}")
    try:
        table, choice = pilot.gp_parameters(history, pilot.histories(args.campaign, field="checkpoints"))
    except ValueError as error:
        raise SystemExit(str(error)) from None
    folder = output(args.campaign)
    table.to_csv(folder / "gp_params.csv")
    (folder / "gp_params_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    log.info("GP-parameter pilot:\n%s", table.round(3).to_string())
    log.info("choice: population %d, %d generations, tournament %d (%d evaluations)%s", choice["population"],
             choice["generations"], choice["tournament"], choice["evaluations"],
             "; its validation Sharpe ratio is more than 0.1 below the best candidate's: discuss" if choice["flag"]
             else "")
    return 0


def pilot_command(args: argparse.Namespace) -> int:
    config = load_config()
    runs = load_runs(args.campaign)
    folder = output(args.campaign)
    benchmarks = load_benchmarks().methods()
    runs_for = {m: (benchmarks[m].runs if m in benchmarks else config.runs_per_configuration)
                for m in runs["rung"].unique()}
    pooled = frozenset(m for m in runs_for if m in benchmarks and benchmarks[m].pooled)
    speed = pilot.timing(runs, stocks=50, years=len(config.period.test_years), runs_for=runs_for, pooled=pooled)
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
