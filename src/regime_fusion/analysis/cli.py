"""Command line for the analysis: ``python -m regime_fusion.analysis <command>``.

Commands (each has --help):

  compare    per-stock-year medians of a campaign's methods and the per-stock paired tests between them
  portfolio  daily returns of each method's equal-weight portfolio; the confirmatory paired tests, SPA,
             Deflated Sharpe, PBO, by regime
  random     each method's percentile among random entries with the same exit rule and trades
  pilot      timing, convergence and the 100-run check of the pilot
  gp-params  the GP-parameter pilot: population, generations and tournament size
  figures    the campaign's figures, as PDFs in reports/figures/<campaign>/ (needs matplotlib)

They read results/<campaign>/runs.parquet (``python -m regime_fusion.engine collect``)
or the run records, and write tables to results/<campaign>/analysis/.

compare, portfolio and random check the runs against the campaign's plans:
every planned run of the methods analysed must have a result (--incomplete
instead leaves out the stock-years with runs missing, and says which), runs
outside the plans are left out, and --plan keeps one plan's runs only (the
pilot's 10 runs, say, without the 100-run check's extra ones). compare and the
portfolios use the stock-years that every method analysed covers.
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
    sub.add_argument("--pairs", nargs="+", metavar="FIRST:SECOND",
                     help="test these comparisons instead (as for the FinBERT check: A0-FB:A0 A1-FB:A1)")
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

    sub = commands.add_parser("figures", help="the campaign's figures, as PDFs in reports/figures/<campaign>/")
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--png", action="store_true", help="a PNG of each figure as well")
    sub.set_defaults(handler=figures_command)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.handler(args) or 0


def add_selection(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--plan", help="only the runs of this plan (default: those of every plan of runs)")
    sub.add_argument("--incomplete", action="store_true",
                     help="leave out the stock-years with planned runs missing instead of stopping")


def planned_of(args: argparse.Namespace) -> pd.DataFrame:
    """The runs the campaign's plans (or --plan) list for the stage analysed."""
    try:
        planned = tasks.planned_runs(args.campaign, args.plan)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    return planned[planned["stage"] == args.stage]


def select(args: argparse.Namespace, runs: pd.DataFrame, methods: list[str],
           planned: pd.DataFrame | None = None) -> pd.DataFrame:
    """The runs of ``methods`` and the stage to analyse, checked against the campaign's plans
    (``planned``, from ``planned_of``; see the module notes)."""
    runs = runs[(runs["stage"] == args.stage) & runs["rung"].isin(methods)]
    planned = planned_of(args) if planned is None else planned
    planned = planned[planned["rung"].isin(methods)]
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
    distinct configurations the plans of ``--trials-from`` list (docs/PROTOCOL.md, section 5). A
    configuration is a method's name with the digest of the settings its plan froze, so a method run
    with other settings in another campaign counts again, and the same settings count once."""
    if args.trials:
        return {"count": args.trials, "from": "given (--trials)"}
    campaigns = args.trials_from or [args.campaign]
    found = {campaign: tasks.planned_methods(campaign) for campaign in campaigns}
    unplanned = [c for c in args.trials_from or [] if not found[c]]
    if unplanned:
        raise SystemExit(f"--trials-from: no plans of runs for {unplanned}")
    digests: dict[str, set[str]] = {}
    for name, digest in set().union(*found.values()):
        digests.setdefault(name, set()).update({digest} - {None})
    # a plan made before plans froze their settings adds its methods' names only where no plan knows more
    configurations = sorted(f"{name}@{d}" for name, known in digests.items() for d in known) + \
        sorted(name for name, known in digests.items() if not known)
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
        raise SystemExit(f"--pairs: {absent} not among the methods analysed")
    return pairs


def shared_stock_years(runs: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """The runs on the stock-years that every method analysed covers (the FinBERT rungs cover fewer),
    saying so when that leaves some out; stops when they share none."""
    shared = results.common_stock_years(runs)
    if shared.empty:
        raise SystemExit(f"the methods analysed share no stock-year: {methods}")
    every = runs[results.KEYS].drop_duplicates()
    if len(shared) == len(every):
        return runs
    fewest = runs.groupby("rung")[results.KEYS].apply(lambda g: len(g.drop_duplicates())).idxmin()
    log.warning("the methods cover different stock-years: the analysis uses the %d that all of them "
                "cover, of %d (%s covers the fewest)", len(shared), len(every), fewest)
    return results.keep_stock_years(runs, shared)


def default_methods(args: argparse.Namespace, runs: pd.DataFrame, planned: pd.DataFrame, option: str) -> list[str]:
    """The methods to analyse when none are named: those ``planned`` (``planned_of``) lists, so that a
    planned method without results stops the analysis instead of dropping out of it; when no plan lists
    any, those with results. Methods planned on different stock-years (the FinBERT rungs cover fewer)
    must be named with ``option``, so that they never shrink the others' units unseen."""
    if planned.empty:
        return sorted(runs.loc[runs["stage"] == args.stage, "rung"].unique())
    cells = {method: frozenset(map(tuple, group[results.KEYS].to_numpy().tolist()))
             for method, group in planned.groupby("rung")}
    if len(set(cells.values())) > 1:
        widest = max(len(c) for c in cells.values())
        fewer = sorted(m for m, c in cells.items() if len(c) < widest) or sorted(cells)
        raise SystemExit(f"the planned methods cover different stock-years ({', '.join(f'{m}: {len(cells[m])}' for m in fewer)}"
                         f", of {widest} at most); name the methods to analyse together with {option}")
    return sorted(cells)


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
    runs, planned = load_runs(args.campaign), planned_of(args)
    rungs = args.rungs or default_methods(args, runs, planned, "--rungs")
    runs = select(args, runs, rungs, planned)
    missing = sorted(set(rungs) - set(runs["rung"]))
    if missing:
        raise SystemExit(f"no stage-{args.stage} runs for {missing}")
    runs = shared_stock_years(runs[runs["rung"].isin(rungs)], rungs)
    folder, name = output(args.campaign), f"{args.measure}.stage{args.stage}"
    medians = results.stock_year_medians(runs, args.measure)[rungs]
    if args.pairs:
        pairs = paired_comparisons(args.pairs, rungs)
    elif args.confirmatory:
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
    runs, planned = load_runs(args.campaign), planned_of(args)
    methods = args.methods or default_methods(args, runs, planned, "--methods")
    runs = select(args, runs, methods, planned)
    absent = sorted(set(methods) - set(runs["rung"]))
    if absent:
        raise SystemExit(f"no stage-{args.stage} runs for {absent}")
    pairs = paired_comparisons(args.pairs, methods)
    every = runs[results.KEYS].drop_duplicates()
    runs = shared_stock_years(runs, methods)
    shared = runs[results.KEYS].drop_duplicates()
    if args.benchmark and args.benchmark not in methods:
        raise SystemExit(f"--benchmark {args.benchmark} is not among the methods analysed")
    returns = portfolio.method_returns(runs, cost)
    benchmarks = [m for m in returns.columns if m.startswith(("RB-", "ML-"))]
    benchmark = args.benchmark or (max(benchmarks, key=lambda m: metrics.sharpe(returns[m].dropna()))
                                   if benchmarks else None)  # a flat portfolio (never trades) counts as zero
    candidates = [m for m in returns.columns if m not in benchmarks and m != benchmark]
    if args.benchmark and not candidates:
        raise SystemExit(f"no GP rung among the methods analysed to test against {args.benchmark}")
    if benchmark is None or not candidates:  # a FinBERT or expanding-window check: GP rungs alone
        log.warning("no SPA test: %s", "no benchmark among the methods analysed (--benchmark names one)"
                    if benchmark is None else "no GP rung to test against the benchmark")
        benchmark, candidates = None, []
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
    if outcome["spa"]:
        log.info("SPA against %s: p = %.4f", benchmark, outcome["spa"]["p"])
    log.info("Deflated Sharpe (%d trials): %s", outcome["n_trials"],
             {k: round(v, 3) for k, v in outcome["deflated_sharpe"].items()})
    if args.methods:
        log.warning("Deflated Sharpe: the trials' variance comes from the %d portfolios analysed (--methods); "
                    "the protocol's comes from all the campaign's methods", returns.shape[1])
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
    counts = run_counts(runs, args.measure)
    counts.to_csv(folder / f"run_count.{args.measure}.csv")
    log.info("seconds per run and campaign CPU hours:\n%s", speed.round(2).to_string())
    log.info("spread of the median of n runs (%s):\n%s", args.measure, counts.round(4).to_string())
    return 0


def run_counts(runs: pd.DataFrame, measure: str) -> pd.DataFrame:
    """The 100-run check, with the same draws for its table and its figure."""
    return pilot.run_count_check(runs, measure, make_rng(load_config().base_seed, "pilot", "run-count"))


def figures_command(args: argparse.Namespace) -> int:
    try:
        from regime_fusion.analysis import figures
    except ImportError:  # matplotlib is optional
        raise SystemExit('the figures need matplotlib: pip install -e ".[figures]"') from None
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # it reports its font subsetting at INFO
    config = load_config()
    folder = store.data_dir() / "reports" / "figures" / args.campaign
    drawn = []
    history = pilot.histories(args.campaign)
    if len(history) and history["rung"].str.extract(pilot.CANDIDATE)["base"].notna().any():
        checkpoints = pilot.histories(args.campaign, field="checkpoints")
        table, choice = pilot.gp_parameters(history, checkpoints)
        drawn += figures.save(figures.gp_parameters(history, checkpoints, table, choice), folder, "gp_parameters",
                              args.png)
    if (store.data_dir() / "results" / args.campaign / "runs.parquet").exists():
        counts = run_counts(load_runs(args.campaign), f"sharpe_{config.primary_cost_bps_per_side:g}bps")
        if len(counts):
            drawn += figures.save(figures.run_count(counts, runs=config.runs_per_configuration), folder,
                                  "run_count", args.png)
    if not drawn:
        raise SystemExit(f"{args.campaign}: nothing to draw yet (so far the figures cover the GP-parameter pilot "
                         "and the 100-run check)")
    for path in drawn:
        log.info("wrote %s (%.0f KB)", path, path.stat().st_size / 1024)
    return 0
