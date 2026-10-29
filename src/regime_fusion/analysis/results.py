"""From a campaign's runs to the study's comparisons (docs/RESEARCH_PLAN.md, evaluation protocol).

The input is the table ``python -m regime_fusion.engine collect`` writes: one
row per run, with its rung (or benchmark), stock, test year, stage, run number,
training fitness and evaluation measures at every cost.

* ``stock_year_medians``: the median over runs of each stock-year and method,
  the unit of every test (the plan's paired design).
* ``best_fitness``: the measure of the run with the best training fitness, as in
  ActTrade's best-tree analysis.
* ``integration``: the whole model's Sharpe ratio minus that of its best single
  branch (the plan's ΔSRC_Max), per run.
* ``summary``: per method, the medians and means over stock-years of several
  measures, and the share of runs that never trade.
* ``compare``: Wilcoxon-Holm tests between named pairs, each with a bootstrap
  confidence interval for the median paired difference, and the Friedman test
  with Conover's comparisons across all methods, on stock-years that every
  method covers. ``CONFIRMATORY`` holds the six comparisons of docs/PROTOCOL.md.
* ``trust_map``: per method and regime state, the mean share of days on which
  each branch is on the model's active path (research question 4).
* ``missing_runs`` and ``common_stock_years``: which planned runs have no
  result, and the stock-years every method has, so that methods are compared
  on the same units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_fusion.analysis import stats

KEYS = ["stock", "test_year"]
RUN_KEYS = ["rung", "stock", "test_year", "stage", "run"]
CONFIRMATORY = [("A1", "A0"), ("A2", "A1"), ("A3", "A2"), ("A4", "A3"), ("A5", "A4"), ("A5", "A0")]


def stock_year_medians(runs: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Rows: (stock, test year); columns: methods; values: the median of ``measure`` over runs."""
    return runs.pivot_table(index=KEYS, columns="rung", values=measure, aggfunc="median")


def best_fitness(runs: pd.DataFrame, measure: str) -> pd.DataFrame:
    """The measure of each stock-year's best-training-fitness run, per method (first run on ties)."""
    ordered = runs.sort_values([*KEYS, "rung", "fitness", "run"], ascending=[True, True, True, False, True])
    best = ordered.groupby([*KEYS, "rung"], sort=False).head(1)
    return best.pivot_table(index=KEYS, columns="rung", values=measure, aggfunc="first")


def integration(runs: pd.DataFrame, cost: str = "10bps") -> pd.Series:
    """Whole-model Sharpe ratio minus the best single branch's, per run (missing without branches)."""
    branches = [c for c in runs.columns if c.endswith(f"_sharpe_{cost}") and c != f"sharpe_{cost}"]
    if not branches:
        return pd.Series(np.nan, index=runs.index, name="integration")
    return (runs[f"sharpe_{cost}"] - runs[branches].max(axis=1, skipna=True)).rename("integration")


def summary(runs: pd.DataFrame, cost: str = "10bps") -> pd.DataFrame:
    """Per method: medians (and the mean Sharpe ratio) over stock-year medians, and never-trading runs."""
    measures = [f"{m}_{cost}" for m in ("sharpe", "annual_return", "max_drawdown", "cvar_95", "trades_per_year")]
    frame = runs.assign(integration=integration(runs, cost), never_trades=runs[f"trades_{cost}"] == 0)
    rows = {}
    for rung, group in frame.groupby("rung"):
        medians = group.groupby(KEYS)[[*measures, "integration"]].median()
        rows[rung] = {
            "stock_years": len(medians),
            "runs": len(group),
            **{f"median_{c}": medians[c].median() for c in medians.columns},
            f"mean_sharpe_{cost}": medians[f"sharpe_{cost}"].mean(),
            "never_trades": group["never_trades"].mean(),
        }
    return pd.DataFrame(rows).T


def ladder(rungs: list[str]) -> list[tuple[str, str]]:
    """Each rung against the one before it: (A1, A0), (A2, A1), ..."""
    return list(zip(rungs[1:], rungs[:-1]))


def median_difference_ci(differences, rng: np.random.Generator, draws: int = 10_000,
                         level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the median of paired differences (units resampled)."""
    values = np.asarray(differences, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    medians = np.median(values[rng.integers(len(values), size=(draws, len(values)))], axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(medians, tail)), float(np.quantile(medians, 1 - tail))


def compare(table: pd.DataFrame, pairs: list[tuple[str, str]],
            rng: np.random.Generator | None = None) -> dict[str, object]:
    """Paired tests on a stock-year table; Friedman and Conover use the stock-years every method has."""
    complete = table.dropna()
    tested = stats.wilcoxon_pairs(table, pairs)
    if rng is not None:
        bounds = [median_difference_ci(table[a] - table[b], rng) for a, b in pairs]
        tested["ci_low"], tested["ci_high"] = [b[0] for b in bounds], [b[1] for b in bounds]
    out: dict[str, object] = {"pairs": tested, "stock_years": len(complete)}
    if table.shape[1] >= 3 and len(complete) >= 2:
        statistic, p = stats.friedman(complete)
        out.update(friedman={"statistic": statistic, "p": p}, conover=stats.conover(complete),
                   mean_ranks=complete.rank(axis=1, ascending=False).mean().sort_values())
    return out


def by_year(runs: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Median over stocks of the stock-year medians, per test year and method."""
    return stock_year_medians(runs, measure).groupby(level="test_year").median()


def trust_map(runs: pd.DataFrame) -> pd.DataFrame:
    """Rows: (method, regime state); columns: branches; values: mean share of days listened to."""
    columns = [c for c in runs.columns if c.startswith("listen_") and "_state" in c]
    if not columns:
        return pd.DataFrame()
    long = runs.melt(id_vars=["rung"], value_vars=columns, var_name="column", value_name="share").dropna()
    parts = long["column"].str.extract(r"listen_(?P<source>[a-z]+)_state(?P<state>\d+)")
    long = pd.concat([long, parts], axis=1)
    return long.pivot_table(index=["rung", "state"], columns="source", values="share", aggfunc="mean")


def missing_runs(runs: pd.DataFrame, planned: pd.DataFrame) -> pd.DataFrame:
    """The planned runs (``RUN_KEYS``) that have no result in ``runs``."""
    merged = planned[RUN_KEYS].merge(runs[RUN_KEYS].drop_duplicates(), on=RUN_KEYS, how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", RUN_KEYS].reset_index(drop=True)


def common_stock_years(runs: pd.DataFrame) -> pd.DataFrame:
    """The stock-years (``KEYS``) that every method in ``runs`` has."""
    cells = runs[["rung", *KEYS]].drop_duplicates()
    counts = cells.groupby(KEYS).size()
    return counts[counts == cells["rung"].nunique()].index.to_frame(index=False)


def keep_stock_years(runs: pd.DataFrame, stock_years: pd.DataFrame, keep: bool = True) -> pd.DataFrame:
    """The runs on ``stock_years`` (or, with ``keep=False``, the runs on the others)."""
    merged = runs.merge(stock_years[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True)
    chosen = merged["_merge"] == "both"
    return merged.loc[chosen if keep else ~chosen].drop(columns="_merge").reset_index(drop=True)

