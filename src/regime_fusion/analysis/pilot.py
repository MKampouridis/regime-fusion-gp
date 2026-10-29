"""Checks for the pilot (Phase 5, on 2015-2018 data only): timing, convergence, number of runs.

* ``timing``: seconds per run for each method, and the CPU hours the main
  campaign would need at that speed, with each method's own number of runs (a
  pooled model's run serves every stock at once).
* ``histories`` and ``convergence``: each run's progress, the share of its
  total rise in best training fitness reached by each generation (well defined
  whatever the sign of the fitness), and the share of runs still improving in
  their last generations, which says whether there are enough generations.
* ``run_count_check``: from stock-years with many runs (the plan's 100-run
  check), how much the median of n runs varies around the median of all of
  them, for several n; it says whether 30 runs are enough or 50 are needed.
* ``gp_parameters``: the GP-parameter pilot's decision (docs/PROTOCOL.md,
  section 9). For each population and tournament size (candidates named
  ``<rung>-p<population>-t<tournament>``, all with the same budget of
  evaluations), the generations are the median convergence generation of its
  runs, rounded up to a multiple of 10; its converged fitness is each run's
  best training fitness at that generation. Of the candidates whose converged
  fitness is, in median paired difference, within ``margin`` of the best
  candidate's, the one with the fewest evaluations (population x generations)
  wins. The validation Sharpe ratio at that generation is reported beside it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from regime_fusion.data import store

CELL = ["rung", "stock", "test_year", "stage"]
CANDIDATE = re.compile(r"^(?P<base>.+)-p(?P<population>\d+)-t(?P<tournament>\d+)$")


def timing(runs: pd.DataFrame, *, stocks: int = 50, years: int = 7, runs_for: dict[str, int] | int = 30,
           pooled: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Seconds per run for each method, and the CPU hours of a campaign of that size.

    ``runs_for`` gives each method's runs per stock-year (one number for all, or
    one per method). A pooled method's record holds the seconds of the run that
    served every stock, so it counts once per test year, stage and run.
    """
    rows = {}
    for rung, group in runs.groupby("rung"):
        count = runs_for if isinstance(runs_for, int) else runs_for.get(rung, 30)
        if rung in pooled:
            seconds = group.drop_duplicates(["test_year", "stage", "run"])["seconds"]
            hours = seconds.median() * years * count / 3600
        else:
            seconds = group["seconds"]
            hours = seconds.median() * stocks * years * count / 3600
        rows[rung] = {"runs": len(seconds), "median_seconds": seconds.median(),
                      "p90_seconds": seconds.quantile(0.9), "runs_per_stock_year": count,
                      "campaign_cpu_hours": hours}
    return pd.DataFrame(rows).T


def histories(campaign: str, root: Path | None = None, field: str = "history") -> pd.DataFrame:
    """The generation-by-generation ``history`` (or validation ``checkpoints``) of every GP run."""
    base = (root or store.data_dir()) / "results" / campaign
    frames = []
    for path in sorted(base.glob("*/*/*.stage*.run*.json")):
        record = json.loads(path.read_text())
        if not record.get(field):
            continue  # benchmarks have no generations; test-year runs no checkpoints
        frame = pd.DataFrame(record[field])
        for key in CELL + ["run"]:
            frame[key] = record[key]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def progress(history: pd.DataFrame) -> pd.DataFrame:
    """Adds each run's progress: (best fitness - its first value) / (its last value - its first value).

    Progress runs from 0 to 1 whatever the sign of the fitness (the best fitness
    never falls, thanks to elitism); it is missing for a run that never improved.
    """
    keys = CELL + ["run"]
    ordered = history.sort_values([*keys, "generation"])
    first = ordered.groupby(keys)["best_fitness"].transform("first")
    last = ordered.groupby(keys)["best_fitness"].transform("last")
    rise = (last - first).where(last > first)
    return ordered.assign(progress=(ordered["best_fitness"] - first) / rise)


def convergence(history: pd.DataFrame, last: int = 10, gain: float = 0.01) -> dict[str, pd.DataFrame]:
    """Per method: the median progress of each generation, and the share of runs that gained more
    than ``gain`` of their total rise over their last ``last`` generations (still improving)."""
    keys = CELL + ["run"]
    runs = progress(history)
    curve = runs.groupby(["rung", "generation"])["progress"].median().unstack("rung")
    final_generation = runs.groupby(keys)["generation"].transform("max")
    before_end = runs.loc[runs["generation"] == final_generation - last].set_index(keys)["progress"]
    improving = (1.0 - before_end > gain).groupby(level="rung").mean().rename("still_improving")
    return {"curve": curve, "still_improving": improving.to_frame()}


def convergence_generation(history: pd.DataFrame, tolerance: float = 0.01) -> pd.Series:
    """Per run, the first generation by which the best training fitness had made all but
    ``tolerance`` of its total rise (generation 0 for a run that never improved)."""
    keys = CELL + ["run"]
    runs = progress(history)
    reached = runs[(runs["progress"] >= 1.0 - tolerance) | runs["progress"].isna()]
    return reached.groupby(keys)["generation"].min().rename("convergence_generation")


def run_count_check(runs: pd.DataFrame, measure: str, rng: np.random.Generator,
                    sizes: tuple[int, ...] = (10, 20, 30, 50), draws: int = 1000) -> pd.DataFrame:
    """How far the median of n runs strays from the median of all runs, over cells with many runs.

    For each cell (method, stock, test year, stage) with at least max(sizes) runs,
    n runs are drawn without replacement ``draws`` times. Reported per n: the
    median over cells of the standard deviation of the n-run median, and of its
    95th-percentile absolute distance from the all-run median.
    """
    rows = []
    for cell, group in runs.groupby(CELL):
        values = group[measure].dropna().to_numpy()
        if len(values) < max(sizes):
            continue
        full = np.median(values)
        for n in sizes:
            medians = np.array([np.median(rng.choice(values, n, replace=False)) for _ in range(draws)])
            rows.append({"cell": cell, "n": n, "sd": medians.std(ddof=1),
                         "p95_distance": np.quantile(np.abs(medians - full), 0.95)})
    if not rows:
        return pd.DataFrame(columns=["cells", "sd", "p95_distance"])
    frame = pd.DataFrame(rows)
    return frame.groupby("n").agg(cells=("cell", "nunique"), sd=("sd", "median"),
                                  p95_distance=("p95_distance", "median"))


def gp_parameters(history: pd.DataFrame, checkpoints: pd.DataFrame, *, tolerance: float = 0.01,
                  margin: float = 0.05, every: int = 10) -> tuple[pd.DataFrame, dict]:
    """The GP-parameter pilot's table (one row per population and tournament size) and its choice."""
    parts = history["rung"].str.extract(CANDIDATE)
    if parts["base"].isna().all():
        raise ValueError("no GP-parameter pilot candidates (rungs named <rung>-p<population>-t<tournament>)")
    history = history.assign(**parts)[parts["base"].notna()]
    history = history.astype({"population": int, "tournament": int})
    reached = convergence_generation(history, tolerance).rename("generation").reset_index()
    units = ["base", "stock", "test_year", "stage", "run"]
    rows, fitness = {}, {}
    for (population, tournament), group in history.groupby(["population", "tournament"]):
        cap = int(group["generation"].max())
        mine = reached[reached["rung"].isin(group["rung"].unique())]
        generations = min(cap, max(every, every * math.ceil(mine["generation"].median() / every)))
        at = group[group["generation"] == generations]
        fitness[(population, tournament)] = at.set_index(units)["best_fitness"]
        marks = checkpoints[(checkpoints["rung"].isin(group["rung"].unique()))
                            & (checkpoints["generation"] == generations)]
        rows[(population, tournament)] = {
            "population": population, "tournament": tournament, "runs": len(at),
            "generations": generations, "evaluations": population * generations,
            "converged_fitness": float(at["best_fitness"].median()),
            "validation_sharpe": float(marks["sharpe"].median()) if len(marks) else float("nan"),
            "still_rising_at_cap": float((mine["generation"] >= cap - every).mean()),
        }
    table = pd.DataFrame(rows.values()).set_index(["population", "tournament"]).sort_index()
    best = table["converged_fitness"].idxmax()
    table["fitness_vs_best"] = [float((fitness[key] - fitness[best]).median()) for key in table.index]
    eligible = table[table["fitness_vs_best"] >= -margin]
    chosen = eligible.sort_values(["evaluations", "validation_sharpe"], ascending=[True, False]).index[0]
    row = table.loc[chosen]
    choice = {"population": int(chosen[0]), "tournament": int(chosen[1]), "generations": int(row["generations"]),
              "evaluations": int(row["evaluations"]), "converged_fitness": float(row["converged_fitness"]),
              "validation_sharpe": float(row["validation_sharpe"]),
              "best_validation_sharpe": float(table["validation_sharpe"].max()),
              "flag": bool(table["validation_sharpe"].max() - row["validation_sharpe"] > 0.1)}
    return table, choice

