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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_fusion.data import store

CELL = ["rung", "stock", "test_year", "stage"]


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


def histories(campaign: str, root: Path | None = None) -> pd.DataFrame:
    """The generation-by-generation history of every GP run of a campaign."""
    base = (root or store.data_dir()) / "results" / campaign
    frames = []
    for path in sorted(base.glob("*/*/*.stage*.run*.json")):
        record = json.loads(path.read_text())
        if not record.get("history"):
            continue  # benchmarks have no generations
        frame = pd.DataFrame(record["history"])
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
