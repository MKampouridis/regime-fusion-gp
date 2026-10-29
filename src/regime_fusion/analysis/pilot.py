"""Checks for the pilot (Phase 5, on 2015-2018 data only): timing, convergence, number of runs.

* ``timing``: seconds per run for each method, and the CPU hours the main
  campaign would need at that speed.
* ``histories`` and ``convergence``: the best training fitness of each
  generation relative to the last, and the share of runs still improving in
  their last generations, which says whether 50 generations are enough.
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


def timing(runs: pd.DataFrame, *, stocks: int = 50, years: int = 7, runs_per: int = 30) -> pd.DataFrame:
    """Seconds per run for each method, and the CPU hours of a campaign of that size."""
    grouped = runs.groupby("rung")["seconds"]
    out = pd.DataFrame({"runs": grouped.size(), "median_seconds": grouped.median(),
                        "p90_seconds": grouped.quantile(0.9)})
    out["campaign_cpu_hours"] = out["median_seconds"] * stocks * years * runs_per / 3600
    return out


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


def convergence(history: pd.DataFrame, last: int = 10, gain: float = 0.01) -> dict[str, pd.DataFrame]:
    """Per method: the median best fitness of each generation relative to the run's final best,
    and the share of runs whose best fitness still rose by more than ``gain`` (of its final
    size) over the last ``last`` generations."""
    keys = CELL + ["run"]
    final_generation = history.groupby(keys)["generation"].transform("max")
    final = history.loc[history["generation"] == final_generation].set_index(keys)["best_fitness"]
    relative = history.join(final.rename("final"), on=keys)
    relative["share_of_final"] = relative["best_fitness"] / relative["final"].where(relative["final"] != 0)
    curve = relative.groupby(["rung", "generation"])["share_of_final"].median().unstack("rung")
    earlier = relative.loc[relative["generation"] == final_generation - last].set_index(keys)["best_fitness"]
    rise = (final - earlier.reindex(final.index)) / final.abs().where(final != 0)
    improving = (rise > gain).groupby(level="rung").mean().rename("still_improving")
    return {"curve": curve, "still_improving": improving.to_frame()}


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
