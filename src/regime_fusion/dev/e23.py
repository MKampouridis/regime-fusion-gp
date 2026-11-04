"""E2 x E3: separate or pooled training, crossed with three trade designs (docs/DECISIONS.md, 4 Nov 2026).

Development data only: two folds, each training on the years before a validation year inside
2015-2018. Rung A1 (fusion functions, no regimes, tournament selection, active trading), with a
smaller budget than the campaigns (population 500, 20 generations): E1 showed that validation
results do not change with the amount of search (populations 250-2,000, up to 400 generations).

* separate: one GP per stock, trained and scored on that stock (as the campaigns do);
* pooled: one GP per fold, trained on all the stocks' training rows, its fitness computed from the
  equal-weight portfolio of its signals on every stock (``designs.PooledMarket``).

Either way the result of a run is each stock's daily returns on the validation year, at the primary
cost. Seeds depend on the unit (stock, or "pooled"), the fold and the run, never on the design.
Records go to ``results/dev-e23/<mode>/<design>/<fold>/``, each with provenance.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import StudyConfig, load_config
from regime_fusion.data import store
from regime_fusion.data.universe import load_universe
from regime_fusion.dev.designs import DESIGNS, Design, PooledMarket, StockMarket
from regime_fusion.engine.fitness import Evaluator
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.panel import PRICE_COLUMNS, Panel, load_panel
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.tree import Grammar, describe, signals, to_record
from regime_fusion.seeds import make_rng
from regime_fusion.windows import DateRange

CAMPAIGN = "dev-e23"
RUNG = "A1"
BUDGET = {"population": 500, "generations": 20}
RUNS = 10
MODES = ("separate", "pooled")
FOLDS = {  # validation year: (training, validation)
    "2017": (DateRange(dt.date(2015, 2, 1), dt.date(2016, 12, 31)), DateRange(dt.date(2017, 1, 1), dt.date(2017, 12, 31))),
    "2018": (DateRange(dt.date(2015, 2, 1), dt.date(2017, 12, 31)), DateRange(dt.date(2018, 1, 1), dt.date(2018, 12, 31))),
}


@dataclass(frozen=True)
class Job:
    mode: str
    design: str
    fold: str
    run: int
    stock: str | None = None   # separate mode only

    @property
    def unit(self) -> str:
        return self.stock or "pooled"

    def path(self) -> Path:
        return store.data_dir() / "results" / CAMPAIGN / self.mode / self.design / self.fold / \
            f"{self.unit}.run{self.run:02d}.json"


def check_development(config: StudyConfig) -> None:
    for train, validation in FOLDS.values():
        if validation.end > config.period.development_end or train.start < config.period.data_start:
            raise ValueError("the development folds must lie inside data_start .. development_end")


def stocks() -> list[str]:
    return [s.id for s in load_universe().select("main")]


def jobs(modes=MODES, designs=DESIGNS, folds=tuple(FOLDS), runs: int = RUNS) -> list[Job]:
    out = []
    for mode in modes:
        for design in designs:
            for fold in folds:
                for run in range(runs):
                    if mode == "pooled":
                        out.append(Job(mode, design, fold, run))
                    else:
                        out.extend(Job(mode, design, fold, run, s) for s in stocks())
    # pooled runs are the long ones: start them first
    return sorted(out, key=lambda j: j.mode != "pooled")


def _market(design: Design, panel: Panel, days: pd.DatetimeIndex) -> StockMarket:
    frame = panel.prices.loc[days]
    return StockMarket(design, *(frame[c].to_numpy(dtype=np.float64) for c in PRICE_COLUMNS))


def _all_names(panels: dict[str, Panel]) -> dict[str, list[str]]:
    """Every indicator any stock has, by group. Stocks differ in their fundamental indicators, so
    keeping only the shared ones would leave a pooled GP with one or two; a stock without an
    indicator has missing values for it, and a comparison with a missing value is false."""
    groups: dict[str, list[str]] = {}
    for panel in panels.values():
        for group, names in panel.names().items():
            known = groups.setdefault(group, [])
            known.extend(n for n in names if n not in known)
    return groups


def _arrays(panel: Panel, days: pd.DatetimeIndex, names: list[str]) -> dict[str, np.ndarray]:
    found = panel.arrays(days)
    return {n: found[n] if n in found else np.full(len(days), np.nan) for n in names}


def _stacked(panels: dict[str, Panel], names: list[str], part: str) -> dict[str, np.ndarray]:
    parts = [_arrays(panel, getattr(panel, part), names) for panel in panels.values()]
    return {n: np.concatenate([p[n] for p in parts]) for n in names}


def _pooled_market(design: Design, panels: dict[str, Panel], part: str) -> tuple[PooledMarket, pd.DatetimeIndex]:
    days = pd.DatetimeIndex(sorted(set().union(*(set(getattr(p, part)) for p in panels.values()))))
    markets = [_market(design, p, getattr(p, part)) for p in panels.values()]
    positions = [days.get_indexer(getattr(p, part)) for p in panels.values()]
    return PooledMarket(markets, positions, len(days)), days


def run(job: Job, config: StudyConfig | None = None, write: bool = True) -> dict[str, Any]:
    config = config or load_config()
    check_development(config)
    started = time.perf_counter()
    rung = load_rungs()[RUNG]
    gp_config = replace(rung.gp, **BUDGET)
    train, validation = FOLDS[job.fold]
    design = Design(job.design, config.primary_cost_bps_per_side / 1e4)
    units = [job.stock] if job.mode == "separate" else stocks()
    panels = {s: load_panel(s, train, validation, rung.sources, None, rung.scaling) for s in units}
    names = _all_names(panels)
    grammar = Grammar(names, fusion_functions=rung.fusion, sources=rung.sources,
                      max_branch_depth=rung.max_branch_depth)
    flat = [n for group in names.values() for n in group]
    if job.mode == "separate":
        panel = panels[job.stock]
        market = _market(design, panel, panel.train)
        settings = rung.fitness
    else:
        market, _ = _pooled_market(design, panels, "train")
        settings = replace(rung.fitness, min_trades=rung.fitness.min_trades * len(units))
    data = _stacked(panels, flat, "train")
    evaluator = Evaluator(data, market, settings, cases=False)
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, job.unit, job.fold, job.run)
    evolution = evolve(grammar, evaluator, gp_config, rng)
    best = evolution.best
    out = {}
    for stock, panel in panels.items():
        later = signals(best, _arrays(panel, panel.evaluate, flat))
        daily, trades = _market(design, panel, panel.evaluate).daily(later["C"])
        out[stock] = {"days": [d.strftime("%Y-%m-%d") for d in panel.evaluate],
                      "daily": [round(float(x), 8) for x in daily], "trades": int(trades)}
    record = {
        **job.__dict__, "campaign": CAMPAIGN, "rung": RUNG, "budget": BUDGET,
        "train": [str(train.start), str(train.end)], "validation": [str(validation.start), str(validation.end)],
        "cost_bps": config.primary_cost_bps_per_side, "stocks": units,
        "model": describe(best), "tree": to_record(best), "fitness": best.fitness,
        "train_stats": best.stats, "validation_returns": out, "history": evolution.history,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, rung_digest=rung.digest(overrides=BUDGET),
                                                  rung_spec=rung.spec, experiment=CAMPAIGN, job=job.__dict__)
        provenance.write(job.path(), record)
    return record


def run_job(job: Job) -> str:
    """``run`` for a worker process (defined here, not in ``__main__``, so that spawned workers find it)."""
    record = run(job)
    return f"{job.mode} {job.design} {job.fold} {job.unit} run {job.run}: {record['seconds']:.0f} s"


def buy_and_hold(fold: str, config: StudyConfig | None = None) -> pd.Series:
    """Equal-weight portfolio of holding every stock through the validation year (the ``long``
    design with a signal that never fires), at the primary cost."""
    config = config or load_config()
    train, validation = FOLDS[fold]
    design = Design("long", config.primary_cost_bps_per_side / 1e4)
    series = []
    for s in stocks():
        panel = load_panel(s, train, validation, ("TA",))
        daily, _ = _market(design, panel, panel.evaluate).daily(np.zeros(len(panel.evaluate), dtype=bool))
        series.append(pd.Series(daily, index=panel.evaluate))
    return pd.concat(series, axis=1).mean(axis=1)
