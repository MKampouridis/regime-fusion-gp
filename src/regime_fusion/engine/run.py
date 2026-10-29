"""One GP run, and the choice of exit settings.

A run evolves a model of one rung for one stock on a training range and
evaluates it on the following period (``regime_fusion.windows``):

* stage 1: train on the three training years, evaluate on the validation year
  (used to choose the exit settings);
* stage 2: train on all four years, evaluate on the test year.

The model trades with the stock-year's exit settings (d, r). Fitness uses the
primary cost; the model is then evaluated at every cost level. Evaluation starts
without a position, so its first possible purchase is on its second day. Seeds
depend on stock, test year, run number and stage only, never on the rung.

Exit settings are chosen per stock and test year on the validation year with
rung A0: every candidate of the grid gets a few stage-1 runs, and the one with
the best median validation Sharpe ratio at the primary cost wins. Every method
then uses them, so any advantage of tuning goes to the baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from regime_fusion import provenance
from regime_fusion.config import StudyConfig
from regime_fusion.data import store
from regime_fusion.engine import metrics
from regime_fusion.engine.backtest import backtest
from regime_fusion.engine.fitness import Evaluator
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.panel import load_panel
from regime_fusion.engine.rungs import Rung
from regime_fusion.engine.tree import Grammar, describe, signals
from regime_fusion.seeds import make_rng
from regime_fusion.windows import walk_forward


@dataclass(frozen=True)
class RunSpec:
    stock: str
    test_year: int
    run: int
    stage: int          # 1: training -> validation; 2: training and validation -> test
    exit_days: int
    exit_rate: float


def run(spec: RunSpec, rung: Rung, config: StudyConfig, *, campaign: str = "dev",
        sources: tuple[str, ...] | None = None, overrides: dict[str, Any] | None = None,
        write: bool = True) -> dict[str, Any]:
    """Evolve and evaluate one model; returns (and writes) the result record."""
    started = time.perf_counter()
    window = next(w for w in walk_forward(config) if w.test_year == spec.test_year)
    train, evaluate = ((window.train, window.validation) if spec.stage == 1
                       else (window.final_train, window.test))
    sources = tuple(sources or rung.sources)
    regimes = None
    if rung.regimes:
        regimes = store.layer("processed", "regimes", rung.regimes, f"{spec.test_year}.stage{spec.stage}.parquet")
    panel = load_panel(spec.stock, train, evaluate, sources, regimes, rung.scaling)
    grammar = Grammar(panel.names(), fusion_functions=rung.fusion, sources=sources,
                      max_branch_depth=rung.max_branch_depth)
    gp_config = rung.gp if not overrides else type(rung.gp)(**{**rung.gp.__dict__, **overrides})
    if len(sources) != 3 and gp_config.active_trading:
        gp_config = type(gp_config)(**{**gp_config.__dict__, "active_trading": False})
    primary = config.primary_cost_bps_per_side
    evaluator = Evaluator(panel.arrays(panel.train),
                          panel.market(panel.train, spec.exit_days, spec.exit_rate, primary), rung.fitness)
    rng = make_rng(config.base_seed, "run", spec.stock, spec.test_year, spec.run, spec.stage)
    evolution = evolve(grammar, evaluator, gp_config, rng)
    best = evolution.best

    later = signals(best, panel.arrays(panel.evaluate))
    prices = panel.prices.loc[panel.evaluate]
    outcome = {}
    for cost in config.costs_bps_per_side:
        result = backtest(later["C"], prices, spec.exit_days, spec.exit_rate, cost)
        outcome[f"{cost:g}bps"] = metrics.summary(result)
    branches = {s: metrics.summary(backtest(later[s], prices, spec.exit_days, spec.exit_rate, primary))
                for s in sources}
    record = {
        **spec.__dict__,
        "rung": rung.name,
        "campaign": campaign,
        "sources": list(sources),
        "train": [str(train.start), str(train.end)],
        "evaluate": [str(evaluate.start), str(evaluate.end)],
        "indicators": {k: len(v) for k, v in panel.names().items()},
        "model": describe(best),
        "fitness": best.fitness,
        "train_stats": {c: v for c, v in best.stats.items() if c != "cases"},
        "evaluation": outcome,
        "branches": branches,
        "history": evolution.history,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, rung_digest=rung.digest(), rung_spec=rung.spec,
                                                  overrides=overrides or {})
        provenance.write(result_path(campaign, rung.name, spec), record)
    return record


def result_path(campaign: str, rung: str, spec: RunSpec) -> Path:
    name = f"{spec.test_year}.stage{spec.stage}.run{spec.run:02d}.json"
    return store.data_dir() / "results" / campaign / rung / spec.stock / name


def choose_exits(stock: str, test_year: int, rung: Rung, config: StudyConfig, grid: dict[str, Any], *,
                 campaign: str = "dev", sources: tuple[str, ...] | None = None,
                 overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pick (d, r) for one stock-year on the validation year (see the module notes)."""
    primary = f"{config.primary_cost_bps_per_side:g}bps"
    scores = []
    for days in grid["days"]:
        for rate in grid["rates"]:
            sharpes = []
            for number in range(int(grid["runs"])):
                spec = RunSpec(stock, test_year, number, 1, int(days), float(rate))
                record = run(spec, rung, config, campaign=campaign, sources=sources, overrides=overrides,
                             write=False)
                sharpes.append(record["evaluation"][primary]["sharpe"])
            scores.append({"days": int(days), "rate": float(rate), "median_sharpe": float(np.median(sharpes)),
                           "sharpes": sharpes})
    best = max(scores, key=lambda s: (s["median_sharpe"], -s["days"]))
    choice = {"stock": stock, "test_year": test_year, "days": best["days"], "rate": best["rate"],
              "rung": rung.name, "candidates": scores}
    path = store.data_dir() / "results" / campaign / "exits" / f"{stock}.{test_year}.json"
    provenance.write(path, {**choice, "provenance": provenance.collect(config, rung_digest=rung.digest())})
    return choice
