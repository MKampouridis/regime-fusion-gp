"""E8: the long/short GP on the broad universe (``dev.broad``), with six validation years.

As E6/E7 (``dev.longshort``), on the ~430 S&P 500 members of January 2015 outside the hold-out draw,
fundamental and technical indicators only (no news, so two branches and no active trading
operator, which needs three), as trailing ranks. Six folds: each validation year from 2013 to 2018,
trained on the three calendar years before it; all inside ``development_end``. Data come from
``dev.broad.BROAD_DIR``. Variants: ``d5`` (rung A1's depth) and ``d2-seeded`` (two comparisons per
branch, every median-split rule in the starting population); ``simple`` is the median-split rule
with the best training fitness. Seeds depend on the fold, horizon, variant name and run, never on
anything else; results go to ``<BROAD_DIR>/results/dev-e8``.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.dev import broad
from regime_fusion.dev.e23 import BUDGET, RUNG, _all_names
from regime_fusion.dev.features import load_stationary_panel
from regime_fusion.dev.fitness import LowerBoundEvaluator
from regime_fusion.dev.longshort import LongShortMarket, SeededGrammar, _wide, common_indicators, complete, median_rules
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.tree import describe, signals, to_record
from regime_fusion.seeds import make_rng
from regime_fusion.windows import DateRange

CAMPAIGN = "dev-e8"
SOURCES = ("FA", "TA")
HORIZONS = (5, 20)
RUNS = 10
VARIANTS = {"d5": (None, False), "d2-seeded": (2, True)}
FOLDS = {str(y): (DateRange(dt.date(y - 3, 1, 1), dt.date(y - 1, 12, 31)), DateRange(dt.date(y, 1, 1), dt.date(y, 12, 31)))
         for y in range(2013, 2019)}


def usable() -> list[str]:
    return json.loads((broad.BROAD_DIR / "reports" / "broad_build.json").read_text())["usable"]


def _panels(fold: str) -> dict:
    train, validation = FOLDS[fold]
    panels = {}
    for s in usable():
        try:
            panel = load_stationary_panel(s, train, validation, SOURCES)
        except (ValueError, KeyError, FileNotFoundError):
            continue   # listed too late for this window, or no indicator with enough training values
        if len(panel.train) > 200 and len(panel.evaluate) > 0 and all(
                any(c.startswith(f"{t.lower()}:") for c in panel.indicators.columns) for t in SOURCES):
            panels[s] = panel
    return panels


def path(fold: str, horizon: int, run: int, variant: str) -> Path:
    return broad.BROAD_DIR / "results" / CAMPAIGN / variant / f"h{horizon}" / fold / f"run{run:02d}.json"


def _setup(fold: str, horizon: int):
    config = load_config()
    rung = load_rungs()[RUNG]
    panels = _panels(fold)
    groups = _all_names(panels)
    groups = {k: v for k, v in groups.items() if k in SOURCES or k == "RG"}
    names = [n for k in SOURCES for n in groups[k]]
    _, train_data, train_opens = _wide(panels, names, "train")
    names = common_indicators({n: train_data[n] for n in names})
    groups = {k: [n for n in v if n in names] for k, v in groups.items()}
    train_data = {n: train_data[n] for n in names}
    days, val_data, val_opens = _wide(panels, names, "evaluate")
    cost = config.primary_cost_bps_per_side / 1e4
    return config, rung, panels, groups, train_data, \
        LongShortMarket(train_opens, horizon, cost, complete(train_data, names)), \
        days, val_data, LongShortMarket(val_opens, horizon, cost, complete(val_data, names))


def run(fold: str, horizon: int, run_number: int, variant: str, write: bool = True) -> dict[str, Any]:
    broad.use_broad_dir()
    started = time.perf_counter()
    config, rung, panels, groups, train_data, market, days, val_data, later = _setup(fold, horizon)
    depth, seeded = VARIANTS[variant]
    grammar = SeededGrammar(groups, fusion_functions=rung.fusion, sources=SOURCES,
                            max_branch_depth=depth or rung.max_branch_depth,
                            seeds=median_rules(groups, SOURCES) if seeded else ())
    settings = replace(rung.fitness, min_trades=max(1, len(market.decisions) // 2))
    evaluator = LowerBoundEvaluator(train_data, market, settings, cases=False)
    gp_config = replace(rung.gp, **BUDGET, active_trading=False)
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, variant, fold, horizon, run_number)
    best = evolve(grammar, evaluator, gp_config, rng).best
    signal = signals(best, val_data)["C"]
    net, active, _ = later.simulate(signal)
    record = {
        "campaign": CAMPAIGN, "variant": variant, "fold": fold, "horizon": horizon, "run": run_number,
        "stocks": len(panels), "model": describe(best), "tree": to_record(best), "fitness": best.fitness,
        "validation_returns": [round(float(x), 8) for x in net], "validation_active": active,
        "validation_long_share": float(np.nanmean(signal[later.decisions])),
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, rung_digest=rung.digest(overrides=BUDGET), rung_spec=rung.spec,
                                                  experiment=CAMPAIGN, variant=variant, fold=fold, horizon=horizon,
                                                  run=run_number, sources=list(SOURCES))
        provenance.write(path(fold, horizon, run_number, variant), record)
    return record


def run_job(job: tuple) -> str:
    record = run(*job)
    return f"{job}: {record['stocks']} stocks, {record['seconds']:.0f} s"


def jobs() -> list[tuple]:
    broad.use_broad_dir()
    return [(f, h, r, v) for v in VARIANTS for h in HORIZONS for f in FOLDS for r in range(RUNS)
            if not path(f, h, r, v).exists()]


def simple_baseline() -> pd.DataFrame:
    broad.use_broad_dir()
    rows = []
    for fold in FOLDS:
        for h in HORIZONS:
            config, rung, panels, groups, train_data, market, days, val_data, later = _setup(fold, h)
            settings = replace(rung.fitness, min_trades=max(1, len(market.decisions) // 2))
            evaluator = LowerBoundEvaluator(train_data, market, settings, cases=False)
            rules = median_rules(groups, SOURCES)
            for rule in rules:
                evaluator.evaluate(rule)
            best = max(rules, key=lambda r: r.fitness)
            net, _, _ = later.simulate(signals(best, val_data)["C"])
            branch = best.fusion[0][2]
            rows.append({"horizon": h, "fold": fold, "stocks": len(panels),
                         "rule": describe(best).split("\n")[1 + SOURCES.index(branch)].split(": ", 1)[1],
                         "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(252 / h)),
                         "return_pct": 100 * (np.prod(1 + net) - 1)})
    table = pd.DataFrame(rows)
    table.to_csv(broad.BROAD_DIR / "results" / "e8_simple.csv", index=False)
    print(table.round(3).to_string(index=False))
    return table


def report() -> pd.DataFrame:
    broad.use_broad_dir()
    rows = []
    for p in sorted((broad.BROAD_DIR / "results" / CAMPAIGN).glob("*/h*/*/run*.json")):
        if p.name.endswith(".provenance.json"):
            continue
        r = json.loads(p.read_text())
        rows.append({"variant": r["variant"], "horizon": r["horizon"], "fold": r["fold"],
                     "returns": np.array(r["validation_returns"])})
    t = pd.DataFrame(rows)
    out = []
    for (variant, h, fold), cell in t.groupby(["variant", "horizon", "fold"]):
        ens = np.mean(np.vstack(cell["returns"]), axis=0)
        sharpe = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252 / h)) if x.std(ddof=1) > 0 else 0.0
        out.append({"variant": variant, "horizon": h, "fold": fold, "runs": len(cell), "ensemble_sharpe": sharpe(ens),
                    "ensemble_return_pct": 100 * (np.prod(1 + ens) - 1),
                    "runs_positive": float(np.mean([sharpe(x) > 0 for x in cell["returns"]]))})
    summary = pd.DataFrame(out)
    wide = summary.pivot_table(index=["variant", "horizon"], columns="fold", values="ensemble_sharpe")
    wide["positive_years"] = (wide > 0).sum(axis=1)
    pd.set_option("display.width", 200)
    print(wide.round(2).to_string())
    summary.to_csv(broad.BROAD_DIR / "results" / "e8_summary.csv", index=False)
    return summary
