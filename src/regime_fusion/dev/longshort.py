"""E6: a long/short GP across stocks (docs/DECISIONS.md, 4 November 2026).

The information-coefficient screen (``dev.ic``) found no signal for timing one stock, but a few weak,
stable cross-sectional ones (news attention reversal over 60 days, improving profitability). Here the
GP's Boolean signal splits the stocks instead of timing them: every ``h`` sessions, at the close, the
portfolio goes long the stocks whose signal is on and short those whose signal is off, equal weights
on each side (100% long, 100% short), and holds from the next open to the open after the next
rebalance. A rebalance with fewer than ``MIN_SIDE`` stocks on either side is flat. Costs: the primary
cost per side on the traded value (the change of weights). The trees act day by day, so a tree is
evaluated once on day x stock arrays.

E7 (``VARIANTS``) reins the GP in: ``d2`` caps branches at two comparisons (depth 2, against 5);
``seeded`` puts every median-split rule (one indicator above or below 0, its trailing median, as the
whole model) into the starting population; ``simple`` is no GP at all: the median-split rule with
the best training fitness, then scored on the validation year (the fair version of a rule picked
from the information-coefficient screen, which saw the validation years).

Fitness: the PSR lower bound (``dev.fitness``) of the period returns. Every rule is invested in every
active period, so the number of periods is the evidence; a rule flat in more than half of the
periods scores ``FLOOR``. Inputs: trailing ranks (``dev.features``). Rung A1's grammar and operators,
population 500, 20 generations, 10 runs, the two development folds. Seeds depend on the fold, the
horizon and the run only.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.dev.e23 import BUDGET, FOLDS, RUNG, _all_names, stocks
from regime_fusion.dev.features import load_stationary_panel
from regime_fusion.dev.fitness import FLOOR, LowerBoundEvaluator
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.tree import COMPARISONS, ERC, LEAF, FUSION, VAR, Grammar, Individual, describe, signals, to_record
from regime_fusion.seeds import make_rng

CAMPAIGN = "dev-e6"
HORIZONS = (5, 20)
MIN_SIDE = 5
RUNS = 10
VARIANTS = {            # name: (max branch depth or None for the rung's, seeded)
    "d5": (None, False),
    "d2": (2, False),
    "d2-seeded": (2, True),
    "d5-seeded": (None, True),
}


def median_rules(groups: dict[str, list[str]], sources: tuple[str, ...]) -> list[Individual]:
    """Every rule "indicator above (or below) its trailing median" as a whole model: the fusion is
    that branch alone; the other branches hold a placeholder comparison of their own type."""
    placeholder = {s: [(COMPARISONS[0], s, None), (VAR, s, groups[s][0]), (ERC, None, 0.0)] for s in sources}
    rules = []
    for s in sources:
        for name in groups[s]:
            for comparison in COMPARISONS:
                branches = {t: list(placeholder[t]) for t in sources}
                branches[s] = [(comparison, s, None), (VAR, s, name), (ERC, None, 0.0)]
                rules.append(Individual([(LEAF, FUSION, s)], branches))
    return rules


class SeededGrammar(Grammar):
    """A grammar whose starting population begins with given individuals (the rest as usual)."""

    def __init__(self, *args, seeds: tuple = (), **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "seeds", tuple(seeds))

    def ramped_population(self, size, initial_depth, rng):
        population = super().ramped_population(size, initial_depth, rng)
        seeds = [ind.copy() for ind in self.seeds][:size]
        return seeds + population[len(seeds):]


COVERAGE = 0.8   # an indicator is used only if most stocks have it


def common_indicators(data: dict[str, np.ndarray], coverage: float = COVERAGE) -> list[str]:
    """Indicators known for at least ``coverage`` of the stocks on the median training day (banks
    have no R&D, say). Days on which a stock has no value at all (before its listing, or during the
    ranks' warm-up) count for nothing, so the test is about stocks, not about the start of the data."""
    listed = np.logical_or.reduce([np.isfinite(v) for v in data.values()])
    keep = []
    for n, values in data.items():
        known = np.isfinite(values).sum(axis=1)
        alive = listed.sum(axis=1)
        share = np.divide(known, alive, out=np.full(len(known), np.nan), where=alive > 0)
        if np.nanmedian(share) >= coverage:
            keep.append(n)
    return keep


def complete(data: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Day x stock: whether every indicator in ``names`` is known. An incomplete stock-day is left
    out of both sides, so that a missing value (False in every comparison) cannot put a stock on
    the short side."""
    return np.logical_and.reduce([np.isfinite(data[n]) for n in names])


class LongShortMarket:
    """Day x stock open prices over a period; ``simulate`` scores a day x stock Boolean signal.
    ``usable`` (day x stock) marks the stock-days that may be held; others are left out."""

    def __init__(self, opens: np.ndarray, horizon: int, cost: float, usable: np.ndarray | None = None):
        self.opens, self.h, self.cost = opens, horizon, cost
        n = opens.shape[0]
        # decision days whose holding period (next open to the open h sessions later) ends inside
        self.decisions = np.arange(0, n - horizon - 1, horizon)
        a, b = opens[self.decisions + 1], opens[self.decisions + 1 + horizon]
        self.returns = b / a - 1.0       # per decision and stock; NaN where a stock has no price
        if usable is not None:
            self.returns = np.where(usable[self.decisions], self.returns, np.nan)

    def weights(self, on: np.ndarray, known: np.ndarray) -> np.ndarray:
        longs, shorts = on & known, ~on & known
        nl, ns = longs.sum(axis=1, keepdims=True), shorts.sum(axis=1, keepdims=True)
        active = (nl >= MIN_SIDE) & (ns >= MIN_SIDE)
        w = np.where(longs, 1.0 / np.maximum(nl, 1), 0.0) - np.where(shorts, 1.0 / np.maximum(ns, 1), 0.0)
        return np.where(active, w, 0.0)

    def simulate(self, signal: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """Period returns net of costs, the number of active periods, and the period returns again
        (the evidence of the PSR lower bound)."""
        on = np.asarray(signal, dtype=bool)[self.decisions]
        known = np.isfinite(self.returns)
        w = self.weights(on, known)
        gross = np.nansum(w * np.nan_to_num(self.returns), axis=1)
        traded = np.abs(np.diff(np.vstack([np.zeros(w.shape[1]), w]), axis=0)).sum(axis=1)
        net = gross - traded * self.cost
        active = int((np.abs(w).sum(axis=1) > 0).sum())
        return net, active, net

    def daily(self, signal: np.ndarray) -> tuple[np.ndarray, int]:
        net, active, _ = self.simulate(signal)
        return net, active


def _wide(panels: dict, names: list[str], part: str) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], np.ndarray]:
    days = pd.DatetimeIndex(sorted(set().union(*(set(getattr(p, part)) for p in panels.values()))))
    data = {n: np.full((len(days), len(panels)), np.nan) for n in names}
    opens = np.full((len(days), len(panels)), np.nan)
    for j, panel in enumerate(panels.values()):
        idx = days.get_indexer(getattr(panel, part))
        frame = panel.indicators.loc[getattr(panel, part)]
        for n in names:
            if n in frame:
                data[n][idx, j] = frame[n].to_numpy()
        opens[idx, j] = panel.prices.loc[getattr(panel, part), "adj_open"].to_numpy()
    return days, data, opens


def path(fold: str, horizon: int, run: int, variant: str = "d5") -> Path:
    base = store.data_dir() / "results" / CAMPAIGN / f"h{horizon}"
    return (base if variant == "d5" else base / variant) / fold / f"run{run:02d}.json"


def run(fold: str, horizon: int, run_number: int, variant: str = "d5", write: bool = True) -> dict[str, Any]:
    config = load_config()
    started = time.perf_counter()
    rung = load_rungs()[RUNG]
    train, validation = FOLDS[fold]
    cost = config.primary_cost_bps_per_side / 1e4
    panels = {s: load_stationary_panel(s, train, validation, rung.sources) for s in stocks()}
    groups = _all_names(panels)
    names = [n for g in groups.values() for n in g]
    _, train_data, train_opens = _wide(panels, names, "train")
    names = common_indicators(train_data)
    groups = {k: [n for n in v if n in names] for k, v in groups.items()}
    train_data = {n: train_data[n] for n in names}
    depth, seeded = VARIANTS[variant]
    seeds = median_rules(groups, rung.sources) if seeded else ()
    grammar = SeededGrammar(groups, fusion_functions=rung.fusion, sources=rung.sources,
                            max_branch_depth=depth or rung.max_branch_depth, seeds=seeds)
    market = LongShortMarket(train_opens, horizon, cost, complete(train_data, names))
    settings = replace(rung.fitness, min_trades=max(1, len(market.decisions) // 2))
    evaluator = LowerBoundEvaluator(train_data, market, settings, cases=False)
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, fold, horizon, run_number)
    evolution = evolve(grammar, evaluator, replace(rung.gp, **BUDGET), rng)
    best = evolution.best
    days, val_data, val_opens = _wide(panels, names, "evaluate")
    later = LongShortMarket(val_opens, horizon, cost, complete(val_data, names))
    net, active, _ = later.simulate(signals(best, val_data)["C"])
    on_share = float(np.nanmean(signals(best, val_data)["C"][later.decisions]))
    record = {
        "campaign": CAMPAIGN, "variant": variant, "fold": fold, "horizon": horizon, "run": run_number, "rung": RUNG, "budget": BUDGET,
        "train": [str(train.start), str(train.end)], "validation": [str(validation.start), str(validation.end)],
        "cost_bps": config.primary_cost_bps_per_side, "stocks": list(panels),
        "model": describe(best), "tree": to_record(best), "fitness": best.fitness, "train_stats": best.stats,
        "validation_periods": [d.strftime("%Y-%m-%d") for d in days[later.decisions]],
        "validation_returns": [round(float(x), 8) for x in net], "validation_active": active,
        "validation_long_share": on_share, "history": evolution.history,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, rung_digest=rung.digest(overrides=BUDGET), rung_spec=rung.spec,
                                                  experiment=CAMPAIGN, variant=variant, fold=fold, horizon=horizon,
                                                  run=run_number)
        provenance.write(path(fold, horizon, run_number, variant), record)
    return record


def run_job(job: tuple) -> str:
    record = run(*job)
    return f"{job}: {record['seconds']:.0f} s"


def jobs(variants=tuple(VARIANTS)) -> list[tuple]:
    return [(f, h, r, v) for v in variants for h in HORIZONS for f in FOLDS for r in range(RUNS)
            if not path(f, h, r, v).exists()]


def simple_baseline() -> pd.DataFrame:
    """The median-split rule with the best training fitness, scored on the validation year."""
    config = load_config()
    rung = load_rungs()[RUNG]
    cost = config.primary_cost_bps_per_side / 1e4
    rows = []
    for fold, (train, validation) in FOLDS.items():
        panels = {s: load_stationary_panel(s, train, validation, rung.sources) for s in stocks()}
        groups = _all_names(panels)
        names = [n for g in groups.values() for n in g]
        _, train_data, train_opens = _wide(panels, names, "train")
        names = common_indicators(train_data)
        groups = {k: [n for n in v if n in names] for k, v in groups.items()}
        train_data = {n: train_data[n] for n in names}
        _, val_data, val_opens = _wide(panels, names, "evaluate")
        for h in HORIZONS:
            market = LongShortMarket(train_opens, h, cost, complete(train_data, names))
            settings = replace(rung.fitness, min_trades=max(1, len(market.decisions) // 2))
            evaluator = LowerBoundEvaluator(train_data, market, settings, cases=False)
            rules = median_rules(groups, rung.sources)
            for rule in rules:
                evaluator.evaluate(rule)
            best = max(rules, key=lambda r: r.fitness)
            net, active, _ = LongShortMarket(val_opens, h, cost, complete(val_data, names)).simulate(
                signals(best, val_data)["C"])
            per_year = 252 / h
            rows.append({"horizon": h, "fold": fold, "rule": describe(best).split("\n")[1 + rung.sources.index(
                best.fusion[0][2])].split(": ", 1)[1], "train_fitness": best.fitness,
                "validation_sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(per_year)),
                "validation_return_pct": 100 * (np.prod(1 + net) - 1)})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    return table


def report() -> pd.DataFrame:
    import json
    rows = []
    found = sorted((store.data_dir() / "results" / CAMPAIGN).glob("h*/*/run*.json")) + \
        sorted((store.data_dir() / "results" / CAMPAIGN).glob("h*/*/*/run*.json"))
    for p in found:
        if p.name.endswith(".provenance.json"):
            continue
        r = json.loads(p.read_text())
        rows.append({"variant": r.get("variant", "d5"), "horizon": r["horizon"], "fold": r["fold"], "run": r["run"], "fitness": r["fitness"],
                     "returns": np.array(r["validation_returns"]), "active": r["validation_active"],
                     "long_share": r["validation_long_share"], "fusion": r["model"].split("\n")[0]})
    t = pd.DataFrame(rows)
    out = []
    for (variant, h, fold), cell in t.groupby(["variant", "horizon", "fold"]):
        per_year = 252 / h
        ensemble = np.mean(np.vstack(cell["returns"]), axis=0)     # the runs' portfolios averaged
        sharpe = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(per_year)) if x.std(ddof=1) > 0 else 0.0
        runs = np.array([sharpe(x) for x in cell["returns"]])
        out.append({"variant": variant, "horizon": h, "fold": fold, "runs": len(cell), "ensemble_sharpe": sharpe(ensemble),
                    "ensemble_return_pct": 100 * (np.prod(1 + ensemble) - 1),
                    "median_run_sharpe": float(np.median(runs)), "runs_positive": float((runs > 0).mean()),
                    "active_periods": float(cell["active"].median()), "long_share": float(cell["long_share"].median()),
                    "train_fitness": float(cell["fitness"].median()),
                    "fused": float((~cell["fusion"].str.match(r"fusion: (FA|SA|TA)$")).mean())})
    summary = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    return summary
