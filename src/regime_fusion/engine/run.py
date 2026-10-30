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

Every result carries a *fingerprint*: digests of the study configuration, of
the method's settings and of the data it read, each file's rows over the
sessions used (``fingerprint``). A campaign uses them to refuse results made
under other settings or from other data (``engine/tasks.py``). A GP result also
keeps its model exactly (``tree.to_record``) besides the readable text.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import StudyConfig
from regime_fusion.data import store
from regime_fusion.engine import metrics
from regime_fusion.engine.backtest import backtest
from regime_fusion.engine.fitness import Evaluator
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.panel import frame_digest, load_panel
from regime_fusion.engine.rungs import Rung, finbert_source
from regime_fusion.engine.tree import Grammar, active_sources, describe, signals, to_record
from regime_fusion.regimes import market as regime_inputs
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
    regimes = regimes_file(rung.regimes, spec.stock, spec.test_year, spec.stage, config.windows.scheme)
    headlines = finbert_source(spec.test_year) if rung.finbert else None
    panel = load_panel(spec.stock, train, evaluate, sources, regimes, rung.scaling, finbert=headlines)
    grammar = Grammar(panel.names(), fusion_functions=rung.fusion, sources=sources,
                      max_branch_depth=rung.max_branch_depth)
    gp_config = rung.gp if not overrides else type(rung.gp)(**{**rung.gp.__dict__, **overrides})
    if len(sources) != 3 and gp_config.active_trading:
        gp_config = type(gp_config)(**{**gp_config.__dict__, "active_trading": False})
    primary = config.primary_cost_bps_per_side
    evaluator = Evaluator(panel.arrays(panel.train),
                          panel.market(panel.train, spec.exit_days, spec.exit_rate, primary), rung.fitness)
    rng = make_rng(config.base_seed, "run", spec.stock, spec.test_year, spec.run, spec.stage)
    primary_market = panel.prices.loc[panel.evaluate]
    evaluation_data = panel.arrays(panel.evaluate)

    def checkpoint(individual) -> dict:
        """What stopping now would give on the validation year (primary cost); it never feeds back
        into evolution. Test-year runs (stage 2) record none, so no one can tune on them."""
        result = backtest(signals(individual, evaluation_data)["C"], primary_market, spec.exit_days,
                          spec.exit_rate, primary)
        return {"sharpe": metrics.sharpe(result.daily), "trades": len(result.trades)}

    evolution = evolve(grammar, evaluator, gp_config, rng, checkpoint=checkpoint if spec.stage == 1 else None)
    best = evolution.best

    later = signals(best, panel.arrays(panel.evaluate))
    signal = np.asarray(later["C"], dtype=bool)
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
        "scheme": config.windows.scheme,
        "sources": list(sources),
        "finbert": headlines,
        "train": [str(train.start), str(train.end)],
        "evaluate": [str(evaluate.start), str(evaluate.end)],
        "indicators": {k: len(v) for k, v in panel.names().items()},
        "model": describe(best),
        "tree": to_record(best),
        "fitness": best.fitness,
        "train_stats": {c: v for c, v in best.stats.items() if c != "cases"},
        "evaluation": outcome,
        "signal": pack_signal(signal),
        "listening": listening(best, panel),
        "branches": branches,
        "history": evolution.history,
        "checkpoints": evolution.checkpoints,
        "fingerprint": fingerprint(config, rung.digest(sources, overrides), panel.inputs, prices),
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, rung_digest=rung.digest(), rung_spec=rung.spec,
                                                  sources=list(sources or rung.sources), overrides=overrides or {})
        provenance.write(result_path(campaign, rung.name, spec), record)
    return record


FINGERPRINT_FORMAT = 2  # 2: inputs keyed by file and year (1: by file and the sessions read)


def fingerprint(config: StudyConfig, method: str, inputs: dict[str, str], evaluated: pd.DataFrame) -> dict[str, Any]:
    """What a result depends on: the configuration, the method's settings (``method``, a digest) and
    the data read (``panel.Panel.inputs``). ``evaluated`` holds the prices of the evaluation days; the
    analysis checks them before it rebuilds the run's daily returns."""
    return {"format": FINGERPRINT_FORMAT, "config": config.digest(), "method": method,
            "inputs": dict(sorted(inputs.items())), "evaluation_prices": frame_digest(evaluated)}


def listening(individual, panel) -> dict[str, Any]:
    """Share of evaluation days on which each branch is on the model's active path, overall and
    in each regime (the most probable state of the run's regime probabilities)."""
    data = panel.arrays(panel.evaluate)
    active = active_sources(individual, data)
    out: dict[str, Any] = {"all": {s: float(v.mean()) for s, v in active.items()}}
    columns = panel.names()["RG"]
    if columns:
        values = np.column_stack([data[c] for c in columns])
        known = ~np.isnan(values).all(axis=1)
        state = np.full(len(values), -1)
        state[known] = np.nan_to_num(values[known], nan=-np.inf).argmax(axis=1)
        out["by_state"] = {str(k): {s: float(v[state == k].mean()) for s, v in active.items()}
                           for k in range(len(columns)) if (state == k).any()}
        out["state_share"] = {str(k): float((state == k).mean()) for k in range(len(columns))}
    return out


def pack_signal(signal) -> str:
    """A daily buy signal in compact text (one bit per day), so that daily returns can be rebuilt."""
    values = np.asarray(signal, dtype=bool)
    return f"{len(values)}:" + base64.b64encode(np.packbits(values).tobytes()).decode()


def unpack_signal(text: str) -> np.ndarray:
    length, _, payload = text.partition(":")
    bits = np.unpackbits(np.frombuffer(base64.b64decode(payload), dtype=np.uint8))
    return bits[:int(length)].astype(bool)


def regimes_file(name: str | None, stock: str, test_year: int, stage: int,
                 scheme: str = "rolling") -> Path | None:
    """The filtered regime probabilities a run of ``stock`` uses (none without regimes)."""
    if not name:
        return None
    own = stock if regime_inputs.per_stock(name, regime_inputs.load_settings()) else None
    return regime_inputs.probabilities_path(name, test_year, stage, own, scheme=scheme)


def result_path(campaign: str, rung: str, spec: RunSpec) -> Path:
    name = f"{spec.test_year}.stage{spec.stage}.run{spec.run:02d}.json"
    return store.data_dir() / "results" / campaign / rung / spec.stock / name


def exits_path(campaign: str, stock: str, test_year: int) -> Path:
    return store.data_dir() / "results" / campaign / "exits" / f"{stock}.{test_year}.json"


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
              "rung": rung.name, "scheme": config.windows.scheme, "candidates": scores}
    provenance.write(exits_path(campaign, stock, test_year), {**choice, "provenance": provenance.collect(
        config, rung_digest=rung.digest(), method_digest=rung.digest(sources, overrides),
        sources=list(sources or rung.sources), overrides=overrides or {})})
    return choice
