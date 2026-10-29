"""One machine-learning benchmark on one stock-year, with the GP's protocol.

Stage 1 tunes on the validation year. Each candidate of the model's grid is
trained on the three training years and predicts the validation year. Each
candidate is paired with each threshold quantile q, under which the model buys
when its predicted probability reaches the q-quantile of its own training
predictions. Every pair is backtested on the validation year with the
stock-year's exit settings, and the pair with the highest Sharpe ratio wins
(fewer than ``min_trades`` trades score zero, as in the GP's fitness).

Stage 2 retrains the chosen candidate on all four years, sets its threshold at
the chosen quantile of its new training predictions, and trades the test year.
Stage-1 results report the chosen pair on the validation year.

Features are the GP's (``regime_fusion.engine.panel``: the same indicators,
scaled on the training rows) plus the regime probabilities. Labels come from the
training rows only (``regime_fusion.ml.labels``), at the primary cost. Seeds
come from ``regime_fusion.seeds`` with the GP's keys (stock, test year, run,
stage), so they never depend on the model.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from regime_fusion import provenance
from regime_fusion.config import StudyConfig
from regime_fusion.engine import metrics
from regime_fusion.engine.backtest import backtest
from regime_fusion.engine.panel import Panel, load_panel
from regime_fusion.engine.run import RunSpec, pack_signal, regimes_file, result_path
from regime_fusion.ml.labels import trade_returns
from regime_fusion.ml.models import ModelSpec, Settings, build
from regime_fusion.seeds import make_rng
from regime_fusion.windows import walk_forward


def run_ml(spec: RunSpec, model: ModelSpec, settings: Settings, config: StudyConfig, *,
           campaign: str = "dev", write: bool = True) -> dict[str, Any]:
    """Tune (stage 1) and, for stage 2, retrain and test one model; returns (and writes) the record."""
    started = time.perf_counter()
    window = next(w for w in walk_forward(config) if w.test_year == spec.test_year)
    primary = config.primary_cost_bps_per_side
    tuning = panel_for(spec, settings, window.train, window.validation, stage=1)
    choice = tune(tuning, spec, model, settings, primary,
                  seed(config, spec, stage=1))
    if spec.stage == 1:
        final, signal, fitted_on, evaluated_on = tuning, choice["validation_signal"], window.train, window.validation
        train_signal = choice["train_signal"]
    else:
        final = panel_for(spec, settings, window.final_train, window.test, stage=2)
        fitted = fit(final, spec, model, choice["params"], primary, seed(config, spec, stage=2))
        threshold = float(np.quantile(fitted["train_prediction"], choice["quantile"]))
        signal = fitted["evaluate_prediction"] >= threshold
        train_signal = fitted["train_prediction"] >= threshold
        fitted_on, evaluated_on = window.final_train, window.test
    prices = final.prices.loc[final.evaluate]
    evaluation = {f"{cost:g}bps": metrics.summary(backtest(signal, prices, spec.exit_days, spec.exit_rate, cost))
                  for cost in config.costs_bps_per_side}
    train_result = backtest(train_signal, final.prices.loc[final.train], spec.exit_days, spec.exit_rate, primary)
    record = {
        **spec.__dict__,
        "rung": model.name,
        "campaign": campaign,
        "sources": list(settings.sources),
        "train": [str(fitted_on.start), str(fitted_on.end)],
        "evaluate": [str(evaluated_on.start), str(evaluated_on.end)],
        "indicators": {k: len(v) for k, v in final.names().items()},
        "model": f"{model.name} {choice['params']}, buy at or above the {choice['quantile']:g} quantile",
        "fitness": metrics.sharpe(train_result.daily),
        "validation_sharpe": choice["sharpe"],
        "candidates": choice["table"],
        "evaluation": evaluation,
        "signal": pack_signal(signal),
        "branches": {},
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, ml_settings_digest=settings.digest(),
                                                  ml_model=model.__dict__)
        provenance.write(result_path(campaign, model.name, spec), record)
    return record


def seed(config: StudyConfig, spec: RunSpec, stage: int) -> int:
    rng = make_rng(config.base_seed, "run", spec.stock, spec.test_year, spec.run, stage)
    return int(rng.integers(2**31 - 1))


def panel_for(spec: RunSpec, settings: Settings, train, evaluate, stage: int) -> Panel:
    regimes = regimes_file(settings.regimes, spec.stock, spec.test_year, stage)
    return load_panel(spec.stock, train, evaluate, settings.sources, regimes, settings.scaling)


def fit(panel: Panel, spec: RunSpec, model: ModelSpec, params: dict[str, Any], cost_bps: float,
        seed_value: int) -> dict[str, np.ndarray]:
    """Train on the panel's training rows; predictions for its training and evaluation rows."""
    train_x, evaluate_x = features(panel, panel.train), features(panel, panel.evaluate)
    outcome = trade_returns(panel.prices.loc[panel.train], spec.exit_days, spec.exit_rate, cost_bps)
    labelled = outcome.notna().to_numpy()
    y = (outcome.to_numpy() > 0).astype(int)
    train_states, evaluate_states = states(panel, panel.train), states(panel, panel.evaluate)
    learner = build(model, params, seed_value, list(panel.indicators.columns)).fit(
        train_x[labelled], y[labelled], None if train_states is None else train_states[labelled])
    return {"train_prediction": learner.predict(train_x, train_states),
            "evaluate_prediction": learner.predict(evaluate_x, evaluate_states)}


def tune(panel: Panel, spec: RunSpec, model: ModelSpec, settings: Settings, cost_bps: float,
         seed_value: int) -> dict[str, Any]:
    """The candidate and threshold quantile with the best validation Sharpe ratio."""
    prices = panel.prices.loc[panel.evaluate]
    best, table = None, []
    for params in model.candidates():
        fitted = fit(panel, spec, model, params, cost_bps, seed_value)
        for quantile in settings.quantiles:
            threshold = float(np.quantile(fitted["train_prediction"], quantile))
            signal = fitted["evaluate_prediction"] >= threshold
            result = backtest(signal, prices, spec.exit_days, spec.exit_rate, cost_bps)
            trades = len(result.trades)
            score = metrics.sharpe(result.daily) if trades >= settings.min_trades else 0.0
            table.append({**params, "quantile": quantile, "sharpe": score, "trades": trades})
            if best is None or score > best["sharpe"]:
                best = {"params": params, "quantile": quantile, "sharpe": score, "validation_signal": signal,
                        "train_signal": fitted["train_prediction"] >= threshold}
    return {**best, "table": table}


def features(panel: Panel, days) -> np.ndarray:
    return panel.indicators.loc[days].to_numpy(dtype=np.float64)


def states(panel: Panel, days) -> np.ndarray | None:
    """The most probable regime of each day, or None without regimes."""
    columns = panel.names()["RG"]
    if not columns:
        return None
    values = panel.indicators.loc[days, columns].to_numpy()
    known = ~np.isnan(values).all(axis=1)
    out = np.full(len(values), -1)  # -1: no regime yet; the pooled model predicts
    out[known] = np.nan_to_num(values[known], nan=-np.inf).argmax(axis=1)
    return out
