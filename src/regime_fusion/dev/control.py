"""The positive control (docs/E9_PLAN.md, part 1): plant a signal of known strength and ask each design to find it.

1A, ranking stocks (the design E9 uses). The small-cap panel's hedged returns are shuffled across the stocks of
each date, which removes any real link with the 17 factors (they stay in the inputs as decoys). A slow
characteristic is planted as one more input, and a component proportional to a planted function ``g`` is added
to the returns, scaled so that the mean rank correlation of ``g`` with the returns equals a target IC. Three
plants: P1 (returns rise with one characteristic), P2 (returns higher only when a fundamental-type and a
technical-type characteristic are both high), P3 (returns follow one characteristic in one market state and
another in the other). The E9 GP (zero costs: the control is about detection) and three reference learners
(best single input, ridge, LightGBM) are scored on the last 24 dates.

1B, timing one stock (the campaigns' design). A daily indicator is planted in the prices of one stock, so that
each day's return gains a component proportional to the previous day's indicator, and the campaigns' rung A1 is
asked to find it (training 2015-2017, evaluation 2018).

Seeds depend on the sample (size, run, stock), never on the plant or the strength, so that a plant and a
strength change only the planted part. Development data only. Results go to
``<small dir>/results/dev-control/1a/...`` and ``<data dir>/results/dev-control/1b/...``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import zlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.dev import e9
from regime_fusion.dev import score
from regime_fusion.dev import small
from regime_fusion.dev.designs import Design
from regime_fusion.engine.fitness import Evaluator, sharpe
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.panel import PRICE_COLUMNS, Panel, load_panel
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.tree import Grammar, describe, signals, to_record
from regime_fusion.seeds import make_rng
from regime_fusion.windows import DateRange

CAMPAIGN = "dev-control"
PLANTS = ("P1", "P2", "P3", "P4")
DEFAULT_PLANTS = ("P1", "P2", "P3")   # the plants of the plan's first control (P4 is asked for by name)
ICS = (0.0, 0.02, 0.03, 0.05, 0.10)
SIZES = (50, 300, 700)
RUNS_1A = 20
PANEL_STOCKS = 694            # the small-cap panel's size when the plan was written: subsets are shares of it
TEST_START = 57               # the first test date's position (24 test dates with 81 decision dates)
TOP = 0.2                     # the reference learners' and the oracle's book: the top fifth
AR = 0.9
PLANTED = ("fa:planted", "ta:planted", "rg:planted_state")
LONG_SHARE = 0.7              # long samples: the first 70% of the dates train, the rest test
MEAN_BLOCK = 6.0              # long samples: the stationary bootstrap's mean block length (dates)
REFERENCES = ("best_single", "ridge", "lightgbm")
REFERENCE_LABELS = {"best_single": "best single input", "ridge": "ridge", "lightgbm": "LightGBM"}
IC_TOLERANCE_1A = 0.001
RIDGE_PENALTY = 0.05

# 1B
STOCKS_1B = ("ABT", "DIS", "GD", "MS", "NFLX", "NSC", "SO", "SPG", "UPS", "WFC")
ICS_1B = (0.0, 0.05, 0.10, 0.20, 0.30)
RUNS_1B = 10
TRAIN_1B = DateRange(dt.date(2015, 2, 1), dt.date(2017, 12, 31))
EVALUATE_1B = DateRange(dt.date(2018, 1, 1), dt.date(2018, 12, 31))
EXIT_DAYS, EXIT_RATE = 20, 0.03
HORIZON_1B = 20               # the indicator is scored against the return from the open of t+1 to the open of t+21
IC_TOLERANCE_1B = 0.002


def ic_label(ic: float) -> str:
    return f"ic{ic * 100:03.0f}"


def _clean(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _nan(x) -> float:
    return float("nan") if x is None else float(x)


# ------------------------------------------------------------------------------------------ planting, 1A
def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row, the Pearson correlation over the columns where both are finite (NaN with under 3 or no spread)."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    safe = np.maximum(n, 1)
    da = np.where(ok, a0 - (a0.sum(axis=1) / safe)[:, None], 0.0)
    db = np.where(ok, b0 - (b0.sum(axis=1) / safe)[:, None], 0.0)
    num, va, vb = (da * db).sum(axis=1), (da * da).sum(axis=1), (db * db).sum(axis=1)
    den = np.sqrt(va * vb)
    with np.errstate(all="ignore"):
        out = np.where((n >= 3) & (den > 0), num / den, np.nan)
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks along each row; NaN stays NaN."""
    return pd.DataFrame(x).rank(axis=1).to_numpy(dtype=float)


def _spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both = np.isfinite(a) & np.isfinite(b)
    return _row_corr(_ranks(np.where(both, a, np.nan)), _ranks(np.where(both, b, np.nan)))


def mean_spearman(g: np.ndarray, returns: np.ndarray) -> float:
    """Mean over the dates of the rank correlation of ``g`` and ``returns`` across the stocks with both."""
    out = _spearman_rows(g, returns)
    return float(np.nanmean(out)) if np.isfinite(out).any() else float("nan")


def bisect_lambda(f, ic: float, tolerance: float, start: float, limit: float = math.inf) -> float:
    """lambda with ``f(lambda)`` within ``tolerance`` of ``ic``, ``f`` rising with lambda, found by bisection
    from 0 upwards. lambda is not negative unless the unplanted data already give ``f(0)`` above ``ic`` by
    chance (possible in small samples and for a single stock), when only a negative lambda can reach ``ic``.
    ``start`` is the first step of the search, ``limit`` the largest |lambda| allowed."""
    sign = 1.0 if f(0.0) <= ic else -1.0
    far = min(start, limit)
    while sign * (f(sign * far) - ic) < 0:
        if far >= limit:
            raise ValueError(f"an IC of {ic} cannot be reached by planting")
        far = min(2.0 * far, limit)
    lo, hi = (0.0, far) if sign > 0 else (-far, 0.0)
    best = sign * far
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = f(mid)
        best = mid
        if abs(value - ic) <= tolerance / 2:
            break
        lo, hi = (mid, hi) if value < ic else (lo, mid)
    if abs(f(best) - ic) > tolerance:
        raise ValueError(f"could not calibrate lambda to an IC of {ic} within {tolerance}")
    return float(best)


def calibrate_1a(shuffled: np.ndarray, g: np.ndarray, ic: float, tolerance: float = IC_TOLERANCE_1A) -> float:
    """lambda such that the mean rank correlation of ``g`` with ``shuffled + lambda g`` is ``ic`` (within
    ``tolerance``); 0 for ic 0 (negative only in the chance case of ``bisect_lambda``)."""
    if ic == 0:
        return 0.0
    ranked_g = _ranks(g)
    return bisect_lambda(lambda lam: float(np.nanmean(_row_corr(ranked_g, _ranks(shuffled + lam * g)))),
                         ic, tolerance, 0.01, 1e6)


def _rank_input(values: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """Each date's percentile rank among the tradable stocks, in (-1, 1]; NaN elsewhere."""
    frame = pd.DataFrame(np.where(tradable, values, np.nan))
    return 2.0 * frame.rank(axis=1, pct=True).to_numpy(dtype=float) - 1.0


def long_train(dates: int) -> int:
    """The number of training dates of a long sample: the first 70% of its dates, rounded."""
    return int(round(LONG_SHARE * dates))


def block_positions(n: int, length: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """``length`` positions into ``n`` dates by the stationary bootstrap: each step either starts a new block at
    a random position (probability 1 / ``mean_block``) or follows the one before, wrapping round at the end.
    (``analysis.stats.stationary_bootstrap`` returns exactly ``n`` positions; this is the same rule for any length.)"""
    first = rng.integers(n)
    restart = rng.random(length) < 1.0 / mean_block
    starts = rng.integers(n, size=length)
    restart[0], starts[0] = True, first
    steps = np.arange(length)
    began = np.maximum.accumulate(np.where(restart, steps, 0))
    return (starts[began] + steps - began) % n


def planted_data(ctx: e9.Context, plant: str, ic: float, stocks: int, run_number: int,
                 dates: int | None = None) -> dict[str, Any]:
    """One replicate of the planted data from the small-cap context (see the module docstring). Everything
    random is drawn from one generator, in the same order whatever the plant and strength. With ``dates`` (a long
    sample) the real dates are first resampled to that many by a block bootstrap with its own generator, which
    depends on ``dates`` and the run only; everything else then runs along the new timeline. ``positions`` in the
    result are the real dates' positions that the rows stand for."""
    if plant not in PLANTS:
        raise ValueError(f"plant must be one of {PLANTS}")
    config = load_config()
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, "1a", stocks, run_number)
    hedged = np.asarray(ctx.hedged, dtype=float)
    positions = np.arange(len(hedged))
    if dates is not None:
        long_rng = make_rng(config.base_seed, "dev", CAMPAIGN, "1a-long", dates, run_number)
        positions = block_positions(len(hedged), dates, MEAN_BLOCK, long_rng)
        hedged = hedged[positions]
    dates, n = hedged.shape
    # 1. the stocks
    subset = np.ones(n, dtype=bool)
    if stocks < 700:
        subset = np.zeros(n, dtype=bool)
        subset[rng.choice(n, size=min(n, round(n * stocks / PANEL_STOCKS)), replace=False)] = True
    tradable = np.isfinite(hedged) & subset[None, :]
    # 2. the shuffle, date by date
    shuffled = np.full((dates, n), np.nan)
    for d in range(dates):
        where = np.flatnonzero(tradable[d])
        shuffled[d, where] = hedged[d, where][rng.permutation(len(where))]
    # 3. two slow characteristics per stock
    noise = rng.standard_normal((2, dates, n))
    chars = np.empty_like(noise)
    chars[:, 0] = noise[:, 0]
    scale = math.sqrt(1.0 - AR ** 2)
    for t in range(1, dates):
        chars[:, t] = AR * chars[:, t - 1] + scale * noise[:, t]
    u1, u2 = _rank_input(chars[0], tradable), _rank_input(chars[1], tradable)
    # 4. the market state
    first, flips = rng.random(), rng.random(max(dates - 1, 0)) >= 0.9
    state = np.empty(dates)
    state[0] = 1.0 if first < 0.5 else -1.0
    for t in range(1, dates):
        state[t] = -state[t - 1] if flips[t - 1] else state[t - 1]
    # 5. the planted component
    if plant == "P1":
        g = u1.copy()
    elif plant == "P2":
        both = np.where(tradable, ((u1 > 0.2) & (u2 > 0.2)).astype(float), np.nan)
        g = both - np.nanmean(both, axis=1, keepdims=True)
    elif plant == "P3":
        g = np.where((state > 0)[:, None], u1, u2)
    else:                                  # P4: the characteristic's sign follows the state
        g = state[:, None] * u1
    g = np.where(tradable, g, np.nan)
    # 6. the strength
    lam = calibrate_1a(shuffled, g, ic)
    returns = shuffled + lam * g
    # 7. the inputs
    names = [f"fa:{f}" for f in e9.FA_FACTORS] + [f"ta:{f}" for f in e9.TA_FACTORS] + list(e9.CONTEXT)
    inputs = {name: np.where(subset[None, :], np.asarray(ctx.inputs[name])[positions], np.nan) for name in names}
    inputs["fa:planted"], inputs["ta:planted"] = u1, u2
    inputs["rg:planted_state"] = np.repeat(state[:, None], n, axis=1)
    return {"inputs": inputs, "returns": returns, "g": g, "lambda": lam, "tradable": tradable,
            "positions": positions}


def groups_1a(inputs: dict[str, np.ndarray]) -> dict[str, list[str]]:
    return {"FA": [k for k in inputs if k.startswith("fa:")], "TA": [k for k in inputs if k.startswith("ta:")],
            "RG": [k for k in inputs if k.startswith("rg:")]}


def detected(plant: str, used) -> bool:
    """Whether the rule's deciding part uses what the plant needs: P1 the characteristic, P2 both, P3 both and the
    state, P4 the characteristic and the state."""
    used = set(used)
    need = {"P1": {"fa:planted"}, "P2": {"fa:planted", "ta:planted"}, "P3": set(PLANTED),
            "P4": {"fa:planted", "rg:planted_state"}}[plant]
    return need <= used


def uses_planted(used) -> bool:
    return bool(set(used) & set(PLANTED))


def top_book(score: np.ndarray, tradable: np.ndarray, share: float = TOP) -> np.ndarray:
    """Per date, the top ``share`` of the tradable stocks with a score (at least one; NaN scores never held)."""
    out = np.zeros(score.shape, dtype=bool)
    for d in range(score.shape[0]):
        ok = np.flatnonzero(tradable[d])
        if len(ok) == 0:
            continue
        k = max(1, int(round(share * len(ok))))
        s = score[d, ok]
        order = np.argsort(-np.where(np.isfinite(s), s, -np.inf), kind="stable")[:k]
        out[d, ok[order[np.isfinite(s[order])]]] = True
    return out


def oracle_signal(plant: str, g: np.ndarray, tradable: np.ndarray) -> np.ndarray:
    """The best book that knows the plant: P2 the stocks with both characteristics high, the others (P1, P3 and
    P4) the top fifth by ``g``."""
    if plant == "P2":
        return np.where(np.isfinite(g), g, -1.0) > 0
    return top_book(g, tradable)


# Versions of the GP in 1A. "fitness": the plan's first version (model = fittest, fitness on raw returns).
# "holdback": the model is chosen on held-back training dates. "clipped", "ranks": the fitness scores books
# on clipped or ranked training returns (``e9.fitness_returns``). Test results always use the returns as planted.
# "score-book", "score-rank": the scoring GP of E10 (``score.evolve_score``) with its book or rank fitness, on the
# planted returns as they are (it ranks them itself); its book is the top fifth of its score, like the references'.
SCORE_FITNESS = {"score-book": "book", "score-rank": "rank"}
SELECTIONS = ("fitness", "holdback", "clipped", "ranks", *SCORE_FITNESS)


def part_1a(selection: str = "fitness") -> str:
    """The results folder of 1A for a way of choosing the model: ``1a``, or ``1a-holdback`` for the repair."""
    if selection not in SELECTIONS:
        raise ValueError(f"unknown selection {selection!r}")
    return "1a" if selection == "fitness" else f"1a-{selection}"


def path_1a(plant: str, ic: float, stocks: int, train: int, run_number: int, selection: str = "fitness",
            dates: int | None = None) -> Path:
    """Where a run's record goes; a long sample (``dates``) has an extra folder ``d<dates>`` after the stocks'."""
    folder = small.SMALL_DIR / "results" / CAMPAIGN / part_1a(selection) / plant / ic_label(ic) / f"n{stocks}"
    if dates is not None:
        folder = folder / f"d{dates}"
    return folder / f"t{train}" / f"run{run_number:02d}.json"


def _cells_1a() -> list[tuple[str, float, int, int]]:
    cells = [(p, ic, s, 57) for p in DEFAULT_PLANTS for ic in ICS for s in SIZES]
    cells += [(p, ic, 700, 36) for p in DEFAULT_PLANTS for ic in ICS]
    return cells


CELLS_1A = _cells_1a()
CELLS_P4 = [("P4", ic, s, 57) for ic in ICS for s in SIZES] + [("P4", ic, 700, 36) for ic in ICS]
DATES_LONG = (150, 315)
RUNS_LONG = 10
CELLS_LONG = [(p, ic, 700, long_train(d), d) for p in ("P3", "P4") for ic in (0.03, 0.05, 0.10) for d in DATES_LONG]


# ----------------------------------------------------------------------------------- reference learners
def _train_ics(X: dict[str, np.ndarray], returns: np.ndarray) -> dict[str, float]:
    return {name: float(np.nanmean(_spearman_rows(x, returns))) if np.isfinite(_spearman_rows(x, returns)).any()
            else float("nan") for name, x in X.items()}


def _reference_scores(inputs: dict[str, np.ndarray], returns: np.ndarray, train: np.ndarray, test: np.ndarray,
                      ) -> dict[str, dict[str, Any]]:
    """Fit the three references on the training dates; give each one's score on the test dates (date x stock)
    and what it picked."""
    import lightgbm as lgb

    names = list(inputs)                                                   # 19 factor inputs, then 3 RG
    X = np.stack([inputs[n] for n in names], axis=2)                       # date x stock x input
    target = 2.0 * pd.DataFrame(returns).rank(axis=1, pct=True).to_numpy(dtype=float) - 1.0
    ok = np.isfinite(target[train])
    xtr, ytr = np.nan_to_num(X[train])[ok], target[train][ok]
    known = np.isfinite(X[test]).any(axis=2)
    out: dict[str, dict[str, Any]] = {}

    ics = _train_ics({n: inputs[n][train] for n in names}, returns[train])
    usable = {n: abs(v) for n, v in ics.items() if np.isfinite(v)}
    chosen = max(usable, key=usable.get) if usable else names[0]
    sign = 1.0 if not np.isfinite(ics.get(chosen, np.nan)) or ics[chosen] >= 0 else -1.0
    out["best_single"] = {"score": sign * inputs[chosen][test], "chosen": chosen,
                          "detected": chosen == "fa:planted"}

    w = np.linalg.solve(xtr.T @ xtr + RIDGE_PENALTY * len(ytr) * np.eye(len(names)), xtr.T @ ytr)
    score = np.where(known, np.nan_to_num(X[test]) @ w, np.nan)
    out["ridge"] = {"score": score, "chosen": names[int(np.argmax(np.abs(w)))],
                    "detected": names[int(np.argmax(np.abs(w)))] == "fa:planted"}

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=8, min_child_samples=200,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=0, verbose=-1,
                              n_jobs=1, importance_type="gain").fit(xtr, ytr)
    gain = np.asarray(model.feature_importances_, dtype=float)
    top = names[int(np.argmax(gain))] if gain.max() > 0 else None
    pred = np.full(known.shape, np.nan)
    flat = np.nan_to_num(X[test])
    if known.any():
        pred[known] = model.predict(flat[known])
    out["lightgbm"] = {"score": pred, "chosen": top, "detected": top == "fa:planted"}
    return out


# ------------------------------------------------------------------------------------------------ runs, 1A
def run_1a(plant: str, ic: float, stocks: int, train: int, run_number: int, write: bool = True, *,
           ctx: e9.Context | None = None, budget: dict[str, int] | None = None,
           test_start: int = TEST_START, selection: str = "fitness", dates: int | None = None) -> dict[str, Any]:
    """Plant, evolve on the first ``train`` dates, score the GP and the references on the dates from
    ``test_start``. With ``selection="holdback"`` the GP evolves on the earlier training dates only and its
    model is the candidate that does best on the latest ones (``e9.choose_on_holdback``); the references
    always use every training date. With "score-book" or "score-rank" the GP is the scoring GP of E10 and its
    book is the top fifth of its score, as for the references. With ``dates`` the sample is a long one
    (``planted_data``): ``train`` must be ``long_train(dates)`` and the test dates start there.
    ``ctx``, ``budget`` and ``test_start`` are for tests."""
    part_1a(selection)
    if dates is not None:
        if train != long_train(dates):
            raise ValueError(f"a long sample of {dates} dates trains on {long_train(dates)} dates, not {train}")
        test_start = train
    started = time.perf_counter()
    config = load_config()
    base = load_rungs()[e9.BASE_RUNG]
    ctx = e9.context("small") if ctx is None else ctx
    budget = e9.BUDGET if budget is None else budget
    data = planted_data(ctx, plant, ic, stocks, run_number, dates)
    inputs, returns, tradable, positions = data["inputs"], data["returns"], data["tradable"], data["positions"]
    n_dates = len(returns)
    train_pos, test_pos = np.arange(train), np.arange(test_start, n_dates)
    zero_cost = np.zeros_like(returns)
    groups = groups_1a(inputs)
    names = [n for g in groups.values() for n in g]
    fit_pos, held_pos = e9.holdback_split(train_pos) if selection == "holdback" else (train_pos, train_pos[:0])
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, "1a-gp", stocks, train, run_number)   # not the version's
    holdback_score = None
    test_inputs = {n: inputs[n][test_pos] for n in names}
    if selection in SCORE_FITNESS:
        evidence = "ranks"
        best = score.evolve_score({n: inputs[n][train_pos] for n in names},
                                  {"FA": groups["FA"], "TA": groups["TA"]}, returns[train_pos], zero_cost[train_pos],
                                  fitness=SCORE_FITNESS[selection], if_inputs=[n for n in names if n.startswith("rg:")],
                                  budget=budget, rng=rng)
        used = best.inputs_used()
    else:
        evidence = selection if selection in e9.EVIDENCE else "raw"
        market = e9.BookMarket(e9.fitness_returns(returns[fit_pos], evidence), zero_cost[fit_pos], e9.LOW, e9.HIGH)
        settings = replace(base.fitness, min_trades=math.ceil(e9.COVERAGE * len(fit_pos)))
        evaluator = e9.KeepingEvaluator({n: inputs[n][fit_pos] for n in names}, market, settings, cases=False)
        grammar = Grammar(groups, fusion_functions=("AND", "OR", "IF"), sources=e9.SOURCES, max_branch_depth=3)
        gp_config = replace(base.gp, **budget, active_trading=False)
        best = evolve(grammar, evaluator, gp_config, rng).best
        if selection == "holdback":
            found = e9.choose_on_holdback(evaluator.final, {n: inputs[n][held_pos] for n in names},
                                          e9.BookMarket(returns[held_pos], zero_cost[held_pos], e9.LOW, e9.HIGH))
            if found is not None:
                best, holdback_score = found
        used = e9.inputs_used(best)

    later = e9.BookMarket(returns[test_pos], zero_cost[test_pos], e9.LOW, e9.HIGH)
    test_tradable = tradable[test_pos]

    def result(signal: np.ndarray) -> float:
        return float(later.gross(signal).mean())

    if selection in SCORE_FITNESS:
        gp_result = result(top_book(best.score(test_inputs, test_tradable), test_tradable))
        model_text, model_tree = best.describe(), best.to_record()
    else:
        gp_result = result(signals(best, test_inputs)["C"])
        model_text, model_tree = describe(best), to_record(best)
    oracle = result(oracle_signal(plant, data["g"][test_pos], test_tradable))

    def captured(x: float) -> float | None:
        return _clean(x / oracle) if ic > 0 and oracle > 0 else None

    refs = {}
    for name, found in _reference_scores(inputs, returns, train_pos, test_pos).items():
        value = result(top_book(found["score"], test_tradable))
        refs[name] = {"test_mean": value, "captured": captured(value), "chosen": found["chosen"],
                      "detected": bool(found["detected"]) if plant == "P1" else None}
    record = {
        "campaign": CAMPAIGN, "part": "1a", "plant": plant, "ic": ic, "stocks": stocks, "train": train, "run": run_number,
        "budget": budget, "selection": selection, "evidence": evidence, "fit_dates": int(len(fit_pos)), "held_dates": int(len(held_pos)),
        "holdback_score": holdback_score, "lambda": data["lambda"], "n_tradable_mean": float(tradable.sum(axis=1).mean()),
        "realised_ic": _clean(mean_spearman(data["g"], returns)),
        "model": model_text, "tree": model_tree, "fitness": best.fitness, "inputs_used": used,
        "detected": detected(plant, used), "uses_planted": uses_planted(used),
        "gp_test_mean": gp_result, "oracle_test_mean": oracle, "captured": captured(gp_result),
        "references": refs, "test_dates": [d.strftime("%Y-%m-%d") for d in ctx.dates[positions[test_pos]]],
        "seconds": round(time.perf_counter() - started, 2),
    }
    extra = {} if dates is None else {"dates": dates}
    record.update(extra)
    record["provenance"] = provenance.collect(
        config, rung_digest=base.digest(overrides=budget), rung_spec=base.spec, experiment=CAMPAIGN, part="1a",
        plant=plant, ic=ic, stocks=stocks, train=train, run=run_number, selection=selection,
        sources=list(e9.SOURCES), **extra)
    if write:
        provenance.write(path_1a(plant, ic, stocks, train, run_number, selection, dates), record)
    return record


def jobs_1a(runs: int = RUNS_1A, selection: str = "fitness", plants=None, ics=None, long: bool = False) -> list[tuple]:
    """Unfinished runs as (plant, ic, stocks, train, run), with the selection added when it is not "fitness".
    ``plants`` default to P1-P3 (P4's cells, ``CELLS_P4``, come when it is named), ``ics`` limits the strengths.
    With ``long`` only the long samples (``CELLS_LONG``, at most ``RUNS_LONG`` runs each, ``plants`` default P3 and
    P4) come, as (plant, ic, stocks, train, run, selection, dates)."""
    wanted = None if ics is None else {float(i) for i in ics}
    if long:
        plants = ("P3", "P4") if plants is None else plants
        return [(p, ic, s, t, k, selection, d) for (p, ic, s, t, d) in CELLS_LONG
                if p in plants and (wanted is None or ic in wanted) for k in range(min(runs, RUNS_LONG))
                if not path_1a(p, ic, s, t, k, selection, d).exists()]
    plants = DEFAULT_PLANTS if plants is None else plants
    extra = () if selection == "fitness" else (selection,)
    cells = CELLS_1A + (CELLS_P4 if "P4" in plants else [])
    return [(p, ic, s, t, k, *extra) for (p, ic, s, t) in cells
            if p in plants and (wanted is None or ic in wanted) for k in range(runs)
            if not path_1a(p, ic, s, t, k, selection).exists()]


def shard_of(job: tuple, parts: int) -> int:
    """The part (1 to ``parts``) of a job array that does this run: fixed by the run itself, so it does not
    move when other runs finish, and two parts never do the same run."""
    return zlib.crc32(repr(tuple(job)).encode()) % parts + 1


def run_job_1a(job: tuple) -> str:
    """Run a job of ``jobs_1a``: 5 elements (plant, ic, stocks, train, run), 6 with the selection, 7 with the dates."""
    record = run_1a(*job[:5], selection=job[5] if len(job) > 5 else "fitness", dates=job[6] if len(job) > 6 else None)
    return (f"{job}: fitness {record['fitness']:.3f}, detected {record['detected']}, "
            f"{record['seconds']:.0f} s")


# ------------------------------------------------------------------------------------------- planting, 1B
def ar1(rng: np.random.Generator, n: int, coefficient: float = AR) -> np.ndarray:
    """A stationary AR(1) series with standard normal values."""
    noise = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = noise[0]
    scale = math.sqrt(1.0 - coefficient ** 2)
    for t in range(1, n):
        out[t] = coefficient * out[t - 1] + scale * noise[t]
    return out


def plant_prices(prices: pd.DataFrame, s: np.ndarray, lam: float) -> pd.DataFrame:
    """Every price of day t times the product up to t of (1 + lam s[j-1]); the first day's factor is 1. So a
    day's return gains lam x the previous day's indicator, and day t's prices use s up to t-1 only."""
    steps = 1.0 + lam * np.asarray(s, dtype=float)[:-1]
    if (steps <= 0).any():
        raise ValueError("lambda is too large: a daily factor would not be positive")
    factor = np.concatenate([[1.0], np.cumprod(steps)])
    return prices.mul(factor, axis=0)


def _spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
    ranks = pd.DataFrame({"a": a, "b": b}).rank()
    return float(np.corrcoef(ranks["a"], ranks["b"])[0, 1])


def planted_ic_1b(prices: pd.DataFrame, s: np.ndarray, lam: float, train_days: int,
                  horizon: int = HORIZON_1B) -> float:
    """Over the training sessions t (those whose return to the open of t + horizon + 1 is still inside
    training), the rank correlation of s[t] with the planted prices' return from the open of t+1."""
    planted = plant_prices(prices, s, lam)["adj_open"].to_numpy(dtype=float)
    t = np.arange(0, train_days - horizon - 1)
    forward = planted[t + horizon + 1] / planted[t + 1] - 1.0
    return _spearman_1d(np.asarray(s)[t], forward)


def calibrate_1b(prices: pd.DataFrame, s: np.ndarray, ic: float, train_days: int,
                 tolerance: float = IC_TOLERANCE_1B) -> float:
    if ic == 0:
        return 0.0
    return bisect_lambda(lambda lam: planted_ic_1b(prices, s, lam, train_days), ic, tolerance, 1e-4,
                         0.99 / float(np.abs(s).max()))


def plant_panel(panel: Panel, s: np.ndarray, lam: float) -> Panel:
    """The panel with planted prices and ``ta:planted`` (the indicator scaled to [-1, 1] by its training
    minimum and maximum, like the other indicators). The real technical indicators stay as they were."""
    prices = plant_prices(panel.prices, s, lam)
    fit = np.asarray(s, dtype=float)[:len(panel.train)]
    low, high = fit.min(), fit.max()
    scaled = pd.Series(2.0 * (np.asarray(s, dtype=float) - low) / (high - low) - 1.0, index=panel.prices.index)
    indicators = panel.indicators.assign(**{"ta:planted": scaled})
    return replace(panel, prices=prices, indicators=indicators)


def path_1b(stock: str, ic: float, run_number: int) -> Path:
    return store.data_dir() / "results" / CAMPAIGN / "1b" / stock / ic_label(ic) / f"run{run_number:02d}.json"


def run_1b(stock: str, ic: float, run_number: int, write: bool = True, *, panel: Panel | None = None,
           budget: dict[str, int] | None = None) -> dict[str, Any]:
    """Plant the indicator in one stock's prices, evolve rung A1 on 2015-2017, score 2018. ``panel`` and
    ``budget`` are for tests."""
    started = time.perf_counter()
    config = load_config()
    rung = load_rungs()["A1"]
    budget = {"population": 500, "generations": 20} if budget is None else budget
    cost = config.primary_cost_bps_per_side
    if panel is None:
        panel = load_panel(stock, TRAIN_1B, EVALUATE_1B, rung.sources, None, rung.scaling)
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, "1b", stock, run_number)
    s = ar1(rng, len(panel.prices))
    lam = calibrate_1b(panel.prices, s, ic, len(panel.train))
    planted = plant_panel(panel, s, lam)

    names = planted.names()
    grammar = Grammar(names, fusion_functions=rung.fusion, sources=rung.sources, max_branch_depth=rung.max_branch_depth)
    market = planted.market(planted.train, EXIT_DAYS, EXIT_RATE, cost)
    evaluator = Evaluator(planted.arrays(planted.train), market, rung.fitness, cases=False)
    gp_config = replace(rung.gp, **budget)
    gp_rng = make_rng(config.base_seed, "dev", CAMPAIGN, "1b-gp", stock, run_number)
    best = evolve(grammar, evaluator, gp_config, gp_rng).best

    later = planted.market(planted.evaluate, EXIT_DAYS, EXIT_RATE, cost)
    model_daily, model_trades = later.daily(signals(best, planted.arrays(planted.evaluate))["C"])
    position = panel.prices.index.get_indexer(planted.evaluate)
    rule_daily, rule_trades = later.daily(np.asarray(s)[position] > 0)
    frame = planted.prices.loc[planted.evaluate]
    hold = Design("long", cost / 1e4).daily(np.zeros(len(planted.evaluate), dtype=bool),
                                           *(frame[c].to_numpy(dtype=np.float64) for c in PRICE_COLUMNS))[0]
    used = e9.inputs_used(best)
    record = {
        "campaign": CAMPAIGN, "part": "1b", "stock": stock, "ic": ic, "run": run_number, "budget": budget,
        "lambda": lam, "realised_ic": _clean(planted_ic_1b(panel.prices, s, lam, len(panel.train))),
        "model": describe(best), "tree": to_record(best), "fitness": best.fitness, "inputs_used": used,
        "detected": "ta:planted" in used, "gp_sharpe": float(sharpe(model_daily)), "gp_trades": int(model_trades),
        "rule_sharpe": float(sharpe(rule_daily)), "rule_trades": int(rule_trades),
        "hold_sharpe": float(sharpe(hold)), "seconds": round(time.perf_counter() - started, 2),
    }
    record["provenance"] = provenance.collect(
        config, rung_digest=rung.digest(overrides=budget), rung_spec=rung.spec, experiment=CAMPAIGN, part="1b",
        stock=stock, ic=ic, run=run_number, panel_inputs=panel.inputs)
    if write:
        provenance.write(path_1b(stock, ic, run_number), record)
    return record


def jobs_1b(runs: int = RUNS_1B) -> list[tuple]:
    return [(s, ic, k) for s in STOCKS_1B for ic in ICS_1B for k in range(runs) if not path_1b(s, ic, k).exists()]


def run_job_1b(job: tuple) -> str:
    record = run_1b(*job)
    return (f"{job}: lambda {record['lambda']:.4f}, detected {record['detected']}, GP Sharpe {record['gp_sharpe']:.2f}, "
            f"{record['seconds']:.0f} s")


# ------------------------------------------------------------------------------------------------ report
def _records(root: Path) -> list[dict]:
    files = [p for p in sorted(root.glob("**/run*.json")) if not p.name.endswith(".provenance.json")]
    return [json.loads(p.read_text()) for p in files]


def _flat_1a(r: dict) -> dict:
    row = {"plant": r["plant"], "ic": r["ic"], "stocks": r["stocks"], "train": r["train"], "run": r["run"],
           "dates": r.get("dates"),
           "detected": bool(r["detected"]), "uses_planted": bool(r["uses_planted"]), "fitness": r["fitness"],
           "gp_test": r["gp_test_mean"], "gp_captured": _nan(r["captured"])}
    for k in REFERENCES:
        ref = r["references"][k]
        row[f"{k}_test"], row[f"{k}_captured"] = ref["test_mean"], _nan(ref["captured"])
        row[f"{k}_detected"] = float("nan") if ref["detected"] is None else float(ref["detected"])
    return row


LEARNERS = {"GP": "gp_test", **{REFERENCE_LABELS[k]: f"{k}_test" for k in REFERENCES}}


def table_1a(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["plant", "ic", "stocks", "train"]
    rows = []
    for key, g in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update({"runs": len(g), "gp_detection": float(g["detected"].mean()),
                    "gp_captured_median": float(g["gp_captured"].median())})
        for k in REFERENCES:
            row[f"{k}_captured_median"] = float(g[f"{k}_captured"].median())
        row["gp_test_mean"] = float(g["gp_test"].mean())
        for k in REFERENCES:
            row[f"{k}_test_mean"] = float(g[f"{k}_test"].mean())
        row["gp_train_fitness_mean"] = float(g["fitness"].mean())
        if key[0] == "P1":
            for k in REFERENCES:
                row[f"{k}_detection"] = float(g[f"{k}_detected"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def table_long(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per plant, strength and number of dates of the long samples."""
    keys = ["plant", "ic", "dates"]
    rows = []
    for key, g in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update({"runs": len(g), "gp_detection": float(g["detected"].mean()),
                    "gp_captured_median": float(g["gp_captured"].median())})
        for k in REFERENCES:
            row[f"{k}_captured_median"] = float(g[f"{k}_captured"].median())
        rows.append(row)
    return pd.DataFrame(rows, columns=keys + ["runs", "gp_detection", "gp_captured_median",
                                              *[f"{k}_captured_median" for k in REFERENCES]])


def _holds(flag: bool) -> str:
    return "holds" if flag else "does not hold"


def conditions_1a(frame: pd.DataFrame, table: pd.DataFrame) -> list[str]:
    lines = []
    zero = frame[frame["ic"] == 0]
    # at ic 0 the three plants give the same data and the same GP run, so count each sample once
    zero = zero.sort_values("plant").drop_duplicates(["stocks", "train", "run"])
    if len(zero) < 2:
        lines.append("  (1) nothing planted: too few runs with IC 0 to judge.")
    else:
        parts, flag = [], True
        for label, column in LEARNERS.items():
            x = zero[column].to_numpy(dtype=float)
            mean, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
            inside = abs(mean) <= 2 * se
            flag &= inside
            parts.append(f"{label} {100 * mean:+.3f}% (2 SE {100 * 2 * se:.3f}%)")
        lines.append(f"  (1) nothing planted: the mean test result of every learner is within two standard errors of "
                     f"zero over {len(zero)} runs; per date, {'; '.join(parts)}: {_holds(flag)}.")
        noise = zero.groupby(["stocks", "train"]).agg(runs=("run", "size"), planted=("uses_planted", "mean"),
                                                      fitness=("fitness", "mean"))
        lines.append(f"      With nothing planted, {100 * zero['uses_planted'].mean():.0f}% of the GP's final rules use a "
                     f"planted input; the mean training fitness (the noise level) is {zero['fitness'].mean():.3f}.")
        for (s, t), row in noise.iterrows():
            lines.append(f"        {s} stocks, train {t}: {int(row['runs'])} runs, {100 * row['planted']:.0f}% use a planted "
                         f"input, mean training fitness {row['fitness']:.3f}")
    cell = table[(table["plant"] == "P1") & (table["ic"] == 0.05) & (table["stocks"] == 700) & (table["train"] == 57)]
    if cell.empty:
        lines.append("  (2) P1, IC 0.05, 700 stocks, 57 dates: no runs yet.")
    else:
        rate = float(cell["gp_detection"].iloc[0])
        lines.append(f"  (2) P1, IC 0.05, 700 stocks, training 57 dates: the GP's detection rate is {100 * rate:.0f}% "
                     f"over {int(cell['runs'].iloc[0])} runs (at least 80% needed): {_holds(rate >= 0.8)}.")
    p1 = table[(table["plant"] == "P1") & (table["ic"] > 0)]
    if p1.empty:
        lines.append("  (3) P1 cells: no runs yet.")
    else:
        breaks = []
        for _, row in p1.iterrows():
            best = max(row[f"{k}_detection"] for k in REFERENCES)
            if row["gp_detection"] < best - 0.2 - 1e-9:
                breaks.append(f"IC {row['ic']}, {int(row['stocks'])} stocks, train {int(row['train'])}: GP "
                              f"{100 * row['gp_detection']:.0f}% against {100 * best:.0f}%")
        lines.append(f"  (3) in no P1 cell with IC above 0 is the GP's detection rate more than 20 points below the best "
                     f"reference learner's ({len(p1)} cells): {_holds(not breaks)}.")
        lines += [f"        breaks it: {b}" for b in breaks]
    return lines


def table_1b(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ic, g in frame.groupby("ic", sort=True):
        rows.append({"ic": ic, "runs": len(g), "detection": float(g["detected"].mean()),
                     "gp_sharpe_median": float(g["gp_sharpe"].median()),
                     "rule_sharpe_median": float(g["rule_sharpe"].median()),
                     "hold_sharpe_median": float(g["hold_sharpe"].median()),
                     "share_above_hold": float((g["gp_sharpe"] > g["hold_sharpe"]).mean()),
                     "share_never_trade": float((g["gp_trades"] == 0).mean())})
    return pd.DataFrame(rows)


def report(selection: str = "fitness") -> dict[str, pd.DataFrame]:
    """Print and write the tables of 1A and 1B, and apply the plan's three conditions to 1A. Works on whatever
    runs exist. ``selection`` picks which version of 1A is read (its table goes to ``control_<part>.csv``)."""
    part = part_1a(selection)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    out: dict[str, pd.DataFrame] = {}
    records = _records(small.SMALL_DIR / "results" / CAMPAIGN / part)
    print(f"Control 1A (ranking stocks), version {selection!r}: {len(records)} runs found")
    long_records = [r for r in records if r.get("dates") is not None]
    records = [r for r in records if r.get("dates") is None]      # the conditions are about the 81 real dates
    if records:
        frame = pd.DataFrame([_flat_1a(r) for r in records])
        table = table_1a(frame)
        out["1a"] = table
        shown = table.copy()
        for c in shown.columns:
            if c.endswith(("_detection", "_test_mean")):
                shown[c] = 100 * shown[c]
        print("(detection in %; test means in % per date, before costs; captured = share of the oracle's result)")
        print(shown.round(3).to_string(index=False))
        print("\nThe plan's conditions before E9 runs:")
        for line in conditions_1a(frame, table):
            print(line)
        target = small.SMALL_DIR / "results" / f"control_{part.replace('-', '_')}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
    if long_records:
        longs = table_long(pd.DataFrame([_flat_1a(r) for r in long_records]))
        out["1a_long"] = longs
        shown = longs.assign(gp_detection=100 * longs["gp_detection"])
        print(f"\nLong samples (block-bootstrapped dates, 70% training): {len(long_records)} runs found "
              f"(detection in %; captured = share of the oracle's result)")
        print(shown.round(3).to_string(index=False))
        target = small.SMALL_DIR / "results" / f"control_{part.replace('-', '_')}_long.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        longs.to_csv(target, index=False)
    records = _records(store.data_dir() / "results" / CAMPAIGN / "1b")
    print(f"\nControl 1B (timing one stock): {len(records)} runs found")
    if records:
        table = table_1b(pd.DataFrame(records))
        out["1b"] = table
        shown = table.assign(detection=100 * table["detection"], share_above_hold=100 * table["share_above_hold"],
                             share_never_trade=100 * table["share_never_trade"])
        print("(detection and shares in %; Sharpe ratios are 2018's, annualised, after costs)")
        print(shown.round(3).to_string(index=False))
        target = store.data_dir() / "results" / "control_1b.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
    return out


__all__ = ["CELLS_1A", "CELLS_P4", "CELLS_LONG", "planted_data", "run_1a", "jobs_1a", "run_job_1a", "run_1b", "jobs_1b", "run_job_1b", "report"]
