"""E9: the cross-sectional GP on the small caps, measured on a hedged basis (docs/E9_PLAN.md, part 2).

Every 20 sessions the GP's Boolean rule picks a book from the stocks in the universe: the stocks for which
the rule is true, equal weights, long only. The book's result on a date is its hedged return less the mean
hedged return of the stocks that could be traded that day, less the cost of the trades (tiered by value
traded, ``dev.crosssection``). A rule must hold between 10% and 40% of the tradable stocks on at least 90% of
the training dates, otherwise it scores the floor. Inputs are the 17 factors as ranks among the members
(unsigned: the literature's signs are not given to the GP), two stock-context ranks and three market-state
percentiles, all known at the close of the decision date.

Folds: for each test year 2014 to 2018, test = the decision dates of that year; train = every earlier
date except the last one before the test year (its holding period runs into the test year). 2014 is an
extra year, kept apart from the main years 2015 to 2018 in the report.

Seeds depend on the universe, the test year and the run number, never on the rung. Results go to
``<universe dir>/results/dev-e9/<rung>/<year>/run<NN>.json``. ``benchmarks`` rebuilds the plan's simple
alternatives on the same folds, hedge and costs; ``report`` applies the plan's four rules. Development data
only (to ``period.development_end``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.analysis import stats
from regime_fusion.config import load_config
from regime_fusion.dev import broad
from regime_fusion.dev import crosssection as C
from regime_fusion.dev import factors as F
from regime_fusion.dev import small
from regime_fusion.dev.fitness import FLOOR, LowerBoundEvaluator, psr_lower_bound
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.tree import Grammar, describe, signals, to_record
from regime_fusion.seeds import make_rng

CAMPAIGN = "dev-e9"
SOURCES = ("FA", "TA")
BASE_RUNG = "A1"
BUDGET = {"population": 500, "generations": 20}
RUNS = 20
LOW, HIGH = 0.10, 0.40            # share of the tradable stocks the book must hold
COVERAGE = 0.9                    # share of the training dates on which it must
TEST_YEARS = (2014, 2015, 2016, 2017, 2018)
MAIN_YEARS = (2015, 2016, 2017, 2018)

FA_FACTORS = ("earnings_price", "book_price", "sales_price", "cashflow_price", "op_profitability", "roe",
              "accruals", "leverage", "asset_growth", "share_issuance", "earnings_change")
TA_FACTORS = ("mom_12_1", "rev_1m", "high_52w", "volatility_60", "beta_252", "size")
CONTEXT = ("rg:value_traded", "rg:size")
STATE = ("rg:market_volatility", "rg:market_return", "rg:market_drawdown")
RUNGS = {
    "C1": {"fusion": ("AND", "OR"), "regime": ()},
    "C2": {"fusion": ("AND", "OR", "IF"), "regime": CONTEXT},
    "C3": {"fusion": ("AND", "OR", "IF"), "regime": CONTEXT + STATE},
}
BANDED = "composite (literature signs), banded 20/40"
STATE_WINDOWS = {"vol_window": 60, "return_window": 250, "min_history": 250}


# ------------------------------------------------------------------- choosing a run's model on held-back dates
HOLDBACK = 0.3                    # share of the training dates, the latest ones, kept out of the evolution
CANDIDATES = 50                   # the fittest distinct individuals of the last generation that are compared


class KeepingEvaluator(LowerBoundEvaluator):
    """``LowerBoundEvaluator`` that remembers the last population the evolution showed it (``final``)."""

    final: list = []

    def retain(self, population) -> None:
        self.final = list(population)
        super().retain(population)


def holdback_split(positions: np.ndarray, share: float = HOLDBACK) -> tuple[np.ndarray, np.ndarray]:
    """The training positions split in time: the earlier part to evolve on, the latest ``share`` held back."""
    held = min(max(2, round(share * len(positions))), len(positions) - 1)
    return positions[:-held], positions[-held:]


def choose_on_holdback(population, data: dict[str, np.ndarray], market: "BookMarket",
                       coverage: float | None = None, candidates: int = CANDIDATES):
    """Among the ``candidates`` fittest distinct individuals (those at the floor left out), the one whose
    book has the highest PSR lower bound on the held-back dates (``data`` and ``market`` cover those dates
    only). A book outside the size bounds on more than ``1 - coverage`` of them scores the floor. Returns
    the individual and its held-back score; ties, and the case where every candidate is at the floor, go to
    the fittest. None if no individual is above the floor."""
    coverage = COVERAGE if coverage is None else coverage
    floor = math.ceil(coverage * len(market.returns))
    ranked, seen = [], set()
    for individual in sorted(population, key=lambda i: -i.fitness):
        if individual.fitness <= FLOOR or individual.key() in seen:
            continue
        seen.add(individual.key())
        ranked.append(individual)
        if len(ranked) == candidates:
            break
    chosen, best = None, -math.inf
    for individual in ranked:
        net, active, _ = market.simulate(signals(individual, data)["C"])
        score = psr_lower_bound(net) if active >= floor else FLOOR
        if score > best:
            chosen, best = individual, float(score)
    return (chosen, best) if chosen is not None else None


# ------------------------------------------------------------- the returns the training fitness is shown
EVIDENCE = ("raw", "clipped", "ranks")
CLIP = (5.0, 95.0)                # percentiles of a date's returns for "clipped"


def fitness_returns(hedged: np.ndarray, how: str = "raw") -> np.ndarray:
    """The date x stock returns a training fitness scores books on. ``raw``: unchanged. ``clipped``: each
    date's returns limited to that date's 5th and 95th percentiles. ``ranks``: each return replaced by its
    rank among the date's stocks, from -1 to 1. Each date uses its own stocks only, and NaN stays NaN.
    A few extreme returns dominate a book's raw result and hide a weak signal; the other two limit that.
    Only the fitness is meant to see these: results are always measured on the returns themselves."""
    if how not in EVIDENCE:
        raise ValueError(f"unknown evidence {how!r}")
    hedged = np.asarray(hedged, dtype=float)
    if how == "raw":
        return hedged
    out = np.full(hedged.shape, np.nan)
    for i, row in enumerate(hedged):
        ok = np.isfinite(row)
        if not ok.any():
            continue
        if how == "clipped":
            low, high = np.percentile(row[ok], CLIP)
            out[i, ok] = np.clip(row[ok], low, high)
        else:
            n = int(ok.sum())
            ranks = pd.Series(row[ok]).rank().to_numpy() - 1.0
            out[i, ok] = 2.0 * ranks / (n - 1) - 1.0 if n > 1 else 0.0
    return out


# ---------------------------------------------------------------------------------------- the book market
class BookMarket:
    """Hedged returns and costs of a date x stock grid; ``simulate`` scores a date x stock Boolean signal.

    ``hedged``: hedged return over each date's holding period, NaN where the stock cannot be traded;
    ``cost``: cost per side (NaN counts as ``crosssection.COST_BELOW``). On each date the book holds, with
    equal weights, the tradable stocks whose signal is on (none if there are none). The result is the sum
    of weight x hedged return, less the mean hedged return of the tradable stocks that date, less the sum
    of |weight change| x cost; the first date trades from ``initial`` (one weight per stock, default zeros).
    A date with an empty book scores exactly 0 before costs (cash, hedged); leaving a book still costs. A stock that stops being
    tradable has weight 0, so its exit is charged. ``active`` counts the dates on which the share of
    tradable stocks held lies in [``low``, ``high``]."""

    def __init__(self, hedged: np.ndarray, cost: np.ndarray, low: float = LOW, high: float = HIGH,
                 initial: np.ndarray | None = None):
        self.returns = np.asarray(hedged, dtype=float)
        self.tradable = np.isfinite(self.returns)
        self.cost = np.where(np.isfinite(cost), cost, C.COST_BELOW)
        self.low, self.high = low, high
        self.initial = np.zeros(self.returns.shape[1]) if initial is None else np.asarray(initial, dtype=float)
        self.count = self.tradable.sum(axis=1)
        total = np.where(self.tradable, self.returns, 0.0).sum(axis=1)
        self.base = np.divide(total, self.count, out=np.zeros(len(total)), where=self.count > 0)

    def held(self, signal: np.ndarray) -> np.ndarray:
        return np.asarray(signal, dtype=bool) & self.tradable

    def weights(self, signal: np.ndarray) -> np.ndarray:
        held = self.held(signal)
        n = held.sum(axis=1, keepdims=True)
        return np.where(n > 0, held / np.maximum(n, 1), 0.0)

    def share(self, signal: np.ndarray) -> np.ndarray:
        """Per date, the share of the tradable stocks held (0 when none can be traded)."""
        n = self.held(signal).sum(axis=1)
        return np.divide(n, self.count, out=np.zeros(len(n)), where=self.count > 0)

    def gross(self, signal: np.ndarray) -> np.ndarray:
        """Per date, the result before costs."""
        w = self.weights(signal)
        book = (w * np.where(self.tradable, self.returns, 0.0)).sum(axis=1) - self.base
        return np.where(self.held(signal).any(axis=1), book, 0.0)

    def _change(self, signal: np.ndarray) -> np.ndarray:
        return np.abs(np.diff(np.vstack([self.initial, self.weights(signal)]), axis=0))

    def turnover(self, signal: np.ndarray) -> np.ndarray:
        return self._change(signal).sum(axis=1)

    def costs(self, signal: np.ndarray) -> np.ndarray:
        return (self._change(signal) * self.cost).sum(axis=1)

    def daily(self, signal: np.ndarray) -> tuple[np.ndarray, int]:
        net, active, _ = self.simulate(signal)
        return net, active

    def simulate(self, signal: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """Per-date results net of costs, the number of active dates, and the results again (the evidence
        of the PSR lower bound)."""
        net = self.gross(signal) - self.costs(signal)
        n = self.held(signal).sum(axis=1)
        eps = 1e-9
        inside = (self.count > 0) & (n >= self.low * self.count - eps) & (n <= self.high * self.count + eps)
        return net, int(inside.sum()), net


def folds(dates: pd.DatetimeIndex, years=TEST_YEARS) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Test year -> (training positions, test positions). The last date before the test year is left out
    of training: its holding period runs into the test year."""
    out = {}
    for year in years:
        test = np.flatnonzero(np.asarray(dates.year) == year)
        if len(test) == 0 or test[0] < 2:
            continue
        out[year] = (np.arange(0, test[0] - 1), test)
    return out


# --------------------------------------------------------------------------------------------- the inputs
def _percentile_state(x: pd.Series, min_history: int) -> pd.Series:
    """Each value's percentile among the values up to and including its own day (expanding), mapped to
    [-1, 1]; NaN until ``min_history`` values exist."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    seen = 0
    for t in range(len(v)):
        if np.isfinite(v[t]):
            seen += 1
            if seen >= min_history:
                past = v[:t + 1]
                past = past[np.isfinite(past)]
                out[t] = 2.0 * float((past <= v[t]).mean()) - 1.0
    return pd.Series(out, index=x.index)


def market_state(daily_market_return: pd.Series, vol_window: int = 60, return_window: int = 250,
                 min_history: int = 250) -> pd.DataFrame:
    """The market's state on each session, from its daily equal-weight return, using that day and earlier
    only: ``market_volatility`` (standard deviation over ``vol_window`` sessions), ``market_return`` (the
    return over ``return_window`` sessions) and ``market_drawdown`` (the level over its ``return_window``
    high, less 1), each as its percentile among its own values to date, in [-1, 1]."""
    r = daily_market_return.astype(float)
    level = (1.0 + r.fillna(0.0)).cumprod()
    raw = pd.DataFrame({
        "market_volatility": r.rolling(vol_window, min_periods=vol_window).std(),
        "market_return": level / level.shift(return_window) - 1.0,
        "market_drawdown": level / level.rolling(return_window, min_periods=return_window).max() - 1.0,
    })
    return raw.apply(lambda col: _percentile_state(col, min_history))


def universe_dir(universe: str) -> Path:
    if universe not in ("small", "broad"):
        raise ValueError("universe must be 'small' or 'broad'")
    return small.SMALL_DIR if universe == "small" else broad.BROAD_DIR


def daily_market_return(ids, directory: Path) -> pd.Series:
    """Mean over the stocks of the daily adjusted-close return (clipped as in ``crosssection``), each file cut
    at the end of the development period."""
    closes = pd.concat({s: small.read_until(directory / "interim" / "prices" / f"{s}.parquet",
                                            columns=["adj_close"])["adj_close"] for s in ids}, axis=1)
    return closes.pct_change(fill_method=None).clip(*C.RETURN_CLIP).mean(axis=1)


def build_inputs(panel: C.Panel, daily_market_return_: pd.Series | None = None,
                 universe: str = "small") -> dict[str, np.ndarray]:
    """The GP's inputs, date x stock, in [-1, 1], NaN where unknown. Ranks are among the members."""
    member = panel.member
    out: dict[str, np.ndarray] = {}

    def rank(frame: pd.DataFrame) -> np.ndarray:
        return F.cross_rank(frame.where(member)).to_numpy(dtype=float)

    for f in FA_FACTORS:
        out[f"fa:{f}"] = rank(panel.factors[f])
    for f in TA_FACTORS:
        out[f"ta:{f}"] = rank(panel.factors[f])
    out["rg:value_traded"] = rank(panel.value)
    out["rg:size"] = rank(panel.factors["size"])
    if daily_market_return_ is None:
        daily_market_return_ = daily_market_return(list(panel.forward.columns), universe_dir(universe))
    state = market_state(daily_market_return_, **STATE_WINDOWS).reindex(panel.dates)
    for name in ("market_volatility", "market_return", "market_drawdown"):
        out[f"rg:{name}"] = np.repeat(state[name].to_numpy(dtype=float)[:, None], member.shape[1], axis=1)
    return out


# ---------------------------------------------------------------------------------------------- the data
@dataclass(frozen=True)
class Context:
    """Everything a run needs, built once per process."""

    panel: C.Panel
    inputs: dict[str, np.ndarray]
    hedged: np.ndarray
    cost: np.ndarray

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.panel.dates

    @classmethod
    def from_panel(cls, panel: C.Panel, inputs: dict[str, np.ndarray]) -> Context:
        return cls(panel, inputs, panel.hedged().to_numpy(dtype=float),
                   C.cost_per_side(panel.value).to_numpy(dtype=float))


_CONTEXTS: dict[str, Context] = {}


def context(universe: str) -> Context:
    """The panel and inputs of a universe, loaded once per process (about 25 seconds)."""
    if universe not in _CONTEXTS:
        panel = C.small_panel() if universe == "small" else C.broad_panel()
        _CONTEXTS[universe] = Context.from_panel(panel, build_inputs(panel, universe=universe))
    return _CONTEXTS[universe]


# ---------------------------------------------------------------------------------------------- runs
def path(universe: str, rung: str, year: int, run_number: int) -> Path:
    return universe_dir(universe) / "results" / CAMPAIGN / rung / str(year) / f"run{run_number:02d}.json"


def _names(rung: str) -> dict[str, list[str]]:
    groups = {"FA": [f"fa:{f}" for f in FA_FACTORS], "TA": [f"ta:{f}" for f in TA_FACTORS]}
    if RUNGS[rung]["regime"]:
        groups["RG"] = list(RUNGS[rung]["regime"])
    return groups


def inputs_used(individual) -> list[str]:
    """The inputs that can affect the signal: those of the branches the fusion tree names, and of its IF
    conditions."""
    named = {n[2] for n in individual.fusion if n[0] == "LEAF"}
    used = {n[2] for n in individual.fusion if n[0] == "VAR"}
    for source in named:
        used |= {n[2] for n in individual.branches[source] if n[0] == "VAR"}
    return sorted(used)


def _carried(universe: str, rung: str, year: int, run_number: int, ctx: Context) -> np.ndarray:
    """The final weights of the previous year's record of this run, as one weight per stock."""
    before = path(universe, rung, year - 1, run_number)
    if not before.exists():
        raise FileNotFoundError(f"E9 {universe} {rung} run {run_number}, {year}: the book starts from the final "
                                f"holdings of {year - 1}, but {before} does not exist (run the years in order)")
    final = json.loads(before.read_text())["final_weights"]
    columns = list(ctx.panel.forward.columns)
    weights = np.zeros(len(columns))
    for stock, w in final.items():
        weights[columns.index(stock)] = w
    return weights


def run(universe: str, rung: str, year: int, run_number: int, write: bool = True, *,
        ctx: Context | None = None, budget: dict[str, int] | None = None,
        initial: np.ndarray | None | str = "carry") -> dict[str, Any]:
    """Evolve on the fold's training dates, apply the best individual to the test dates. The book starts
    from the final weights of the same universe, rung and run number in the year before (``"carry"``, an
    error if that record is missing); the first test year starts from zero, and ``initial=None`` or an array
    of weights (one per stock) sets the start by hand. ``ctx`` and ``budget`` are for tests."""
    started = time.perf_counter()
    config = load_config()
    base = load_rungs()[BASE_RUNG]
    ctx = context(universe) if ctx is None else ctx
    budget = BUDGET if budget is None else budget
    train, test = folds(ctx.dates)[year]
    groups = _names(rung)
    names = [n for g in groups.values() for n in g]
    train_data = {n: ctx.inputs[n][train] for n in names}
    test_data = {n: ctx.inputs[n][test] for n in names}
    market = BookMarket(ctx.hedged[train], ctx.cost[train], LOW, HIGH)
    settings = replace(base.fitness, min_trades=math.ceil(COVERAGE * len(train)))
    evaluator = LowerBoundEvaluator(train_data, market, settings, cases=False)
    grammar = Grammar(groups, fusion_functions=RUNGS[rung]["fusion"], sources=SOURCES, max_branch_depth=3)
    gp_config = replace(base.gp, **budget, active_trading=False)
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, universe, year, run_number)
    best = evolve(grammar, evaluator, gp_config, rng).best
    signal = signals(best, test_data)["C"]
    if isinstance(initial, str):
        initial = None if year == TEST_YEARS[0] else _carried(universe, rung, year, run_number, ctx)
    later = BookMarket(ctx.hedged[test], ctx.cost[test], LOW, HIGH, initial)
    net, active, _ = later.simulate(signal)
    record = {
        "campaign": CAMPAIGN, "universe": universe, "rung": rung, "year": year, "run": run_number, "budget": budget,
        "model": describe(best), "tree": to_record(best), "fitness": best.fitness,
        "train_dates": int(len(train)), "train_active": int(best.stats["C"]["trades"]),
        "test_dates": [d.strftime("%Y-%m-%d") for d in ctx.dates[test]],
        "test_net": [round(float(x), 8) for x in net], "test_gross": [round(float(x), 8) for x in later.gross(signal)],
        "test_share": [round(float(x), 6) for x in later.share(signal)],
        "test_turnover": [round(float(x), 6) for x in later.turnover(signal)], "test_active": active,
        "final_weights": {str(c): round(float(w), 10) for c, w in
                          zip(ctx.panel.forward.columns, later.weights(signal)[-1]) if w != 0.0},
        "inputs_used": inputs_used(best), "seconds": round(time.perf_counter() - started, 2),
    }
    record["provenance"] = provenance.collect(
        config, rung_digest=base.digest(overrides=budget), rung_spec=base.spec, experiment=CAMPAIGN,
        universe=universe, e9_rung=rung, year=year, run=run_number, sources=list(SOURCES))
    if write:
        provenance.write(path(universe, rung, year, run_number), record)
    return record


def run_job(job: tuple) -> str:
    record = run(*job)
    return f"{job}: fitness {record['fitness']:.3f}, {record['seconds']:.0f} s"


def jobs(universe: str, rungs=tuple(RUNGS), years=TEST_YEARS, runs: int = RUNS) -> list[tuple]:
    """Unfinished runs as (universe, rung, year, run) tuples, in year order. A year's runs need the previous
    year's records (their final holdings), so run one year's jobs to the end before the next year's."""
    years = sorted(set(years))
    return [(universe, r, y, k) for y in years for r in rungs for k in range(runs)
            if not path(universe, r, y, k).exists()]


# ---------------------------------------------------------------------------------------------- benchmarks
def _evaluate_book(score: pd.DataFrame, resid: pd.DataFrame, costs: pd.DataFrame, banded: bool) -> pd.DataFrame:
    """Long book of the top fifth of ``score`` (banded 20/40, or rebuilt in full), from zero holdings at the
    first row (so the rows should be consecutive test dates): per date ``gross`` and ``net`` (as excess over the tradable mean), ``turnover`` and ``share``."""
    tradable = resid.notna()
    w = (C.banded_weights(score, tradable, long_only=True) if banded
         else C.quantile_weights(score, tradable, long_only=True))
    book = C.portfolio_returns(w, resid, costs)
    base = resid.where(tradable).mean(axis=1).fillna(0.0)
    count = tradable.sum(axis=1)
    base = base.where(w.sum(axis=1) > 0, 0.0)     # an empty book is cash: no excess to subtract
    out = pd.DataFrame({"gross": book["gross"] - base, "net": book["net"] - base, "turnover": book["turnover"],
                        "share": (w > 0).sum(axis=1) / count.where(count > 0)}).fillna({"share": 0.0})
    return out


def _signed_ranks(panel: C.Panel) -> dict[str, pd.DataFrame]:
    return {f: F.SIGNS[f] * F.cross_rank(panel.factors[f].where(panel.member)) for f in F.SIGNS}


def _scores(panel: C.Panel, resid: pd.DataFrame, fold_map: dict) -> dict[str, pd.DataFrame]:
    """The benchmark scores, each a date x stock frame filled on the test dates of every fold."""
    import lightgbm as lgb

    names = list(F.SIGNS)
    signed = _signed_ranks(panel)
    X = np.stack([signed[f].to_numpy(dtype=float) for f in names], axis=2)      # date x stock x factor
    target = F.cross_rank(resid).to_numpy(dtype=float)
    ic = pd.DataFrame({f: C.information_coefficient(signed[f], resid) for f in names}).reindex(panel.dates)
    blank = np.full(target.shape, np.nan)
    scores = {k: blank.copy() for k in ("composite", "past IC weights", "best single factor", "ridge", "LightGBM")}

    def weighted(w, rows):
        with np.errstate(all="ignore"):
            num = np.nansum(X[rows] * w, axis=2)
            den = np.nansum(np.isfinite(X[rows]) * np.abs(w), axis=2)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    for year, (train, test) in fold_map.items():
        past = ic.iloc[train].mean().fillna(0.0).to_numpy()
        scores["past IC weights"][test] = weighted(past, test)
        best = int(np.argmax(np.abs(past)))
        scores["best single factor"][test] = np.sign(past[best]) * X[test][:, :, best]
        ok = np.isfinite(target[train])
        xtr = np.nan_to_num(X[train])[ok]
        ytr = target[train][ok]
        w = np.linalg.solve(xtr.T @ xtr + 0.05 * len(ytr) * np.eye(len(names)), xtr.T @ ytr)
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=8,
                                  min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                                  random_state=0, verbose=-1, n_jobs=2).fit(xtr, ytr)
        for k in test:
            known = np.isfinite(X[k]).any(axis=1)
            scores["ridge"][k] = np.where(known, np.nan_to_num(X[k]) @ w, np.nan)
            pred = np.full(X.shape[1], np.nan)
            if known.any():
                pred[known] = model.predict(np.nan_to_num(X[k][known]))
            scores["LightGBM"][k] = pred
    scores["composite"] = C.composite(panel).to_numpy(dtype=float)   # needs no fitting: the library's version
    return {k: pd.DataFrame(v, index=panel.dates, columns=panel.forward.columns) for k, v in scores.items()}


LABELS = {"composite": "composite (literature signs)", "past IC weights": "past-IC weights",
          "best single factor": "best single factor so far", "ridge": "ridge", "LightGBM": "LightGBM"}


def benchmarks(universe: str, ctx: Context | None = None) -> dict[str, pd.DataFrame]:
    """Per-date test results (``gross``, ``net``, ``turnover``, ``share``) of the long book of the top fifth
    of each score, rebuilt in full and banded 20/40, over the folds of 2014 to 2018 with the GP's hedged
    returns, costs and excess-over-tradable-mean convention. The test years follow one another, so each score
    is built continuously from the first 2014 test date (banded state and holdings carry across years); only
    that date starts from zero holdings."""
    ctx = context(universe) if ctx is None else ctx
    panel = ctx.panel
    resid = pd.DataFrame(ctx.hedged, index=panel.dates, columns=panel.forward.columns)
    costs = pd.DataFrame(ctx.cost, index=panel.dates, columns=panel.forward.columns)
    fold_map = folds(panel.dates)
    scores = _scores(panel, resid, fold_map)
    rows = panel.dates[np.concatenate([test for _, test in fold_map.values()])]
    out = {}
    for key, score in scores.items():
        for banded in (False, True):
            out[f"{LABELS[key]}, {'banded 20/40' if banded else 'rebuilt'}"] = _evaluate_book(
                score.loc[rows], resid.loc[rows], costs.loc[rows], banded)
    return out


def random_books(universe: str, share: float, draws: int = 1000, rng: np.random.Generator | None = None,
                 ctx: Context | None = None) -> dict[str, pd.DataFrame]:
    """Per-date results (dates x draws, ``gross`` and ``net``) of books of random tradable stocks holding
    ``share`` of the tradable stocks, rebuilt at random on every date (holdings start from zero on the first
    test date only), over the test dates of 2014 to 2018."""
    ctx = context(universe) if ctx is None else ctx
    if rng is None:
        rng = make_rng(load_config().base_seed, "dev", CAMPAIGN, "random-books", universe)
    fold_map = folds(ctx.dates)
    rows = np.concatenate([test for _, test in fold_map.values()])
    firsts = {int(rows[0])}
    gross, net = np.zeros((len(rows), draws)), np.zeros((len(rows), draws))
    previous = np.zeros((draws, ctx.hedged.shape[1]))
    for i, k in enumerate(rows):
        returns = ctx.hedged[k]
        tradable = np.isfinite(returns)
        n = int(tradable.sum())
        if k in firsts:
            previous = np.zeros_like(previous)
        if n == 0:
            weights = np.zeros_like(previous)
        else:
            size = max(1, int(round(share * n)))
            draw = np.where(tradable, rng.random((draws, len(returns))), np.inf)
            rank = draw.argsort(axis=1).argsort(axis=1)
            weights = (rank < size) / size
        base = np.nanmean(returns) if n else 0.0
        gross[i] = weights @ np.where(tradable, returns, 0.0) - base
        cost = np.where(np.isfinite(ctx.cost[k]), ctx.cost[k], C.COST_BELOW)
        net[i] = gross[i] - (np.abs(weights - previous) * cost).sum(axis=1)
        previous = weights
    index = ctx.dates[rows]
    return {"gross": pd.DataFrame(gross, index=index), "net": pd.DataFrame(net, index=index)}


# ------------------------------------------------------------------------------------------------ report
def bootstrap_mean_p(x, rng: np.random.Generator, mean_block: float = 3, resamples: int = 10_000) -> float:
    """One-sided p that the mean of ``x`` is above zero: the share of stationary-bootstrap resamples of the
    periods whose mean is zero or less. A zero-mean series gives about 0.5."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    means = np.array([x[stats.stationary_bootstrap(len(x), mean_block, rng)].mean() for _ in range(resamples)])
    return float(np.mean(means <= 0.0))


def _two_sided(p_one: float) -> float:
    return float(min(1.0, 2.0 * min(p_one, 1.0 - p_one)))


def gp_frames(universe: str, rungs=tuple(RUNGS)) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    """Per rung, the per-date results averaged over its runs (the runs' books averaged), and the records."""
    frames, records = {}, {}
    for rung in rungs:
        files = sorted((universe_dir(universe) / "results" / CAMPAIGN / rung).glob("*/run*.json"))
        files = [p for p in files if not p.name.endswith(".provenance.json")]
        recs = [json.loads(p.read_text()) for p in files]
        if not recs:
            continue
        long = pd.concat([pd.DataFrame({"gross": r["test_gross"], "net": r["test_net"], "turnover": r["test_turnover"],
                                        "share": r["test_share"]}, index=pd.DatetimeIndex(r["test_dates"]))
                          for r in recs])
        frames[rung] = long.groupby(level=0).mean().sort_index()
        records[rung] = recs
    return frames, records


def _row(label: str, frame: pd.DataFrame) -> dict:
    main = frame[frame.index.year.isin(MAIN_YEARS)]
    per_year = 252 / C.H
    by_year = main["net"].groupby(main.index.year).sum()
    row = {"line": label, "gross_return": main["gross"].mean() * per_year, "net_return": main["net"].mean() * per_year,
           "sharpe_net": C.summarise(main["net"])["sharpe"], "turnover": main["turnover"].mean(),
           "share_held": main["share"].mean() if "share" in main else np.nan}
    for y in MAIN_YEARS:
        row[f"net_{y}"] = float(by_year.get(y, np.nan))
    extra = frame[frame.index.year == 2014]["net"]
    row["net_2014_extra"] = float(extra.sum()) if len(extra) else np.nan
    return row


def _load_benchmarks(universe: str) -> dict[str, pd.DataFrame]:
    path_ = universe_dir(universe) / "results" / "e9_benchmarks.csv"
    if path_.exists():
        table = pd.read_csv(path_, parse_dates=["date"])
        return {k: g.drop(columns="benchmark").set_index("date") for k, g in table.groupby("benchmark", sort=False)}
    return benchmarks(universe)


def save_benchmarks(universe: str) -> dict[str, pd.DataFrame]:
    out = benchmarks(universe)
    table = pd.concat({k: v.rename_axis("date") for k, v in out.items()}, names=["benchmark"]).reset_index()
    path_ = universe_dir(universe) / "results" / "e9_benchmarks.csv"
    path_.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path_, index=False)
    return out


def report(universe: str) -> pd.DataFrame:
    """Print and write ``<dir>/results/e9_report.csv``: every GP rung and benchmark over 2015-2018, then the
    plan's four rules, the most frequent inputs and three example rules."""
    config = load_config()
    frames, records = gp_frames(universe)
    bench = _load_benchmarks(universe)
    rows = [_row(f"GP {r} (mean of {len(records[r])} runs)", f) for r, f in frames.items()]
    rows += [_row(k, v) for k, v in bench.items()]
    if "C1" in frames:
        share = float(frames["C1"]["share"].mean())
        rb = random_books(universe, share)
        mask = rb["net"].index.year.isin(MAIN_YEARS)
        random_frame = pd.DataFrame({"gross": rb["gross"].mean(axis=1), "net": rb["net"].mean(axis=1),
                                     "turnover": np.nan, "share": share})
        rows.append(_row(f"random books, share {share:.2f} (mean of 1000)", random_frame))
        per_year = 252 / C.H
        net_main = rb["net"][mask].mean(axis=0) * per_year
        print(f"Random books of the GP's size: net return a year, 5th to 95th percentile of 1000 draws: "
              f"{100 * net_main.quantile(0.05):.1f}% to {100 * net_main.quantile(0.95):.1f}%")
    table = pd.DataFrame(rows)
    shown = table.assign(**{c: 100 * table[c] for c in ("gross_return", "net_return")})
    shown = shown.rename(columns={"gross_return": "gross %/yr", "net_return": "net %/yr"})
    for y in MAIN_YEARS:
        shown[f"net_{y}"] = 100 * shown[f"net_{y}"]
    shown["net_2014_extra"] = 100 * shown["net_2014_extra"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\nE9, {universe} caps, long books hedged, tiered costs, test years 2015-2018 "
          f"(by-year net sums in %; 2014 is an extra year)")
    print(shown.round(2).to_string(index=False))

    def main_part(frame):
        return frame[frame.index.year.isin(MAIN_YEARS)]

    def rng_for(name):
        return make_rng(config.base_seed, "dev", CAMPAIGN, "report", universe, name)

    print("\nThe plan's rules (block bootstrap of the periods, mean block 3, 10,000 resamples):")
    if "C1" not in frames:
        print("  no runs of C1 found, so rules 1 to 3 cannot be applied")
    else:
        c1 = main_part(frames["C1"])
        p1 = bootstrap_mean_p(c1["gross"].to_numpy(), rng_for("rule1"))
        print(f"  Rule 1 (skill): C1's result before costs is {100 * c1['gross'].mean() * 252 / C.H:.2f}% a year; "
              f"one-sided p = {p1:.4f}: {'holds' if p1 < 0.05 else 'does not hold'}.")
        by_year = c1["net"].groupby(c1.index.year).sum()
        total, positive = float(c1["net"].sum()), int((by_year > 0).sum())
        holds = total > 0 and positive >= 3
        print(f"  Rule 2 (profit): C1's net sum over 2015-2018 is {100 * total:.1f}%, positive in {positive} of "
              f"{len(by_year)} years ({', '.join(f'{y}: {100 * v:.1f}%' for y, v in by_year.items())}): "
              f"{'holds' if holds else 'does not hold'}.")
        if BANDED in bench:
            b = main_part(bench[BANDED])
            diff = (c1["net"] - b["net"].reindex(c1.index)).dropna()
            p_one = bootstrap_mean_p(diff.to_numpy(), rng_for("rule3"))
            p_two = _two_sided(p_one)
            verdict = "beats" if diff.mean() > 0 and p_two < 0.05 else "fails" if diff.mean() < 0 and p_two < 0.05 else "matches"
            print(f"  Rule 3 (against the banded composite): C1 less the composite, net, is "
                  f"{100 * diff.mean() * 252 / C.H:+.2f}% a year, two-sided p = {p_two:.4f}: C1 {verdict} the benchmark.")
    pairs = [("C2", "C1"), ("C3", "C2")]
    if all(a in frames and b in frames for a, b in pairs):
        diffs = [(main_part(frames[a])["net"] - main_part(frames[b])["net"]).dropna() for a, b in pairs]
        raw = [bootstrap_mean_p(d.to_numpy(), rng_for(f"rule4-{a}-{b}")) for d, (a, b) in zip(diffs, pairs)]
        adjusted = stats.holm(raw)
        for d, (a, b), p, adj in zip(diffs, pairs, raw, adjusted):
            worse = int((d.groupby(d.index.year).sum() < 0).sum())
            improves = d.mean() > 0 and adj < 0.05 and worse <= 1
            print(f"  Rule 4 ({a} against {b}): net difference {100 * d.mean() * 252 / C.H:+.2f}% a year, one-sided "
                  f"p = {p:.4f} (Holm {adj:.4f}); {a} is worse in {worse} of 4 years: "
                  f"{'an improvement' if improves else 'not an improvement'}.")
    else:
        print("  Rule 4 needs runs of C1, C2 and C3.")

    for rung, recs in records.items():
        counts = pd.Series([n for r in recs for n in r["inputs_used"]]).value_counts()
        top = ", ".join(f"{n} ({c})" for n, c in counts.head(8).items())
        print(f"\n{rung}: inputs most often used in the {len(recs)} final models: {top}")
    if "C1" in records:
        best = sorted(records["C1"], key=lambda r: r["fitness"], reverse=True)[:3]
        print("\nThree C1 rules with the best training fitness:")
        for r in best:
            print(f"  test {r['year']}, run {r['run']}, fitness {r['fitness']:.3f}\n    " + r["model"].replace("\n", "\n    "))
    path_ = universe_dir(universe) / "results" / "e9_report.csv"
    path_.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path_, index=False)
    return table


__all__ = ["BookMarket", "folds", "market_state", "build_inputs", "run", "run_job", "jobs", "benchmarks",
           "random_books", "report", "bootstrap_mean_p"]
