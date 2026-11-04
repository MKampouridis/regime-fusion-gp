"""Replication of ActTrade (Christodoulaki and Kampouridis, Applied Soft Computing) on our data.

The paper's protocol and its code (github.com/avechris/ActTrade, read 4 November 2026):

* 1 January 2015 to 31 January 2020, split by sessions 60% / 20% / 20% (training, validation,
  test); the exit settings (d, r) chosen per company on the validation set, then the GP retrained
  on training + validation and evaluated on the test set (February 2019 to January 2020).
* A buy signal at day i buys at day i's close. Every signal day opens its own trade, so trades
  overlap. A trade is sold at the first close within days i .. i+d-1 that is more than r above the
  buy price, otherwise at the close of day i+d-1 (or the segment's last close). Its return is
  (sell - buy) / buy - 0.025%.
* Sharpe ratio per trade: (mean trade return - 0.022%) / standard deviation of the trade returns,
  not annualised; the plain mean return when there are fewer than two trades. Rate of return: the
  mean trade return. Fitness: the sum of the Sharpe ratios of the whole tree and of its three
  branches; a component with fewer than 3 trades has a mean return of zero.
* ActTrade's GP: the fixed fusion ALL(FA, SA, TA), the active trading operator, population 1,000,
  50 generations, tournament 4, maximum depth 6 (our rung A0's grammar and operators).

Inputs are ours (SEC fundamentals, GDELT sentiment, Yahoo prices), so this replicates the method,
not the paper's data. For each run, the test set is scored three ways: the paper's metric; the
paper's metric for a signal that is on every day (the control: what the metric gives for simply
being in the market); and a realistic one (one position at a time, the next open, 10 bps per side,
annualised Sharpe of daily returns), against buy-and-hold.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regime_fusion import provenance
from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.engine.backtest import backtest
from regime_fusion.engine import metrics
from regime_fusion.engine.fitness import Evaluator, Individual, boolean, fused, _branch_key
from regime_fusion.engine.gp import evolve
from regime_fusion.engine.panel import load_panel, stock_prices
from regime_fusion.engine.rungs import load_rungs
from regime_fusion.engine.tree import Grammar, describe, signals, to_record
from regime_fusion.seeds import make_rng
from regime_fusion.windows import DateRange

CAMPAIGN = "dev-acttrade"
START, END = "2015-01-01", "2020-01-31"
COST = 0.00025          # of the selling price, once per trade
RISK_FREE = 0.00022     # per trade
MIN_TRADES = 3          # round trips (the code's MIN_NO_OF_TRADES = 6 counts buys and sells)
GRID_DAYS = (5, 10, 20, 30)
GRID_RATES = (0.01, 0.02, 0.03, 0.05)
TUNING_RUNS = 5
RUNS = 50
GP = {"population": 1000, "generations": 50}
ACTTRADE_FILE = Path(__file__).resolve().parents[3] / "configs" / "acttrade.yaml"


def stocks() -> list[str]:
    raw = yaml.safe_load(ACTTRADE_FILE.read_text())["stocks"]
    usable = []
    for entry in raw:
        sid = entry["id"]
        if store.layer("interim", "prices", f"{sid}.parquet").exists():
            usable.append(sid)
    return usable


def segments(stock: str) -> dict[str, DateRange]:
    """The paper's 60/20/20 split of the sessions from START to END."""
    days = stock_prices(stock).loc[START:END].index
    a, b = int(round(0.6 * len(days))), int(round(0.8 * len(days)))
    as_range = lambda d: DateRange(d[0].date(), d[-1].date())
    return {"train": as_range(days[:a]), "validation": as_range(days[a:b]), "test": as_range(days[b:]),
            "train+validation": as_range(days[:b])}


def trade_returns(close: np.ndarray, days: int, rate: float) -> np.ndarray:
    """For every day i, the return of a trade bought at close i (the paper's rule, inside one segment)."""
    n = len(close)
    out = np.empty(n)
    for i in range(n):
        window = close[i:i + days]
        hit = np.nonzero((window - close[i]) / close[i] > rate)[0]
        sell = window[hit[0]] if len(hit) else window[-1]
        out[i] = (sell - close[i]) / close[i] - COST
    return out


def paper_sharpe(r: np.ndarray) -> float:
    """The paper's per-trade Sharpe ratio (its code: ``sharpe_ratiototal``)."""
    if len(r) == 0:
        return 0.0
    mean = float(np.mean(r))
    if len(r) < 2:
        return mean
    spread = float(np.std(r, ddof=1))
    return (mean - RISK_FREE) / spread if spread != 0 else mean


class TradeMarket:
    """Scores a signal by the paper's rule: every signal day is a trade."""

    def __init__(self, close: np.ndarray, days: int, rate: float):
        self.returns = trade_returns(close, days, rate)

    def trades(self, signal: np.ndarray) -> np.ndarray:
        return self.returns[np.asarray(signal, dtype=bool)]


class PaperEvaluator(Evaluator):
    """The paper's fitness: the sum of the per-trade Sharpe ratios of the tree and its branches."""

    def component(self, key: tuple, signal_of) -> tuple:
        found = self._cache.get(key)
        if found is None:
            signal = signal_of()
            r = self.market.trades(signal)
            found = (signal, len(r), paper_sharpe(r) if len(r) >= MIN_TRADES else 0.0, None)
            self._cache[key] = found
        return found

    def evaluate(self, individual: Individual) -> None:
        stats, branch_signals = {}, {}
        for s, tree in individual.branches.items():
            stats[s] = self.component(_branch_key(s, tree), lambda t=tree: boolean(t, self.data)[0])
            branch_signals[s] = stats[s][0]
        stats["C"] = self.component(("C", *individual.key()),
                                    lambda: fused(individual.fusion, branch_signals, self.data)[0])
        components = ("C", *individual.branches)
        individual.fitness = float(sum(stats[c][2] for c in components))
        individual.stats = {c: {"sharpe": float(stats[c][2]), "trades": int(stats[c][1])} for c in components}


def path(stock: str, phase: str, days: int, rate: float, run: int) -> Path:
    return store.data_dir() / "results" / CAMPAIGN / phase / stock / f"d{days}_r{rate:g}.run{run:02d}.json"


def run(stock: str, phase: str, days: int, rate: float, run_number: int, write: bool = True) -> dict[str, Any]:
    """phase "tuning": train on training, score on validation; phase "final": train on training +
    validation, score on test."""
    config = load_config()
    started = time.perf_counter()
    rung = load_rungs()["A0"]
    seg = segments(stock)
    train, evaluate = (seg["train"], seg["validation"]) if phase == "tuning" else (seg["train+validation"], seg["test"])
    panel = load_panel(stock, train, evaluate, rung.sources, None, rung.scaling)
    grammar = Grammar(panel.names(), fusion_functions=rung.fusion, sources=rung.sources,
                      max_branch_depth=rung.max_branch_depth)
    close_train = panel.prices.loc[panel.train, "adj_close"].to_numpy()
    evaluator = PaperEvaluator(panel.arrays(panel.train), TradeMarket(close_train, days, rate),
                               replace(rung.fitness, min_trades=MIN_TRADES), cases=False)
    rng = make_rng(config.base_seed, "dev", CAMPAIGN, stock, phase, run_number)
    best = evolve(grammar, evaluator, replace(rung.gp, **GP), rng).best
    later = signals(best, panel.arrays(panel.evaluate))["C"]
    prices = panel.prices.loc[panel.evaluate]
    market = TradeMarket(prices["adj_close"].to_numpy(), days, rate)
    trades = market.trades(later)
    always = market.trades(np.ones(len(prices), dtype=bool))
    realistic = backtest(later, prices, days, rate, config.primary_cost_bps_per_side)
    hold = prices["adj_close"].to_numpy()
    record = {
        "campaign": CAMPAIGN, "stock": stock, "phase": phase, "days": days, "rate": rate, "run": run_number,
        "train": [str(train.start), str(train.end)], "evaluate": [str(evaluate.start), str(evaluate.end)],
        "model": describe(best), "tree": to_record(best), "fitness": best.fitness, "train_stats": best.stats,
        "paper": {"sharpe": paper_sharpe(trades), "rate_of_return": float(np.mean(trades)) if len(trades) else 0.0,
                  "risk": float(np.std(trades, ddof=1)) if len(trades) > 1 else 0.0, "trades": int(len(trades))},
        "always_on": {"sharpe": paper_sharpe(always), "rate_of_return": float(np.mean(always)), "trades": int(len(always))},
        "realistic": {"sharpe": metrics.sharpe(realistic.daily), "return": float((1 + realistic.daily).prod() - 1),
                      "trades": int(len(realistic.trades))},
        "buy_and_hold": {"return": float(hold[-1] / hold[0] - 1),
                         "sharpe": metrics.sharpe(pd.Series(hold).pct_change().dropna())},
        "seconds": round(time.perf_counter() - started, 2),
    }
    if write:
        record["provenance"] = provenance.collect(config, rung_digest=rung.digest(overrides=GP), rung_spec=rung.spec,
                                                  experiment=CAMPAIGN, stock=stock, phase=phase, days=days, rate=rate,
                                                  run=run_number)
        provenance.write(path(stock, phase, days, rate, run_number), record)
    return record


def run_job(job: tuple) -> str:
    r = run(*job)
    return f"{job}: paper Sharpe {r['paper']['sharpe']:.2f}, {r['seconds']:.0f} s"


def tuning_jobs() -> list[tuple]:
    return [(s, "tuning", d, r, k) for s in stocks() for d in GRID_DAYS for r in GRID_RATES for k in range(TUNING_RUNS)
            if not path(s, "tuning", d, r, k).exists()]


def chosen_exits() -> dict[str, tuple[int, float]]:
    """Per stock, the (d, r) with the best median validation Sharpe (paper's metric) over the tuning runs."""
    out = {}
    for s in stocks():
        scores = {}
        for d in GRID_DAYS:
            for r in GRID_RATES:
                vals = [json.loads(path(s, "tuning", d, r, k).read_text())["paper"]["sharpe"]
                        for k in range(TUNING_RUNS) if path(s, "tuning", d, r, k).exists()]
                if vals:
                    scores[(d, r)] = float(np.median(vals))
        if scores:
            out[s] = max(scores, key=scores.get)
    return out


def final_jobs() -> list[tuple]:
    exits = chosen_exits()
    return [(s, "final", d, r, k) for s, (d, r) in exits.items() for k in range(RUNS)
            if not path(s, "final", d, r, k).exists()]


PAPER_ACTTRADE = {  # Table 6 (Sharpe) and Table 9 (rate of return), ActTrade column
    "AAPL": (7.2507, 0.0242), "AMZN": (0.0, 0.0), "KO": (1.1766, 0.0119), "EBAY": (0.1362, 0.0026),
    "EL": (1.6824, 0.0053), "F": (3.2462, 0.0182), "GM": (0.0, 0.0), "GOOGL": (14.1991, 0.0103),
    "IBM": (1.7280, 0.0250), "INTC": (-0.0318, -0.0026), "KODK": (1.2845, 0.0390), "MCD": (-1.2986, -0.0563),
    "NFLX": (9.3962, 0.0525), "NYT": (0.1232, 0.0087), "TSLA": (2.7183, 0.0601), "WMT": (1.5442, 0.0026),
    "XRX": (0.3188, 0.0174),
}


def report() -> pd.DataFrame:
    rows = []
    for p in sorted((store.data_dir() / "results" / CAMPAIGN / "final").glob("*/*.json")):
        if p.name.endswith(".provenance.json"):
            continue
        r = json.loads(p.read_text())
        rows.append({"stock": r["stock"], "d": r["days"], "r": r["rate"], "sharpe": r["paper"]["sharpe"],
                     "ror": r["paper"]["rate_of_return"], "trades": r["paper"]["trades"],
                     "always_sharpe": r["always_on"]["sharpe"], "always_ror": r["always_on"]["rate_of_return"],
                     "real_sharpe": r["realistic"]["sharpe"], "real_return": r["realistic"]["return"],
                     "bh_return": r["buy_and_hold"]["return"], "bh_sharpe": r["buy_and_hold"]["sharpe"]})
    t = pd.DataFrame(rows)
    g = t.groupby("stock")
    out = pd.DataFrame({
        "d": g["d"].first(), "r": g["r"].first(),
        "paper_sharpe": t.assign(s=lambda x: x.sharpe).groupby("stock")["s"].mean(),
        "paper_sharpe_trading_runs": t[t.trades > 0].groupby("stock")["sharpe"].mean(),
        "paper_ror": g["ror"].mean(), "trades": g["trades"].median(), "never_trade": g["trades"].apply(lambda x: (x == 0).mean()),
        "always_on_sharpe": g["always_sharpe"].first(), "always_on_ror": g["always_ror"].first(),
        "realistic_sharpe": g["real_sharpe"].mean(), "realistic_return": g["real_return"].mean(),
        "bh_return": g["bh_return"].first(), "bh_sharpe": g["bh_sharpe"].first(),
    })
    out["published_sharpe"] = [PAPER_ACTTRADE.get(s, (np.nan, np.nan))[0] for s in out.index]
    out["published_ror"] = [PAPER_ACTTRADE.get(s, (np.nan, np.nan))[1] for s in out.index]
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    trading = t[t.trades > 0]
    print(f"\nall runs: paper Sharpe mean {t.sharpe.mean():.2f}, median {t.sharpe.median():.2f}; trading runs only: "
          f"mean {trading.sharpe.mean():.2f}, median {trading.sharpe.median():.2f}; mean trade return {trading.ror.mean():.4f}")
    print(f"always-on signal, same metric: mean Sharpe {out.always_on_sharpe.mean():.2f}, mean trade return {out.always_on_ror.mean():.4f}")
    print(f"realistic: mean Sharpe {t.real_sharpe.mean():.2f}, mean return {100*t.real_return.mean():.1f}% "
          f"vs buy-and-hold {100*out.bh_return.mean():.1f}% (Sharpe {out.bh_sharpe.mean():.2f})")
    out.to_csv(store.data_dir() / "results" / CAMPAIGN / "replication.csv")
    return out
