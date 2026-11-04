"""Summary of the development experiments: per mode, design and fold, the validation year's
equal-weight portfolio at the primary cost, against buy-and-hold."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from regime_fusion.data import store
from regime_fusion.dev import e23
from regime_fusion.engine.metrics import sharpe


def load_e23() -> pd.DataFrame:
    """One row per (mode, design, fold, run, stock): validation daily returns, trades, training fitness."""
    rows = []
    for path in sorted((store.data_dir() / "results" / e23.CAMPAIGN).glob("*/*/*/*.json")):
        if path.name.endswith(".provenance.json"):
            continue
        record = json.loads(path.read_text())
        for stock, v in record["validation_returns"].items():
            rows.append({"mode": record["mode"], "design": record["design"], "fold": record["fold"],
                         "run": record["run"], "stock": stock, "fitness": record["fitness"],
                         "trades": v["trades"],
                         "daily": pd.Series(v["daily"], index=pd.DatetimeIndex(v["days"]))})
    return pd.DataFrame(rows)


def e23_report() -> pd.DataFrame:
    table = load_e23()
    if table.empty:
        print("no E2 x E3 results yet")
        return table
    out = []
    bh = {fold: e23.buy_and_hold(fold) for fold in sorted(table["fold"].unique())}
    for (mode, design, fold), cell in table.groupby(["mode", "design", "fold"]):
        runs = sorted(cell["run"].unique())
        # the protocol's portfolio: average the runs within each stock, then the stocks day by day
        per_stock = [pd.concat(list(group["daily"]), axis=1).mean(axis=1) for _, group in cell.groupby("stock")]
        portfolio = pd.concat(per_stock, axis=1).mean(axis=1)
        by_run = [pd.concat(list(cell.loc[cell.run == r, "daily"]), axis=1).mean(axis=1) for r in runs]
        run_sharpes = np.array([sharpe(s) for s in by_run])
        out.append({"mode": mode, "design": design, "fold": fold, "runs": len(runs),
                    "stocks": cell["stock"].nunique(),
                    "sharpe": sharpe(portfolio), "buy_and_hold": sharpe(bh[fold]),
                    "return_pct": 100 * ((1 + portfolio).prod() - 1),
                    "bh_return_pct": 100 * ((1 + bh[fold]).prod() - 1),
                    "runs_above_bh": float((run_sharpes > sharpe(bh[fold])).mean()),
                    "median_run_sharpe": float(np.median(run_sharpes)),
                    "never_trade": float((cell["trades"] == 0).mean()),
                    "trades_per_stock": float(cell["trades"].median()),
                    "train_fitness": float(cell.groupby("run")["fitness"].first().median())})
    summary = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    folder = store.data_dir() / "results" / e23.CAMPAIGN
    summary.to_csv(folder / "summary.csv", index=False)
    return summary
