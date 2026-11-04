"""Do the GP's inputs predict returns at all? Information coefficients on development data.

For each indicator (raw, and as its trailing rank, ``dev.features``) and horizon h, the Spearman
correlation between its value at the close of day t and the return from the open of t+1 to the open
of t+1+h (what a trade decided at t could earn), over the 50 main stocks, from the first session of
February 2015 to the end of ``development_end`` (2018):

* time series: per stock, over its days; summarised as the mean over stocks, the share of stocks
  with the mean's sign, and the IC of each half (2015-16, 2017-18) to see whether it holds;
* cross section: per day, over the stocks; summarised as the mean daily IC, its t statistic with
  a Newey-West correction for the h - 1 days by which consecutive forward returns overlap, and the
  mean of each half.

Several hundred correlations are computed, so a few will look strong by chance: the report counts
how many pass |t| > 3 and keeps the same sign in both halves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_fusion.config import load_config
from regime_fusion.data import store
from regime_fusion.dev.e23 import stocks
from regime_fusion.dev.features import DROPPED, trailing_rank
from regime_fusion.engine.panel import PROCESSED, stock_prices

HORIZONS = (5, 20, 60)
START = pd.Timestamp("2015-02-01")
HALF = pd.Timestamp("2016-12-31")


def forward_returns(prices: pd.DataFrame, h: int) -> pd.Series:
    """Return from the open of t+1 to the open of t+1+h, indexed by t."""
    opens = prices["adj_open"]
    return opens.shift(-(h + 1)) / opens.shift(-1) - 1.0


def indicators(stock: str, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for source, folder in PROCESSED.items():
        table = pd.read_parquet(store.layer("processed", folder, f"{stock}.parquet"))
        table = table.loc[table.index <= end, [c for c in table.columns if c not in DROPPED]]
        raw = table.add_prefix(f"{source.lower()}:")
        frames += [raw.add_suffix("|raw"), trailing_rank(table).add_prefix(f"{source.lower()}:").add_suffix("|rank")]
    return pd.concat(frames, axis=1)


def newey_west_t(series: pd.Series, lags: int) -> float:
    x = series.dropna().to_numpy()
    n = x.size
    if n < 10:
        return np.nan
    d = x - x.mean()
    variance = d @ d / n
    for k in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return float(x.mean() / np.sqrt(variance / n)) if variance > 0 else np.nan


def build(end: pd.Timestamp) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Indicators stacked by (day, stock), and forward returns for each horizon, inside [START, end].
    Forward returns are cut at ``end``, so nothing after the development period is used."""
    values, returns = [], {h: [] for h in HORIZONS}
    for stock in stocks():
        prices = stock_prices(stock)
        prices = prices.loc[prices.index <= end]
        table = indicators(stock, end).reindex(prices.index)
        keep = prices.index >= START
        values.append(table.loc[keep].assign(stock=stock).set_index("stock", append=True))
        for h in HORIZONS:
            returns[h].append(forward_returns(prices, h).loc[keep].rename(stock))
    panel = pd.concat(values).sort_index()
    return panel, {h: pd.concat(r, axis=1) for h, r in returns.items()}


def screen() -> pd.DataFrame:
    end = pd.Timestamp(load_config().period.development_end)
    panel, forward = build(end)
    rows = []
    for column in panel.columns:
        wide = panel[column].unstack("stock")
        for h in HORIZONS:
            fwd = forward[h].reindex(wide.index)
            # time series, per stock
            ts = {s: wide[s].corr(fwd[s], method="spearman") for s in wide.columns if wide[s].nunique() > 2}
            ts_first = {s: wide.loc[:HALF, s].corr(fwd.loc[:HALF, s], method="spearman") for s in ts}
            ts_second = {s: wide.loc[HALF:, s].corr(fwd.loc[HALF:, s], method="spearman") for s in ts}
            ts = pd.Series(ts).dropna()
            # cross section, per day
            cs = wide.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1, method="pearson")
            cs = cs[wide.notna().sum(axis=1).where(fwd.notna().sum(axis=1) > 0, 0) >= 20]
            rows.append({
                "indicator": column.split("|")[0], "form": column.split("|")[1], "h": h,
                "ts_ic": ts.mean(), "ts_same_sign": float((np.sign(ts) == np.sign(ts.mean())).mean()) if len(ts) else np.nan,
                "ts_ic_2015_16": pd.Series(ts_first).mean(), "ts_ic_2017_18": pd.Series(ts_second).mean(),
                "cs_ic": cs.mean(), "cs_t": newey_west_t(cs, h - 1),
                "cs_ic_2015_16": cs.loc[:HALF].mean(), "cs_ic_2017_18": cs.loc[HALF:].mean(),
                "cs_days": int(cs.notna().sum()), "ts_stocks": len(ts),
            })
    table = pd.DataFrame(rows)
    folder = store.data_dir() / "results" / "dev-ic"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "ic.csv", index=False)
    return table


def report(table: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    t = table.copy()
    t["cs_stable"] = np.sign(t.cs_ic_2015_16) == np.sign(t.cs_ic_2017_18)
    t["ts_stable"] = np.sign(t.ts_ic_2015_16) == np.sign(t.ts_ic_2017_18)
    print(f"{len(t)} indicator x form x horizon combinations")
    print("cross section: |t| > 3 and the same sign in both halves:",
          int(((t.cs_t.abs() > 3) & t.cs_stable).sum()), "| |t| > 2:", int((t.cs_t.abs() > 2).sum()))
    print("\nstrongest cross-sectional ICs (by |t|):")
    print(t.reindex(t.cs_t.abs().sort_values(ascending=False).index).head(20)[
        ["indicator", "form", "h", "cs_ic", "cs_t", "cs_ic_2015_16", "cs_ic_2017_18"]].round(3).to_string(index=False))
    print("\nstrongest time-series ICs (by |mean| with >= 70% of stocks agreeing):")
    s = t[t.ts_same_sign >= 0.7]
    print(s.reindex(s.ts_ic.abs().sort_values(ascending=False).index).head(20)[
        ["indicator", "form", "h", "ts_ic", "ts_same_sign", "ts_ic_2015_16", "ts_ic_2017_18"]].round(3).to_string(index=False))
    print("\ndistribution of |IC|: time series median %.3f, 90th pct %.3f; cross section median %.3f, 90th pct %.3f"
          % (t.ts_ic.abs().median(), t.ts_ic.abs().quantile(.9), t.cs_ic.abs().median(), t.cs_ic.abs().quantile(.9)))
