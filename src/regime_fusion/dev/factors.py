"""A library of documented return predictors ("factors") for the broad universe (MK, 4 November 2026).

The GP's inputs so far (ActTrade's indicators) carry no timing information and only weak
cross-sectional information. These factors come from the finance literature, chosen before looking
at our data, and are built from the prices and SEC filings ``dev.broad`` already holds. Each is a
per-stock daily series known by the close of day t (prices up to t; fundamentals as filed, from the
session after their filing date, ``data.fundamentals``); stocks are compared with each other only
when the factors are used, by ranking across stocks each day.

Signs are the literature's (``SIGNS``: +1 where a high value predicts higher returns), fixed here
before the screen, and used for the equal-weight composite.

======================  =====================================================  =====
factor                  definition                                             sign
======================  =====================================================  =====
mom_12_1                return from t-252 to t-21 (skipping the last month)    +1
rev_1m                  return over the last 21 sessions                       -1
high_52w                close / highest close of the last 252 sessions         +1
earnings_price          trailing-12-month net income / market value            +1
book_price              equity / market value                                  +1
sales_price             trailing-12-month revenue / market value               +1
cashflow_price          trailing-12-month operating cash flow / market value   +1
op_profitability        trailing-12-month operating income / assets            +1
roe                     trailing-12-month net income / equity                  +1
accruals                (net income - operating cash flow) / assets            -1
leverage                liabilities / assets                                   -1
asset_growth            assets / assets 252 sessions earlier - 1              -1
share_issuance          shares outstanding / 252 sessions earlier - 1         -1
volatility_60           standard deviation of daily returns, 60 sessions       -1
beta_252                beta on the universe's equal-weight return, 252 days   -1
earnings_change         (net income ttm - a year earlier) / assets             +1
size                    log market value                                       -1
======================  =====================================================  =====

References: Jegadeesh and Titman (1993); George and Hwang (2004); Fama and French (1992, 2015);
Novy-Marx (2013), Ball et al. (2016) for operating profitability (cost of revenue is not in our
SEC extract, so gross profitability cannot be built); Sloan (1996); Pontiff and Woodgate (2008);
Ang et al. (2006); Frazzini and Pedersen (2014); Bernard and Thomas (1989); Banz (1981).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from regime_fusion.data import store
from regime_fusion.dev import broad

SIGNS = {
    "mom_12_1": 1, "rev_1m": -1, "high_52w": 1, "earnings_price": 1, "book_price": 1, "sales_price": 1,
    "cashflow_price": 1, "op_profitability": 1, "roe": 1, "accruals": -1, "leverage": -1,
    "asset_growth": -1, "share_issuance": -1, "volatility_60": -1, "beta_252": -1, "earnings_change": 1,
    "size": -1,
}
YEAR = 252


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b > 0)


def stock_factors(prices: pd.DataFrame, fundamentals: pd.DataFrame, market: pd.Series | None = None) -> pd.DataFrame:
    """The per-stock factors on the price sessions; ``market`` is the universe's daily return (for beta)."""
    close, traded = prices["adj_close"], prices["traded_close"]
    f = fundamentals.reindex(prices.index)
    returns = close.pct_change(fill_method=None)
    mcap = traded * f["shares_outstanding"]
    out = pd.DataFrame(index=prices.index)
    out["mom_12_1"] = close.shift(21) / close.shift(YEAR) - 1
    out["rev_1m"] = close / close.shift(21) - 1
    out["high_52w"] = prices["close"] / prices["close"].rolling(YEAR, min_periods=200).max()
    out["earnings_price"] = _ratio(f["net_income_ttm"], mcap)
    out["book_price"] = _ratio(f["equity"], mcap)
    out["sales_price"] = _ratio(f["revenue_ttm"], mcap)
    out["cashflow_price"] = _ratio(f["operating_cash_flow_ttm"], mcap)
    out["op_profitability"] = _ratio(f["operating_income_ttm"], f["assets"])
    out["roe"] = _ratio(f["net_income_ttm"], f["equity"])
    out["accruals"] = _ratio(f["net_income_ttm"] - f["operating_cash_flow_ttm"], f["assets"])
    out["leverage"] = _ratio(f["liabilities"], f["assets"])
    out["asset_growth"] = _ratio(f["assets"], f["assets"].shift(YEAR)) - 1
    out["share_issuance"] = _ratio(f["shares_outstanding"], f["shares_outstanding"].shift(YEAR)) - 1
    out["volatility_60"] = returns.rolling(60, min_periods=40).std()
    if market is not None:
        m = market.reindex(prices.index)
        cov = returns.rolling(YEAR, min_periods=200).cov(m)
        out["beta_252"] = cov / m.rolling(YEAR, min_periods=200).var()
    out["earnings_change"] = _ratio(f["net_income_ttm"] - f["net_income_ttm_prev"], f["assets"])
    out["size"] = np.log(mcap.where(mcap > 0))
    return out.replace([np.inf, -np.inf], np.nan)


def build() -> list[str]:
    """Factors for every usable stock of the broad universe, in ``processed/factors``."""
    broad.use_broad_dir()
    stocks = json.loads((broad.BROAD_DIR / "reports" / "broad_build.json").read_text())["usable"]
    prices = {s: pd.read_parquet(store.layer("interim", "prices", f"{s}.parquet")) for s in stocks}
    daily = pd.concat({s: p["adj_close"].pct_change(fill_method=None) for s, p in prices.items()}, axis=1)
    market = daily.mean(axis=1)   # equal-weight universe return, day t uses day t's returns only
    done = []
    for s in stocks:
        path = store.layer("interim", "fundamentals", f"{s}.parquet")
        if not path.exists():
            continue
        table = stock_factors(prices[s], pd.read_parquet(path), market)
        store.write_parquet(store.layer("processed", "factors", f"{s}.parquet"), table, stock=s,
                            factors=list(table.columns))
        done.append(s)
    return done


def panel(start: str, end: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Factors as {factor: day x stock}, open prices (day x stock), and each stock's sector."""
    broad.use_broad_dir()
    sectors = {s.id: s.sector for s in broad.universe()}
    tables, opens = {}, {}
    for path in sorted(store.layer("processed", "factors").glob("*.parquet")):
        s = path.stem
        tables[s] = pd.read_parquet(path).loc[start:end]
        opens[s] = pd.read_parquet(store.layer("interim", "prices", f"{s}.parquet"))["adj_open"].loc[start:end]
    days = pd.DatetimeIndex(sorted(set().union(*(t.index for t in tables.values()))))
    factors = {f: pd.concat({s: t[f] for s, t in tables.items()}, axis=1).reindex(days) for f in SIGNS}
    return factors, pd.concat(opens, axis=1).reindex(days), {s: sectors.get(s, "") for s in tables}


def cross_rank(frame: pd.DataFrame, sectors: dict[str, str] | None = None) -> pd.DataFrame:
    """Each day's percentile rank across stocks, in [-1, 1]; within sector when ``sectors`` is given."""
    if not sectors:
        return 2 * frame.rank(axis=1, pct=True) - 1
    out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    groups: dict[str, list[str]] = {}
    for s in frame.columns:
        groups.setdefault(sectors.get(s, ""), []).append(s)
    for members in groups.values():
        out[members] = 2 * frame[members].rank(axis=1, pct=True) - 1
    return out


def composite(factors: dict[str, pd.DataFrame], sectors: dict[str, str] | None = None) -> pd.DataFrame:
    """Equal-weight composite: the mean of the signed cross-sectional ranks of the factors known."""
    ranked = [SIGNS[f] * cross_rank(t, sectors) for f, t in factors.items()]
    stacked = np.stack([r.to_numpy(dtype=float) for r in ranked])
    with np.errstate(all="ignore"):
        mean = np.nanmean(stacked, axis=0)
    return pd.DataFrame(mean, index=ranked[0].index, columns=ranked[0].columns)


def screen(horizons=(20, 60), start: str = "2009-01-01", first_year: int = 2010, end: str = "2018-12-31") -> pd.DataFrame:
    """Cross-sectional information coefficients of every factor (and the composite), per year,
    2010-2018: the Spearman correlation across stocks between the factor at the close of t and the
    return from the open of t+1 to the open of t+1+h, with Newey-West t statistics."""
    from regime_fusion.dev.ic import newey_west_t
    factors, opens, sectors = panel(start, end)
    inputs = {**{f: (t, None) for f, t in factors.items()},
              **{f"{f} (in sector)": (t, sectors) for f, t in factors.items()}}
    rows = []
    for h in horizons:
        forward = opens.shift(-(h + 1)) / opens.shift(-1) - 1      # cut at ``end``: no later prices
        fwd_rank = forward.rank(axis=1)
        signals = {name: SIGNS[name.split(" ")[0]] * cross_rank(t, sec) for name, (t, sec) in inputs.items()}
        signals["composite"] = composite(factors)
        signals["composite (in sector)"] = composite(factors, sectors)
        for name, sig in signals.items():
            ic = sig.rank(axis=1).corrwith(fwd_rank, axis=1)
            ic = ic[(sig.notna() & forward.notna()).sum(axis=1) >= 50]
            ic = ic[ic.index.year >= first_year]
            by_year = ic.groupby(ic.index.year).mean()
            rows.append({"factor": name, "h": h, "ic": ic.mean(), "t": newey_west_t(ic, h - 1),
                         "years_positive": int((by_year > 0).sum()), "years": len(by_year),
                         **{str(y): v for y, v in by_year.items()}})
    table = pd.DataFrame(rows)
    out = broad.BROAD_DIR / "results" / "factor_screen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table
