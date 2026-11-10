"""Cross-sectional factor portfolios, measured on a market-hedged basis with liquidity-dependent costs.

This is the tested version of the read-only study scripts of 10 November 2026. Development data only:
every file is cut at ``period.development_end`` (31 December 2018) straight after reading.

* Decision dates come every 20 sessions. The factors are known at the close of the decision day t;
  the return runs from the open of t+1 to the open of t+21. A date is used only if its return ends
  on or before the last session of the development period, so no later price is ever needed.
* "Hedged" means each stock's return less its beta times the universe's equal-weight return over
  the same window. A dollar-neutral portfolio of low-beta longs and high-beta shorts is short the
  market, so in a rising market it looks worse than its stock selection deserves; the hedge takes
  that out.
* Betas come from 5-session returns over the last 250 sessions, not from daily returns. Thinly
  traded small stocks react to the market a few days late, so daily returns understate their betas
  and compress the spread between stocks.
* Costs depend on the value traded (the median of close times volume over 60 sessions). The tiers in
  ``COST_TIERS`` are an assumption made for sensitivity checks, not a measurement of real spreads.
* Portfolios hold equal weights in the top and bottom fifth of a score (or, banded, keep a stock
  until it falls out of the top or bottom two fifths, which saves trading).

Functions are pure and take date x stock frames; only ``small_panel``, ``broad_panel`` and ``report``
touch the disk. ``python -m regime_fusion.dev hedged-report --universe small`` prints the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_fusion.config import load_config
from regime_fusion.dev import broad
from regime_fusion.dev import factors as F
from regime_fusion.dev import small

H = 20                         # sessions between decision dates, and the holding period
BETA_WINDOW = 250              # sessions of history for a beta
BETA_STEP = 5                  # sessions in one return used for a beta
VALUE_WINDOW = 60              # sessions in the value-traded median
VALUE_MIN_PERIODS = 30
RETURN_CLIP = (-0.9, 3.0)      # keeps data errors from dominating
BETA_CLIP = (-1.0, 4.0)
# Assumed cost per side by median daily value traded in dollars: (threshold, cost) from the top down.
COST_TIERS = ((10e6, 0.0010), (5e6, 0.0020), (2e6, 0.0035), (1e6, 0.0060), (0.5e6, 0.0100))
COST_BELOW = 0.0200
LIQUID = 2e6                   # dollars a day: the split used in the small-cap report


def decision_positions(sessions: pd.DatetimeIndex, start, end, horizon: int = H) -> np.ndarray:
    """Positions t of the decision dates: every ``horizon``-th session from the first session on or
    after ``start``, keeping those with t + 1 + horizon at or before the last session on or before ``end``."""
    first = int(np.searchsorted(sessions, pd.Timestamp(start), side="left"))
    last = int(np.searchsorted(sessions, pd.Timestamp(end), side="right")) - 1
    return np.arange(first, last - horizon, horizon, dtype=int)


def forward_returns(opens: pd.DataFrame, positions: np.ndarray, horizon: int = H) -> pd.DataFrame:
    """open[t + 1 + horizon] / open[t + 1] - 1 for each position t, indexed by the date of t."""
    values = opens.to_numpy(dtype=float)
    out = values[positions + 1 + horizon] / values[positions + 1] - 1.0
    return pd.DataFrame(out, index=opens.index[positions], columns=opens.columns)


def weekly_beta(returns: pd.DataFrame, positions: np.ndarray | None = None, eligible: pd.DataFrame | None = None,
                window: int = BETA_WINDOW, step: int = BETA_STEP) -> pd.DataFrame:
    """Beta of each stock on the equal-weight market, from ``step``-session returns.

    ``returns`` are daily simple returns (session x stock). For a position t only rows t-window+1 to t
    are used: clipped to ``RETURN_CLIP``, logged, and summed into ``window / step`` non-overlapping
    returns. Stocks with a missing value in the window, or not ``eligible`` at t, get NaN. The market is
    the equal-weight mean of the remaining stocks. ``positions=None`` means every row; rows before a
    full window are NaN."""
    if window % step:
        raise ValueError("window must be a multiple of step")
    logs = np.log1p(returns.clip(*RETURN_CLIP).to_numpy(dtype=float))
    ok_rows = None if eligible is None else eligible.reindex_like(returns).fillna(False).to_numpy(dtype=bool)
    rows = np.arange(len(returns)) if positions is None else np.asarray(positions, dtype=int)
    out = np.full((len(rows), logs.shape[1]), np.nan)
    for k, t in enumerate(rows):
        if t < window - 1:
            continue
        block = logs[t - window + 1:t + 1].reshape(window // step, step, -1).sum(axis=1)
        keep = np.isfinite(block).all(axis=0)
        if ok_rows is not None:
            keep &= ok_rows[t]
        if not keep.any():
            continue
        x = block[:, keep]
        market = x.mean(axis=1)
        centred = market - market.mean()
        spread = centred @ centred
        if spread > 0:
            out[k, keep] = (x - x.mean(axis=0)).T @ centred / spread
    return pd.DataFrame(out, index=returns.index[rows], columns=returns.columns)


def value_traded(close: pd.DataFrame, volume: pd.DataFrame, window: int = VALUE_WINDOW,
                 min_periods: int = VALUE_MIN_PERIODS) -> pd.DataFrame:
    """Rolling median of close x volume, one row per session. ``close`` and ``volume`` are both
    split-adjusted in this project's price files, so the product is the money traded."""
    return (close * volume).rolling(window, min_periods=min_periods).median()


def hedged(forward: pd.DataFrame, beta: pd.DataFrame, market: pd.Series | None = None) -> pd.DataFrame:
    """Forward returns less beta times the market's return. ``market`` defaults to the row mean of the
    clipped forward returns; missing betas count as 1."""
    clipped = forward.clip(*RETURN_CLIP)
    if market is None:
        market = clipped.mean(axis=1)
    b = beta.reindex_like(clipped).clip(*BETA_CLIP).fillna(1.0)
    return clipped - b.mul(market, axis=0)


def information_coefficient(score: pd.DataFrame, returns: pd.DataFrame, min_names: int = 100) -> pd.Series:
    """Per date, the Spearman correlation across the stocks that have both values; dates with fewer
    than ``min_names`` such stocks are dropped."""
    both = score.notna() & returns.notna()
    ic = score.where(both).rank(axis=1).corrwith(returns.where(both).rank(axis=1), axis=1)
    return ic[both.sum(axis=1) >= min_names]


def _percentile(score: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    return score.where(score.notna() & tradable.reindex_like(score).fillna(False).astype(bool)).rank(axis=1, pct=True)


def _equal(book: pd.DataFrame) -> pd.DataFrame:
    return book.div(book.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def quantile_weights(score: pd.DataFrame, tradable: pd.DataFrame, quantile: float = 0.2,
                     long_only: bool = False) -> pd.DataFrame:
    """Equal weights summing to +1 on the top ``quantile`` of the tradable stocks with a score and,
    unless ``long_only``, to -1 on the bottom ``quantile``."""
    pct = _percentile(score, tradable)
    weights = _equal(pct > 1 - quantile)
    if not long_only:
        weights = weights - _equal(pct <= quantile)
    return weights


def banded_weights(score: pd.DataFrame, tradable: pd.DataFrame, enter: float = 0.2, leave: float = 0.4,
                   long_only: bool = False, short_allowed: pd.DataFrame | None = None) -> pd.DataFrame:
    """Like ``quantile_weights`` but a stock enters a book in the top (bottom) ``enter`` share and stays
    while it is in the top (bottom) ``leave`` share. A stock with no score, or not tradable, leaves. A date
    where a needed book is empty has zero weights (the holdings carry on to the next date)."""
    pct_frame = _percentile(score, tradable)
    pct = pct_frame.to_numpy()
    known = np.isfinite(pct)
    allowed = None if short_allowed is None else short_allowed.reindex_like(score).fillna(False).to_numpy(dtype=bool)
    weights = np.zeros(pct.shape)
    long_held = np.zeros(pct.shape[1], dtype=bool)
    short_held = np.zeros(pct.shape[1], dtype=bool)
    with np.errstate(invalid="ignore"):
        for k in range(len(pct)):
            p, ok = pct[k], known[k]
            long_held = ok & ((p > 1 - enter) | (long_held & (p > 1 - leave)))
            short_held = ok & ((p <= enter) | (short_held & (p <= leave)))
            if allowed is not None:
                short_held &= allowed[k]
            if long_only:
                if long_held.any():
                    weights[k] = long_held / long_held.sum()
            elif long_held.any() and short_held.any():
                weights[k] = long_held / long_held.sum() - short_held / short_held.sum()
    return pd.DataFrame(weights, index=score.index, columns=score.columns)


def cost_per_side(value: pd.DataFrame) -> pd.DataFrame:
    """Assumed cost per side of each stock-date from its value traded (see ``COST_TIERS``);
    ``COST_BELOW`` under the last threshold or when the value is missing."""
    v = value.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        cost = np.select([v >= threshold for threshold, _ in COST_TIERS], [c for _, c in COST_TIERS], COST_BELOW)
    return pd.DataFrame(cost, index=value.index, columns=value.columns)


def portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame, cost: float | pd.DataFrame) -> pd.DataFrame:
    """Per date: ``gross`` (sum of weight x return, missing returns count as 0), ``turnover`` (sum of
    absolute weight changes from the previous date; the first date from zero), ``cost`` paid (a scalar per
    side or a date x stock frame, times the weight changes) and ``net`` = gross - cost."""
    gross = (weights * returns.reindex_like(weights).fillna(0.0)).sum(axis=1)
    change = weights.diff().abs()
    change.iloc[0] = weights.iloc[0].abs()
    per_side = cost.reindex_like(weights).fillna(COST_BELOW) if isinstance(cost, pd.DataFrame) else cost
    paid = (change * per_side).sum(axis=1)
    return pd.DataFrame({"gross": gross, "turnover": change.sum(axis=1), "cost": paid, "net": gross - paid})


def summarise(returns: pd.Series, periods_per_year: float = 252 / H) -> dict:
    """Annualised mean and standard deviation of per-period returns, their ratio, the t statistic of the
    mean, the number of periods, and the calendar years whose summed return is positive."""
    x = returns.dropna()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    by_year = x.groupby(x.index.year).sum()
    return {"return": mean * periods_per_year, "volatility": sd * np.sqrt(periods_per_year),
            "sharpe": mean / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
            "t": mean / sd * np.sqrt(len(x)) if sd > 0 else np.nan, "periods": len(x),
            "years_positive": int((by_year > 0).sum()), "years": len(by_year)}


@dataclass(frozen=True)
class Panel:
    """Everything at the decision dates, as date x stock frames.

    ``factors``: raw factor values (NaN outside the universe); ``forward``: raw forward returns of the
    members (NaN for others); ``member``: Boolean; ``beta``: weekly betas; ``value``: median daily value
    traded in dollars; ``market``: row mean of the clipped forward returns of the members."""

    dates: pd.DatetimeIndex
    factors: dict[str, pd.DataFrame]
    forward: pd.DataFrame
    member: pd.DataFrame
    beta: pd.DataFrame
    value: pd.DataFrame
    market: pd.Series

    def hedged(self) -> pd.DataFrame:
        return hedged(self.forward, self.beta, self.market)


def _end(end) -> pd.Timestamp:
    return pd.Timestamp(load_config().period.development_end if end is None else end)


def _assemble(opens, closes, traded, volumes, tables, positions, member_daily) -> Panel:
    """``closes`` are adjusted closes (for returns); ``traded`` and ``volumes`` give the money traded."""
    days = opens.index
    columns = opens.columns
    dates = days[positions]
    member = member_daily.iloc[positions]
    forward = forward_returns(opens, positions).where(member)
    factors = {}
    for name in F.SIGNS:
        daily = pd.concat({s: t[name] for s, t in tables.items()}, axis=1).reindex(days)[columns]
        factors[name] = pd.DataFrame(daily.to_numpy()[positions], index=dates, columns=columns).where(member)
    beta = weekly_beta(closes.pct_change(fill_method=None), positions, eligible=member_daily)
    value = value_traded(traded, volumes).iloc[positions]
    market = forward.clip(*RETURN_CLIP).mean(axis=1)
    return Panel(dates, factors, forward, member, beta, value, market)


def _load(directory, ids, end) -> tuple:
    columns = ["adj_open", "adj_close", "close", "volume"]
    prices = {s: small.read_until(directory / "interim" / "prices" / f"{s}.parquet", end, columns=columns) for s in ids}
    tables = {s: small.read_until(directory / "processed" / "factors" / f"{s}.parquet", end) for s in ids}
    return tuple(pd.concat({s: p[c] for s, p in prices.items()}, axis=1) for c in columns) + (tables,)


def small_panel(end=None) -> Panel:
    """The small-cap universe: members from ``reports/membership.csv`` (July to the next June), decision
    dates from the first session of July 2012."""
    end = _end(end)
    ids = sorted(pd.read_csv(small.SMALL_DIR / "reports" / "membership.csv")["id"].unique())
    opens, closes, traded, volumes, tables = _load(small.SMALL_DIR, ids, end)
    positions = decision_positions(opens.index, "2012-07-01", end)
    member_daily = small.member_mask(opens.index, list(opens.columns))
    return _assemble(opens, closes, traded, volumes, tables, positions, member_daily)


def broad_panel(end=None) -> Panel:
    """The large-cap universe: the ``usable`` stocks of ``reports/broad_build.json`` that have a factor
    file, all members throughout, decision dates from the first session of 2010."""
    end = _end(end)
    usable = json.loads((broad.BROAD_DIR / "reports" / "broad_build.json").read_text())["usable"]
    ids = [s for s in usable if (broad.BROAD_DIR / "processed" / "factors" / f"{s}.parquet").exists()]
    opens, closes, traded, volumes, tables = _load(broad.BROAD_DIR, ids, end)
    positions = decision_positions(opens.index, "2010-01-01", end)
    member_daily = pd.DataFrame(True, index=opens.index, columns=opens.columns)
    return _assemble(opens, closes, traded, volumes, tables, positions, member_daily)


def composite(panel: Panel, factors: list[str] | None = None, group: pd.DataFrame | None = None) -> pd.DataFrame:
    """Equal-weight mean of the signed cross-sectional ranks of ``factors`` (default all 17), each ranked
    inside ``group`` (a Boolean frame, default the members)."""
    group = panel.member if group is None else group
    names = list(F.SIGNS) if factors is None else list(factors)
    ranked = [F.SIGNS[f] * F.cross_rank(panel.factors[f].where(group)) for f in names]
    with np.errstate(all="ignore"):
        mean = np.nanmean(np.stack([r.to_numpy(dtype=float) for r in ranked]), axis=0)
    return pd.DataFrame(mean, index=panel.dates, columns=panel.forward.columns).where(group)


def _row(line: str, result: pd.DataFrame, column: str) -> dict:
    return {"line": line, **summarise(result[column]), "turnover": float(result["turnover"].mean())}


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def report(universe: str) -> pd.DataFrame:
    """The hedged report for the composite (top fifth minus bottom fifth unless stated); printed and
    written to ``<universe dir>/reports/hedged_report.csv``."""
    if universe not in ("small", "broad"):
        raise ValueError("universe must be 'small' or 'broad'")
    panel = small_panel() if universe == "small" else broad_panel()
    directory = small.SMALL_DIR if universe == "small" else broad.BROAD_DIR
    raw = panel.forward.clip(*RETURN_CLIP)
    resid = panel.hedged()
    costs = cost_per_side(panel.value)
    score = composite(panel)
    tradable = resid.notna()
    rows = []

    def add(line, weights, returns, cost, column):
        rows.append(_row(line, portfolio_returns(weights, returns, cost), column))

    full = quantile_weights(score, tradable)
    banded = banded_weights(score, tradable)
    add("(a) dollar-neutral, gross", full, raw, 0.0, "gross")
    add("(b) hedged, gross", full, resid, 0.0, "gross")
    add("(c) hedged, flat 10 bps per side", full, resid, 0.0010, "net")
    add("(c) hedged, flat 25 bps per side", full, resid, 0.0025, "net")
    add("(d) hedged, tiered costs, rebuilt in full", full, resid, costs, "net")
    add("(e) hedged, tiered costs, banded 20/40", banded, resid, costs, "net")
    book = portfolio_returns(banded_weights(score, tradable, long_only=True), resid, costs)
    base = resid.where(panel.member).mean(axis=1)
    book[["gross", "net"]] = book[["gross", "net"]].sub(base, axis=0)
    rows.append(_row("(f) long book only, banded 20/40, tiered costs, excess over members", book, "net"))
    if universe == "small":
        liquid = panel.member & (panel.value >= LIQUID)
        thin = panel.member & (panel.value < LIQUID)
        for label, group in (("value traded >= $2m a day", liquid), ("value traded < $2m a day", thin)):
            sub = composite(panel, group=group)
            w = quantile_weights(sub, tradable & group)
            add(f"(g) {label}: hedged, gross", w, resid, 0.0, "gross")
            add(f"(g) {label}: hedged, tiered costs, rebuilt in full", w, resid, costs, "net")
    table = pd.DataFrame(rows)
    path = directory / "reports" / "hedged_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

    shown = table.assign(**{"return %": 100 * table["return"], "volatility %": 100 * table["volatility"]})
    print(f"\nComposite, {universe} caps, {panel.dates[0].date()} to {panel.dates[-1].date()}, "
          f"{len(panel.dates)} decision dates, top fifth minus bottom fifth")
    print(shown[["line", "return %", "volatility %", "sharpe", "t", "years_positive", "years", "turnover"]]
          .round(2).to_string(index=False))
    ic_raw, ic_hedged = information_coefficient(score, raw), information_coefficient(score, resid)
    print(f"\nIC of the composite: against raw returns {ic_raw.mean():.3f} (t {_t(ic_raw):.1f}); "
          f"against hedged returns {ic_hedged.mean():.3f} (t {_t(ic_hedged):.1f})")
    beta = panel.beta.clip(*BETA_CLIP)
    difference = (beta.where(full > 0).mean(axis=1) - beta.where(full < 0).mean(axis=1)).mean()
    print(f"Mean weekly beta of the long book less the short book: {difference:.2f}")
    return table
