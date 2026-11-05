"""A rough small-cap universe from free data, as a development check only (MK, 4 November 2026).

Do the documented factors carry more signal among small stocks than among large ones? CRSP through
WRDS is the proper source; this is the free approximation, and it is biased:

* Candidates: the companies the SEC lists today on NYSE or Nasdaq (``company_tickers_exchange``).
  Firms that were listed in 2010-2018 but have since been acquired, gone bankrupt or delisted are
  missing, and Yahoo no longer has their prices. This survivorship bias is large for small stocks
  and favours strategies that buy cheap or distressed firms; it cannot be corrected here.
* Membership: each year at the last session of June, as the Russell indices are reconstituted, the
  candidates are ranked by market value (the day's close times the shares outstanding last filed),
  and ranks 1,001-3,000 form the universe from July to the next June. This uses only data known
  then, so the ranking itself has no look-ahead; the candidate list does.
* Data: SEC company facts processed with the campaign pipeline (``data.sec``, ``data.fundamentals``)
  and discarded after processing, to save disk; Yahoo prices aligned to the NYSE calendar, kept from
  2009 to 2019 (the development period and the warm-up). Everything lives in ``SMALL_DIR``.

``build`` reports how many candidates have no usable data, so the bias is visible.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from regime_fusion.data import fundamentals, prices, sec, store
from regime_fusion.data.calendar import study_calendar
from regime_fusion.data.settings import load_settings
from regime_fusion.dev import factors as F

log = logging.getLogger("regime_fusion.dev.small")

SMALL_DIR = Path("~/rfgp-small").expanduser()
DOWNLOAD_START = dt.date(2009, 1, 1)
KEEP_UNTIL = "2019-12-31"
RANKS = (1001, 3000)
MIN_MARKET_VALUE = 100e6   # about the bottom of the Russell 2000; below it, micro-caps and share-unit errors
MIN_PRICE = 1.0
LIST = "company_tickers_exchange.json"
FUNDAMENTALS = ["shares_outstanding", "net_income_ttm", "net_income_ttm_prev", "equity", "revenue_ttm",
                "operating_cash_flow_ttm", "operating_income_ttm", "assets", "liabilities"]   # what dev.factors uses
PRICES = ["close", "adj_open", "adj_close", "traded_close", "volume"]


def use_small_dir() -> None:
    os.environ[store.DATA_DIR_ENV] = str(SMALL_DIR)


def _settings():
    return dataclasses.replace(load_settings(), download_start=DOWNLOAD_START, yahoo_pause_seconds=1.5)


def candidates() -> pd.DataFrame:
    """One row per company on NYSE or Nasdaq today: id (``C<cik>``), CIK, Yahoo symbol, name.
    Where a company lists several tickers (share classes, preferreds), the shortest is kept."""
    use_small_dir()
    raw = json.loads(store.layer("raw", "sec", LIST).read_text())
    table = pd.DataFrame(raw["data"], columns=raw["fields"])
    table = table[table["exchange"].isin(["NYSE", "Nasdaq"]) & table["ticker"].notna()]
    table = table.assign(length=table["ticker"].str.len()).sort_values(["cik", "length", "ticker"])
    table = table.drop_duplicates("cik")
    table["yahoo"] = table["ticker"].str.replace(".", "-", regex=False)
    table["id"] = "C" + table["cik"].astype(str)
    return table[["id", "cik", "yahoo", "name", "exchange"]].reset_index(drop=True)


def build_fundamentals() -> dict:
    """SEC company facts of every candidate, processed to a daily panel; the raw file is not kept."""
    use_small_dir()
    settings = _settings()
    calendar = study_calendar(settings)
    sessions = calendar.sessions[calendar.sessions <= pd.Timestamp(KEEP_UNTIL)]
    concepts = fundamentals.load_concepts()
    client = sec.Client(settings.sec_user_agent, settings.sec_max_requests_per_second)
    summary = {"done": 0, "no_facts": [], "no_us_gaap": []}
    for row in candidates().itertuples():
        out = store.layer("interim", "fundamentals", f"{row.id}.parquet")
        if out.exists():
            summary["done"] += 1
            continue
        try:
            facts = sec.report_facts(json.loads(client.get(sec.COMPANYFACTS_URL.format(cik=row.cik))))
        except sec.NotFound:
            summary["no_facts"].append(row.id)
            continue
        except Exception as problem:   # malformed or unusual filers (funds, foreign issuers)
            log.warning("%s (%s): company facts unusable (%s)", row.id, row.yahoo, problem)
            summary["no_us_gaap"].append(row.id)
            continue
        if facts.empty:
            summary["no_us_gaap"].append(row.id)
            continue
        facts["available"] = calendar.session_after_date(facts["filed"])
        store.write_parquet(out, fundamentals.daily_panel(facts, concepts, sessions)[FUNDAMENTALS], stock=row.id,
                            cik=int(row.cik))
        summary["done"] += 1
    _write_report("fundamentals", summary)
    return summary


def build_prices() -> dict:
    """Yahoo prices of every candidate with fundamentals, aligned, kept from 2009 to 2019."""
    use_small_dir()
    settings = _settings()
    calendar = study_calendar(settings)
    summary = {"done": 0, "no_prices": []}
    for row in candidates().itertuples():
        out = store.layer("interim", "prices", f"{row.id}.parquet")
        if out.exists():
            summary["done"] += 1
            continue
        try:
            raw = prices.download(row.yahoo, DOWNLOAD_START, max_retries=settings.yahoo_max_retries,
                                  pause=settings.yahoo_pause_seconds)
            if len(raw) < 250 or raw.index.min().year > 2018:
                raise ValueError(f"{len(raw)} bars from {raw.index.min().date() if len(raw) else None}")
            aligned = prices.align(raw, calendar)
        except Exception as problem:
            summary["no_prices"].append(row.id)
            log.info("%s (%s): no usable prices (%s)", row.id, row.yahoo, str(problem)[:80])
            continue
        store.write_parquet(out, aligned.loc[:KEEP_UNTIL, PRICES], series=row.id, symbol=row.yahoo)
        summary["done"] += 1
    _write_report("prices", summary)
    return summary


def _write_report(name: str, summary: dict) -> None:
    path = SMALL_DIR / "reports" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("%s: %s", name, {k: (len(v) if isinstance(v, list) else v) for k, v in summary.items()})


def membership(years=range(2012, 2019)) -> pd.DataFrame:
    """Per June reconstitution, the ids ranked 1,001-3,000 by market value among candidates with data."""
    use_small_dir()
    rows = []
    ids = [p.stem for p in store.layer("interim", "prices").glob("*.parquet")
           if store.layer("interim", "fundamentals", f"{p.stem}.parquet").exists()]
    caps = {}
    for s in ids:
        p = pd.read_parquet(store.layer("interim", "prices", f"{s}.parquet"))
        f = pd.read_parquet(store.layer("interim", "fundamentals", f"{s}.parquet"), columns=["shares_outstanding"])
        cap = (p["traded_close"] * f["shares_outstanding"].reindex(p.index))
        caps[s] = cap.where(p["close"] >= MIN_PRICE).dropna()
    for year in years:
        day = pd.Timestamp(f"{year}-06-30")
        values = {}
        for s, cap in caps.items():
            known = cap.loc[:day]
            if len(known) and (day - known.index[-1]).days <= 7 and known.iloc[-1] >= MIN_MARKET_VALUE:
                values[s] = float(known.iloc[-1])
        ranked = pd.Series(values).sort_values(ascending=False)
        members = ranked.iloc[RANKS[0] - 1:RANKS[1]]
        rows += [{"year": year, "id": s, "rank": i + RANKS[0], "market_value": v} for i, (s, v) in enumerate(members.items())]
        log.info("%d: %d candidates of at least %.0f m; members ranked %d-%d: %d (largest %.0f m, smallest %.0f m)",
                 year, len(ranked), MIN_MARKET_VALUE / 1e6, RANKS[0], RANKS[1], len(members),
                 members.max() / 1e6 if len(members) else np.nan, members.min() / 1e6 if len(members) else np.nan)
    table = pd.DataFrame(rows)
    table.to_csv(SMALL_DIR / "reports" / "membership.csv", index=False)
    return table


def build_factors() -> list[str]:
    """``dev.factors`` for every stock that is ever a member, with the members' equal-weight return
    as the market for beta."""
    use_small_dir()
    members = membership()
    ids = sorted(members["id"].unique())
    px = {s: pd.read_parquet(store.layer("interim", "prices", f"{s}.parquet")) for s in ids}
    market = pd.concat({s: p["adj_close"].pct_change(fill_method=None) for s, p in px.items()}, axis=1).mean(axis=1)
    for s in ids:
        table = F.stock_factors(px[s], pd.read_parquet(store.layer("interim", "fundamentals", f"{s}.parquet")), market)
        store.write_parquet(store.layer("processed", "factors", f"{s}.parquet"), table, stock=s)
    return ids


def member_mask(index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    """Day x stock: True while the stock is in the universe (July of a year to June of the next)."""
    members = pd.read_csv(SMALL_DIR / "reports" / "membership.csv")
    mask = pd.DataFrame(False, index=index, columns=columns)
    for year, group in members.groupby("year"):
        days = (index >= pd.Timestamp(f"{year}-07-01")) & (index <= pd.Timestamp(f"{year + 1}-06-30"))
        cols = [c for c in group["id"] if c in mask.columns]
        mask.loc[days, cols] = True
    return mask


def screen(horizons=(20, 60), first_year: int = 2010) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The factor screen of ``dev.factors`` on the small-cap universe, members only (July to June
    after each June ranking), and median-split long/short portfolios by year at 10 bps."""
    from regime_fusion.dev.ic import newey_west_t
    from regime_fusion.dev.longshort import LongShortMarket
    use_small_dir()
    ids = sorted(pd.read_csv(SMALL_DIR / "reports" / "membership.csv")["id"].unique())
    tables = {s: pd.read_parquet(store.layer("processed", "factors", f"{s}.parquet")).loc["2009":"2019-06-30"] for s in ids}
    opens = pd.concat({s: pd.read_parquet(store.layer("interim", "prices", f"{s}.parquet"))["adj_open"].loc["2009":"2019-06-30"]
                       for s in ids}, axis=1)
    days = opens.index
    mask = member_mask(days, list(opens.columns))
    factors = {f: pd.concat({s: t[f] for s, t in tables.items()}, axis=1).reindex(days)[opens.columns].where(mask)
               for f in F.SIGNS}
    signals = {f: F.SIGNS[f] * F.cross_rank(t) for f, t in factors.items()}
    signals["composite"] = F.composite(factors)
    rows = []
    for h in horizons:
        forward = opens.shift(-(h + 1)) / opens.shift(-1) - 1
        fwd_rank = forward.rank(axis=1)
        for name, sig in signals.items():
            ic = sig.rank(axis=1).corrwith(fwd_rank, axis=1)
            ic = ic[(sig.notna() & forward.notna()).sum(axis=1) >= 200]
            ic = ic[(ic.index.year >= first_year) & (ic.index <= "2018-12-31")]
            by_year = ic.groupby(ic.index.year).mean()
            rows.append({"factor": name, "h": h, "ic": ic.mean(), "t": newey_west_t(ic, h - 1),
                         "years_positive": int((by_year > 0).sum()), **{str(y): v for y, v in by_year.items()}})
    ic_table = pd.DataFrame(rows)
    ls_rows = []
    for name in ["composite", "mom_12_1", "rev_1m", "book_price", "earnings_price", "op_profitability",
                 "share_issuance", "volatility_60", "size"]:
        sig = signals[name]
        for year in range(first_year, 2019):
            sel = days.year == year
            o, sgn = opens[sel].to_numpy(), sig[sel].to_numpy()
            for cost in (0.0, 0.001):
                net, _, _ = LongShortMarket(o, 20, cost, np.isfinite(sgn)).simulate(sgn > 0)
                ls_rows.append({"signal": name, "year": year, "cost": cost, "return": float(np.prod(1 + net) - 1),
                                "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(252 / 20)) if net.std(ddof=1) > 0 else 0.0})
    ls_table = pd.DataFrame(ls_rows)
    ic_table.to_csv(SMALL_DIR / "reports" / "factor_screen.csv", index=False)
    ls_table.to_csv(SMALL_DIR / "reports" / "long_short.csv", index=False)
    return ic_table, ls_table
