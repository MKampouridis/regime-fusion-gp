"""A broad universe with a longer history, for the development experiments (MK, 4 November 2026).

The S&P 500 members of January 2015 (``configs/constituents_2015-01.csv``), less the 50 hold-out
stocks, which stay untouched for a final test, and less the stocks the campaign universe excludes.
Fundamental and technical indicators only (no news), from 2009: downloads start in January 2008,
so the technical indicators and the trailing ranks have a year of history before the first
training day.

Everything lives in its own data directory, ``BROAD_DIR`` (``~/rfgp-broad``), so nothing of the
campaigns' data or configuration changes. The steps reuse the campaign pipeline's functions:
Yahoo prices aligned to the NYSE calendar, SEC company facts, point-in-time fundamentals, and
ActTrade's indicators. Two differences: no price corrections (``configs/price_corrections.yaml``
covers the campaign stocks only), and stocks Yahoo no longer has are recorded as missing.

Survivorship: the list is point in time for January 2015. For training years before 2015 it
holds only firms that survived to 2015; firms delisted after 2015 lose their prices at Yahoo.
``build`` reports both counts.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from pathlib import Path

import pandas as pd

from regime_fusion.data import fa, fundamentals, prices, sec, store, ta
from regime_fusion.data.calendar import study_calendar
from regime_fusion.data.settings import load_settings
from regime_fusion.data.universe import Stock, load_universe

log = logging.getLogger("regime_fusion.dev.broad")

BROAD_DIR = Path("~/rfgp-broad").expanduser()
DOWNLOAD_START = dt.date(2008, 1, 1)
CONSTITUENTS = Path(__file__).resolve().parents[3] / "configs" / "constituents_2015-01.csv"


# January 2015 tickers that trade under another symbol today, whose price history Yahoo keeps under
# the new one (renames, or the continuing entity of a reverse merger). Checked by the build: a
# symbol whose history does not reach back to 2015 is reported, not used silently.
RENAMED = {
    "GPS": "GAP", "ABC": "COR", "ZMH": "ZBH", "HCN": "WELL", "HCP": "DOC", "COH": "TPR",
    "MHFI": "SPGI", "JEC": "J", "BHI": "BKR", "BLL": "BALL", "PKI": "RVTY", "CBG": "CBRE",
    "DPS": "KDP", "TMK": "GL", "LUK": "JEF", "HRS": "LHX", "MYL": "VTRS", "KRFT": "KHC",
    "CHK": "EXE", "WYN": "TNL", "VIAB": "PARA", "SCG": "SCG", "HOT": "HOT",
}


def use_broad_dir() -> None:
    os.environ[store.DATA_DIR_ENV] = str(BROAD_DIR)


def universe() -> list[Stock]:
    """The broad universe: campaign entries where they exist (their checked CIKs and Yahoo
    symbols), the constituents list otherwise; hold-out and excluded stocks left out."""
    known = {s.id: s for s in load_universe().stocks}
    out = []
    for row in pd.read_csv(CONSTITUENTS).itertuples():
        ticker = str(row.ticker)
        stock = known.get(ticker)
        if stock is not None:
            if stock.role in ("holdout", "excluded") or not stock.yahoo:
                continue
            out.append(stock)
        else:
            out.append(Stock(id=ticker, name=str(row.name), sector=str(row.sector), ciks=(int(row.cik),),
                             yahoo=RENAMED.get(ticker, ticker.replace(".", "-")), role="candidate"))
    return out


def _settings():
    return dataclasses.replace(load_settings(), download_start=DOWNLOAD_START, yahoo_pause_seconds=3.0)


def build(steps=("prices", "sec", "indicators"), refresh: bool = False) -> dict:
    """Download and build every step for the broad universe; returns (and writes) a summary."""
    use_broad_dir()
    settings = _settings()
    calendar = study_calendar(settings)
    stocks = universe()
    summary = {"stocks": len(stocks), "missing_prices": [], "late_prices": [], "missing_sec": []}
    if "prices" in steps:
        for stock in stocks:
            raw_path = store.layer("raw", "yahoo", "stocks", f"{stock.id}.parquet")
            out = store.layer("interim", "prices", f"{stock.id}.parquet")
            if out.exists() and not refresh:
                continue
            try:
                if raw_path.exists() and not refresh:
                    raw = pd.read_parquet(raw_path)
                else:
                    raw = prices.download(stock.yahoo, DOWNLOAD_START, max_retries=settings.yahoo_max_retries,
                                          pause=settings.yahoo_pause_seconds)
                    if len(raw) < 250 or raw.index.min().year > 2015:   # throttled, or not the 2015 firm
                        raise ValueError(f"{len(raw)} bars from {raw.index.min().date() if len(raw) else None}")
                    store.write_parquet(raw_path, raw, source="Yahoo Finance", symbol=stock.yahoo)
            except Exception as problem:   # Yahoo no longer has it, or the symbol changed
                log.warning("%s (%s): no prices (%s)", stock.id, stock.yahoo, problem)
                summary["missing_prices"].append(stock.id)
                continue
            try:
                aligned = prices.align(raw, calendar)
            except ValueError as problem:   # e.g. bars only before 2008 or after the calendar
                log.warning("%s (%s): prices unusable (%s)", stock.id, stock.yahoo, problem)
                summary["missing_prices"].append(stock.id)
                continue
            store.write_parquet(out, aligned, series=stock.id, symbol=stock.yahoo)
    if "sec" in steps:
        client = sec.Client(settings.sec_user_agent, settings.sec_max_requests_per_second)
        for stock in stocks:
            out = store.layer("interim", "sec", f"{stock.id}.parquet")
            if out.exists() and not refresh:
                continue
            frames = []
            for cik in stock.ciks:
                name = f"CIK{cik:010d}.json.gz"
                path = store.layer("raw", "sec", "companyfacts", name)
                try:
                    if refresh or not path.exists():
                        url = sec.COMPANYFACTS_URL.format(cik=cik)
                        store.write_bytes(path, sec.compress(client.get(url)), source="SEC EDGAR", url=url)
                    frames.append(sec.report_facts(sec.load_json(path)))
                except sec.NotFound:
                    continue
            if not frames:
                log.warning("%s: no SEC company facts", stock.id)
                summary["missing_sec"].append(stock.id)
                continue
            table = pd.concat(frames, ignore_index=True)
            table["available"] = calendar.session_after_date(table["filed"])
            store.write_parquet(out, table, stock=stock.id, ciks=list(stock.ciks))
    if "indicators" in steps:
        concepts = fundamentals.load_concepts()
        for stock in stocks:
            price_path = store.layer("interim", "prices", f"{stock.id}.parquet")
            if not price_path.exists():
                continue
            aligned = pd.read_parquet(price_path)
            first = aligned["adj_close"].first_valid_index()
            if first is None or first > pd.Timestamp("2010-01-01"):
                summary["late_prices"].append(stock.id)
            aligned = aligned.loc[first:] if first is not None else aligned
            store.write_parquet(store.layer("processed", "ta", f"{stock.id}.parquet"), ta.indicators(aligned),
                                stock=stock.id)
            sec_path = store.layer("interim", "sec", f"{stock.id}.parquet")
            if not sec_path.exists():
                continue
            panel = fundamentals.daily_panel(pd.read_parquet(sec_path), concepts, calendar.sessions)
            store.write_parquet(store.layer("interim", "fundamentals", f"{stock.id}.parquet"), panel, stock=stock.id)
            splits = fa.true_splits(aligned["split"], [])
            store.write_parquet(store.layer("processed", "fa", f"{stock.id}.parquet"),
                                fa.indicators(panel, aligned["traded_close"], splits), stock=stock.id)
    summary["usable"] = sorted(s.id for s in stocks
                               if store.layer("processed", "fa", f"{s.id}.parquet").exists()
                               and store.layer("processed", "ta", f"{s.id}.parquet").exists())
    path = BROAD_DIR / "reports" / "broad_build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("broad universe: %d stocks; %d usable; no prices %d %s; prices only from 2010 or later %d; no SEC %d %s",
             len(stocks), len(summary["usable"]), len(summary["missing_prices"]), summary["missing_prices"],
             len(summary["late_prices"]), len(summary["missing_sec"]), summary["missing_sec"])
    return summary
