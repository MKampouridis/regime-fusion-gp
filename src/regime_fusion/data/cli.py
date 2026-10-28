"""Command line for the data pipeline: ``python -m regime_fusion.data <command>``.

Commands (each has --help):

  constituents  rebuild configs/constituents_2015-01.csv from pinned Wikipedia revisions
  prices        download Yahoo prices for the universe and align them to the calendar
  market        download the S&P 500 index, SPY and the VIX, and align them

Data go to the folder named by RFGP_DATA_DIR (see docs/DATA.md). Downloads are
cached: a raw file already there is not fetched again unless --refresh is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import pandas as pd

from regime_fusion import __version__
from regime_fusion.config import StudyConfig, load_config
from regime_fusion.data import constituents, market, prices, store
from regime_fusion.data.calendar import TradingCalendar, study_calendar
from regime_fusion.data.settings import DataSettings, load_settings
from regime_fusion.data.universe import load_universe

CONSTITUENTS_FILE = Path(__file__).resolve().parents[3] / "configs" / "constituents_2015-01.csv"

log = logging.getLogger("regime_fusion.data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m regime_fusion.data", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("constituents", help="rebuild the January 2015 constituents list")
    sub.set_defaults(run=run_constituents)

    sub = commands.add_parser("prices", help="download and align Yahoo prices for the universe")
    sub.add_argument("--stocks", nargs="+", metavar="ID",
                     help="stock ids (January 2015 tickers); default: every main and candidate stock")
    sub.add_argument("--include-holdout", action="store_true",
                     help="also download the hold-out stocks")
    sub.add_argument("--refresh", action="store_true", help="download again even if cached")
    sub.set_defaults(run=run_prices)

    sub = commands.add_parser("market", help="download and align the S&P 500 index, SPY and the VIX")
    sub.add_argument("--refresh", action="store_true", help="download again even if cached")
    sub.set_defaults(run=run_market)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.run(args) or 0


def user_agent() -> str:
    """Identifies our requests to services that ask for a contact."""
    return f"regime-fusion-gp/{__version__} ({load_settings().sec_user_agent})"


def run_constituents(args: argparse.Namespace) -> int:
    texts = {}
    for key, revision in constituents.SNAPSHOT.items():
        payload = constituents.fetch(revision, user_agent())
        store.write_bytes(
            store.layer("raw", "wikipedia", f"{key}_{revision.revid}.json"), payload,
            source="Wikipedia API", page=revision.page, revid=revision.revid,
        )
        texts[key] = constituents.wikitext_of(payload, revision)
    table = constituents.combine(
        constituents.parse_sp100(texts["sp100"]), constituents.parse_sp500(texts["sp500"])
    )
    table.to_csv(CONSTITUENTS_FILE, index=False)
    log.info("wrote %d S&P 500 members (%d in the S&P 100) to %s",
             len(table), int(table["sp100"].sum()), CONSTITUENTS_FILE)
    return 0


def run_prices(args: argparse.Namespace) -> int:
    settings, config = load_settings(), load_config()
    settings.check_covers(config)
    calendar = study_calendar(settings)
    universe = load_universe()
    if args.stocks:
        stocks = [universe.get(stock_id) for stock_id in args.stocks]
        if any(stock.role == "holdout" for stock in stocks) and not args.include_holdout:
            raise SystemExit("hold-out stocks need --include-holdout")
    else:
        roles = ("main", "candidate") + (("holdout",) if args.include_holdout else ())
        stocks = universe.select(*roles, include_holdout=args.include_holdout)

    checks, failed = {}, []
    for stock in stocks:
        if stock.yahoo is None:
            log.info("%s: skipped, no Yahoo symbol (%s)", stock.id, stock.notes)
            continue
        raw_path = store.layer("raw", "yahoo", "stocks", f"{stock.id}.parquet")
        try:
            raw = cached_prices(stock.yahoo, raw_path, settings, args.refresh, stock=stock.id)
        except prices.MissingPrices as problem:
            log.error("%s: %s", stock.id, problem)
            failed.append(stock.id)
            continue
        checks[stock.id] = align_and_check(
            stock.id, raw, store.layer("interim", "prices", f"{stock.id}.parquet"),
            calendar, config, symbol=stock.yahoo,
        )
    write_checks("price_checks.json", checks)
    flagged = sorted(name for name, result in checks.items() if result.flags)
    moves = sum(len(result.big_moves) for result in checks.values())
    log.info("prices: %d aligned; %d with problems %s; %d failed %s; "
             "%d daily moves above %.0f%% listed in reports/price_checks.json for checking",
             len(checks), len(flagged), flagged, len(failed), failed, moves, 100 * prices.BIG_MOVE)
    return 1 if failed else 0


def run_market(args: argparse.Namespace) -> int:
    settings, config = load_settings(), load_config()
    calendar = study_calendar(settings)
    checks = {}
    for key, symbol in market.MARKET_SYMBOLS.items():
        raw = cached_prices(
            symbol, store.layer("raw", "yahoo", "market", f"{key}.parquet"), settings,
            args.refresh, series=key,
        )
        checks[key] = align_and_check(
            key, raw, store.layer("interim", "market", f"{key}.parquet"), calendar, config,
            symbol=symbol,
        )
    write_checks("market_checks.json", checks)

    vix_path = store.layer("raw", "cboe", "VIX_History.csv")
    if args.refresh or not vix_path.exists():
        content = market.fetch_vix(settings.vix_url, user_agent())
        store.write_bytes(vix_path, content, source="Cboe", url=settings.vix_url)
    vix, counts = market.align_vix(market.parse_vix(vix_path.read_bytes()), calendar)
    store.write_parquet(
        store.layer("interim", "market", "VIX.parquet"), vix,
        derived_from="raw/cboe/VIX_History.csv", **counts,
    )
    problems = {name: count for name, count in counts.items() if count and name != "off_calendar_rows"}
    (log.warning if problems else log.info)(
        "VIX: %d sessions, %s; %d values on exchange holidays dropped (Cboe's extended hours)",
        len(vix), problems or "no gaps", counts["off_calendar_rows"],
    )
    return 0


def cached_prices(symbol: str, path: Path, settings: DataSettings, refresh: bool,
                  **details) -> pd.DataFrame:
    """Raw Yahoo bars, downloaded only when not already cached."""
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frame = prices.download(
        symbol, settings.download_start,
        max_retries=settings.yahoo_max_retries, pause=settings.yahoo_pause_seconds,
    )
    store.write_parquet(
        path, frame, source="Yahoo Finance (yfinance)", symbol=symbol,
        requested_start=settings.download_start.isoformat(), **details,
    )
    time.sleep(settings.yahoo_pause_seconds)
    return frame


def align_and_check(name: str, raw: pd.DataFrame, path: Path, calendar: TradingCalendar,
                    config: StudyConfig, **details) -> prices.PriceCheck:
    aligned = prices.align(raw, calendar)
    store.write_parquet(path, aligned, series=name, **details)
    result = prices.check(raw, aligned, config.period.data_start, config.period.data_end, calendar)
    for flag in result.flags:
        log.warning("%s: %s", name, flag)
    return result


def write_checks(filename: str, checks: dict[str, prices.PriceCheck]) -> None:
    report = {name: dataclasses.asdict(result) for name, result in checks.items()}
    content = json.dumps(report, indent=2, default=str).encode()
    store.write_bytes(store.layer("reports", filename), content, kind="price checks")
