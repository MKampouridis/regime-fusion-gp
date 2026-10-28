"""Command line for the data pipeline: ``python -m regime_fusion.data <command>``.

Commands (each has --help):

  constituents  rebuild configs/constituents_2015-01.csv from pinned Wikipedia revisions
  prices        download Yahoo prices for the universe and align them to the calendar
  market        download the S&P 500 index, SPY and the VIX, and align them
  sec           download SEC company facts and filer details; keep report facts
  fundamentals  build daily point-in-time accounting values from the report facts
  draw          draw the main universe and the hold-out stocks (once; see draw.py)
  gdelt         stream GDELT's GKG files and keep hourly news sums for the universe
  fnspid        stream FNSPID's news files and keep the universe's headlines
  fa            ActTrade's twelve fundamental indicators, daily, from the steps above
  ta            ActTrade's twelve technical indicators, daily, from the prices
  regimes       market regimes: HMMs on the S&P 500 per walk-forward window (filtered)

Data go to the folder named by RFGP_DATA_DIR (see docs/DATA.md). Downloads are
cached: a raw file already there is not fetched again unless --refresh is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import io
import json
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import __version__
from regime_fusion.config import StudyConfig, load_config
from regime_fusion.data import (
    constituents, draw, fa, fnspid, fundamentals, gdelt, market, prices, sec, store, ta,
)
from regime_fusion.data.calendar import TradingCalendar, study_calendar
from regime_fusion.data.settings import DataSettings, load_settings
from regime_fusion.data.universe import (
    DEFAULT_UNIVERSE, Stock, load_acttrade, load_universe, write_universe,
)
from regime_fusion.seeds import make_rng

CONSTITUENTS_FILE = Path(__file__).resolve().parents[3] / "configs" / "constituents_2015-01.csv"
UNIVERSE_FILE = DEFAULT_UNIVERSE
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

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
    add_stock_options(sub, refresh=True)
    sub.set_defaults(run=run_prices)

    sub = commands.add_parser("market", help="download and align the S&P 500 index, SPY and the VIX")
    sub.add_argument("--refresh", action="store_true", help="download again even if cached")
    sub.set_defaults(run=run_market)

    sub = commands.add_parser("sec", help="download SEC company facts and keep report facts")
    add_stock_options(sub, refresh=True)
    sub.set_defaults(run=run_sec)

    sub = commands.add_parser("fundamentals", help="build daily point-in-time accounting values")
    add_stock_options(sub, refresh=False)
    sub.set_defaults(run=run_fundamentals)

    sub = commands.add_parser("draw", help="draw the main universe and the hold-out stocks")
    sub.add_argument("--redraw", action="store_true", help="draw again although a draw exists")
    sub.set_defaults(run=run_draw)

    sub = commands.add_parser("gdelt", help="stream GDELT's GKG files; keep hourly news sums")
    sub.add_argument("--start", default="2015-02-18", help="first UTC day (default: GDELT's first)")
    sub.add_argument("--end", help="last UTC day (default: the study's last day)")
    sub.add_argument("--workers", type=int, default=8,
                     help="days processed in parallel; 0 runs in this process")
    sub.add_argument("--min-free-gb", type=float, default=5.0,
                     help="stop when the data disk has less free space than this")
    sub.add_argument("--retry-failed", action="store_true", help="redo days with failed files")
    sub.set_defaults(run=run_gdelt)

    sub = commands.add_parser("fa", help="ActTrade's twelve fundamental indicators, daily")
    add_stock_options(sub, refresh=False)
    sub.set_defaults(run=run_fa)

    sub = commands.add_parser("ta", help="ActTrade's twelve technical indicators, daily")
    add_stock_options(sub, refresh=False)
    sub.set_defaults(run=run_ta)

    sub = commands.add_parser("regimes", help="market regimes per walk-forward window")
    sub.add_argument("--inputs", nargs="+", default=None, help="input sets (default: all in the settings)")
    sub.set_defaults(run=run_regimes)

    sub = commands.add_parser("fnspid", help="stream FNSPID's news files; keep the universe's headlines")
    sub.add_argument("--files", nargs="+", default=list(fnspid.FILES), help="files in the dataset")
    sub.add_argument("--refresh", action="store_true", help="filter again even if done")
    sub.set_defaults(run=run_fnspid)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.run(args) or 0


def add_stock_options(sub: argparse.ArgumentParser, *, refresh: bool) -> None:
    sub.add_argument("--stocks", nargs="+", metavar="ID",
                     help="stock ids (January 2015 tickers); default: every main and candidate stock")
    sub.add_argument("--include-holdout", action="store_true", help="also the hold-out stocks")
    if refresh:
        sub.add_argument("--refresh", action="store_true", help="download again even if cached")


def selected_stocks(args: argparse.Namespace) -> list[Stock]:
    """Stocks named on the command line, or every main and candidate stock; excluded ones are skipped."""
    universe = load_universe(UNIVERSE_FILE)
    if args.stocks:
        unknown = sorted(set(args.stocks) - {stock.id for stock in universe.stocks})
        if unknown:
            raise SystemExit(f"not in configs/universe.yaml: {unknown} (ids are January 2015 tickers)")
        stocks = [universe.get(stock_id) for stock_id in args.stocks]
        if any(stock.role == "holdout" for stock in stocks) and not args.include_holdout:
            raise SystemExit("hold-out stocks need --include-holdout")
    else:
        roles = ("main", "candidate") + (("holdout",) if args.include_holdout else ())
        stocks = universe.select(*roles, include_holdout=args.include_holdout)
    for stock in stocks:
        if stock.role == "excluded":
            log.info("%s: skipped, excluded (%s)", stock.id, stock.notes)
    return [stock for stock in stocks if stock.role != "excluded"]


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
    corrections = prices.load_corrections()
    checks, failed = {}, []
    for stock in selected_stocks(args):
        raw_path = store.layer("raw", "yahoo", "stocks", f"{stock.id}.parquet")
        try:
            raw = cached_prices(stock.yahoo, raw_path, settings, args.refresh, stock=stock.id)
        except prices.MissingPrices as problem:
            log.error("%s: %s", stock.id, problem)
            failed.append(stock.id)
            continue
        fixes = [c for c in corrections if c.stock == stock.id]
        checks[stock.id] = align_and_check(
            stock.id, raw, store.layer("interim", "prices", f"{stock.id}.parquet"),
            calendar, config, symbol=stock.yahoo,
            fixes=[(c, children_for(c, settings, calendar, args.refresh)) for c in fixes],
        )
    write_report("price_checks.json", checks)
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
    write_report("market_checks.json", checks)

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


def run_sec(args: argparse.Namespace) -> int:
    settings, config = load_settings(), load_config()
    calendar = study_calendar(settings)
    client = sec.Client(settings.sec_user_agent, settings.sec_max_requests_per_second)
    years = range(settings.download_start.year, config.period.data_end.year + 1)
    checks, failed = {}, []
    for stock in selected_stocks(args):
        frames, names = [], []
        for cik in stock.ciks:
            name = f"CIK{cik:010d}.json.gz"
            try:
                facts = cached_edgar(client, sec.COMPANYFACTS_URL.format(cik=cik),
                                     store.layer("raw", "sec", "companyfacts", name), args.refresh)
                filer = cached_edgar(client, sec.SUBMISSIONS_URL.format(cik=cik),
                                     store.layer("raw", "sec", "submissions", name), args.refresh)
            except sec.NotFound as missing:
                log.warning("%s: CIK %d not found on EDGAR (%s)", stock.id, cik, missing)
                continue
            frames.append(sec.report_facts(facts))
            names += sec.filer_names(filer)
        if not frames:
            log.error("%s: no company facts for any of its CIKs %s", stock.id, list(stock.ciks))
            failed.append(stock.id)
            continue
        table = pd.concat(frames, ignore_index=True)
        table["available"] = calendar.session_after_date(table["filed"])
        store.write_parquet(store.layer("interim", "sec", f"{stock.id}.parquet"), table,
                            stock=stock.id, ciks=list(stock.ciks), forms=list(sec.FORMS))
        checks[stock.id] = sec.check_filings(stock.name, table, names, years)
        for flag in checks[stock.id].flags:
            log.warning("%s: %s", stock.id, flag)
    write_report("sec_checks.json", checks)
    flagged = sorted(name for name, result in checks.items() if result.flags)
    log.info("sec: %d stocks; %d flagged %s; %d failed %s",
             len(checks), len(flagged), flagged, len(failed), failed)
    return 1 if failed else 0


def run_fundamentals(args: argparse.Namespace) -> int:
    settings, config = load_settings(), load_config()
    calendar = study_calendar(settings)
    concepts = fundamentals.load_concepts()
    study = calendar.between(config.period.data_start, config.period.data_end)
    report, missing = {}, []
    for stock in selected_stocks(args):
        source = store.layer("interim", "sec", f"{stock.id}.parquet")
        if not source.exists():
            log.error("%s: no report facts; run the sec command first", stock.id)
            missing.append(stock.id)
            continue
        panel = fundamentals.daily_panel(pd.read_parquet(source), concepts, calendar.sessions)
        store.write_parquet(store.layer("interim", "fundamentals", f"{stock.id}.parquet"), panel,
                            stock=stock.id, concepts=[c.name for c in concepts])
        summary, flags = fundamentals.coverage(panel, concepts, study)
        report[stock.id] = {"concepts": summary, "flags": flags}
        for flag in flags:
            log.warning("%s: %s", stock.id, flag)
    write_report("fundamentals_coverage.json", report)
    flagged = sorted(name for name, entry in report.items() if entry["flags"])
    log.info("fundamentals: %d stocks; %d flagged %s", len(report), len(flagged), flagged)
    return 1 if missing else 0


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


def cached_edgar(client: sec.Client, url: str, path: Path, refresh: bool) -> dict[str, Any]:
    """An EDGAR JSON document, downloaded only when not already cached."""
    if refresh or not path.exists():
        store.write_bytes(path, sec.compress(client.get(url)), source="SEC EDGAR", url=url)
    return sec.load_json(path)


def align_and_check(name: str, raw: pd.DataFrame, path: Path, calendar: TradingCalendar,
                    config: StudyConfig, fixes=(), **details) -> prices.PriceCheck:
    aligned = prices.align(raw, calendar)
    applied = []
    for correction, children in fixes:
        true_return = prices.corrected_return(aligned, children, correction)
        day = correction.date
        before = aligned.index[aligned.index.get_loc(day) - 1]
        yahoo_return = aligned.loc[day, "adj_close"] / aligned.loc[before, "adj_close"] - 1.0
        aligned = prices.apply_correction(aligned, day, true_return)
        applied.append({"date": day.date().isoformat(), "event": correction.event,
                        "yahoo_return": round(float(yahoo_return), 4),
                        "corrected_return": round(true_return, 4)})
        log.info("%s %s: return %+.4f corrected to %+.4f (%s)", name, day.date(),
                 yahoo_return, true_return, correction.event)
    store.write_parquet(path, aligned, series=name, corrections=applied, **details)
    result = prices.check(raw, aligned, config.period.data_start, config.period.data_end, calendar)
    for flag in result.flags:
        log.warning("%s: %s", name, flag)
    return result


def children_for(correction: prices.Correction, settings: DataSettings,
                 calendar: TradingCalendar, refresh: bool) -> dict[str, pd.DataFrame]:
    """Aligned prices of the firms spun off in a correction."""
    children = {}
    for symbol in correction.received:
        path = store.layer("raw", "yahoo", "corrections", f"{symbol}.parquet")
        raw = cached_prices(symbol, path, settings, refresh, purpose=f"correction for {correction.stock}")
        children[symbol] = prices.align(raw, calendar)
    return children


def write_report(filename: str, entries: dict[str, Any], *, merge: bool = True) -> None:
    """Write a report; by default entries from earlier runs for other names are kept."""
    path = store.layer("reports", filename)
    report = json.loads(path.read_text()) if merge and path.exists() else {}
    report.update({
        name: dataclasses.asdict(entry) if dataclasses.is_dataclass(entry) else entry
        for name, entry in entries.items()
    })
    content = json.dumps(dict(sorted(report.items())), indent=2, default=str).encode()
    store.write_bytes(path, content, kind=filename.removesuffix(".json"))


UNIVERSE_HEADER = """\
Stock universe, written by `python -m regime_fusion.data draw` (src/regime_fusion/data/draw.py);
field meanings are in src/regime_fusion/data/universe.py.

main      ActTrade's usable US stocks (configs/acttrade.yaml) and a seeded draw from the other
          usable January 2015 S&P 100 members, topping the sector mix up to the S&P 100's
holdout   a seeded draw from the January 2015 S&P 500 members outside the S&P 100, used
          only in the final campaign
unused    eligible S&P 100 members that were not drawn
excluded  unusable firms, with the reason in notes

Names, sectors and CIKs as listed in January 2015 (configs/constituents_2015-01.csv). Later
corporate events of the S&P 100 members were checked against SEC EDGAR and Yahoo Finance
in October 2026; the downloads check every entry again (docs/DATA.md).
"""


def run_draw(args: argparse.Namespace) -> int:
    settings, config = load_settings(), load_config()
    calendar = study_calendar(settings)
    start = calendar.between(config.period.data_start, config.period.data_end)[0]
    universe = load_universe(UNIVERSE_FILE)
    if any(stock.role in ("main", "holdout") for stock in universe.stocks) and not args.redraw:
        raise SystemExit("the universe has been drawn already; pass --redraw to draw it again")
    table = pd.read_csv(CONSTITUENTS_FILE)
    rows = table.set_index("ticker")
    existing = {stock.id: stock for stock in universe.stocks}
    client = sec.Client(settings.sec_user_agent, settings.sec_max_requests_per_second)
    tickers_by_cik, cik_by_ticker = current_sec_tickers(client)
    checked: dict[str, bool] = {}

    def has_prices(symbol: str) -> bool:
        """Whether Yahoo has prices for a symbol from the study's first session."""
        if symbol not in checked:
            path = store.layer("raw", "yahoo", "draw", f"{symbol}.parquet")
            try:
                raw = cached_prices(symbol, path, settings, False, purpose="universe draw")
                checked[symbol] = prices.first_bar(raw) <= start
            except prices.MissingPrices:
                checked[symbol] = False
        return checked[symbol]

    # ActTrade's US stocks open the main universe.
    acttrade = [acttrade_stock(entry, rows, existing) for entry in load_acttrade()]
    fixed, final = {}, {}
    for stock in acttrade:
        if stock.yahoo and has_prices(stock.yahoo):
            fixed[stock.id] = stock.sector
            final[stock.id] = dataclasses.replace(stock, role="main")
        else:
            final[stock.id] = dataclasses.replace(stock, role="excluded", notes=stock.notes or
                                                  "Yahoo has no prices from the study start.")
    taken_ciks = {cik for stock in acttrade for cik in stock.ciks}

    # The other usable S&P 100 members top the sector mix up.
    members = table[table["sp100"]]
    sp100 = [existing[i] for i in sorted(members["ticker"])
             if i in existing and i not in final and not taken_ciks & set(existing[i].ciks)]
    candidates = {s.id: s.sector for s in sp100 if s.role != "excluded"}
    shares = (members["sector"].value_counts() / len(members)).to_dict()
    drawn, skipped = draw.draw_main(
        candidates, fixed, shares, draw.MAIN_SIZE, make_rng(config.base_seed, "universe", "main"),
        usable=lambda stock_id: has_prices(existing[stock_id].yahoo),
    )
    for stock in sp100:
        role = "main" if stock.id in drawn else "excluded" if stock.role == "excluded" else "unused"
        notes = stock.notes or ("No Yahoo prices from the study start." if stock.id in skipped else "")
        final[stock.id] = dataclasses.replace(stock, role="excluded" if stock.id in skipped else role,
                                              acttrade=False, notes=notes)

    # Hold-out stocks: the other January 2015 S&P 500 members, in a random order.
    taken_ciks |= {cik for stock in final.values() for cik in stock.ciks}
    pool = table[~table["sp100"] & ~table["cik"].isin(list(taken_ciks)) & ~table["ticker"].isin(list(final))]
    pool = pool.sort_values("ticker").drop_duplicates("cik")
    order = draw.shuffled(pool["ticker"], make_rng(config.base_seed, "universe", "holdout"))
    firms_2015 = dict(zip(table["cik"].astype(int), table["ticker"]))
    report = {"main": {"acttrade": sorted(fixed), "drawn": sorted(drawn), "skipped": skipped,
                       "sector_shares": shares},
              "holdout": {"order": order, "accepted": [], "skipped": {}}}
    for ticker in order:
        if len(report["holdout"]["accepted"]) == draw.HOLDOUT_SIZE:
            break
        row = rows.loc[ticker]
        symbol, ciks, how = resolve_holdout(ticker, row, tickers_by_cik, cik_by_ticker, has_prices,
                                            firms_2015)
        stock = Stock(id=ticker, name=str(row["name"]), sector=str(row["sector"]), ciks=ciks,
                      yahoo=symbol, role="holdout" if symbol else "excluded", acttrade=False,
                      notes=how)
        final[ticker] = stock
        if symbol:
            report["holdout"]["accepted"].append(ticker)
        else:
            report["holdout"]["skipped"][ticker] = how
    if len(report["holdout"]["accepted"]) < draw.HOLDOUT_SIZE:
        raise SystemExit("not enough usable hold-out stocks")

    rank = {"main": 0, "holdout": 1, "unused": 2, "excluded": 3}
    stocks = sorted(final.values(), key=lambda s: (rank[s.role], s.id))
    write_universe(UNIVERSE_FILE, stocks, UNIVERSE_HEADER)
    write_report("universe_draw.json", report, merge=False)
    counts = {role: sum(s.role == role for s in stocks) for role in rank}
    log.info("universe drawn: %s; written to %s", counts, UNIVERSE_FILE)
    return 0


def acttrade_stock(entry: dict[str, Any], rows: pd.DataFrame, existing: dict[str, Stock]) -> Stock:
    """One of ActTrade's stocks, from its entry, the S&P 100 universe or the constituents list."""
    known = existing.get(entry["id"])
    row = rows.loc[entry["id"]] if entry["id"] in rows.index else None

    def pick(field: str, from_row: str):
        if field in entry:
            return entry[field]
        if known is not None:
            return getattr(known, field)
        if row is not None and from_row:
            return row[from_row]
        raise ValueError(f"ActTrade stock {entry['id']}: no {field} given")

    ciks = entry.get("ciks") or (list(known.ciks) if known else [int(row["cik"])])
    yahoo = entry["yahoo"] if "yahoo" in entry else (known.yahoo if known else entry["id"].replace(".", "-"))
    earlier = known.notes if known and not known.acttrade else ""  # not from an earlier draw
    notes = " ".join(n for n in (entry.get("notes", ""), earlier) if n)
    return Stock(id=entry["id"], name=str(pick("name", "name")), sector=str(pick("sector", "sector")),
                 ciks=tuple(int(c) for c in ciks), yahoo=yahoo, role="main", acttrade=True,
                 notes=f"ActTrade label {entry['label']}. {notes}".strip())


def resolve_holdout(ticker: str, row: pd.Series, tickers_by_cik: dict[int, list[str]],
                    cik_by_ticker: dict[str, int], has_prices,
                    firms_2015: dict[int, str]) -> tuple[str | None, tuple[int, ...], str]:
    """Yahoo symbol, CIKs and an explanation for a firm drawn into the hold-out set.

    A firm whose January 2015 CIK no longer files keeps its old ticker only if Yahoo's
    name for it matches and the ticker has not passed to another January 2015 firm
    (ACE took Chubb's ticker, CB, when it bought Chubb in 2016).
    """
    cik = int(row["cik"])
    current = tickers_by_cik.get(cik, [])
    if current:
        for symbol in draw.ticker_candidates(ticker, current):
            if has_prices(symbol):
                return symbol, (cik,), f"Yahoo symbol {symbol}: SEC's current ticker for the January 2015 CIK."
        return None, (cik,), "No Yahoo prices from the study start under the CIK's current tickers."
    symbol = ticker.replace(".", "-")
    if not has_prices(symbol):
        return None, (cik,), "The January 2015 CIK no longer files; Yahoo has no prices under the January 2015 ticker."
    new = cik_by_ticker.get(symbol)
    if new and new != cik and new in firms_2015:
        return None, (cik,), (f"The January 2015 ticker now belongs to another January 2015 firm "
                              f"({firms_2015[new]}, CIK {new}).")
    name = prices.yahoo_name(symbol)
    if name and sec.names_match(str(row["name"]), [name]):
        ciks = (cik,) + ((new,) if new and new != cik else ())
        return symbol, ciks, f"January 2015 ticker; Yahoo's name {name!r} matches."
    return None, (cik,), f"The January 2015 ticker is now {name!r} on Yahoo (the CIK no longer files)."


def current_sec_tickers(client: sec.Client) -> tuple[dict[int, list[str]], dict[str, int]]:
    """Today's SEC tickers by CIK, and CIK by ticker (Yahoo spelling: '-' for '.')."""
    path = store.layer("raw", "sec", "company_tickers.json.gz")
    data = cached_edgar(client, SEC_TICKERS_URL, path, refresh=False)
    by_cik: dict[int, list[str]] = {}
    by_ticker: dict[str, int] = {}
    for entry in data.values():
        symbol = str(entry["ticker"]).replace(".", "-")
        by_cik.setdefault(int(entry["cik_str"]), []).append(symbol)
        by_ticker[symbol] = int(entry["cik_str"])
    return by_cik, by_ticker


def run_gdelt(args: argparse.Namespace) -> int:
    config = load_config()
    universe = load_universe(UNIVERSE_FILE)
    stocks = [s.id for s in universe.select("main", "holdout", include_holdout=True)]
    matcher = gdelt.load_matcher(stocks=stocks)
    first = args.start.replace("-", "")
    last = (args.end or config.period.data_end.isoformat()).replace("-", "")
    files = gdelt.master_list(gdelt_master_text(), first, last)
    done = {day for day in files["day"].unique() if gdelt_day_done(day, args.retry_failed)}
    todo = files[~files["day"].isin(done)]
    total_bytes = int(todo["size"].sum())
    log.info("gdelt: %d days to do (%d done), %d files, %.0f GB to stream; anchors %s",
             todo["day"].nunique(), len(done), len(todo), total_bytes / 1e9, matcher.digest)
    jobs = [(day, list(zip(group["url"], group["md5"]))) for day, group in todo.groupby("day")]
    started, streamed, finished = time.monotonic(), 0, 0
    min_free = args.min_free_gb * 1e9

    def save(day: str, hourly: pd.DataFrame, titled: pd.DataFrame, summary: dict) -> None:
        nonlocal streamed, finished
        store.write_parquet(gdelt_path(day, "titles"), titled, **summary)
        store.write_parquet(gdelt_path(day, "hourly"), hourly, **summary)  # written last: marks the day done
        streamed += summary["bytes"]
        finished += 1
        rate = streamed / max(time.monotonic() - started, 1e-9)
        left = (total_bytes - streamed) / rate / 3600 if rate else float("nan")
        (log.warning if summary["failed"] else log.info)(
            "%s: %d records, %d hourly rows, %d titled, %d failed files; %d/%d days; "
            "%.1f MB/s; about %.1f hours left", day, summary["records_used"], summary["hourly_rows"],
            summary["titled_records"], len(summary["failed"]), finished, len(jobs), rate / 1e6, left)

    def disk_full() -> bool:
        free = shutil.disk_usage(store.data_dir()).free
        if free < min_free:
            log.error("gdelt: stopping, only %.1f GB free on the data disk (minimum %.1f GB); "
                      "run the same command again to resume", free / 1e9, args.min_free_gb)
            return True
        return False

    if args.workers == 0:
        for day, day_files in jobs:
            if disk_full():
                return 1
            save(*gdelt.process_day(day, day_files, matcher))
        return 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(gdelt.process_day, day, day_files, matcher) for day, day_files in jobs]
        try:
            for future in as_completed(futures):
                if disk_full():
                    pool.shutdown(cancel_futures=True)
                    return 1
                save(*future.result())
        except KeyboardInterrupt:
            log.warning("gdelt: interrupted; finished days are kept, run again to resume")
            pool.shutdown(cancel_futures=True)
            return 1
    return 0


def gdelt_path(day: str, kind: str) -> Path:
    return store.layer("raw", "gdelt", "gkg", day[:4], f"{day}.{kind}.parquet")


def gdelt_day_done(day: str, retry_failed: bool) -> bool:
    path = gdelt_path(day, "hourly")
    if not path.exists():
        return False
    return not (retry_failed and json.loads(store.sidecar(path).read_text()).get("failed"))


def gdelt_master_text() -> str:
    """GDELT's master file list, downloaded once and kept compressed."""
    path = store.layer("raw", "gdelt", "masterfilelist.txt.gz")
    if not path.exists():
        import requests

        response = requests.get(gdelt.MASTER_URL, headers={"User-Agent": user_agent()}, timeout=600)
        response.raise_for_status()
        store.write_bytes(path, gzip.compress(response.content, mtime=0), source="GDELT",
                          url=gdelt.MASTER_URL)
    return gzip.decompress(path.read_bytes()).decode()


def run_fnspid(args: argparse.Namespace) -> int:
    import requests

    universe = load_universe(UNIVERSE_FILE)
    tickers = fnspid.ticker_map(universe.select("main", "holdout", include_holdout=True))
    session = requests.Session()
    session.headers["User-Agent"] = user_agent()
    for name in args.files:
        path = store.layer("raw", "fnspid", Path(name).name.replace(".csv", ".parquet"))
        if path.exists() and not args.refresh:
            log.info("fnspid: %s already filtered", name)
            continue
        url = fnspid.BASE_URL + name
        log.info("fnspid: streaming %s", url)
        stream = io.BufferedReader(fnspid.ResumableStream(url, session), buffer_size=1 << 20)
        frame, seen = fnspid.filter_csv(stream, tickers)
        store.write_parquet(path, frame, source="FNSPID (Hugging Face, Zihan1004/FNSPID)", url=url,
                            licence="CC BY-NC 4.0", rows_read=seen, tickers=sorted(tickers))
        log.info("fnspid: %s: kept %d of %d rows", name, len(frame), seen)
    return 0


def run_fa(args: argparse.Namespace) -> int:
    config = load_config()
    corrections = prices.load_corrections()
    calendar = study_calendar()
    study = calendar.between(config.period.data_start, config.period.data_end)
    report, missing = {}, []
    for stock in selected_stocks(args):
        panel_path = store.layer("interim", "fundamentals", f"{stock.id}.parquet")
        price_path = store.layer("interim", "prices", f"{stock.id}.parquet")
        if not (panel_path.exists() and price_path.exists()):
            log.error("%s: run the prices and fundamentals commands first", stock.id)
            missing.append(stock.id)
            continue
        panel, aligned = pd.read_parquet(panel_path), pd.read_parquet(price_path)
        splits = fa.true_splits(aligned["split"], [c for c in corrections if c.stock == stock.id])
        table = fa.indicators(panel, aligned["traded_close"], splits)
        store.write_parquet(store.layer("processed", "fa", f"{stock.id}.parquet"), table,
                            stock=stock.id, splits={str(d.date()): r for d, r in splits.items()})
        share = table.reindex(study)[list(fa.INDICATORS)].notna().mean().round(4).to_dict()
        report[stock.id] = {"share_of_days": share,
                            "missing": sorted(k for k, v in share.items() if v < 0.9)}
    write_report("fa_coverage.json", report)
    log.info("fa: %d stocks; indicators on fewer than 90%% of days: %s", len(report),
             {k: v["missing"] for k, v in report.items() if v["missing"]})
    return 1 if missing else 0


def run_ta(args: argparse.Namespace) -> int:
    config = load_config()
    study = study_calendar().between(config.period.data_start, config.period.data_end)
    report, missing = {}, []
    for stock in selected_stocks(args):
        source = store.layer("interim", "prices", f"{stock.id}.parquet")
        if not source.exists():
            log.error("%s: run the prices command first", stock.id)
            missing.append(stock.id)
            continue
        table = ta.indicators(pd.read_parquet(source))
        store.write_parquet(store.layer("processed", "ta", f"{stock.id}.parquet"), table, stock=stock.id)
        share = table.reindex(study).notna().mean().round(4).to_dict()
        report[stock.id] = {"share_of_days": share,
                            "missing": sorted(k for k, v in share.items() if v < 0.99)}
    write_report("ta_coverage.json", report)
    log.info("ta: %d stocks; indicators on fewer than 99%% of days: %s", len(report),
             {k: v["missing"] for k, v in report.items() if v["missing"]})
    return 1 if missing else 0


def run_regimes(args: argparse.Namespace) -> int:
    from regime_fusion.regimes import market as regimes
    from regime_fusion.windows import walk_forward

    config = load_config()
    settings = regimes.load_settings()
    source = store.layer("interim", "market", f"{settings['series']}.parquet")
    close = pd.read_parquet(source)["close"]
    report = {}
    for name in args.inputs or list(settings["inputs"]):
        features = regimes.inputs(close, name, settings)
        for fit in regimes.walk_forward(features, walk_forward(config), settings, config.base_seed):
            path = store.layer("processed", "regimes", name, f"{fit.test_year}.stage{fit.stage}.parquet")
            store.write_parquet(path, fit.probabilities, inputs=name, test_year=fit.test_year,
                                stage=fit.stage, fitted_on=[str(fit.fitted_on.start), str(fit.fitted_on.end)],
                                settings=settings)
            model = fit.model
            fitted = fit.probabilities.loc[str(fit.fitted_on.start):str(fit.fitted_on.end)]
            report[f"{name} {fit.test_year} stage {fit.stage}"] = {
                "fitted_on": [str(fit.fitted_on.start), str(fit.fitted_on.end)],
                "means": model.means.round(4).tolist(),
                "sds": np.sqrt(model.variances).round(4).tolist(),
                "expected_duration_days": model.expected_durations().round(1).tolist(),
                "share_of_fit_days": fitted.idxmax(axis=1).value_counts(normalize=True).round(3).to_dict(),
                "log_likelihood": round(model.log_likelihood, 2),
            }
        log.info("regimes: %s fitted for %d test years", name, len(config.period.test_years))
    write_report("regimes.json", report, merge=False)
    return 0
