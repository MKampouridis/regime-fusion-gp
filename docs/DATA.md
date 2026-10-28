# Data pipeline

How the data for the study are obtained, stored and aligned. The sources and the point-in-time rules are set in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md); this file records how the code applies them. Open questions are listed in [`DECISIONS.md`](DECISIONS.md).

## Where data live

Nothing under the data folder is committed. The folder is named by the environment variable `RFGP_DATA_DIR` (default `./data` in the repository, which git ignores). The repository sits in a Box-synchronised folder, so `RFGP_DATA_DIR` should point outside it: Box would otherwise upload gigabytes of licensed data.

| Layer | Contents |
| --- | --- |
| `raw/` | Files exactly as downloaded, never edited |
| `interim/` | Cleaned tables aligned to the trading calendar |
| `processed/` | Model-ready daily panels |
| `reports/` | Coverage and quality reports |

Every file is written atomically and gets a provenance sidecar, `<name>.provenance.json`, with its SHA-256 checksum, the code version (git commit), package versions, the time of writing and how the file was obtained (`regime_fusion.data.store`).

Download settings live in [`configs/data_sources.yaml`](../configs/data_sources.yaml). Downloads start in January 2014, a year before the study period, so that indicators with long look-backs are defined from February 2015. The earlier data serve only as this warm-up.

## When information may be used

The clock is the New York Stock Exchange (calendar `XNYS` of the `exchange_calendars` package, which includes holidays, the closures of 5 December 2018 and 9 January 2025, and the 13:00 early closes). A value used on session *t* must be available by the close of *t*; trades execute on *t*+1. `regime_fusion.data.calendar.TradingCalendar` applies two rules:

| Item | First session on which it may be used | Example |
| --- | --- | --- |
| Has a timestamp (news) | The first session whose close is strictly later than the timestamp | 15:59 New York time counts for that day; 16:00 or later counts for the next session; on an early-close day the cut-off is 13:00 |
| Has a date but no time (SEC filing date, news stamped only with a date) | The first session strictly after the date | A 10-Q filed on Monday is first used on Tuesday |

Feature code built on daily rows is tested with `regime_fusion.testing.check_no_lookahead`. Daily tables built from dated items (news, filed values) are tested with `regime_fusion.testing.check_point_in_time`: withholding every item not yet available on day *t* must leave row *t* unchanged.

## Universe

The universe is fixed as of January 2015, so firms later acquired or delisted stay in it.

- [`configs/constituents_2015-01.csv`](../configs/constituents_2015-01.csv) lists the S&P 500 members at the end of January 2015 (502 share classes of 500 firms), with GICS sector and sub-industry, SEC CIK, and S&P 100 membership. It is rebuilt from pinned Wikipedia revisions with `python -m regime_fusion.data constituents`: the S&P 100 page as of 16 January 2015 (revision 642684730) and the S&P 500 list as of 27 January 2015 (revision 644366768). Wikipedia is a secondary source (CC BY-SA); an official list would be a welcome check.
- [`configs/universe.yaml`](../configs/universe.yaml) holds the stocks of the study. It currently lists the 100 S&P 100 members as candidates. Eight are excluded because Yahoo Finance no longer has their prices: Anadarko, Dow Chemical, EMC, Twenty-First Century Fox, Monsanto, Raytheon, Time Warner and Walgreens Boots Alliance. This is the survivorship bias of free price data that the paper states.
- Each stock keeps its January 2015 ticker as its identifier, in file names and seeds. The entry records every CIK that filed its accounts from 2014 to 2025 (six firms changed CIK through holding-company reorganisations, e.g. Google into Alphabet) and the Yahoo symbol of its price history today (FB is now META, UTX is RTX, BK is BNY).
- Tickers are reused: Yahoo's EMC, FOXA and DOW are now other securities. Price downloads are therefore checked to cover the study period from its start.
- Hold-out stocks are returned only by `Universe.select("holdout", include_holdout=True)`, which only the downloads and the final campaign use.
