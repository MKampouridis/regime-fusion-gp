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
