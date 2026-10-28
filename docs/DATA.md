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
- Each stock keeps its January 2015 ticker as its identifier, in file names and seeds. The entry records every CIK that filed its accounts from 2014 to 2025 (six firms have two, mostly after holding-company reorganisations such as Google into Alphabet) and the Yahoo symbol of its price history today (FB is now META, UTX is RTX, BK is BNY).
- Tickers are reused: Yahoo's EMC, FOXA and DOW are now other securities. Price downloads are therefore checked to cover the study period from its start.
- Hold-out stocks are returned only by `Universe.select("holdout", include_holdout=True)`, which only the downloads and the final campaign use.

## Prices

`python -m regime_fusion.data prices` downloads daily bars from Yahoo Finance for every main and candidate stock (`--include-holdout` adds the hold-out stocks) into `raw/yahoo/stocks/<id>.parquet`. It then writes one row per session to `interim/prices/<id>.parquet` and the checks to `reports/price_checks.json`.

- **Raw files** keep the output of `yfinance` (`auto_adjust=False, actions=True`) from January 2014 to the download date. Yahoo adjusts the whole history for later splits, so splits after 2025 are needed to recover traded prices.
- **Interim tables** hold split-adjusted open, high, low and close (as Yahoo gives them), total-return prices (`adj_*`) from which returns are computed, volume, dividends, splits, and `traded_close`, the price actually paid that day. Valuation ratios need `traded_close`, because filed per-share values are not split-adjusted. Apple's traded close on 28 August 2020 comes out as $499.23, its actual close before the 4-for-1 split.
- **Spin-offs.** Yahoo records some spin-offs as fractional splits (eBay 2.376 for PayPal in July 2015, HP 2.202 for Hewlett Packard Enterprise in November 2015, RTX 1.589 in April 2020), so total-return prices do not drop on those days.
- **Scale.** Yahoo's adjustments rescale the past by later events. Ratios within a look-back window are unaffected, so technical indicators must be scale-invariant (returns, price ratios, relative DC thresholds) and never use price levels.
- **Checks.** Coverage of the study period from its first session; sessions without a bar (kept as missing rows, never filled); bars on days that are not sessions; impossible bars (non-positive prices, a close outside the day's range); zero volume; runs of five or more unchanged closes. Daily moves above 15% are listed for a one-off hand check. Most are genuine (earnings, March 2020), but a spin-off that Yahoo does not adjust would also show up there. In a ten-stock trial every such move was genuine except possibly DuPont's +17.8% on 3 June 2019, the day Corteva was spun off.

## Market series

`python -m regime_fusion.data market` downloads the S&P 500 index (`^GSPC`, a price index, for market regimes) and SPY (whose adjusted close is a total-return benchmark) from Yahoo and processes them like stocks. The VIX comes from Cboe's own history file (`raw/cboe/VIX_History.csv`). Since mid-2022 that file also has values on US exchange holidays, from Cboe's extended trading hours; alignment drops them.

The VIX is calculated until 16:15 New York time, a quarter of an hour after the stock close, so its close on day *t* is not strictly available by the close of *t*. The interim table keeps each value on its own day; whether features use the close of *t* or of *t*−1 is an open question.

## Fundamentals

`python -m regime_fusion.data sec` downloads, for every CIK of each stock, the XBRL company facts and the filer details from SEC EDGAR (`raw/sec/companyfacts/`, `raw/sec/submissions/`, gzip-compressed JSON). It keeps the facts from 10-K and 10-Q reports and their amendments (`interim/sec/<id>.parquet`), and checks the SEC filer names against the universe and the number of reports per year (`reports/sec_checks.json`). Requests name the contact in `configs/data_sources.yaml` and are paced at five a second (the SEC allows ten).

`python -m regime_fusion.data fundamentals` turns the facts into daily point-in-time values (`interim/fundamentals/<id>.parquet`) for the concepts in [`configs/fundamentals.yaml`](../configs/fundamentals.yaml), and reports their coverage (`reports/fundamentals_coverage.json`).

- **Availability.** A fact is used from the first session after its filing date. A period's value on day *t* comes from the latest filing available by *t*, so a restatement replaces the original only once it is filed. Apple's 2008 net income, for example, was first filed as $4.83bn and restated to $6.12bn in January 2010.
- **Balance-sheet items** take the value at the latest period end known. **Income and cash-flow items** take the trailing twelve months (TTM), and the TTM a year earlier for growth rates. 10-Qs report years to date, so TTM = previous fiscal year + year to date − the same span a year earlier. Fiscal years of 52 or 53 weeks and 12-week quarters (PepsiCo, Costco) are handled.
- **Earnings releases** (8-K facts) are dropped, because few firms file them in XBRL. The data are therefore conservative: the market usually learns earnings from the press release a few weeks before the 10-Q.
- **Period ends.** Each value comes with the end of the period it describes (`<concept>_end`), so the feature layer can drop stale values, such as those left by a tag a firm stopped using.
- **Check on real filings.** For eight firms with different fiscal calendars (Microsoft, Walmart, PepsiCo, Costco, Alphabet, Medtronic, Disney, JPMorgan), FY2019 revenue and net income in the panel equal the published figures, each first used on the session after its filing.
- **Draft concepts.** The concept list is a draft until it is matched to ActTrade's fundamental indicators. The coverage report shows what remains to be done: banks have no operating income or current assets; several firms (Walmart, Disney) never report total liabilities, so it must be derived; some tags fall out of use.
- **Valuation ratios.** Filed per-share values and share counts are not split-adjusted, so they must be combined with `traded_close`. Yahoo records some spin-offs as splits, so its split column cannot be used to adjust share counts.

## News

Sentiment and news attention come from GDELT (main source) and FNSPID (robustness check). The timing rules are in `regime_fusion.data.news`. The downloads are not built yet; they wait on the open questions about GDELT access and the universe.

- **Timing.** A GDELT record counts from the time of the 15-minute update that first carried it, which is never before publication. An FNSPID item counts from its publication time. Both go to the first session whose close is strictly later. FNSPID items stamped exactly at midnight UTC carry only a date, so they count for the first session after it. News sums are kept per UTC hour; the NYSE closes on the hour, so this loses nothing at the cut-off.
- **GDELT cost.** The Loughran–McDonald word counts are in the GKG's GCAM field, about 8.5 KB per record, and GDELT has roughly 200,000–320,000 records a day. In BigQuery, one pass over 2015–2025 that reads GCAM would process about 8–11 TB: eight to eleven months of the sandbox's free 1 TB a month. Organisations, tone, URL and date alone come to about 0.3 TB. The raw GKG files from 18 February 2015 (the first) to 31 December 2025 are 373,743 files, 2.75 TB compressed. Streaming them and keeping only records that name our firms needs no quota and almost no disk, and takes one to two days of downloading.
- **Company matching.** GDELT lists organisations in lower case without punctuation ("jpmorgan chase co", "exxonmobil corp", "alphabet inc"), with noise ("ford foundation", "apple certified refurbished"). Each firm needs a curated list of name variants matched against whole entries, and hand-checked samples. Many records name a firm only in passing, such as JPMorgan as the analyst in a story about another company. The position of the first mention (`V2Organizations` has offsets) or the page title can separate articles about a firm from passing mentions.
- **Page titles.** GDELT records carry the page title (in the `Extras` field) in March 2024 but not in March 2015; when titles start is still to be checked. If early enough, they could supply the 2024–25 headlines for the FinBERT check.
- **FNSPID** is on Hugging Face as two files, `All_external.csv` (5.7 GB) and `nasdaq_exteral_data.csv` (23 GB, with article text). Both are larger than the free disk on Michael's Mac, so they will be streamed and filtered to our tickers.
