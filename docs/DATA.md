# Data pipeline

How the data for the study are obtained, stored and aligned. The sources and the point-in-time rules are set in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md); this file records how the code applies them. Open questions are listed in [`DECISIONS.md`](DECISIONS.md).

## Where data live

Nothing under the data folder is committed. The folder is named by the environment variable `RFGP_DATA_DIR` (default `./data` in the repository, which git ignores). On Michael's Mac it is `~/rfgp-data`, outside the Box-synchronised folder that holds the repository: Box would otherwise upload gigabytes of licensed data. The Python environment is `~/.venvs/rfgp`.

| Layer | Contents |
| --- | --- |
| `raw/` | Files exactly as downloaded, never edited |
| `interim/` | Cleaned tables aligned to the trading calendar |
| `processed/` | Model-ready daily panels |
| `reports/` | Coverage and quality reports |

Every file is written atomically and gets a provenance sidecar, `<name>.provenance.json`, with its SHA-256 checksum, the code version (git commit), package versions, the time of writing and how the file was obtained (`regime_fusion.data.store`).

Download settings live in [`configs/data_sources.yaml`](../configs/data_sources.yaml). Downloads start in January 2014, a year before the study period, so that indicators with long look-backs are defined from February 2015. The earlier data serve only as this warm-up.

## Running the pipeline

Run on a machine that can reach the sources (Michael's Mac), with the `download` extra installed (README):

```bash
source ~/.venvs/rfgp/bin/activate
export RFGP_DATA_DIR=~/rfgp-data                          # outside Box
python -m regime_fusion.data prices --include-holdout     # Yahoo prices, about 2 minutes
python -m regime_fusion.data market                       # S&P 500 index, SPY, VIX
python -m regime_fusion.data sec --include-holdout        # SEC company facts, about 2 minutes
python -m regime_fusion.data fundamentals --include-holdout  # point-in-time accounting values
python -m regime_fusion.data fa --include-holdout         # ActTrade's twelve fundamental indicators
python -m regime_fusion.data ta --include-holdout         # ActTrade's twelve technical indicators
python -m regime_fusion.data fnspid                       # FNSPID headlines, about an hour of streaming
python -m regime_fusion.data gdelt --workers 12           # GDELT news, about 3 days of streaming
python -m regime_fusion.data sa --include-holdout         # twelve sentiment indicators from the GDELT news
python -m regime_fusion.data regimes --include-holdout    # regimes per walk-forward window (A4's need the news)
python -m regime_fusion.data check --include-holdout      # one view of every check: reports/data_check.md
python -m regime_fusion.data finbert                      # robustness: FinBERT sentiment of FNSPID headlines
```

Each step caches its downloads (`--refresh` fetches again), prints what needs a look, and writes its checks to `reports/`. The `check` step gathers them into one page, `reports/data_check.md`, listing per stock what needs a look (gaps, flags, daily moves over 25%, thin indicators, thin news, missing regime files); it is the basis for signing off the data (Phase 2). `--stocks AAPL MSFT` limits a step to some stocks (ids are January 2015 tickers). `python -m regime_fusion.data constituents` and `draw` rebuilt the committed constituents list and universe; they are needed again only if those change.

The GDELT step is meant to run in the background, kept awake with `caffeinate`, with its log in `~/rfgp-data/logs/`:

```bash
cd ~ && RFGP_DATA_DIR=~/rfgp-data nohup caffeinate -ims ~/.venvs/rfgp/bin/python \
    -m regime_fusion.data gdelt --workers 12 >> ~/rfgp-data/logs/gdelt.log 2>&1 &
tail -f ~/rfgp-data/logs/gdelt.log           # progress: days done and hours left
pkill -f "regime_fusion.data gdelt"          # stop (its workers stop too); the same start command resumes
```

"Days" in its log are days of news data (3,952 from February 2015 to December 2025), not days of running time. It was started on 28 October 2026. Runs started before 29 October 2026 left their worker processes running when stopped this way; `pgrep -fl multiprocessing.spawn` shows any that are left.

## When information may be used

The clock is the New York Stock Exchange (calendar `XNYS` of the `exchange_calendars` package, which includes holidays, the closures of 5 December 2018 and 9 January 2025, and the 13:00 early closes). A value used on session *t* must be available by the close of *t*; trades execute on *t*+1. `regime_fusion.data.calendar.TradingCalendar` applies two rules:

| Item | First session on which it may be used | Example |
| --- | --- | --- |
| Has a timestamp (news) | The first session whose close is strictly later than the timestamp | 15:59 New York time counts for that day; 16:00 or later counts for the next session; on an early-close day the cut-off is 13:00 |
| Has a date but no time (SEC filing date, news stamped only with a date) | The first session strictly after the date | A 10-Q filed on Monday is first used on Tuesday |

Feature code built on daily rows is tested with `regime_fusion.testing.check_no_lookahead`. Daily tables built from dated items (news, filed values) are tested with `regime_fusion.testing.check_point_in_time`: withholding every item not yet available on day *t* must leave row *t* unchanged.

## Universe

The universe is fixed as of January 2015, so firms later acquired or delisted stay in it where prices exist.

- [`configs/constituents_2015-01.csv`](../configs/constituents_2015-01.csv) lists the S&P 500 members at the end of January 2015 (502 share classes of 500 firms), with GICS sector and sub-industry, SEC CIK, and S&P 100 membership. It is rebuilt from pinned Wikipedia revisions with `python -m regime_fusion.data constituents`: the S&P 100 page as of 16 January 2015 (revision 642684730) and the S&P 500 list as of 27 January 2015 (revision 644366768). Wikipedia is a secondary source (CC BY-SA); an official list would be a welcome check.
- [`configs/acttrade.yaml`](../configs/acttrade.yaml) holds ActTrade's 18 US stocks, taken from the ActTrade paper and Table 4.3 of Christodoulaki's thesis (the same 42 companies). The paper's "COKE" is The Coca-Cola Company (KO), as Michael confirmed. Four of the 18 were not in the S&P 500 in January 2015 (Kodak, the New York Times, Tesla, Activision).
- [`configs/universe.yaml`](../configs/universe.yaml) is written by `python -m regime_fusion.data draw` (`regime_fusion.data.draw`), which refuses to run again once a draw exists unless given `--redraw`. Both draws use generators from `regime_fusion.seeds`, keyed by the draw's name only.
  - **Main (50):** ActTrade's 17 usable US stocks, plus 33 drawn from the other usable S&P 100 members. Each GICS sector's target is its share of the S&P 100 times 50, and the draws go to the sectors short of their targets, so the 50 stocks match the S&P 100's sector mix to within one stock per sector.
  - **Hold-out (50):** January 2015 S&P 500 members outside the S&P 100 and the main universe, one share class per firm, taken in a seeded random order. Firms without Yahoo prices from the study start were passed over: 15 of the first 65, all acquired or merged since 2015.
  - **Unused (48):** usable S&P 100 members that were not drawn.
  - **Excluded (24):** with the reason in `notes`. Yahoo no longer has the prices of acquired firms (Activision, Anadarko, Monsanto, Time Warner, Walgreens Boots Alliance and others): this is the survivorship bias of free price data that the paper states.
- Each stock keeps its January 2015 ticker as its identifier, in file names and seeds. Its entry records every CIK that filed its accounts from 2014 to 2025 (holding-company reorganisations such as Google into Alphabet add one) and the Yahoo symbol of its price history today (FB is now META, UTX is RTX, BK is BNY, ANTM is ELV).
- **Symbols for hold-out firms** come from SEC's current tickers for the firm's January 2015 CIK. When that CIK no longer files, the old ticker is kept only if Yahoo's name for it matches the firm and the ticker has not passed to another January 2015 firm: ACE took Chubb's ticker, CB, when it bought Chubb in 2016.
- **Reused tickers:** Yahoo's EMC, FOXA and DOW are now other securities, and every price series is checked to start before the study does.
- **Hold-out guard:** hold-out stocks are returned only by `Universe.select("holdout", include_holdout=True)`, which only the downloads and the final campaign use.

## Prices

`python -m regime_fusion.data prices` downloads daily bars from Yahoo Finance for every main stock (`--include-holdout` adds the hold-out stocks) into `raw/yahoo/stocks/<id>.parquet`. It then writes one row per session to `interim/prices/<id>.parquet` and the checks to `reports/price_checks.json`.

- **Raw files** keep the output of `yfinance` (`auto_adjust=False, actions=True`) from January 2014 to the download date. Yahoo adjusts the whole history for later splits, so splits after 2025 are needed to recover traded prices.
- **Interim tables** hold split-adjusted open, high, low and close (as Yahoo gives them), total-return prices (`adj_*`) from which returns are computed, volume, dividends, splits, and `traded_close`, the price actually paid that day. Valuation ratios need `traded_close`, because filed per-share values are not split-adjusted. Apple's traded close on 28 August 2020 comes out as $499.23, its actual close before the 4-for-1 split.
- **Spin-offs.** Yahoo records some spin-offs as fractional splits (eBay 2.376 for PayPal in July 2015, HP 2.202 for Hewlett Packard Enterprise in November 2015, RTX 1.589 in April 2020), so total-return prices do not drop on those days.
- **Spin-off corrections.** Yahoo's ratio is wrong on three spin-off days in the main universe, which gives returns that never happened: Xerox on 3 January 2017 (+37.6%; Conduent), DuPont on 3 June 2019 (+17.8%; Corteva, with a 1-for-3 reverse split) and RTX on 3 April 2020 (−7.8%; Carrier and Otis). [`configs/price_corrections.yaml`](../configs/price_corrections.yaml) lists them. The day's total return is recomputed from the traded closes of the parent and the spun-off firms (+10.4%, +10.2% and +5.2%), and earlier total-return prices are rescaled, which leaves every other day's return unchanged. They were found by comparing the return on each of Yahoo's 22 fractional splits in the universe with the market's move that day; the other 19 agree with traded prices.
- **Scale.** Yahoo's adjustments rescale the past by later events. Ratios within a look-back window are unaffected, so technical indicators must be scale-invariant (returns, price ratios, relative DC thresholds) and never use price levels.
- **Checks.** Coverage of the study period from its first session; sessions without a bar (kept as missing rows, never filled); bars on days that are not sessions; impossible bars (non-positive prices, a close outside the day's range); zero volume; runs of five or more unchanged closes. Daily moves above 15% are listed for a one-off hand check. Most are genuine (earnings, March 2020), but a spin-off that Yahoo does not adjust would also show up there. Across the 100 stocks the only problem flagged is a day of zero volume (Symantec). The 307 moves above 15% fall almost all on market-wide days (16 March 2020 alone has 35) or in volatile stocks (Kodak 38, Tesla 16); the only artefacts among them were the spin-off days above.

## Market series

`python -m regime_fusion.data market` downloads the S&P 500 index (`^GSPC`, a price index, for market regimes) and SPY (whose adjusted close is a total-return benchmark) from Yahoo and processes them like stocks. The VIX comes from Cboe's own history file (`raw/cboe/VIX_History.csv`). Since mid-2022 that file also has values on US exchange holidays, from Cboe's extended trading hours; alignment drops them.

The VIX is calculated until 16:15 New York time, a quarter of an hour after the stock close, so its close on day *t* is not strictly available by the close of *t*. The interim table keeps each value on its own day, and features on day *t* use the close of *t*−1 (decided on 28 October 2026).

## Fundamentals

`python -m regime_fusion.data sec` downloads, for every CIK of each stock, the XBRL company facts and the filer details from SEC EDGAR (`raw/sec/companyfacts/`, `raw/sec/submissions/`, gzip-compressed JSON). It keeps the facts from 10-K and 10-Q reports and their amendments (`interim/sec/<id>.parquet`), and checks the SEC filer names against the universe and the number of reports per year (`reports/sec_checks.json`). Requests name the contact in `configs/data_sources.yaml` and are paced at five a second (the SEC allows ten).

`python -m regime_fusion.data fundamentals` turns the facts into daily point-in-time values (`interim/fundamentals/<id>.parquet`) for the concepts in [`configs/fundamentals.yaml`](../configs/fundamentals.yaml), and reports their coverage (`reports/fundamentals_coverage.json`).

- **Availability.** A fact is used from the first session after its filing date. A period's value on day *t* comes from the latest filing available by *t*, so a restatement replaces the original only once it is filed. Apple's 2008 net income, for example, was first filed as $4.83bn and restated to $6.12bn in January 2010.
- **Balance-sheet items** take the value at the latest period end known. **Income and cash-flow items** take the trailing twelve months (TTM), and the TTM a year earlier for growth rates. 10-Qs report years to date, so TTM = previous fiscal year + year to date − the same span a year earlier. Fiscal years of 52 or 53 weeks and 12-week quarters (PepsiCo, Costco) are handled.
- **Earnings releases** (8-K facts) are dropped, because few firms file them in XBRL. The data are therefore conservative: the market usually learns earnings from the press release a few weeks before the 10-Q.
- **Period ends.** Each value comes with the end of the period it describes (`<concept>_end`), so the feature layer can drop stale values, such as those left by a tag a firm stopped using.
- **Check on real filings.** For eight firms with different fiscal calendars (Microsoft, Walmart, PepsiCo, Costco, Alphabet, Medtronic, Disney, JPMorgan), FY2019 revenue and net income in the panel equal the published figures, each first used on the session after its filing.
- **Inputs.** The concept list covers the inputs of ActTrade's twelve indicators: revenue, operating and pre-tax income, net income, interest expense, depreciation and amortisation, operating cash flow, capital expenditure, R&D, equity, current assets and liabilities, inventory, long- and short-term debt, and the diluted share count (a period average, taken from the latest quarter). Banks' revenue comes from net interest income plus non-interest income.

## Fundamental indicators

`python -m regime_fusion.data fa` computes ActTrade's twelve indicators for each session (`regime_fusion.data.fa`, `processed/fa/<id>.parquet`, coverage in `reports/fa_coverage.json`): net profit ratio, return on equity, quick ratio, P/E, P/B and P/S ratios, debt to equity, total revenues, EBITDA, levered free cash flow, diluted EPS and R&D expenses. On day *t* every input is the one known at the close of *t*; values are raw, and normalisation is fitted on training windows later.

- **Market value** is the traded close times the diluted share count, which covers every share class. Firms with several classes (Estée Lauder, the New York Times, Ralph Lauren, Under Armour, Universal Health Services) file their cover-page share counts per class, which the SEC's API leaves out.
- **Splits.** Reports filed after a stock split restate their share counts, so a count is multiplied only by the true splits after its filing date. Spin-offs that Yahoo records as fractional splits are left out, and DowDuPont's 1-for-3 reverse split, which Yahoo merged with the Corteva spin-off, comes from the corrections file. Market values run on smoothly across splits: Apple $2.17tn to $2.34tn over its 2020 4-for-1 split, Alphabet $1.54tn to $1.57tn over its 2022 20-for-1 split and $1.58tn after the restated 10-Q. Diluted EPS is net income over the same share count, which avoids adding up quarterly EPS figures across splits.
- **Fallbacks,** each using only what is known on the day: EBITDA from pre-tax income plus interest expense when a firm has no operating-income line (oil majors, large drug makers, IBM), and R&D of zero for a firm that has filed no R&D figure so far.
- **Coverage over 2015–2025.** Each indicator has values on at least 90% of days for 84–100 of the 100 stocks. The gaps are structural. The quick ratio is missing for 16 firms, mostly financial firms with unclassified balance sheets (also Ford and GE). Debt to equity is missing for Ford and GM, which report their debt in their own tags. EBITDA is missing for a few firms with neither an operating-income line nor the standard pre-tax tags.

## Technical indicators

`python -m regime_fusion.data ta` computes ActTrade's twelve technical indicators for each session (`regime_fusion.data.ta`, `processed/ta/<id>.parquet`): moving average, momentum, rate of change, volatility, Williams %R and midprice, over 5 and 10 sessions (equations 8–13 of the ActTrade paper), from total-return prices with highs and lows adjusted by the same factor as closes.

- **Levels in the prices of the day.** Moving average, momentum and midprice are price levels. Yahoo's adjusted prices are scaled by every later dividend and split, so on day *t* these are expressed in the prices of day *t* (past prices adjusted for the events up to *t* only). A later dividend therefore never changes an earlier value.
- **Two readings of the paper:** the moving average averages *n* sessions (equation 8 sums *n*+1 prices but divides by *n*), and the midprice is the midpoint (high + low)/2, as its text says (equation 13 prints (high − low)/2).
- Values are raw; ActTrade's scaling to [−1, 1] is fitted on training windows later. All 100 stocks have every indicator on at least 99% of study days.

## Market regimes

`python -m regime_fusion.data regimes` fits the market regimes for every walk-forward window (`regime_fusion.regimes`, `processed/regimes/<inputs>/<test year>.stage<1|2>.parquet`, parameters in `reports/regimes.json`). The regimes come from the S&P 500 index, so all stocks share them. The settings in [`configs/regimes.yaml`](../configs/regimes.yaml) are a draft until the pilot, which uses 2015–2018 data only.

- **Model.** A Gaussian hidden Markov model with diagonal covariances, fitted by EM from ten random starts (seeds from `regime_fusion.seeds`, keyed by test year and stage only). States are ordered by the mean of the first input, so state 0 is the calmest.
- **Only filtered probabilities:** each day's probabilities use the days up to it, never later ones. For each test year, stage 1 fits on the three training years and runs through the validation year, and stage 2 fits on all four years and runs through the test year, so test-day probabilities come from a model that never saw the test year.
- **Inputs.** Rung A2 uses the log realised volatility of the index over 20 sessions. In the 2015–2018 fit it separates a calm state (about 8% annualised volatility, lasting about 88 days on average) from a turbulent one (about 18%, about 52 days). Rung A3 uses directional changes of the index at a 1% threshold: the number of events in the last 20 sessions and the move of the current trend. At this threshold one state is merely "no event in 20 sessions", so the pilot should try smaller thresholds.
- **News attention (rung A4).** A4 adds to A3's inputs the same two inputs computed on news attention (`regime_fusion.regimes.news`): directional changes of the log of 1 + the articles of the last 5 sessions, at an absolute threshold of 0.25 (a rise or fall of about 28%), counted over 20 sessions. Article counts come from the `sa` step, scaled up for missing GDELT files; sessions without enough files are left out of the 5-session mean. As the plan has it, the regimes stay market-wide and shared by all stocks: the attention is that of the 50 main stocks together (`scope: market`). Setting `scope: stock` gives each stock its own regimes from its own news instead (`processed/regimes/dc_news/<id>/`, seeds keyed by stock, test year and stage), which asks whether a strategy should listen to sentiment when its own firm is in the news; this is an option for the pilot. Both thresholds are drafts for the pilot.

## News

Sentiment and news attention come from GDELT (main source) and FNSPID (robustness check). The timing rules are in `regime_fusion.data.news`.

- **Timing.** A GDELT record counts from the time of the 15-minute update that first carried it, which is never before publication. An FNSPID item counts from its publication time. Both go to the first session whose close is strictly later. FNSPID items stamped exactly at midnight UTC carry only a date, so they count for the first session after it. News sums are kept per UTC hour; the NYSE closes on the hour, so this loses nothing at the cut-off.
- **GDELT access** (decided on 28 October 2026). The Loughran–McDonald word counts are in the GKG's GCAM field, about 8.5 KB per record, at roughly 200,000–320,000 records a day: reading it in BigQuery for 2015–2025 would take 8–11 months of the sandbox's free quota. Instead, `python -m regime_fusion.data gdelt` (`regime_fusion.data.gdelt`) streams GDELT's raw GKG files: 373,743 files from 18 February 2015 (the first) to 31 December 2025, 2.75 TB compressed. Each file is checked against the MD5 in GDELT's master list, read in memory and discarded, so the 2.75 TB pass over the network but are never stored. On Michael's connection (about 12 MB/s with parallel downloads) a full pass takes about 2.7 days. The run can be stopped and resumed at any time, and it stops by itself if the data disk has less than 5 GB free.
- **What is kept.** Only records that name a firm: its anchors ([`configs/news_aliases.yaml`](../configs/news_aliases.yaml)) are matched against GDELT's organisations and against its "all names" field. GDELT's organisation tagger misses some large firms: General Motors appeared as an organisation in only 34 records in the first four months of 2015, but the all-names field finds it, and matching both fields keeps 28–62% more records in sample files. Records are summed per UTC hour, stock and *variant set* (the names that matched, such as "apple inc" or "apple daily"). The sums cover article counts, first mentions (the firm is the first organisation named), title mentions, GDELT's tone fields, and the GCAM word count with the six Loughran–McDonald counts, plus per-article shares. Keeping variant sets apart lets the exact matching be decided later without downloading GDELT again. Records whose page title names the firm are also kept whole (time, source, URL, title, tone and Loughran–McDonald counts), for hand checks and for the FinBERT check. One trial day (11 March 2024) took 2 minutes and 0.4 MB on disk: 9,781 records became 2,115 hourly rows and 1,556 titled records. The whole period should need about 1–1.5 GB. GDELT's own archive lacks a few listed files (two on 19 February 2015 return 404); each day's provenance record lists any file that could not be fetched.
- **Company matching.** GDELT lists organisations in lower case without punctuation ("jpmorgan chase co", "exxonmobil corp", "alphabet inc"), with noise ("ford foundation", "apple daily"). Anchors are therefore broad, and the variant sets are screened by hand before the daily features are built. Many records name a firm only in passing, such as JPMorgan as the analyst in a story about another company; first mentions and title mentions help to separate articles about a firm from passing mentions.
- **Page titles.** GDELT records carry the page title from 23 September 2019 on (every record of the noon file of that day, none of the day before). The records whose title names a firm, kept by the extraction, could therefore supply the 2024–25 headlines for the FinBERT check (an open decision).
- **FNSPID** is on Hugging Face as two files, `All_external.csv` (5.7 GB) and `nasdaq_exteral_data.csv` (23 GB, with article text). Both are larger than the free disk on Michael's Mac, so they will be streamed and filtered to our tickers.

## Sentiment indicators

`python -m regime_fusion.data sa` builds twelve sentiment indicators per stock and session (`regime_fusion.data.sa`, `processed/sa/<id>.parquet`) from the GDELT days extracted so far. Days not yet extracted count as missing, so the command can be run again as the extraction proceeds. Session sums, coverage and article counts go to `interim/news/<id>.parquet`, ready for the news-attention DC of rung A4; checks go to `reports/sa_coverage.json`.

- **Which names count.** [`configs/news_names.yaml`](../configs/news_names.yaml) says, per firm, which of the names its anchors matched really are the firm: "apple inc" and "apple pay" count for Apple, "apple daily" (a Hong Kong newspaper) does not; Raytheon counts for United Technologies only from their merger on 3 April 2020. A record counts when at least one of its names does. The first version was drafted from the first months of 2015. The report lists each firm's most frequent names counted and left out in 2015–2018, and the rules will be revised on the full development data. Renamings after 2018 (NortonLifeLock, Elevance Health, RTX) come from the firms' announcements, not from the news data.
- **Sessions and coverage.** A session's news runs from the previous close to its own close, so Monday's includes the weekend's. GDELT's master list lacks about 2% of its 15-minute files: 51 days have fewer than half of theirs, most in October–November 2020 and in a 17-day outage from 14 June 2025, plus 42 hours in October 2015 and 32 hours in March 2023. A session's coverage is the share of its files that were read. Article counts are scaled up by it, and a session with less than half of its files has no sentiment indicators.
- **Indicators**, each over the last 1 and 5 sessions:
  - attention: log(1 + articles);
  - abnormal attention: attention minus its mean over the previous 60 sessions, which removes the slow drift in GDELT's volume;
  - tone: the mean GDELT tone (positive minus negative words, as a percentage of all words);
  - the mean Loughran–McDonald positive, negative and uncertainty shares (percentages of words).

  Means count each article once and are missing when there is no article. A five-session value needs at least three covered sessions, and its article count is scaled to five sessions. News effects on prices fade within days, hence windows of a day and a week.
- **Why not ActTrade's.** ActTrade's twelve sentiment indicators score each article's text, title and summary with TextBlob (polarity and subjectivity), SentiWordNet and AFINN. GDELT gives no article text, so the study uses the text measures GDELT computes on the full text instead: its tone and the Loughran–McDonald dictionary, the standard finance dictionary, as the plan names them, plus attention. This choice is open for Michael in [`DECISIONS.md`](DECISIONS.md).
- **Look-ahead tests.** The indicators are recomputed on truncated data (`check_no_lookahead`), and rebuilt with only the news and GDELT files public by each close (`check_point_in_time`).

## FinBERT headline sentiment (robustness check)

`python -m regime_fusion.data finbert` (`regime_fusion.data.finbert`) scores the universe's FNSPID headlines with FinBERT (ProsusAI/finbert; it needs the `neural` and `finbert` extras and downloads the model, about 440 MB, from Hugging Face on first use). Each distinct headline is scored once and the scores are kept in `interim/finbert/scores.parquet`, with the model's revision in the provenance record.

- **Timing:** as for all news, a headline published after the close of *t* counts for *t*+1, and one with only a date counts for the first session after it. The same headline twice for a stock at the same time counts once.
- **Indicators** (`processed/finbert/<id>.parquet`), over the last 1 and 5 sessions: `finbert_attention` = log(1 + headlines) and `finbert_sentiment` = the mean of P(positive) − P(negative). Sessions after the last day FNSPID covers have no value, rather than zero news.
- **Use:** the rungs `A0-FB` and `A1-FB` (and the best rung, once known) add these four indicators to the sentiment branch (as `sa:finbert_...`), for 2019–2023, as decided (option B). FNSPID's two files cover different years (`All_external.csv` to 2020, `nasdaq_exteral_data.csv` to 2023), so headline counts shift between periods; the scaling on each training window absorbs part of this.

