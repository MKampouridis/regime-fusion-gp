# Decision log

Newest first. "MK" is Michael Kampouridis (supervisor). [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) holds the reasoning behind each decision.

| Date | Decision | Decided by |
| --- | --- | --- |
| 2026-10-30 | The confirmatory tests compare portfolios: for each of the six comparisons, the difference between the two methods' equal-weight portfolio Sharpe ratios over the seven test years, two-sided, by a stationary block bootstrap of the days (a mean block of 10 days, as in the SPA test; 10,000 resamples), Holm-adjusted. The Wilcoxon tests on the stock-year medians (the unit decided on 28 October) stay as the per-stock view, comparable with ActTrade, each with a check clustered by year: a t-test over the years of each year's mean difference, the test of a model with a random year effect. Reason (raised by an outside review of the code): the stocks share each year's market and a stock's years are related, so tests that treat the 350 stock-years as independent give p-values far too small when a gain varies from year to year: seven yearly differences give p = 0.41, the same seven repeated for 50 stocks p ≈ 3 × 10⁻¹² (`PROTOCOL.md`, §4) | MK |
| 2026-10-30 | Deflated Sharpe Ratio: the trials are the distinct configurations (rungs and benchmarks) in the plans of runs of the GP-parameter pilot, the pilot and the main campaign; exit candidates are not counted, since they are part of every method's recipe. The variance of the trials' Sharpe ratios is that of the daily Sharpe ratios of the stage-2 portfolios analysed, as a stand-in, since the pilot's trials were evaluated on 2018 | MK |
| 2026-10-30 | Stock-years of the FinBERT check: a source covers a stock in a year if at most two of the year's months have no headline for it, and a stock-year enters the check only if its source covers every year of its window (the first training year to the test year). For GDELT's titles, which come from a complete archive, this selects stock-years on the amount of news in the test year (not on returns), and the paper says so | MK |
| 2026-10-30 | A debt or R&D line that a firm stops filing is missing, not zero, once its last value is stale (its period ended more than 456 days before the session): a debt paid off cannot be told apart from a tag change | MK |
| 2026-10-29 | Headlines of the FinBERT check: FNSPID for test years 2019–2023 and GDELT page titles for 2024–2025, so the check covers every test year; each walk-forward window uses one source throughout (GDELT's titles begin on 23 September 2019, so they cover the whole window of 2024 and 2025 only) | MK |
| 2026-10-29 | The pilot runs on Michael's MacBook Air, once the 2015–2018 news data are in; CERES runs the main campaign | MK |
| 2026-10-29 | FinBERT scores the headlines on Michael's Mac (the model is about 440 MB); only the daily indicators go to CERES | MK |
| 2026-10-29 | The GP's population, number of generations and tournament size are chosen together in a small pilot on 2015–2018 data, with fitness convergence as the criterion: populations from 250 to 2,000 at an equal budget of evaluations, tournaments of 2 and 4 (`PROTOCOL.md`, §9.1) | MK |
| 2026-10-29 | The draft analysis protocol (`PROTOCOL.md`) is accepted; it is frozen at the end of the pilot | MK |
| 2026-10-29 | Sentiment indicators: twelve from GDELT, over 1 and 5 sessions: attention, abnormal attention, tone, and the Loughran–McDonald positive, negative and uncertainty shares (ActTrade's TextBlob, SentiWordNet and AFINN scores need article text, which GDELT does not give) | MK |
| 2026-10-28 | Trades execute at the open of *t*+1. The take-profit is a limit order at the target price, filled on the first day whose high reaches it (at the open if the day opens above it) | MK |
| 2026-10-28 | ActTrade's "COKE" is The Coca-Cola Company (KO) | MK |
| 2026-10-28 | Fundamental (FA) indicators are ActTrade's twelve: net profit ratio, return on equity, quick ratio, P/E, P/B and P/S ratios, debt to equity, total revenues, EBITDA, levered free cash flow, diluted EPS and R&D expenses | MK |
| 2026-10-28 | The VIX enters the features of day *t* with its close of *t*−1, because its close is set at 16:15, after the stock close | MK |
| 2026-10-28 | GDELT news comes from GDELT's raw GKG files, streamed and filtered to records that name our firms, instead of the BigQuery sandbox, whose free quota would cover the Loughran–McDonald counts only after 8–11 months | MK |
| 2026-10-28 | Hold-out stocks: 50 taken in a seeded random order from the January 2015 S&P 500 members outside the S&P 100 and the main universe, one share class per firm; firms without Yahoo prices from the study start are passed over (15 of the first 65 were) | MK |
| 2026-10-28 | Main universe: ActTrade's 18 US stocks (Table 4.3 of Christodoulaki's thesis and the ActTrade paper; Activision is excluded because Yahoo no longer has its prices), topped up to 50 by a seeded draw from the other usable January 2015 S&P 100 members, so that the sector mix matches the S&P 100's. Seeds come from `regime_fusion.seeds` | MK |
| 2026-10-28 | FinBERT headline sentiment is a robustness check only (option B): A0, A1 and the best rung are re-run with FinBERT scores added to the sentiment branch | MK |
| 2026-10-28 | Main results use rolling 4-year training windows; expanding windows (all data since 2015) are a robustness check | MK |
| 2026-10-28 | Changes are pushed straight to `main` and summarised in chat; no pull requests | MK |
| 2026-10-28 | The SEC contact (User-Agent) and the Google account for the BigQuery sandbox use Kampouridis.michael@gmail.com | MK |
| 2026-10-28 | Strategies are re-evolved every year (walk-forward). For each test year: evolve on 3 years, choose exit settings on the next (validation) year with the A0 baseline, then re-evolve on all 4 years before the test year. The recipe is fixed using 2015–2018 only | MK |
| 2026-10-28 | Test years 2019–2025 and 50 hold-out S&P 500 stocks are run once, after the recipe is frozen at the end of the pilot | MK |
| 2026-10-28 | 30 runs per stock, test year and configuration (the pilot may raise this to 50); the same seeds for every configuration; tests compare per-stock-year medians | MK |
| 2026-10-28 | Universe: about 50 US large caps, the 18 US names from ActTrade's 42 plus others from the January 2015 S&P 100 | MK |
| 2026-10-28 | Period: February 2015 to December 2025 (GDELT GKG 2.0 news begins in February 2015) | MK |
| 2026-10-28 | Backtest holds one position at a time; signals use data up to the close of day t and trade on day t+1 | MK |
| 2026-10-28 | No WRDS access assumed; the survivorship bias of free price data is stated in the paper | MK |
| 2026-10-28 | The original ActTrade data are unavailable, so the A0 rung is ActTrade's logic on new data, not a replication | MK |
| 2026-10-28 | Python only; no paid compute; main experiments on CERES, submitted by MK | MK |

## Open

| Raised | Question | Proposal |
| --- | --- | --- |
| 2026-10-29 | How FinBERT's five-session indicators treat sessions whose GDELT files were mostly not read | As the GDELT sentiment indicators do: such a session adds nothing to the window rather than counting as a day without news, the window needs at least three read sessions, and its headline count is scaled to five sessions. The unread session's own row stays missing. (Before this, unread sessions counted as zero headlines, which understated attention on the four sessions after them.) |
