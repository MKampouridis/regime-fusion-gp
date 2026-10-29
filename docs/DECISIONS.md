# Decision log

Newest first. "MK" is Michael Kampouridis (supervisor). [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) holds the reasoning behind each decision.

| Date | Decision | Decided by |
| --- | --- | --- |
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
| 2026-10-29 | Sentiment indicators. ActTrade scored article text with TextBlob, SentiWordNet and AFINN; GDELT gives no text | Twelve GDELT indicators over 1 and 5 sessions: attention, abnormal attention, tone, and Loughran–McDonald positive, negative and uncertainty shares (`DATA.md`, Sentiment indicators). Built this way until MK decides |
| 2026-10-28 | Free headline source for 2024–25 (FNSPID ends in 2023) | GDELT records carry page titles from 23 September 2019 (checked on 29 October 2026: none in the noon file of 22 September, all in that of 23 September), and the extraction already keeps every record whose title names a firm. Proposal: score those GDELT titles with the same FinBERT for 2024–25, so the check covers 2019–2025; or, for one source throughout, use GDELT titles alone from 2020 |
