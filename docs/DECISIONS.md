# Decision log

Newest first. "MK" is Michael Kampouridis (supervisor). [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) holds the reasoning behind each decision.

| Date | Decision | Decided by |
| --- | --- | --- |
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
| 2026-10-28 | Trade execution price on day t+1 (open or close) | To be settled when the backtester is designed (Phase 4) |
| 2026-10-28 | Free headline source for 2024–25 (FNSPID ends in 2023) | Check whether GDELT records carry page titles; otherwise limit the FinBERT check to 2019–2023 |
| 2026-10-28 | Fundamental (FA) indicators: which ratios and growth rates the FA branch uses | Match ActTrade's FA indicators. `configs/fundamentals.yaml` holds a draft list of accounting inputs (revenue, operating and net income, operating cash flow, assets, liabilities, equity, current items, cash, long-term debt, shares outstanding) |
| 2026-10-28 | The VIX is calculated until 16:15 New York time, after the 16:00 stock close, so its close on day *t* is not strictly available by the close of *t* | Use the VIX close of *t*−1 on day *t*; if trades execute at the *t*+1 open, the close of *t* would also be safe |
