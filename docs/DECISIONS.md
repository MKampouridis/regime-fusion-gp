# Decision log

Newest first. "MK" is Michael Kampouridis (supervisor). The research-plan document holds the reasoning behind each decision.

| Date | Decision | Decided by |
| --- | --- | --- |
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
| 2026-10-28 | FinBERT sentiment: (A) main sentiment set, (B) robustness check, or (C) left out | B after a word with Kingsley, or A if he contributes the FinBERT pipeline |
| 2026-10-28 | Training windows | Rolling 4-year windows for the main results; expanding windows as a robustness check |
| 2026-10-28 | Trade execution price on day t+1 (open or close) | To be settled when the backtester is designed (Phase 4) |
