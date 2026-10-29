# Research plan

Working title: *Learning When to Listen: regime-conditioned genetic programming for fusing fundamental, sentiment and technical signals.*

This is the plan agreed with Michael Kampouridis on 28 October 2026. Later decisions are logged in [`DECISIONS.md`](DECISIONS.md), which takes precedence over this file.

## Idea

ActTrade (Christodoulaki and Kampouridis, *Applied Soft Computing*, 2026) combines fundamental (FA), sentiment (SA) and technical (TA) branches in two fixed ways:

- a fixed 3-input AND at the root of the tree;
- a fixed weighted sum of four Sharpe ratios as the fitness.

Its integrated tree beats the average single branch but not the best one (ΔSRC_Max ≈ 0.3, not significant), and the AND root leaves many runs without a single trade.

This study evolves the fusion itself:

- **Branches.** Typed branches for FA, SA and TA, plus a regime branch. The regime branch drives gating and voting operators, so a strategy learns which sources must agree, and when.
- **Regimes.** Hidden Markov models over directional-change (DC) indicators of prices and of news attention. Only filtered, real-time probabilities are used.
- **Selection.** ε-lexicase selection over (time block × branch) cases replaces the weighted-sum fitness.

## Research questions

1. Does regime-conditioned fusion beat fixed fusion out of sample, net of costs?
2. Do the gains come from regime information, or from flexible fusion alone?
3. Does ε-lexicase selection beat a weighted-sum fitness?
4. Which analysis type does the evolved strategy trust in which regime?

## Ablation ladder

Each rung adds one thing to the rung above, so every gain has a named cause.

| Rung | What changes | Question it answers |
| --- | --- | --- |
| A0 | ActTrade with the corrected backtest | Baseline: fixed 3-AND root, weighted four-Sharpe fitness |
| A1 | Evolvable fusion: AND, OR, 2-of-3 vote; no regime | Does flexible fusion alone help? |
| A2 | + regime branch from realised volatility | Does any regime information help? |
| A3 | + regime branch from price DC (HMM) | Do DC regimes beat simple volatility regimes? |
| A4 | + news-attention DC in the regime branch | Does news-based regime information add anything? |
| A5 | + ε-lexicase selection | Does lexicase beat the weighted-sum fitness? |
| A6 | + regime-indexed subtree archive (optional) | Does a regime-aware active-trading operator add anything? |

GP baselines also include single-type GPs (FA only, SA only, TA only), and DALex as an intermediate selection method.

## Data

- **Universe.** About 50 US large caps: the 18 US names from ActTrade's 42, plus others from the January 2015 S&P 100. Firms later acquired or delisted are kept where prices exist; using today's constituents would bias results upwards.
- **Hold-out stocks.** 50 other S&P 500 stocks, untouched during development and run only in the final campaign.
- **Period.** February 2015 to December 2025. GDELT news begins in February 2015.

| Analysis type | Source | Notes |
| --- | --- | --- |
| Technical | Yahoo Finance daily OHLCV (`yfinance`) | Rate-limited, so download once and cache. Many delisted tickers are missing (survivorship bias, stated in the paper) |
| Market state | S&P 500 index; VIX from Cboe | For regimes and benchmarks |
| Fundamental | SEC EDGAR XBRL (companyfacts API, Financial Statement Data Sets) | As-filed values with filing dates. Usable from the first trading day after filing. Fair access: at most 10 requests per second, with a named User-Agent |
| Sentiment (main) | GDELT GKG 2.0 via the BigQuery sandbox (free, 1 TB of queries a month) | Daily article count (attention), mean tone, Loughran–McDonald positive, negative and uncertainty shares |
| Sentiment (robustness) | FNSPID headlines (to 2023; CC BY-NC 4.0) scored with FinBERT (ProsusAI, trained on pre-2014 text) | Option B: re-run A0, A1 and the best rung with FinBERT scores added to the SA branch. A 2024–25 headline source is still to be found |

**Point-in-time rules.** A value used on day *t* must be available by the close of *t*, and trades execute on *t*+1. News after the close of *t* counts for *t*+1. Normalisation is fitted on training windows only. Every feature module has an automated look-ahead test.

## Evaluation protocol

- **Walk-forward windows.** Strategies are re-evolved every year. For each test year:
  1. evolve strategies on three years;
  2. choose exit settings on the next (validation) year;
  3. re-evolve on all four years just before the test year.

  Main results use these rolling 4-year windows. Expanding windows (all data since 2015) are a robustness check.
- **Frozen recipe.** The recipe (algorithm settings, features, regime model, statistics plan) is designed on 2015–2018 data only and frozen after the pilot. The seven test years 2019–2025 and the hold-out stocks are then run once.
- **Runs.** 30 independent runs per stock, test year and configuration, with the same seeds for every configuration. The pilot's 100-run check may raise this to 50. Each stock-year is summarised by its median before any test. The best-training-fitness run is also reported, as in ActTrade's best-tree analysis.
- **Exit rule.** ActTrade's take-profit within *d* days. (*d*, *r*) are chosen per stock each year on the validation year, using the A0 baseline (5 runs per candidate), and then shared by every method. Any tuning advantage therefore goes to the baseline.
- **Primary metric.** Annualised Sharpe ratio on daily returns, at 10 basis points per side.
- **Secondary metrics.** Return, maximum drawdown, CVaR 95%, turnover, the share of runs that never trade, and integration versus the best single branch (ΔSRC_Max).
- **Costs.** 0, 5, 10 and 25 basis points per side.
- **Statistics.**
  - Wilcoxon signed-rank tests with Holm correction between rungs.
  - Friedman tests with post-hoc comparisons across methods.
  - Deflated Sharpe Ratio counting every configuration tried.
  - Probability of backtest overfitting.
  - Hansen's SPA test against the best benchmark.
  - Every result also broken down by year and by regime.

## Non-GP benchmarks

| Benchmark | Why it is there |
| --- | --- |
| Buy-and-hold | Standard passive reference |
| Random entry with the same exit rule and trade frequency (1,000 draws) | Separates entry skill from the exit rule |
| Moving-average crossover / time-series momentum | Does the GP beat a one-line technical rule? |
| Logistic regression on all FA, SA, TA and regime features | Is non-linear fusion needed at all? |
| LightGBM on the same features | Strongest off-the-shelf learner for tabular data |
| Regime-switching ensemble: HMM regimes, one LightGBM per regime | Does "regimes + standard ML" already explain the gains? |
| Neural gated fusion: one small network per analysis type, mixed by a regime-driven gate | Evolved, readable gating vs neural gating |
| Temporal Fusion Transformer, one model across all stocks | The one advanced deep model; its variable-selection gates are the neural counterpart of "which source to trust when" |

**Fairness rules.**

- Every benchmark gets the same features, windows, exit rule, *t*+1 execution, costs and tuning budget.
- Tuning uses validation data only.
- Labels match the exit rule.
- Decision thresholds maximise validation Sharpe.
- Neural models get 10 runs each.

## Compute

About 168,000 GP runs:

- 50 stocks × 7 test years × 30 runs × about 12 GP configurations;
- plus 42,000 runs on the hold-out stocks.

ActTrade's code takes about 11 ms per tree, which would mean about 25,000 CPU-hours. The rewrite targets about 0.25 ms per tree, roughly 16 s per run, so about 750 CPU-hours (about 1,250 with 50 runs). On top of that:

- about 100 CPU-hours to choose exit settings;
- a few dozen for the non-GP benchmarks.

The main campaign runs on CERES as job arrays, which Michael submits. Downloads run on Michael's Mac, because the cloud workspace cannot reach the data sources.

## Phases

| Phase | Weeks | Content | Check before moving on | Status |
| --- | --- | --- | --- | --- |
| 1. Set-up | 1 | Repository, configuration, seeds, provenance, tests | Decisions settled | Done |
| 2. Data pipeline | 1–4 | Prices, SEC fundamentals, GDELT and FNSPID sentiment, market series; leakage tests; coverage report | Data validated | In progress |
| 3. DC module | 2–4 | Event detection (done, cross-checked against the Java code); causal indicators (done); DC on news attention; HMM regimes with filtered probabilities | Part of the data check | Built; thresholds set in the pilot |
| 4. Engine | 3–7 | Typed GP with extensible grammar; numba backtester (one position, *t*+1, take-profit, costs); daily-return metrics; tournament, weighted-sum, DALex and ε-lexicase selection | Unit tests, including hand-checked backtests | Built, with campaigns, benchmarks (neural ones included) and the analysis |
| 5. Pilot | 7–8 | 2015–2018 only: timing, convergence, light tuning, 100-run check; analysis plan written to `PROTOCOL.md` | Protocol frozen | Tools and draft protocol ready; runs on Michael's Mac once the 2015–2018 news is in (30 October 2026) |
| 6. Main campaign | 9–12 | Ablation ladder, GP baselines, non-GP benchmarks, hold-out stocks, robustness (expanding windows, FinBERT) | | |
| 7. Analysis | 13–14 | Statistics, cost tiers, interpretability maps, case studies | Analysis signed off | |
| 8. Writing | 14–18 | Paper and reproducibility package (code and data-building scripts; the data cannot be redistributed) | | |

## Risks

| Risk | Mitigation |
| --- | --- |
| Regime information adds little beyond flexible fusion (A2–A4 ≈ A1) | Frame the paper as "how should FA, SA and TA be fused"; A1 and A5 stand on their own |
| Too few days per regime | 2–3 regimes; fit regimes on the market index so all stocks share them; lexicase cases from time blocks |
| Look-ahead in DC or sentiment features | Only confirmed DC events; news cut off at the close; automated prefix-invariance tests |
| Overfitting from many configurations | Recipe frozen on 2015–2018; Deflated Sharpe and backtest-overfitting probability; test years and hold-out stocks untouched until the final run |
| Noisy GDELT company matching | Hand-checked samples; cross-check against FNSPID for 2015–2023 |
| Survivorship bias in free prices | Universe fixed as of January 2015; bias stated in the paper |
| CERES queues or runtime overrun | Engine speed measured in the pilot; job arrays; Michael's Mac as fallback |
