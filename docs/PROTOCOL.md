# Analysis protocol

**Status: DRAFT.** Written on 29 October 2026, before any pilot result, from the [research plan](RESEARCH_PLAN.md) and the [decisions](DECISIONS.md). Michael reviews it and freezes it at the end of the pilot (Phase 5). After that, any change is logged in `DECISIONS.md` with its reason and reported in the paper. The tools named here are described in [`ENGINE.md`](ENGINE.md).

## 1. Data used for what

| Data | Use |
| --- | --- |
| 2015–2018, the 50 main stocks | Design and the pilot. The pilot uses stage 1 of the 2019 window: evolve on February 2015 – December 2017, evaluate on 2018 |
| 2019–2025, the 50 main stocks | Test years, run once with the frozen recipe |
| 50 hold-out stocks, 2019–2025 | Run once, after the main campaign, with the same recipe and analysis |

No test-year or hold-out result is computed before the protocol is frozen. Development-only campaigns refuse any task that evaluates after 2018 (`tasks --development`).

## 2. Methods compared

- **GP rungs:** A0 (ActTrade on the corrected backtest), A1 (evolvable fusion), A2 (+ volatility regimes), A3 (DC regimes instead), A4 (+ news-attention DC), A5 (+ ε-lexicase), and A6 if it is built.
- **GP baselines:** A5-dalex, GP-FA, GP-SA, GP-TA.
- **Rule-based benchmarks:** buy-and-hold (RB-BH), ActTrade's TS(*d*, *r*) (RB-TS), moving-average crossover (RB-MA), time-series momentum (RB-TSMOM), and random entry at each method's trade frequency.
- **Machine-learning benchmarks:** logistic regression (ML-LR), LightGBM (ML-LGBM), regime-switching LightGBM (ML-RS-LGBM), neural gated fusion and the Temporal Fusion Transformer.

## 3. Per stock and test year

- Exit settings (*d*, *r*) are chosen on the validation year with A0 (16 candidates, 5 runs each) and shared by every method.
- Each GP configuration gets 30 runs (50 if the pilot says so, §9), with seeds from stock, test year, run and stage only. Deterministic benchmarks get one run; neural benchmarks get 10.
- Stage 2 models (evolved or trained on the four years before the test year) are evaluated on the test year at 0, 5, 10 and 25 basis points per side.

## 4. Primary analysis

- **Measure:** annualised Sharpe ratio of daily returns at 10 basis points per side.
- **Confirmatory tests** (decided on 30 October 2026): for each of the six comparisons below, the difference between the Sharpe ratios of the two methods' equal-weight portfolios (§5) over the seven test years. The test is two-sided, by a stationary block bootstrap of the days (Politis and Romano, 1994; a mean block of 10 days, as in the SPA test, and 10,000 resamples, both portfolios resampled on the same days), with Holm's adjustment across the six at α = 0.05 (`analysis portfolio`):

  | Comparison | Question |
  | --- | --- |
  | A1 vs A0 | Does flexible fusion alone help? |
  | A2 vs A1 | Does regime information help? |
  | A3 vs A2 | Do DC regimes beat volatility regimes? |
  | A4 vs A3 | Does news attention add anything? |
  | A5 vs A4 | Does ε-lexicase beat the weighted-sum fitness? |
  | A5 vs A0 | Does the full method beat ActTrade? |

  Each result is reported with the difference and its 95% bootstrap interval. The tests compare portfolios rather than stock-years because the stocks share each year's market and a stock's years are related: a paired test on the 350 stock-years would count a gain that comes and goes with the market once per stock, and its p-values would come out far too small. Seven yearly differences of 0.3, −0.2, 0.5, −0.4, −0.1, 0.6 and 0.2 give p = 0.41, but the same seven repeated for 50 stocks give p ≈ 3 × 10⁻¹².
- **Per-stock view** (reported with the confirmatory tests, comparable with ActTrade; not confirmatory): two-sided Wilcoxon signed-rank tests on the paired stock-year medians (the median over runs of each stock-year: 50 stocks × 7 years = 350 paired units), Holm-adjusted across the six. Each is reported with the median paired difference, a bootstrap 95% confidence interval (stock-years resampled) and a check clustered by year: a t-test over the seven years of each year's mean difference, the test of a model with a random year effect (`analysis compare --confirmatory`).
- **All methods together** (exploratory): the Friedman test on stock-years that every method covers, with Conover's comparisons, Holm-adjusted. Like the per-stock view, it treats the stock-years as independent.

## 5. Portfolios

For each method, an equal-weight portfolio: within a stock-year the daily returns of all its runs are averaged (no run is picked after the fact), then the stocks are averaged day by day, over the seven test years (`analysis portfolio`). The confirmatory tests of §4 compare these portfolios, and so do the following.

- **Hansen's SPA test:** every GP rung against the best non-GP benchmark, the one with the highest test-period Sharpe ratio. Picking the benchmark after the fact favours the benchmark.
- **Deflated Sharpe Ratio** of each method (decided on 30 October 2026). The number of trials counts every configuration evaluated (rungs and benchmarks), in the GP-parameter pilot, the pilot and the main campaign, as listed in the campaigns' plans of runs (`analysis portfolio --trials-from gp-pilot pilot <main campaign>`). Exit candidates are not counted, since they are part of every method's recipe. The variance of the trials' Sharpe ratios is that of the daily Sharpe ratios of the stage-2 portfolios analysed, as a stand-in, since the pilot's trials were evaluated on 2018.
- **Probability of backtest overfitting** across all methods' portfolios, by combinatorially symmetric cross-validation with 16 blocks.

## 6. Secondary analyses

All reported, none confirmatory:

- annual return, maximum drawdown, CVaR 95%, trades per year and the share of runs that never trade;
- integration: the whole model's Sharpe ratio minus its best single branch's (ΔSRC_Max);
- the primary tests at 0, 5 and 25 basis points;
- results by test year, and by regime (A3's most probable regime of each day);
- ActTrade's best-tree view: the run with the best training fitness of each stock-year;
- random entry: for each method and stock-year, the percentile of its median Sharpe ratio among 1,000 random-entry strategies with the same exit rule and number of trades; reported as the share of stock-years above the 95th percentile;
- which analysis types the regime rungs use under each regime (research question 4).

## 7. Robustness

- **Expanding windows** (all data since 2015 for training) for the six rungs of §4.
- **FinBERT headline sentiment** (option B): A0, A1 and the best rung by §4 re-run with FinBERT scores added to the sentiment branch, 2019–2025: FNSPID headlines for test years 2019–2023 and GDELT page titles for 2024–2025, one source per window. Only stock-years whose whole window the source covers are compared ([`DATA.md`](DATA.md)); the paired tests use those stock-years for both methods (`analysis portfolio --pairs A0-FB:A0 A1-FB:A1`, and `compare` for the per-stock view). For GDELT's titles, which come from a complete archive, a covered test year means headlines for the firm in at least ten of its months, so the check selects stock-years on the amount of news in the test year, though not on returns; the paper says so (decided on 30 October 2026).
- **Hold-out stocks:** every method, with the analysis of §4–6 unchanged.

## 8. Reporting rules

Every rung and benchmark is reported, whatever its result. The confirmatory family of §4 is not changed after test results are seen. Exploratory results are labelled as such.

## 9. Decisions the pilot makes

Each is settled on 2015–2018 data only (stage 1 of the 2019 window), by the rule given here, and every option tried counts as a trial for the Deflated Sharpe Ratio. The GP parameters are settled first, in a smaller pilot of their own (§9.1); the other decisions then use them.

| Decision | Options | Rule |
| --- | --- | --- |
| Runs per configuration | 30 or 50 | 100 runs of A0 and A1 on 10 stocks (ABT, DIS, GD, MS, NFLX, NSC, SO, SPG, UPS, WFC, drawn with the study's seed on 29 October 2026). Keep 30 if the median distance (95th percentile) between a 30-run median and the 100-run median is at most 0.05 in annualised Sharpe ratio, half the smallest difference of interest (0.1); otherwise 50 (`analysis pilot`) |
| Population, generations and tournament size | Populations of 250, 500, 1,000 or 2,000, each with tournaments of 2 or 4 | The GP-parameter pilot, §9.1 |
| DC threshold of the price regimes (A3) | 0.5% (`A3-fine`) or 1% (`A3`) | Options whose fitted regimes give a state less than 15% of the 2015–2017 days are dropped; of the rest, the higher median validation Sharpe ratio of A3 over the 50 stocks, 10 runs each; ties go to the smaller threshold |
| Threshold of the news-attention DC (A4) | 0.25 (`A4`) or 0.5 (`A4-wide`), in log units | As for A3, with A4, after its price threshold is set to A3's choice |
| Scaling of the indicators | min-max (`A1`) or rank (`A1-rank`) | The higher median validation Sharpe ratio of A1, 10 runs each |
| Fitness weights | Equal (0.25 each) | Kept: ActTrade's values are published nowhere, and its texts describe the fitness as the plain sum of the four Sharpe ratios, which equal weights reproduce (`ENGINE.md`, choice 3) |
| Sentiment indicators | The twelve GDELT indicators | Decided by Michael (29 October 2026) |
| GDELT name rules | `configs/news_names.yaml` | Revised on the 2015–2018 names report and the cross-check with FNSPID, before any pilot run (the main firms' rules were first reviewed on the 2015–16 news, 29 October 2026) |

### 9.1 The GP-parameter pilot

A large population usually needs fewer generations, and a small one more; the tournament size should grow with the population (MK). So the population, the number of generations and the tournament size are chosen together, with fitness convergence as the criterion, before the rest of the pilot.

- **Candidates:** populations of 250, 500, 1,000 and 2,000, each with tournaments of 2 and 4 (8 candidates). Every candidate gets the same budget of 100,000 evaluations (population × generations: 400, 200, 100 and 50 generations), twice ActTrade's 1,000 × 50, so that each can run to convergence.
- **Runs:** rungs A1 (evolvable fusion) and A3 (with regimes), on the 10 stocks of the 100-run check, 10 runs each, evaluated on 2018. Every run uses one fixed exit setting (20 days, 3%), the middle of the exit grid, so that this pilot needs no exit selection first. That is 1,600 runs.
- **Convergence:** a run has converged at the first generation by which its best training fitness has made 99% of its total rise. A candidate's number of generations is the median convergence generation of its runs, rounded up to a multiple of 10; its *converged fitness* is each run's best training fitness at that generation.
- **Choice:** of the candidates whose converged fitness is within 0.05 of the best candidate's (median paired difference over the same stocks, rungs and seeds), the one with the fewest evaluations wins; ties go to the higher validation Sharpe ratio. The validation Sharpe ratio at the chosen generation is reported for every candidate; if the winner's is more than 0.1 below the best, Michael decides.
- **Use:** the chosen population, generations and tournament size become the GP settings of every rung (A0 included, so each rung still changes one thing only). Lexicase selection (A5) has no tournament and takes the population and generations.

```bash
E="python -m regime_fusion.engine"
$E tasks --campaign gp-pilot --kind runs --years 2019 --stages 1 --runs 10 --development --exit 20 0.03 \
    --stocks ABT DIS GD MS NFLX NSC SO SPG UPS WFC --rungs \
    A1-p250-t2 A1-p250-t4 A1-p500-t2 A1-p500-t4 A1-p1000-t2 A1-p1000-t4 A1-p2000-t2 A1-p2000-t4 \
    A3-p250-t2 A3-p250-t4 A3-p500-t2 A3-p500-t4 A3-p1000-t2 A3-p1000-t4 A3-p2000-t2 A3-p2000-t4
$E run-all --campaign gp-pilot --kind runs --workers 7
python -m regime_fusion.analysis gp-params --campaign gp-pilot
```

The chosen values then go into `configs/rungs.yaml` (`gp:`), and the decision is logged.

### 9.2 Running the rest of the pilot

Once the GDELT days of 2015–2018 are extracted and the name rules revised:

```bash
python -m regime_fusion.data sa && python -m regime_fusion.data regimes && python -m regime_fusion.data check
E="python -m regime_fusion.engine"
$E tasks --campaign pilot --kind exits --years 2019 --development
$E tasks --campaign pilot --kind runs --years 2019 --stages 1 --runs 10 --development --rungs \
    A0 A1 A1-rank A2 A3 A3-fine A4 A4-wide A5 A5-dalex GP-FA GP-SA GP-TA \
    RB-BH RB-TS RB-MA RB-TSMOM ML-LR ML-LGBM ML-RS-LGBM ML-GATED
$E tasks --campaign pilot --kind runs --plan check100 --rungs A0 A1 --runs 100 --years 2019 --stages 1 \
    --development --stocks ABT DIS GD MS NFLX NSC SO SPG UPS WFC
$E run-all --campaign pilot --kind exits --workers 7     # or as job arrays on CERES
$E run-all --campaign pilot --kind runs --workers 7
$E run-all --campaign pilot --plan check100 --workers 7
$E collect --campaign pilot
python -m regime_fusion.analysis pilot --campaign pilot
python -m regime_fusion.analysis compare --campaign pilot --stage 1 --plan runs --rungs A0 A1 A2 A3 A4 A5
```

Measured on Michael's MacBook Air (Apple M1: four fast and four slow cores) on 29 October 2026, a full-size run takes 4–6 seconds alone (A0 to A4), 13–15 seconds with lexicase or DALex, and about three times as long when seven run at once (with the GDELT extraction also running). Seven at a time, the GP-parameter pilot takes about 2 hours and the rest about 10 (exit settings about 3, the runs about 6, the 100-run check about 1.5): some 12–16 hours in all, as a fanless laptop slows under long loads. Every run evaluates on 2018 only; the planning refuses anything later.

