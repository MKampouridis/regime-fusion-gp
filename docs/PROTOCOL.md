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

- **Measure:** annualised Sharpe ratio of daily returns at 10 basis points per side, in the test year.
- **Unit:** the median over runs of each stock-year: 50 stocks × 7 years = 350 paired units.
- **Confirmatory tests:** two-sided Wilcoxon signed-rank tests on the paired stock-year medians, with Holm's adjustment across these six at α = 0.05:

  | Comparison | Question |
  | --- | --- |
  | A1 vs A0 | Does flexible fusion alone help? |
  | A2 vs A1 | Does regime information help? |
  | A3 vs A2 | Do DC regimes beat volatility regimes? |
  | A4 vs A3 | Does news attention add anything? |
  | A5 vs A4 | Does ε-lexicase beat the weighted-sum fitness? |
  | A5 vs A0 | Does the full method beat ActTrade? |

  Each result is reported with the median paired difference and a bootstrap 95% confidence interval (stock-years resampled).
- **All methods together** (exploratory): the Friedman test on stock-years that every method covers, with Conover's comparisons, Holm-adjusted.

## 5. Portfolio-level tests

For each method, an equal-weight portfolio: within a stock-year the daily returns of all its runs are averaged (no run is picked after the fact), then the stocks are averaged day by day, over the seven test years (`analysis portfolio`).

- **Hansen's SPA test:** every GP rung against the best non-GP benchmark, the one with the highest test-period Sharpe ratio. Picking the benchmark after the fact favours the benchmark.
- **Deflated Sharpe Ratio** of each method. The number of trials counts every configuration evaluated, in the pilot and in the main campaign, as listed in the campaigns' task plans.
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
- **FinBERT headline sentiment** (option B): A0, A1 and the best rung by §4 re-run with FinBERT scores added to the sentiment branch, 2019–2023.
- **Hold-out stocks:** every method, with the analysis of §4–6 unchanged.

## 8. Reporting rules

Every rung and benchmark is reported, whatever its result. The confirmatory family of §4 is not changed after test results are seen. Exploratory results are labelled as such.

## 9. Decisions the pilot makes

Each is settled on 2015–2018 data only (stage 1 of the 2019 window), by the rule given here, and every option tried counts as a trial for the Deflated Sharpe Ratio.

| Decision | Options | Rule |
| --- | --- | --- |
| Runs per configuration | 30 or 50 | 100 runs of A0 and A1 on 10 stocks (ABT, DIS, GD, MS, NFLX, NSC, SO, SPG, UPS, WFC, drawn with the study's seed on 29 October 2026). Keep 30 if the median distance (95th percentile) between a 30-run median and the 100-run median is at most 0.05 in annualised Sharpe ratio, half the smallest difference of interest (0.1); otherwise 50 (`analysis pilot`) |
| Generations | 50 or 75 | Keep 50 unless more than a quarter of A1's runs still raise their best training fitness by more than 1% over the last 10 generations, and the time budget allows 75 |
| Population | 1,000 or 500 | Keep 1,000 unless the median full run takes more than 30 seconds (twice the plan's budget) |
| DC threshold of the price regimes (A3) | 0.5% (`A3-fine`) or 1% (`A3`) | Options whose fitted regimes give a state less than 15% of the 2015–2017 days are dropped; of the rest, the higher median validation Sharpe ratio of A3 over the 50 stocks, 10 runs each; ties go to the smaller threshold |
| Threshold of the news-attention DC (A4) | 0.25 (`A4`) or 0.5 (`A4-wide`), in log units | As for A3, with A4, after its price threshold is set to A3's choice |
| Scaling of the indicators | min-max (`A1`) or rank (`A1-rank`) | The higher median validation Sharpe ratio of A1, 10 runs each |
| Fitness weights | Equal (0.25 each) | Kept unless ActTrade's own values are found |
| Sentiment indicators | The twelve GDELT indicators | Open in `DECISIONS.md` until Michael decides |
| GDELT name rules | `configs/news_names.yaml` | Revised on the 2015–2018 names report, before any pilot run |

### Running the pilot

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
python -m regime_fusion.analysis compare --campaign pilot --stage 1 --rungs A0 A1 A2 A3 A4 A5
```

At the plan's speed this is about 60 CPU hours (exit settings about 18, runs about 30, the 100-run check about 9), roughly 9 hours on seven cores. Every run evaluates on 2018 only; the planning refuses anything later.

