# GP engine

How the strongly-typed GP, the backtester and the runs are built (`src/regime_fusion/engine`). The plan is in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md), decisions in [`DECISIONS.md`](DECISIONS.md). The last section lists the choices made while building the engine, for Michael to review.

## Representation

An individual has one **branch** per analysis type and a **fusion tree** that combines the branches' buy signals.

- **Branches.** A branch of type FA, SA or TA is a Boolean tree, exactly as in ActTrade:
  - AND and OR of the same type join two subtrees;
  - GT and LT compare an indicator of that type with a random constant in [−1, 1].
- **Fusion.** The fusion tree's leaves name a branch.
  - Its functions are AND and OR (two children) and ALL, ANY and VOTE (three children: all, any, or at least two of them).
  - From rung A2 on there is also IF: a regime condition, then two fusion trees. The condition is a Boolean tree over regime probabilities.
  - ActTrade's 3-AND root is the fixed fusion ALL(FA, SA, TA).
- **Indicators** are scaled to [−1, 1] on the training rows only (`engine/panel.py`): by minimum and maximum, as in ActTrade, or by rank as a robust option for the pilot. Regime probabilities *p* enter as 2*p* − 1. A comparison with a missing value is false.

## Evolution (ActTrade's, `engine/gp.py`)

| Item | Setting |
| --- | --- |
| Population, generations | 1000, 50 |
| Initialisation | Ramped half-and-half, branch depth up to 2 (ActTrade's depth 3 counts its root) |
| Maximum branch depth | 5 (ActTrade: 6 with the root) |
| Crossover | Probability 0.95. Subtree crossover in each branch in turn (FA, SA, TA, then the fusion tree when it evolves), with points of the same type and class; function nodes 90% of the time. A child that is too deep keeps the first parent's tree |
| Point mutation | Probability 0.05. One tree is picked; each of its nodes changes with probability 0.5 (AND↔OR, GT↔LT, another indicator of the same type, a new constant; in the fusion tree AND↔OR, ALL/ANY/VOTE, another leaf) |
| Selection | Tournament of 4, with elitism of 1 |
| Active-trading operator | Each generation, three offspring join the next generation (Algorithm 1 of the paper). Each takes one type's highest-Sharpe branch and the other types' most-trading branches, ties going to smaller branches. The first parent of a crossover (or the parent of a mutation) is selected with probability 0.4 and is one of these offspring with probability 0.2 each |
| Fitness | Weighted sum of the Sharpe ratios of the whole tree and of each branch alone. A signal with fewer than 3 trades scores zero |
| Model of a run | The best individual by that fitness in the last generation, for every rung |

## Ablation ladder (`configs/rungs.yaml`)

| Rung | Change | Notes |
| --- | --- | --- |
| A0 | ActTrade on the corrected backtest | Fixed ALL(FA, SA, TA) |
| A1 | Evolvable fusion: AND, OR, ALL, ANY, VOTE | |
| A2 | + IF gating on volatility regimes | Probabilities of the HMM on realised volatility |
| A3 | DC regimes instead of volatility regimes | A3 vs A2 isolates "DC vs volatility" |
| A4 | + news-attention DC in the regime inputs | Attention of the 50 main stocks together, so regimes stay shared (`configs/regimes.yaml`) |
| A5 | + ε-lexicase selection | Cases: annualised mean daily return of each signal (whole tree, each branch) in each of 8 time blocks; ε = median absolute deviation |
| A5-dalex | DALex instead of ε-lexicase | Particularity pressure 20 |
| GP-FA, GP-SA, GP-TA | One branch only, fitness = its Sharpe ratio | No active-trading operator (it needs three branches) |

## Backtest and metrics (`engine/backtest.py`, `engine/metrics.py`)

- **Trading rules.**
  - One long position at a time.
  - A signal at the close of *t* buys at the open of *t*+1.
  - Take-profit is a limit order at entry × (1 + *r*), filled on the first day whose high reaches it (at the open after a gap).
  - Otherwise the position is sold at the close of the *d*-th day.
  - Costs are charged per side.
- **Daily returns** are marked to market and compound exactly to each trade's net return.
- **Sharpe ratio** is annualised from daily returns with no risk-free rate, flat days counting as zero.
- **Other measures:** total and annual return, maximum drawdown, CVaR 95%, trades per year, exposure and hit rate.
- **Speed:** a 1,000-day backtest takes well under a millisecond (numba).

## Runs (`engine/run.py`, `python -m regime_fusion.engine`)

- **Stages.** Stage 1 trains on the three training years and evaluates on the validation year; stage 2 trains on all four years and evaluates on the test year.
- **Fitness and evaluation.** Fitness uses the primary cost (10 bp per side); the model is evaluated at 0, 5, 10 and 25 bp.
- **Seeds** depend on stock, test year, run number and stage, never on the rung, so every configuration sees the same seeds.
- **Exit settings** (*d*, *r*) are chosen per stock and test year on the validation year with A0. Each candidate of a grid (`exit_grid` in `configs/rungs.yaml`) gets 5 runs, and the best median validation Sharpe wins. Every method then uses the choice.
- **Results.** Each run writes `$RFGP_DATA_DIR/results/<campaign>/<rung>/<stock>/<year>.stage<k>.run<NN>.json`. It holds the model in readable form, training statistics, evaluation metrics at each cost, each branch's evaluation, the generation history and a provenance record (code version, configuration and rung digests, seed keys, package versions).
- **Speed.** A full ActTrade-size run (1,000 × 50) on Apple with FA and TA took 6.9 s, about 0.14 ms per tree. The plan's budget is 0.25 ms per tree (16 s per run).

## Campaigns (`engine/tasks.py`)

A campaign is planned once as numbered tasks, which job-array elements on CERES (or a loop on one machine) then run by number:

```bash
python -m regime_fusion.engine tasks --campaign pilot --kind exits --years 2019 --development
python -m regime_fusion.engine tasks --campaign pilot --kind runs --rungs A0 A1 A2 A3 \
    --years 2019 --stages 1 --development
python -m regime_fusion.engine run-task --campaign pilot --kind exits --index 0   # one per array element
python -m regime_fusion.engine status --campaign pilot
python -m regime_fusion.engine collect --campaign pilot     # results/pilot/runs.parquet
```

- **Tasks.** An `exits` task chooses one stock-year's exit settings (the grid's 16 candidates, 5 runs each). A `runs` task evolves one rung on one stock, test year and stage, for all its runs (30, about 8 minutes at full size) or a block of them (`--block`). The exit settings of a stock-year must exist before its runs start.
- **Restarts.** Finished runs and exit choices are skipped, and result files are written whole or not at all, so a task stopped by a time limit is simply submitted again. `status` lists unfinished tasks.
- **Development-only campaigns** (`--development`, the pilot) may evaluate on data up to the end of 2018 only: planning refuses any task that would evaluate later (stage 2 of 2019, or anything for 2020 on), and running checks again.
- **Plans are fixed.** The numbering is saved in `results/<campaign>/tasks.<kind>.json` and is not overwritten while jobs may be using it (`--replace` plans again).
- `collect` gives one row per run: rung, stock, test year, stage, run, exit settings, training fitness, every evaluation measure at every cost (such as `sharpe_10bps`), each branch's Sharpe ratio and trades, and the model in readable form.
- A run whose stock has no usable indicator of a needed type (values on fewer than half of the training days) stops with a message, as A0 does now for sentiment until the GDELT extraction is finished.

## Rule-based benchmarks (`engine/benchmarks.py`)

All of them trade through the same backtest, exit settings, *t*+1 execution and costs as the GP models:

- buy-and-hold;
- ActTrade's TS(*d*, *r*), which buys whenever flat, so the exit rule alone decides;
- random entry at the rate that matches the model's number of trades (1,000 draws), with the model's percentile in that distribution;
- moving-average crossover;
- time-series momentum.

The machine-learning benchmarks follow in a later step.

## Machine-learning benchmarks (`src/regime_fusion/ml`, `configs/ml.yaml`)

They follow the plan's fairness rules and run through the same commands as the GP rungs (`run --rung ML-LGBM`, or as rungs of a campaign):

| Benchmark | Model |
| --- | --- |
| ML-LR | L2-regularised logistic regression (missing values enter as 0, the middle of the scaled range) |
| ML-LGBM | LightGBM gradient-boosted trees |
| ML-RS-LGBM | One LightGBM per regime (the most probable state of the regime probabilities); a regime with fewer than 100 labelled training days uses the model trained on all days |

- **Features:** the GP's indicators (FA, SA and TA, scaled on the training rows as for the GP) plus the regime probabilities of rung A3.
- **Labels match the exit rule:** for each day, whether the trade a buy signal would open (at the next open, sold at the take-profit or at the close of the *d*-th day) gains after costs at the primary level. Labels come from the training rows alone, so the last trades of a training window, which would end in the next period, are left out.
- **Decision threshold:** buy when the predicted probability reaches the *q*-quantile of the model's own training predictions. A quantile rather than a fixed probability carries over when the model is retrained on four years.
- **Tuning on validation only.** In stage 1, every combination of the model's grid (4 for ML-LR, 12 for the LightGBM models) is trained on the three training years, and each is paired with each quantile (0.5 to 0.9). The pair with the highest validation Sharpe ratio wins; fewer than 3 trades score zero, as in the GP's fitness. In stage 2, the chosen pair is retrained on all four years and trades the test year.
- **Runs and seeds.** Seeds use the GP's keys (stock, test year, run, stage). These three models are deterministic, so they get one run each.
- The neural gated fusion and the Temporal Fusion Transformer follow in a later step.

## Choices to review

1. **Backtest.** ActTrade's code opens a new trade on every day with a true signal, even when trades overlap, and buys at that day's close. A0 uses the corrected backtest above, as the plan intends.
2. **Sharpe ratio.** ActTrade's code uses per-trade returns with a risk-free rate of 0.022%. Here every Sharpe ratio, in the fitness too, comes from daily returns, the study's primary metric.
3. **Fitness weights.** The paper does not give *w*<sub>C</sub>, *w*<sub>FA</sub>, *w*<sub>SA</sub> and *w*<sub>TA</sub>; its code reads them from the command line. Equal weights (0.25) are used until the values are known.
4. **Point mutation.** The paper says it acts on one of the three branches; the code mutates every node of the whole tree. The engine follows the paper.
5. **Scaling.** ActTrade scaled indicators to [−1, 1]. Here the scaling is fitted on training rows only, with rank scaling available as a robust alternative for the pilot.
6. **Regime rungs.** A3 replaces the volatility regimes rather than adding to them, so that A3 against A2 answers "do DC regimes beat volatility regimes?". The IF condition is a typed Boolean tree over regime probabilities.
7. **Evaluation start.** Evaluation periods start without a position, so a model's first possible purchase is on the second day.
8. **News regimes.** A4's news attention pools the 50 main stocks, so its regimes are shared like A2's and A3's, as the plan asks. Regimes from each stock's own news are one setting away (`scope: stock`), for the pilot to try.
9. **Benchmark thresholds as quantiles.** "Decision thresholds maximise validation Sharpe" is read as choosing, on the validation year, the quantile of the model's training predictions above which it buys.
