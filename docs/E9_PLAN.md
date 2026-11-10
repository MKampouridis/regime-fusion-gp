# Plan for approval: a positive control and experiment E9

**Status: PROPOSED on 10 November 2026. Not approved. Nothing in this plan runs before Michael approves it.**

Written by Claude at Michael's request, after the study of 10 November 2026 ([`DECISIONS.md`](DECISIONS.md)). It follows the rules of the development experiments (4 November 2026): data up to `development_end` (2018) only, no hold-out stock, rules for success fixed before any run, every variant reported.

## Why these two experiments

The study found three faults in how the long/short experiments (E6–E8) were set up and measured:

- their inputs compared each stock with its own past, never with other stocks;
- their portfolios were dollar-neutral, so the market's direction swamped the signal (a small-cap composite of 17 factors has a Sharpe ratio of 0.56 measured that way and 1.24 once the market is hedged out);
- costs were flat, although most of the signal sits in thinly traded stocks.

It also found that re-weighting the 17 factors does not beat their equal-weight composite, and that no market state predicts the composite's return on seven years of data.

So two questions come before any new data:

1. **Does the GP find a signal when one is there, and how weak a signal can it find?** No experiment so far can answer this: every result was "nothing found", which a faulty search and an empty data set both produce. The positive control answers it.
2. **With the measurement fixed, does the GP produce consistently positive results out of sample, and how does it compare with simple alternatives on identical terms?** E9 answers this.

Both use only data already held (`dev-data/small`, `dev-data/broad`, the 50 main stocks to 2018) and the measurement code in `src/regime_fusion/dev/crosssection.py`.

## Terms

- **Decision date:** every 20th session. Inputs are known at its close; trades happen at the next open and are held to the open 20 sessions later.
- **Hedged return:** a stock's return less its beta times the return of the equal-weight universe over the same window. Betas come from 5-session returns over the last 250 sessions. In practice the hedge is one index position.
- **Book:** the stocks a rule selects on a decision date, equal weights. Its result is its hedged return in excess of the universe's mean hedged return.
- **Tiered costs:** a cost per side that rises as a stock trades less: 10 bps above $10m a day, 20 bps from $5m, 35 bps from $2m, 60 bps from $1m, 100 bps from $0.5m, 200 bps below. These tiers are an assumption; results are also given at flat 10, 25 and 50 bps and with the tiers doubled.
- **IC:** the rank correlation, across stocks on one date, between a signal and the returns that follow.

## Part 1: the positive control

A signal of known strength is planted in the data. Each design is then asked to find it.

### 1A. Ranking stocks (the design E9 uses)

- **Base data:** the small-cap panel, 81 decision dates from July 2012 to November 2018. Each date's hedged returns are shuffled across its stocks. This keeps every date's spread of returns and removes any real link with the 17 factors, which stay in the inputs as decoys.
- **Planted input:** a new slow-moving characteristic for every stock (each period's value is 0.9 of the last plus fresh noise), given to the GP as one more ranked input.
- **Planted return:** a component proportional to the planted signal is added to the shuffled returns, scaled so that the signal's IC equals a target.
- **Three plants:**

  | Plant | What is planted | What finding it needs |
  | --- | --- | --- |
  | P1, one factor | Returns rise with one characteristic | One comparison |
  | P2, fusion | Returns are higher only when a fundamental-type and a technical-type characteristic are both high | AND across the two branches |
  | P3, regime | Returns follow one characteristic in one market state and another in the other state (states last about ten periods) | IF on the market state |

- **Strengths (target IC):** 0 (nothing planted), 0.02, 0.03, 0.05 and 0.10. For comparison, the real composite's IC is 0.03 in stocks trading at least $2m a day, 0.066 in all members and 0.10 in the thinly traded half.
- **Sample sizes:** 50, 300 and 700 stocks (random subsets), training on the first 57 dates and testing on the last 24. For 700 stocks, also training on 36 dates only.
- **Learners:** the E9 GP (below), and three references: the best single input by training IC, ridge regression, LightGBM.
- **Runs:** 20 per cell; 60 cells; about 1,200 GP runs of a few seconds each.
- **Measures:**
  - detection rate: the share of runs whose final rule uses the planted input in the part that decides (for P2 both inputs, for P3 both inputs and the state);
  - share of the return captured: the test-period return of the rule's book as a share of the return of the rule that knows the plant;
  - false discoveries: with nothing planted, the training fitness reached (the noise level E9's fitness must be read against) and the test-period return, which should centre on zero.

### 1B. Timing one stock (the campaigns' design)

- **Data:** the ten stocks of the 100-run check, training 2015–2017, evaluation 2018 (the pilot's window).
- **Plant:** a daily indicator (each day 0.9 of the last plus noise), added to the technical branch. Every day's four prices are scaled so that the day's return gains a component proportional to the previous day's indicator.
- **Strengths:** IC of the indicator with the next 20 sessions' return of 0, 0.05, 0.10, 0.20 and 0.30.
- **GP:** rung A1 as in the development experiments (population 500, 20 generations, exit fixed at 20 days and 3%), 10 runs per stock and strength: 500 runs.
- **Measures:** the share of runs whose rule uses the planted indicator; the 2018 Sharpe ratio against buy-and-hold of the same planted prices and against the rule "buy while the indicator is positive".
- **What it settles:** whether the campaigns' engine can learn a timing signal at all, and from what strength. The theory note expects about 0.14 for an information ratio of 0.5. If the engine misses a signal of 0.30, the fault is in the engine and the null results so far say less than we thought.

### What the control must show before E9 runs

1. With nothing planted, test-period returns centre on zero (within two standard errors) for every learner.
2. With 700 stocks and 57 training dates, the GP detects P1 at IC 0.05 in at least 80% of runs.
3. In no P1 cell is the GP's detection rate more than 20 points below the best reference learner's.

If any of these fails, the GP is repaired first and the control repeated. P2, P3 and part 1B are reported, not pass or fail. P2 and P3 tell us how strong a fusion or regime effect must be before seven years of data can show it, which decides whether the regime question needs a longer history.

## Part 2: experiment E9

### Data and folds

- **Universe:** small caps (`dev-data/small`, survivors only), members by the June ranking, 81 decision dates.
- **Test years:** 2015, 2016, 2017 and 2018. Each is tested with rules evolved on every earlier date from July 2012, leaving out the last date whose holding period runs into the test year.
- **Inputs, all known at the close of the decision date:**
  - the 17 factors of `dev/factors.py`, each as its rank among the members that day. The literature's signs are not given to the GP;
  - fundamental branch (11): earnings, book, sales and cash flow to price, operating profitability, return on equity, accruals, leverage, asset growth, share issuance, earnings change;
  - technical branch (6): momentum, one-month reversal, 52-week high, volatility, beta, size;
  - stock context, for IF conditions: rank of value traded, rank of size;
  - market state, for IF conditions: the universe's volatility over 60 sessions, its return over 250 sessions and its drawdown, each as a percentile of its own history to date.
- No sentiment branch: the small caps have no news data yet.

### From rule to portfolio

- The stocks for which the rule is true form the book, equal weights, rebuilt at each decision date, hedged with the index.
- A rule must select between 10% and 40% of the members on at least 90% of the training dates. Otherwise it scores the floor.
- Only the long book is traded in the main version, so no single stock is sold short.

### The GP

| Item | Setting |
| --- | --- |
| Fitness | The 95% lower bound of the Sharpe ratio (`dev/fitness.py`) of the book's result per period after tiered costs, over the training dates; whole model only |
| Trees | Two branches (fundamental, technical), at most depth 3; fusion by AND and OR |
| Search | Rung A1's operators, population 500, 20 generations, tournaments of 4 |
| Runs | 20 per test year and rung; seeds from the test year and run number only |
| Model of a run | Best fitness in the last generation |
| Reported GP result | The 20 runs' books averaged |

Three rungs, each adding one thing:

| Rung | Adds | Question |
| --- | --- | --- |
| C1 | Rules over ranks against other stocks, hedged, costed | Does the repaired set-up give positive results? |
| C2 | IF on the stock's value traded and size | Does it pay to treat thin and liquid stocks differently? |
| C3 | IF on the market state | Does regime gating help? (The study suggests not, on seven years.) |

### Benchmarks, on identical terms

Same universe, dates, hedge, costs and book size (the top fifth of each score):

- the equal-weight composite with the literature's signs, rebuilt each date and with banded holdings (a stock enters in the top 20% and leaves below the top 40%);
- weights from past ICs; the best single factor so far; ridge regression; LightGBM; each rebuilt and banded;
- 1,000 random books of the GP's book size, as the no-skill distribution.

The benchmark named in advance for the comparison is the **banded composite**. Over 2012–2018 its long book earned 5.7% a year before costs and 3.2% after tiered costs (2015: 3.3%, 2016: 7.0%, 2017: −0.6%, 2018: 0.9%). Over 2015–2018 alone that is 4.6% a year before costs (t 2.1) and 2.7% after (t 1.2).

### Rules for success, fixed now

Differences and significance use a stationary block bootstrap of the periods (mean block of 3 periods, 10,000 resamples).

1. **Skill.** Before costs, the GP book's result over 2015–2018 is above zero with one-sided p < 0.05.
2. **Profit.** After tiered costs, the GP book's result is positive over 2015–2018 together and in at least three of the four years.
3. **Against the benchmark.** The paired difference between the GP (C1) and the banded composite after costs: "beats" if positive with p < 0.05, "fails" if negative with p < 0.05, otherwise "matches".
4. **Rungs.** C2 against C1 and C3 against C2 count as improvements only if the paired difference is positive with p < 0.05 (Holm-adjusted over the two) and the later rung is not worse in more than one year.

E9 counts as a positive result if rules 1 and 2 hold for C1. Four test years give little power after costs: the banded composite itself passes rule 1 (t 2.1) and only just passes rule 2 (three years of four). That is why rule 1 is stated before costs and rule 2 by sign.

### Also reported, not part of the rules

- A long/short version, short positions only in stocks trading at least $1m a day.
- Flat costs of 10, 25 and 50 bps, and the tiers doubled.
- The large caps (`dev-data/broad`, 2010–2018) as a check for false positives: the study found nothing there since 2015, so a GP that reports profits there is suspect.
- 2014 as an extra test year (its rules see only 18 months of training).
- The rules in readable form, and how often each input appears.

### What I expect

The GP should land close to the benchmarks: skill clearly above zero before costs, about 2–4% a year after costs, "matches" under rule 3. I put the chance of "beats" at about one in three, and of C3 improving on C2 at about one in ten. Writing this down first keeps the reading of the results honest.

## Cost

- Code: `dev/control.py` (planting), `dev/e9.py` (the book market, folds, benchmarks, report) and their tests: about three days of work.
- Computing: about 1,700 control runs and 240 E9 runs (plus the large-cap check), a few seconds each: under two hours on this machine. Nothing goes to CERES.

## What happens next, by outcome

| Outcome | Next step |
| --- | --- |
| The control fails | Repair the GP; E9 waits |
| Control passes, E9 fails on skill | The rules do not carry over from training to test even with a known signal in the inputs: look at the fitness and the model choice before anything else |
| Skill holds, profit fails | Costs are the obstacle: add separate entry and exit rules, so that holdings change less often |
| Both hold | Freeze the recipe, then confirm on survivorship-free data (WRDS) and on small caps 2019–2025, which have not been downloaded or looked at |

## Limits to keep in mind

- The small-cap data hold only firms that are still listed today. The direction of the bias is uncertain; WRDS removes it, later.
- The cost tiers are an assumption. Spreads estimated from daily highs and lows would replace them, later.
- The signal the factors carry is the known quality and low-risk premium, strongest where trading is hardest. E9 can show that the GP finds and trades it sensibly. It cannot show that the GP discovers something new.

## For Michael to approve or change

1. The control's two parts and the three conditions before E9 runs.
2. The main portfolio: long book only, hedged with the index.
3. The cost tiers.
4. The four rules for success, in particular rule 2 (positive in at least three of four years) against a stricter "every year".
5. The three rungs C1–C3.
