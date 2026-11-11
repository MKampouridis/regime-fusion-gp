# Plan for approval: a GP that scores stocks, and the regime question on a long history (E10)

**Status: PROPOSED on 11 November 2026. Nothing here is built or run until Michael approves or changes it.**

Written by Claude at Michael's request after E9 ([`DECISIONS.md`](DECISIONS.md), 11 November 2026). It follows the same discipline as [`E9_PLAN.md`](E9_PLAN.md): rules fixed before any run, every variant reported, a positive control before real data.

## What E9 and its control showed

- **The GP has skill before costs** on small caps (5.4% a year, p = 0.011) and none on large caps, as expected.
- **It loses to trading.** A yes/no rule drops a stock the moment it crosses a threshold: turnover 0.25 a period against 0.09–0.17 for banded benchmarks, and 2.9% a year after costs against 4.6–6.2% for them.
- **Its fitness is short-sighted.** A rule is judged on the slice of stocks it buys. Learners that use every stock find a weak factor (IC 0.02) in 95% of samples; the GP in 60%.
- **Regimes cannot be settled on seven years.** A planted regime effect as strong as the whole real composite is found in under half the runs.
- **The grammar cannot reverse a factor between market states.** Each rule has one fundamental and one technical branch; IF only switches between combinations of them.

Two design changes answer the first, second, third and fifth points. A longer history answers the fourth.

## Part 1: the design (E10)

### From yes/no rules to scores

| Item | E9 | E10 |
| --- | --- | --- |
| Output of a rule | Buy or not, per stock | A number per stock, ranked across the stocks of the date |
| Holdings | Stocks for which the rule is true, rebuilt each date | Banded: enter in the top 20% of the score, leave below the top 40% (as the benchmarks) |
| Branches | One Boolean tree per source (fundamental, technical) | One arithmetic tree per source over the same ranked inputs |
| Functions in a branch | AND, OR, comparisons with a constant | add, subtract, multiply, negate, minimum, maximum, constants; IF(condition, a, b) |
| Fusion | AND, OR, IF | weighted sum, minimum, maximum, IF |
| Regime conditions | In the fusion only | In the fusion and inside branches, so a factor can be used one way in one state and the other way in another |
| Size limit | Depth 3 per branch | At most 15 nodes per branch, 7 in the fusion |
| Model of a run | Fittest rule | Fittest rule |
| Reported GP result | The 20 runs' books averaged | One banded book of the 20 runs' mean rank score (an ensemble: less noise, less trading) |

The typed structure of the paper is kept: separate branches per source, fused under conditions on the regime. A sentiment branch slots in when news data exist for the universe.

Rungs, each adding one thing:

| Rung | Adds | Question |
| --- | --- | --- |
| D1 | Scores, banded holdings, no IF | Does scoring close the gap to the benchmarks after costs? |
| D2 | IF on the stock's value traded and size | Does it pay to score thin and liquid stocks differently? |
| D3 | IF on the market state, in fusion and branches | Does regime conditioning help? This is the paper's question. |

### Fitness

Two candidates, both computed on training dates only. The positive control chooses between them; real returns play no part in the choice.

- **Book fitness:** the 95% lower bound of the Sharpe ratio of the banded book's result per period on ranked returns, after costs scaled as approved for E9. It optimises what is traded, costs included.
- **Rank fitness:** the same lower bound applied to the per-date rank correlation between the score and the returns that follow, across all stocks. It uses every stock, so it should be the more sensitive; costs then enter only through the banded holdings.

Rule for the choice, fixed now: the one with the higher mean detection rate over the single-factor cells at IC 0.02 and 0.03; within 5 points, the book fitness.

### The positive control comes first again

The control of E9 is repeated for E10 on the planted data already defined, with two additions:

- **P4, reversal:** returns rise with a characteristic in one market state and fall with it in the other. No E9 rule could express this.
- **Long samples:** cells with 150, 315 and 630 dates, about 12, 25 and 50 years (the 81 real dates resampled in blocks), to measure how much history the regime plants P3 and P4 need. This decides in advance whether the years before 2001 are needed at all (see the periods below).

Conditions before any real data, as first written for E9 and not relaxed: (1) with nothing planted, test results centre on zero; (2) the single factor at IC 0.05 with 700 stocks is found in at least 80% of runs; (3) in no single-factor cell is the GP more than 20 points below the best simple learner. P2, P3 and P4 are reported. If the conditions fail after two repairs, see the stop rules.

### What the small caps of 2012–2018 are now

Development data only. E9's results for 2015–2018 have been seen and have shaped this design, so no claim of out-of-sample performance will be made on them. E10 is run on them to check that turnover falls to the benchmarks' level and that nothing is broken.

## Part 2: the data (WRDS)

### What to obtain

| Data | Use |
| --- | --- |
| CRSP daily and monthly stock files with delisting returns, common shares on NYSE, AMEX and NASDAQ | Returns without survivorship bias, value traded, betas |
| Compustat annual and quarterly fundamentals with report dates, and the CRSP–Compustat link | The 11 fundamental factors, known at the date used |
| Industry codes (SIC) | Sector-neutral ranks, reported as a variant |
| CRSP daily high, low and closing prices | Spread estimates (Abdi and Ranaldo, 2017) in place of the assumed cost tiers |
| Fama–French factor returns | Audit: our value, size, momentum and profitability portfolios against the published ones |

Analyst forecasts (I/B/E/S) and earnings-surprise signals are left for later: they change the inputs, which is a separate question from the one asked here.

### Universe and periods

- **Universe:** common stocks above the 20th percentile of NYSE market value at each June, price at least $1. Stocks below that line are reported separately, since E9's signal sat in thin stocks.
- **Bridge, 2010–2018:** the same years as now, without survivorship bias. This measures how much the bias in the present small-cap data matters. Development data.
- **Clean test A, 2019–2025 (the main test):** not looked at for this universe. Rules evolved on the 15 years before each test year.
- **Clean test B, 2001–2009:** walk-forward, each block of three years tested with rules evolved on the 15 years before it. Nobody on this project has looked at these years. Like test A it lies after decimalisation (2001), so spreads and market structure are comparable with today's.
- **Years before 2001:** training data for test B. Markets then worked differently (prices in fractions, far wider spreads, little algorithmic trading) and most of the 17 factors were discovered on those decades, so no claim of profit is made on them. They serve as a separate test of the regime question only (1975–2000, rules 4 and 5 below), and only if the control's long samples show that the 25 years from 2001 hold too few dates to detect a regime effect. Otherwise that test is dropped.
- **Why any old data at all:** a regime effect is counted in episodes, not days. There are about two bear markets since 2010, four since 2000 and eight since 1975.
- **A limit to state plainly:** the 17 factors come from a literature that studied the decades before 2010, so test B is clean for the GP's design but not for the choice of inputs. The comparison with benchmarks on the same inputs is unaffected by this.
- **Where the paper goes matters:** referees in finance expect long samples and may ask for the early decades; a computing venue will not.

### Practical points for Michael

- **Login:** Michael runs the download with his own WRDS account; Claude prepares the script and never logs in for him.
- **Licence:** WRDS data are licensed and this repository is public. Data stay outside Git, as now. Michael should check whether Essex's WRDS terms allow licensed data to be processed in a session with an AI assistant; if in doubt, Claude works only from aggregated outputs.
- **Size:** the CRSP daily file is several gigabytes. Disk space on the server is checked before the download.
- **Computing:** the walk-forward over two test periods with three rungs is larger than anything run on this server so far; it may go to CERES, where Michael submits the jobs.

## Rules for success on the clean tests (draft, to be frozen before either test is opened)

Applied to test A and test B separately, long books hedged with the index, costs from estimated spreads.

1. **Skill:** D1's result before costs is above zero, one-sided p < 0.05.
2. **Profit:** D1's result after costs is positive overall and in at least 60% of the years of the test.
3. **Against the benchmark named in advance, banded ridge on the same inputs:** beats, matches or fails, at p < 0.05 for the paired difference after costs.
4. **Regimes:** D3 improves on D1 if the paired difference after costs is positive with p < 0.05. This is the paper's central claim.
5. **Conditions named in advance:** the four of E9, with Holm adjustment.

The parts of the design still open (the fitness, any repair after the control) are settled on the control and the development data, and the whole is frozen in writing before either test is opened.

## Stages, costs and stop rules

| Stage | What | New data | Rough effort | Stop rule |
| --- | --- | --- | --- | --- |
| S1 | Build the scoring GP; repeat the control with P4 and long samples | None | 3–4 days of building, a day of runs | Control fails after two repairs: stop the GP line and write the methodological paper |
| S2 | E10 on small caps 2012–2018, development only | None | 1 day | Turnover not near the benchmarks': fix before going on |
| S3 | WRDS download, panel, audit against Fama–French, bridge 2010–2018 | WRDS | 1 week | Audit fails: fix the data code first |
| S4 | Freeze; clean tests A and B | — | 1 week, probably CERES | Rules 1–5 decide what the paper reports |

S1 and S3 do not depend on each other, so the WRDS download can start while S1 is built.

## What I expect

| Outcome on the clean tests | Chance |
| --- | --- |
| D1 matches the benchmarks after costs | about 60% |
| D1 beats banded ridge | about 15–20% |
| D3 improves on D1 (regimes help) | about 25% |
| A condition named in advance counts | about 25% |

Whichever way it falls, the result is reportable: a positive control, a design fixed in advance and two clean tests are more than most studies of GP trading offer.

## For Michael to approve or change

1. The scoring design: arithmetic branches, banded holdings, the ensemble of 20 runs as the reported result.
2. The two candidate fitnesses and the rule for choosing between them on the control.
3. The control's additions (P4 and long samples) and its three conditions, unrelaxed.
4. Small caps 2012–2018 as development data only from now on.
5. The WRDS request, the universe and the two clean tests (2019–2025 as the main one, 2001–2009 as the second; the years before 2001 only as training data, and as a regime-only test if the control shows they are needed).
6. The draft rules for success, in particular banded ridge as the named benchmark.
7. Whether the WRDS download starts now, alongside S1.
