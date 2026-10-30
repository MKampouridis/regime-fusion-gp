"""Statistical tests of the study (docs/RESEARCH_PLAN.md, evaluation protocol).

* ``holm``: Holm's step-down adjustment of p-values.
* ``sharpe_differences``: the confirmatory tests (docs/PROTOCOL.md, section 4):
  the difference between two portfolios' Sharpe ratios, by a stationary
  bootstrap of the days, Holm-adjusted.
* ``wilcoxon_pairs``: Wilcoxon signed-rank tests between methods on paired
  per-stock-year results (medians over runs), Holm-adjusted: the per-stock view.
* ``clustered_by_year``: the same comparisons clustered by year, a t-test over
  the years of each year's mean difference (a model with a random year effect).
* ``friedman`` and ``conover``: the Friedman test across methods and Conover's
  post-hoc comparisons for the Friedman design (Conover, 1999), Holm-adjusted.
* ``deflated_sharpe``: the Deflated Sharpe Ratio (Bailey and Lopez de Prado,
  2014): the probability that the true Sharpe ratio exceeds the best one
  expected by chance among the trials made, allowing for skewness and kurtosis.
* ``pbo``: the probability of backtest overfitting by combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado and Zhu, 2017).
* ``spa``: Hansen's (2005) test for superior predictive ability, consistent
  version, with Politis and Romano's stationary bootstrap.

Sharpe ratios inside the Deflated Sharpe Ratio are per period (daily), not annualised.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def holm(p_values) -> np.ndarray:
    """Holm's adjustment; a missing p-value stays missing and does not count as a test."""
    p = np.asarray(p_values, dtype=float)
    finite = np.flatnonzero(~np.isnan(p))
    adjusted = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(finite[np.argsort(p[finite])]):
        running = max(running, (len(finite) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def wilcoxon_pairs(table: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Rows of ``table`` are matched units (stock-years), columns are methods."""
    rows = []
    for a, b in pairs:
        both = table[a].notna() & table[b].notna()
        first, second = table.loc[both, a], table.loc[both, b]
        difference = first - second
        result = stats.wilcoxon(first, second) if (difference != 0).any() else None
        rows.append({"first": a, "second": b, "n": int(both.sum()), "median_difference": float(difference.median()),
                     "statistic": np.nan if result is None else float(result.statistic),
                     "p": 1.0 if result is None else float(result.pvalue)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"]) if len(out) else []
    return out


def clustered_by_year(table: pd.DataFrame, pairs: list[tuple[str, str]], years) -> pd.DataFrame:
    """The paired comparisons of ``wilcoxon_pairs`` with the units clustered by year, Holm-adjusted.

    Each year's mean difference over its units (stock-years) is tested against zero with a t-test
    over the years. With a random year effect (difference = mu + year effect + noise) this is the
    test of mu (exact when every year has the same units), so a gain that comes and goes with the
    year counts once per year, not once per stock. ``years`` gives the year of each row of ``table``.
    """
    years = np.asarray(years)
    rows = []
    for a, b in pairs:
        difference = (table[a] - table[b]).to_numpy(dtype=float)
        both = ~np.isnan(difference)
        yearly = pd.Series(difference[both]).groupby(years[both]).mean().to_numpy()
        mean = float(yearly.mean()) if len(yearly) else np.nan
        statistic, p = np.nan, np.nan                          # one year has no spread to test against
        if len(yearly) >= 2:
            spread = yearly.std(ddof=1)
            if spread > 0:
                statistic = mean / (spread / np.sqrt(len(yearly)))
                p = float(2 * stats.t.sf(abs(statistic), len(yearly) - 1))
            else:                                              # the same difference every year
                statistic, p = (np.sign(mean) * np.inf, 0.0) if mean != 0 else (np.nan, 1.0)
        rows.append({"first": a, "second": b, "years": len(yearly), "mean_difference": mean,
                     "statistic": float(statistic), "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"]) if len(out) else []
    return out


def friedman(table: pd.DataFrame) -> tuple[float, float]:
    complete = table.dropna()
    result = stats.friedmanchisquare(*(complete[c] for c in complete.columns))
    return float(result.statistic), float(result.pvalue)


def conover(table: pd.DataFrame) -> pd.DataFrame:
    """Conover's pairwise comparisons for the Friedman design, Holm-adjusted.

    When every unit ranks the methods alike, the rank variance left over is zero:
    methods with different rank sums then differ with certainty (p = 0), and
    methods with equal ones not at all (p = 1).
    """
    complete = table.dropna()
    n, k = complete.shape
    ranks = complete.rank(axis=1).to_numpy()
    sums = ranks.sum(axis=0)
    a = (ranks ** 2).sum()
    scale = np.sqrt(max(0.0, 2 * (n * a - (sums ** 2).sum()) / ((n - 1) * (k - 1))))
    df = (n - 1) * (k - 1)
    rows = []
    for i, j in combinations(range(k), 2):
        gap = abs(sums[i] - sums[j])
        t = gap / scale if scale > 0 else (np.inf if gap > 0 else 0.0)
        rows.append({"first": complete.columns[i], "second": complete.columns[j],
                     "rank_sum_difference": float(sums[i] - sums[j]), "p": float(2 * stats.t.sf(t, df))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios with zero mean and the given variance."""
    if n_trials <= 1:
        return 0.0
    first = stats.norm.ppf(1 - 1 / n_trials)
    second = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(variance) * ((1 - EULER_GAMMA) * first + EULER_GAMMA * second))


def probabilistic_sharpe(returns, benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe ratio > benchmark) given the observed returns. Returns that never vary
    (a strategy that never trades) have a Sharpe ratio of zero, with the skewness and kurtosis of a
    normal distribution."""
    r = np.asarray(returns, dtype=float)
    if r.std(ddof=1) == 0:
        sr, skew, kurt = 0.0, 0.0, 3.0
    else:
        sr = r.mean() / r.std(ddof=1)
        skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - benchmark) * np.sqrt(len(r) - 1) / denominator))


def deflated_sharpe(returns, n_trials: int, trial_sharpe_variance: float) -> float:
    """Deflated Sharpe Ratio of the selected strategy's per-period returns."""
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, trial_sharpe_variance))


def pbo(performance: np.ndarray, blocks: int = 16) -> dict[str, float]:
    """Probability of backtest overfitting.

    ``performance`` holds per-period returns, one column per strategy. The periods
    are cut into ``blocks`` blocks; for every half of the blocks taken as in-sample,
    the best in-sample strategy (by Sharpe ratio) is ranked out of sample.
    """
    m = np.asarray(performance, dtype=float)
    if blocks % 2:
        raise ValueError("blocks must be even")
    parts = np.array_split(np.arange(m.shape[0]), blocks)
    logits = []
    for chosen in combinations(range(blocks), blocks // 2):
        inside = np.concatenate([parts[i] for i in chosen])
        outside = np.concatenate([parts[i] for i in range(blocks) if i not in chosen])
        best = int(np.argmax(sharpe_ratios(m[inside])))
        oos = sharpe_ratios(m[outside])
        omega = stats.rankdata(oos)[best] / (m.shape[1] + 1)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "median_logit": float(np.median(logits)),
            "combinations": len(logits)}


def stationary_bootstrap(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of one stationary-bootstrap resample (blocks of geometric length).

    Each day either starts a new block at a random day (with probability 1 / ``mean_block``)
    or follows the day before, wrapping round at the end.
    """
    first = rng.integers(n)
    restart = rng.random(n) < 1.0 / mean_block
    starts = rng.integers(n, size=n)
    restart[0], starts[0] = True, first
    days = np.arange(n)
    began = np.maximum.accumulate(np.where(restart, days, 0))  # where each day's block began
    return (starts[began] + days - began) % n


def sharpe_differences(returns: pd.DataFrame, pairs: list[tuple[str, str]], rng: np.random.Generator,
                       bootstraps: int = 10_000, mean_block: float = 10.0, level: float = 0.95) -> pd.DataFrame:
    """Paired tests of the difference between two portfolios' Sharpe ratios, Holm-adjusted.

    Rows of ``returns`` are days and columns methods. Each resample draws days by the stationary
    bootstrap, the same days for every method, so the tests keep both the dependence between the
    portfolios and that over time (after Ledoit and Wolf, 2008, with the stationary bootstrap and
    without their studentisation). H0: equal Sharpe ratios. The two-sided p-value is the share of
    resampled differences at least as far from the observed one as it lies from zero, and the
    interval is the symmetric one that matches it. Sharpe ratios are annualised (252 days); a
    portfolio that never varies has zero.
    """
    methods = list(dict.fromkeys(method for pair in pairs for method in pair))
    values = returns[methods].dropna().to_numpy()
    observed = sharpe_ratios(values)
    resampled = np.empty((bootstraps, len(methods)))
    for draw in range(bootstraps):
        resampled[draw] = sharpe_ratios(values[stationary_bootstrap(len(values), mean_block, rng)])
    scale = np.sqrt(252)
    rows = []
    for a, b in pairs:
        i, j = methods.index(a), methods.index(b)
        difference = observed[i] - observed[j]
        distance = np.abs(resampled[:, i] - resampled[:, j] - difference)
        half_width = float(np.quantile(distance, level))
        rows.append({"first": a, "second": b, "days": len(values), "sharpe_first": observed[i] * scale,
                     "sharpe_second": observed[j] * scale, "difference": difference * scale,
                     "ci_low": (difference - half_width) * scale, "ci_high": (difference + half_width) * scale,
                     "p": float((1 + np.sum(distance >= abs(difference))) / (1 + bootstraps))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"]) if len(out) else []
    return out


def spa(differences: np.ndarray, rng: np.random.Generator, bootstraps: int = 1000,
        mean_block: float = 10.0) -> dict[str, float]:
    """Hansen's consistent SPA test.

    ``differences`` holds per-period performance of each model minus the benchmark
    (columns are models; positive is better). H0: no model beats the benchmark.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n = d.shape[0]
    means = d.mean(axis=0)
    boot_means = np.empty((bootstraps, d.shape[1]))
    for b in range(bootstraps):
        boot_means[b] = d[stationary_bootstrap(n, mean_block, rng)].mean(axis=0)
    omega = np.sqrt(n) * boot_means.std(axis=0)
    omega = np.where(omega > 0, omega, np.inf)
    statistic = max(0.0, float(np.max(np.sqrt(n) * means / omega)))
    threshold = -np.sqrt(2 * np.log(np.log(n)))
    recentre = np.where(np.sqrt(n) * means / omega >= threshold, means, 0.0)
    centred = np.sqrt(n) * (boot_means - recentre) / omega
    boot_stats = np.maximum(0.0, centred.max(axis=1))
    return {"statistic": statistic, "p": float((boot_stats >= statistic).mean())}


def sharpe_ratios(returns: np.ndarray) -> np.ndarray:
    """Per-period Sharpe ratio of each column; zero for one that never varies (never trades)."""
    spread = returns.std(axis=0, ddof=1)
    return np.where(spread > 0, returns.mean(axis=0) / np.where(spread > 0, spread, 1), 0.0)
