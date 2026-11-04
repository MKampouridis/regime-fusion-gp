"""A robustness-oriented fitness for the development experiments (E4).

The campaigns' fitness (ActTrade's) is the sum of the Sharpe ratios of the whole model and of each
branch alone. A model whose fusion tree is a single branch counts that branch twice, so evolution
drops fusion (86-95% of the pilot's best fusion-rung models were one branch). And a Sharpe ratio
ignores how much evidence it rests on: a dozen well-timed trades, with the stock out of the market
the rest of the time, can score above 2.

``psr_lower_bound`` scores a model by how sure we can be that it is profitable: its Sharpe ratio
minus a margin for statistical uncertainty, the one-sided lower confidence bound behind the
Probabilistic Sharpe Ratio (Bailey and Lopez de Prado, 2012). With Sharpe ratio s of n returns,
skewness g3 and (non-excess) kurtosis g4, the standard error of s is

    sqrt((1 - g3 s + (g4 - 1) / 4 s^2) / (n - 1)),

and the score is s - z sigma, z the one-sided 95% normal quantile, fixed in advance.

The returns are the model's **per-trade net returns**, so n is the number of trades, the evidence a
rule rests on. On daily returns, n would be the number of days, the same for every rule in a
training window: the bound would then be the Sharpe ratio minus a near-constant, and rank rules
almost as the Sharpe ratio does, however few their trades (4 November 2026). Note that the formula
rewards positive skewness, so a few large wins are not penalised as such; fat tails and negative
skewness are. Only the whole model is scored.
A signal with fewer than ``MIN_TRADES`` trades per stock over the training years scores ``FLOOR``,
below any real model: the formula is asymptotic, and with a handful of trades and positive skewness
its variance term turns negative (one AAPL rule scored 6.1 on four trades). Twenty trades is about
seven to ten a year in the training windows. Should the term still be non-positive, the normal-theory
margin, sqrt((1 + s^2 / 2) / (n - 1)), is used instead.
"""

from __future__ import annotations

import numpy as np

from regime_fusion.engine.fitness import Evaluator, Individual, boolean, fused, _branch_key

Z_ONE_SIDED_95 = 1.6448536269514722
FLOOR = -10.0
MIN_TRADES = 20   # per stock, over the training years


def psr_lower_bound(returns: np.ndarray, z: float = Z_ONE_SIDED_95) -> float:
    """One-sided lower confidence bound on the (per-observation) Sharpe ratio of ``returns``."""
    daily = np.asarray(returns, dtype=float)
    n = daily.size
    if n < 3:
        return FLOOR
    mean, spread = daily.mean(), daily.std(ddof=1)
    if spread == 0.0:
        return FLOOR
    s = mean / spread
    centred = (daily - mean) / daily.std(ddof=0)
    g3, g4 = float((centred ** 3).mean()), float((centred ** 4).mean())
    term = 1.0 - g3 * s + (g4 - 1.0) / 4.0 * s * s
    if term <= 0.0:
        term = 1.0 + s * s / 2.0
    sigma = np.sqrt(term / (n - 1))
    return float(s - z * sigma)


class LowerBoundEvaluator(Evaluator):
    """``engine.fitness.Evaluator`` with the whole model's PSR lower bound as fitness.

    Each component's score slot holds its lower bound, so ActTrade's active trading operator,
    which reads it, picks the branches with the best bound."""

    def component(self, key: tuple, signal_of) -> tuple:
        found = self._cache.get(key)
        if found is None:
            signal = signal_of()
            _, trades, returns = self.market.simulate(signal)
            found = (signal, trades, psr_lower_bound(returns), None)
            self._cache[key] = found
        return found

    def evaluate(self, individual: Individual) -> None:
        stats, branch_signals = {}, {}
        for s, tree in individual.branches.items():
            stats[s] = self.component(_branch_key(s, tree), lambda t=tree: boolean(t, self.data)[0])
            branch_signals[s] = stats[s][0]
        stats["C"] = self.component(
            ("C", *individual.key()), lambda: fused(individual.fusion, branch_signals, self.data)[0]
        )
        floor = self.settings.min_trades
        individual.fitness = float(stats["C"][2] if stats["C"][1] >= floor else FLOOR)
        components = ("C", *individual.branches)
        individual.stats = {c: {"sharpe": float(stats[c][2]) if stats[c][1] >= floor else FLOOR,
                                "trades": int(stats[c][1])} for c in components}
