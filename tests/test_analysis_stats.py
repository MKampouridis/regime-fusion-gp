import numpy as np
import pandas as pd
import pytest

from regime_fusion.analysis import stats


def test_holm_by_hand():
    np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert stats.holm([0.6, 0.9]).max() <= 1.0


def results_table(seed=0, n=60):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return pd.DataFrame({"A0": base + rng.normal(0, 0.1, n), "A1": base + 0.5 + rng.normal(0, 0.1, n),
                         "same": base + rng.normal(0, 0.1, n)})


def test_wilcoxon_and_conover_find_the_better_method():
    table = results_table()
    pairs = stats.wilcoxon_pairs(table, [("A1", "A0"), ("same", "A0"), ("A0", "A0")])
    assert pairs.loc[0, "p_holm"] < 0.001 and pairs.loc[0, "median_difference"] > 0.4
    assert pairs.loc[1, "p_holm"] > 0.05 and pairs.loc[2, "p"] == 1.0
    statistic, p = stats.friedman(table)
    assert p < 0.001
    post = stats.conover(table).set_index(["first", "second"])
    assert post.loc[("A0", "A1"), "p_holm"] < 0.001
    assert post.loc[("A0", "same"), "p_holm"] > 0.05


def test_conover_by_hand_and_for_perfectly_consistent_rankings():
    from scipy import stats as scipy_stats

    table = pd.DataFrame([[1, 2, 3], [2, 1, 3], [1, 3, 2], [1, 2, 3]], columns=["A", "B", "C"])
    # Rank sums 5, 8 and 11; sum of squared ranks 56; n = 4, k = 3 (Conover, 1999):
    # scale = sqrt(2 (4 * 56 - (25 + 64 + 121)) / (3 * 2)) = sqrt(28 / 6), with 6 degrees of freedom.
    post = stats.conover(table).set_index(["first", "second"])
    assert post.loc[("A", "C"), "p"] == pytest.approx(2 * scipy_stats.t.sf(6 / np.sqrt(28 / 6), 6))
    assert post.loc[("A", "B"), "rank_sum_difference"] == -3
    consistent = pd.DataFrame({"A": np.ones(10), "B": 2 * np.ones(10), "C": 3 * np.ones(10)})
    assert stats.friedman(consistent)[1] < 1e-4
    assert (stats.conover(consistent)["p_holm"] == 0).all()          # the same order in every unit
    tied = consistent.assign(B=1.0)
    post = stats.conover(tied).set_index(["first", "second"])
    assert post.loc[("A", "B"), "p"] == 1.0 and post.loc[("A", "C"), "p"] == 0.0


def test_a_strategy_that_never_trades_has_a_defined_deflated_sharpe():
    flat = np.zeros(500)
    assert stats.probabilistic_sharpe(flat) == pytest.approx(0.5)            # a Sharpe ratio of exactly zero
    from scipy.stats import norm

    expected = stats.expected_max_sharpe(10, 0.002)                             # zero minus the expected best
    assert stats.deflated_sharpe(flat, 10, 0.002) == pytest.approx(norm.cdf(-expected * np.sqrt(499)))


def test_deflated_sharpe_shrinks_with_the_number_of_trials():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.01, 1000)
    assert stats.expected_max_sharpe(1, 0.01) == 0.0
    assert stats.expected_max_sharpe(100, 0.01) > stats.expected_max_sharpe(10, 0.01) > 0
    psr = stats.probabilistic_sharpe(returns)
    assert 0.9 < psr < 1.0
    assert stats.deflated_sharpe(returns, 1000, 0.002) < stats.deflated_sharpe(returns, 10, 0.002) < psr
    noise = rng.normal(0, 0.01, 5000)
    assert 0.1 < stats.probabilistic_sharpe(noise) < 0.9


def test_backtest_overfitting_probability():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, (1000, 20))
    assert 0.25 < stats.pbo(noise, blocks=8)["pbo"] < 0.75   # selection by luck alone
    skilled = noise.copy()
    skilled[:, 0] += 0.004                                    # one strategy is truly better
    assert stats.pbo(skilled, blocks=8)["pbo"] < 0.05


def test_superior_predictive_ability():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.01, (750, 3))
    better = noise.copy()
    better[:, 1] += 0.003
    assert stats.spa(better, np.random.default_rng(4), bootstraps=500)["p"] < 0.05
    assert stats.spa(noise - 0.001, np.random.default_rng(4), bootstraps=500)["p"] > 0.5
    index = stats.stationary_bootstrap(1000, 10.0, np.random.default_rng(5))
    assert index.min() >= 0 and index.max() < 1000
    assert np.mean(np.diff(index) == 1) > 0.8  # mostly runs of consecutive days
