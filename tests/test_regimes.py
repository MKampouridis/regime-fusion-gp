import itertools

import numpy as np
import pandas as pd
import pytest

from regime_fusion.config import load_config
from regime_fusion.regimes import hmm, market
from regime_fusion.testing import check_no_lookahead
from regime_fusion.windows import walk_forward


def simulate(n=2000, seed=0):
    """Two regimes: calm (mean 0, sd 1) and turbulent (mean 3, sd 2), persistent."""
    rng = np.random.default_rng(seed)
    transition = np.array([[0.98, 0.02], [0.05, 0.95]])
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=transition[states[t - 1]])
    x = np.where(states == 0, rng.normal(0, 1, n), rng.normal(3, 2, n))
    return x, states


def test_fit_recovers_a_known_model_with_states_ordered():
    x, states = simulate()
    model = hmm.fit(x, 2, np.random.default_rng(1), restarts=3)
    assert model.means[:, 0] == pytest.approx([0.0, 3.0], abs=0.25)
    assert np.sqrt(model.variances[:, 0]) == pytest.approx([1.0, 2.0], abs=0.25)
    assert np.diag(model.transition) == pytest.approx([0.98, 0.95], abs=0.03)
    accuracy = ((hmm.filter(model, x)[:, 1] > 0.5) == states).mean()
    assert accuracy > 0.9


def test_em_never_lowers_the_likelihood():
    x, _ = simulate(600, seed=2)
    x2 = x[:, None]
    floor = hmm.VARIANCE_FLOOR * x2.var(axis=0)
    start = hmm._initial(x2, 2, np.random.default_rng(3))
    values = [hmm._em(x2, start, floor, max_iter=k, tol=0.0).log_likelihood for k in range(1, 12)]
    assert all(b >= a - 1e-8 for a, b in zip(values, values[1:]))


def test_filter_matches_enumeration_of_all_paths():
    model = hmm.GaussianHMM(np.array([0.6, 0.4]), np.array([[0.9, 0.1], [0.2, 0.8]]),
                            np.array([[0.0], [2.0]]), np.array([[1.0], [1.0]]), 0.0)
    x = np.array([0.1, 1.9, 2.2, -0.3])
    density = lambda v, k: np.exp(-0.5 * (v - model.means[k, 0]) ** 2)  # noqa: E731
    filtered = hmm.filter(model, x)
    for t in range(len(x)):
        joint = np.zeros(2)
        for path in itertools.product(range(2), repeat=t + 1):
            p = model.start[path[0]] * density(x[0], path[0])
            for s in range(1, t + 1):
                p *= model.transition[path[s - 1], path[s]] * density(x[s], path[s])
            joint[path[-1]] += p
        assert filtered[t] == pytest.approx(joint / joint.sum())


def test_filter_never_looks_ahead_and_carries_missing_days_forward():
    x, _ = simulate(300, seed=4)
    model = hmm.fit(x, 2, np.random.default_rng(5), restarts=2)
    check_no_lookahead(lambda v: pd.DataFrame(hmm.filter(model, v)), x, range(0, 300, 7))
    gappy = x.copy()
    gappy[100] = np.nan
    out = hmm.filter(model, gappy)
    assert out[100] == pytest.approx(out[99] @ model.transition)


def test_fits_are_reproducible_and_need_enough_data():
    x, _ = simulate(500, seed=6)
    a = hmm.fit(x, 2, np.random.default_rng(7), restarts=2)
    b = hmm.fit(x, 2, np.random.default_rng(7), restarts=2)
    np.testing.assert_array_equal(a.means, b.means)
    with pytest.raises(ValueError, match="too few"):
        hmm.fit(x[:15], 2, np.random.default_rng(7))


def market_close(seed=8):
    days = pd.bdate_range("2014-01-02", "2025-12-31")
    rng = np.random.default_rng(seed)
    volatility = np.where((days.year % 3) == 0, 0.02, 0.008)  # a turbulent year now and then
    return pd.Series(2000 * np.exp(np.cumsum(rng.normal(0, volatility))), index=days)


def test_inputs_never_look_ahead():
    close = market_close().iloc[:400]
    check_no_lookahead(lambda c: market.realised_volatility(c, 20).to_frame(), close, range(0, 400, 9))
    check_no_lookahead(lambda c: market.dc_inputs(c, 0.01, 20), close, range(0, 400, 9))


def test_test_year_probabilities_come_from_a_model_that_never_saw_it():
    config = load_config()
    settings = {"n_states": 2, "restarts": 2, "inputs": {"volatility": {"window": 20}}}
    features = market.inputs(market_close(), "volatility", settings)
    window = walk_forward(config)[0]  # test year 2019
    fits = market.walk_forward(features, [window], settings, config.base_seed)
    changed = features.copy()
    changed.loc["2019-01-01":] *= 3.0
    refits = market.walk_forward(changed, [window], settings, config.base_seed)
    for fit, refit in zip(fits, refits):
        np.testing.assert_array_equal(fit.model.means, refit.model.means)
    stage2, restage2 = fits[1].probabilities, refits[1].probabilities
    pd.testing.assert_frame_equal(stage2.loc[:"2018-12-31"], restage2.loc[:"2018-12-31"])
    assert stage2.index[-1] == pd.Timestamp("2019-12-31")
    assert not stage2.loc["2019"].equals(restage2.loc["2019"])
