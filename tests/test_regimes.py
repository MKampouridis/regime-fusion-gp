import itertools
import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.config import load_config
from regime_fusion.data import cli, store
from regime_fusion.data.store import DATA_DIR_ENV
from regime_fusion.data.universe import load_universe
from regime_fusion.regimes import hmm, market, news
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


def test_the_filter_uses_what_a_day_has_and_survives_an_impossible_day():
    model = hmm.GaussianHMM(np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                            np.array([[0.0, 0.0], [5.0, 5.0]]), np.ones((2, 2)), 0.0)
    out = hmm.filter(model, np.array([[0.0, 0.0], [5.0, np.nan], [np.nan, np.nan]]))
    assert out[1, 1] > 0.99                                      # the price input alone points to state 1
    assert out[2] == pytest.approx(out[1] @ model.transition)    # no input at all: carried forward
    absorbing = hmm.GaussianHMM(np.array([1.0, 0.0]), np.eye(2), np.array([[0.0], [100.0]]),
                                np.array([[1e-6], [1.0]]), 0.0)
    out = hmm.filter(absorbing, np.array([0.0, 100.0, 0.0]))     # day 2 is impossible in the only state left
    assert np.isfinite(out).all() and out[1] == pytest.approx([1.0, 0.0])
    x = np.array([[0.0], [100.0], [0.0]])
    alpha, scale, b, log_likelihood = hmm._forward(absorbing, x)
    gamma = alpha * hmm._backward_pass(absorbing.transition, b, scale)
    assert np.isfinite(gamma / gamma.sum(axis=1, keepdims=True)).all()  # the backward pass agrees: no 0 / 0
    assert -1000 < log_likelihood < -600                                 # the impossible day counts at the floor
    refit = hmm._em(x, absorbing, np.array([1e-6]), max_iter=3, tol=0.0)
    assert np.isfinite(refit.means).all() and np.isfinite(refit.log_likelihood)


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


def news_articles(seed=9):
    """Articles per session: busy months now and then; none before GDELT starts; a short outage."""
    days = pd.bdate_range("2014-01-02", "2025-12-31")
    rng = np.random.default_rng(seed)
    articles = pd.Series(rng.poisson(np.where(days.month % 4 == 0, 40, 8)).astype(float), index=days)
    articles[articles.index < "2015-02-19"] = np.nan
    articles.iloc[1000:1004] = np.nan
    return articles


def test_attention_level_skips_uncovered_sessions_and_keeps_the_last_level():
    level = news.attention_level(pd.Series([np.nan, 2.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, 6.0]), 3)
    assert np.isnan(level[0])
    assert level[1] == 1 + 3 * 2.0 and level[2] == 1 + 3 * 3.0
    assert level[3] == 1 + 3 * 3.0  # the mean of the covered sessions in the window
    assert level[4] == 1 + 3 * 4.0
    assert level[5] == level[4]  # no covered session in the window: the last level is kept
    assert level[8] == 1 + 3 * 6.0
    together = news.market_articles({"A": pd.Series([1.0, np.nan]), "B": pd.Series([2.0, np.nan])})
    assert together[0] == 3.0 and np.isnan(together[1])


def test_a_gdelt_outage_is_missing_news_not_calm_news():
    articles = pd.Series(np.r_[np.linspace(10, 40, 60), np.full(12, np.nan), np.full(30, 40.0)],
                         index=pd.bdate_range("2016-01-04", periods=102))
    inputs = news.attention_inputs(articles, 5, 0.25, 20)
    assert inputs.iloc[64:72].isna().all().all()                 # five or more sessions without coverage
    assert inputs.iloc[60:64].notna().all().all() and inputs.iloc[72:].notna().all().all()


def test_news_inputs_never_look_ahead():
    articles = news_articles().iloc[250:700]
    check_no_lookahead(lambda a: news.attention_inputs(a, 5, 0.25, 20), articles, range(0, 450, 11))


def test_dc_news_inputs_add_news_attention_to_the_dc_inputs():
    settings = {"inputs": {"dc_news": {"threshold": 0.01, "window": 20, "attention": {
        "scope": "stock", "sessions": 5, "threshold": 0.25, "window": 20}}, "dc": {"threshold": 0.01}}}
    close = market_close()
    features = market.inputs(close, "dc_news", settings, news_articles())
    assert list(features.columns) == ["log_events", "trend_move", "news_log_events", "news_trend_move"]
    assert features.loc[:"2015-02-18", "news_log_events"].isna().all()
    assert features.loc["2016"].notna().all().all()
    with pytest.raises(ValueError, match="articles"):
        market.inputs(close, "dc_news", settings)
    assert market.per_stock("dc_news", settings) and not market.per_stock("dc", settings)


def test_per_stock_regimes_have_their_own_seeds_and_files(monkeypatch, tmp_path):
    keys = []
    real = market.make_rng
    monkeypatch.setattr(market, "make_rng", lambda *k: keys.append(k) or real(*k))
    config = load_config()
    settings = {"n_states": 2, "restarts": 1, "inputs": {"volatility": {"window": 20}}}
    features = market.inputs(market_close(), "volatility", settings)
    market.walk_forward(features, walk_forward(config)[:1], settings, config.base_seed, stock="AAPL")
    market.walk_forward(features, walk_forward(config)[:1], settings, config.base_seed)
    assert keys[0][1:] == ("regimes", "AAPL", 2019, 1) and keys[2][1:] == ("regimes", 2019, 1)
    assert (market.probabilities_path("dc_news", 2019, 2, "AAPL", root=tmp_path)
            == tmp_path / "processed" / "regimes" / "dc_news" / "AAPL" / "2019.stage2.parquet")
    assert (market.probabilities_path("dc", 2019, 2, root=tmp_path)
            == tmp_path / "processed" / "regimes" / "dc" / "2019.stage2.parquet")


def test_regimes_command_fits_shared_or_per_stock_news_regimes(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    store.write_parquet(store.layer("interim", "market", "GSPC.parquet"), market_close().to_frame("close"))
    main = [s.id for s in load_universe(cli.UNIVERSE_FILE).select("main")]
    for k, stock in enumerate(main):
        store.write_parquet(store.layer("interim", "news", f"{stock}.parquet"),
                            news_articles(seed=k).to_frame("articles"))
    repository = market.load_settings()
    settings = {**repository, "restarts": 1, "inputs": {"dc_news": repository["inputs"]["dc_news"]}}
    monkeypatch.setattr(market, "load_settings", lambda path=None: settings)
    assert settings["inputs"]["dc_news"]["attention"]["scope"] == "market"  # shared regimes, as planned
    assert cli.main(["regimes"]) == 0
    shared = pd.read_parquet(market.probabilities_path("dc_news", 2019, 2))
    assert shared.index[-1] == pd.Timestamp("2019-12-31")

    attention = {**settings["inputs"]["dc_news"]["attention"], "scope": "stock"}
    settings["inputs"]["dc_news"] = {**settings["inputs"]["dc_news"], "attention": attention}
    assert cli.main(["regimes", "--stocks", "AAPL"]) == 0
    assert market.probabilities_path("dc_news", 2025, 2, "AAPL").exists()
    report = json.loads(store.layer("reports", "regimes_per_stock.json").read_text())
    assert "dc_news AAPL 2025 stage 2" in report
    store.layer("interim", "news", "F.parquet").unlink()
    with pytest.raises(SystemExit, match="sa command"):
        cli.main(["regimes", "--stocks", "F"])


def test_regimes_command_keeps_expanding_windows_apart(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    store.write_parquet(store.layer("interim", "market", "GSPC.parquet"), market_close().to_frame("close"))
    settings = {**market.load_settings(), "restarts": 1}
    settings["inputs"] = {"volatility": settings["inputs"]["volatility"]}
    monkeypatch.setattr(market, "load_settings", lambda path=None: settings)
    assert cli.main(["regimes", "--scheme", "expanding"]) == 0
    path = market.probabilities_path("volatility", 2021, 2, scheme="expanding")
    assert path.exists() and not market.probabilities_path("volatility", 2021, 2).exists()
    assert pd.read_parquet(path).index[0] == pd.Timestamp("2015-02-02")  # fitted from the start of the data
    assert json.loads(store.layer("reports", "regimes_expanding.json").read_text())
