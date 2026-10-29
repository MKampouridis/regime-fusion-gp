import numpy as np
import pandas as pd
import pytest

from regime_fusion.engine import benchmarks
from regime_fusion.engine.backtest import backtest
from regime_fusion.testing import check_no_lookahead


def prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    opening = close * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame({"adj_open": opening, "adj_high": np.maximum(opening, close) * 1.005,
                         "adj_low": np.minimum(opening, close) * 0.995, "adj_close": close},
                        index=pd.bdate_range("2016-01-04", periods=n))


def test_buy_and_hold_compounds_to_the_round_trip():
    p = prices()
    daily = benchmarks.buy_and_hold(p, cost_bps=10)
    expected = p["adj_close"].iloc[-1] * 0.999 / (p["adj_open"].iloc[0] * 1.001) - 1
    assert np.prod(1 + daily) - 1 == pytest.approx(expected)


def test_trading_strategy_buys_whenever_flat():
    p = prices()
    result = backtest(benchmarks.always(p), p, days=10, rate=0.03, cost_bps=10)
    gaps = [p.index.get_loc(entry) - p.index.get_loc(exit_) for exit_, entry
            in zip(result.trades["exit"][:-1], result.trades["entry"][1:])]
    assert set(gaps) == {1}  # each trade starts the day after the previous one ends


def test_rule_signals_never_look_ahead():
    p = prices()
    check_no_lookahead(lambda f: pd.DataFrame({"s": benchmarks.moving_average_crossover(f, 10, 30)}),
                       p, range(0, 400, 13))
    check_no_lookahead(lambda f: pd.DataFrame({"s": benchmarks.time_series_momentum(f, 60)}), p, range(0, 400, 13))


def test_random_entry_matches_the_target_trade_count():
    p = prices()
    out = benchmarks.random_entry(p, days=10, rate=0.03, cost_bps=10, target_trades=12,
                                  rng=np.random.default_rng(1), draws=300)
    assert abs(out["mean_trades"] - 12) < 1.0 and out["sharpes"].shape == (300,)
    again = benchmarks.random_entry(p, days=10, rate=0.03, cost_bps=10, target_trades=12,
                                    rng=np.random.default_rng(1), draws=300)
    np.testing.assert_array_equal(out["sharpes"], again["sharpes"])
    assert benchmarks.percentile_of(10.0, out["sharpes"]) == 1.0
    assert benchmarks.percentile_of(0.0, np.array([-1.0, 0.0, 1.0])) == 0.5


@pytest.fixture
def stock_prices(monkeypatch, tmp_path):
    """A stock that trends up in 2018 and 2019, so the trend rules have something to find."""
    from regime_fusion.data.store import DATA_DIR_ENV

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    days = pd.bdate_range("2014-01-02", "2019-12-31")
    rng = np.random.default_rng(3)
    drift = np.where(days.year >= 2018, 0.002, 0.0)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, len(days))))
    frame = pd.DataFrame({"adj_open": close, "adj_high": close * 1.005, "adj_low": close * 0.995,
                          "adj_close": close}, index=days)
    (tmp_path / "interim" / "prices").mkdir(parents=True)
    frame.to_parquet(tmp_path / "interim" / "prices" / "XYZ.parquet")
    return frame


def test_rules_run_like_rungs_with_parameters_chosen_on_validation(stock_prices):
    from regime_fusion.config import load_config
    from regime_fusion.engine.run import RunSpec
    from regime_fusion.ml.models import load_settings

    config, settings = load_config(), load_settings()
    spec = RunSpec("XYZ", 2019, 0, 2, 10, 0.03)
    ma = benchmarks.run_rule(spec, settings.rules["RB-MA"], settings, config, write=False)
    assert len(ma["candidates"]) == 9 and ma["validation_sharpe"] == max(c["sharpe"] for c in ma["candidates"])
    assert ma["evaluate"] == ["2019-01-01", "2019-12-31"] and ma["train"] == ["2018-01-01", "2018-12-31"]
    chosen = {k: v for k, v in ma["candidates"][[c["sharpe"] for c in ma["candidates"]].index(
        ma["validation_sharpe"])].items() if k in ("fast", "slow")}
    test = stock_prices.loc["2019"]
    signal = benchmarks.moving_average_crossover(stock_prices, chosen["fast"], chosen["slow"])
    expected = backtest(signal[stock_prices.index.year == 2019], test, 10, 0.03, 10)
    assert ma["evaluation"]["10bps"]["sharpe"] == pytest.approx(metrics_sharpe(expected))
    hold = benchmarks.run_rule(spec, settings.rules["RB-BH"], settings, config, write=False)
    assert hold["evaluation"]["0bps"]["total_return"] == pytest.approx(
        test["adj_close"].iloc[-1] / test["adj_open"].iloc[0] - 1)
    assert hold["evaluation"]["0bps"]["trades"] == 1.0 and hold["evaluation"]["0bps"]["exposure"] == 1.0
    always = benchmarks.run_rule(RunSpec("XYZ", 2019, 0, 1, 10, 0.03), settings.rules["RB-TS"], settings,
                                 config, write=False)
    assert always["evaluate"] == ["2018-01-01", "2018-12-31"]
    assert always["evaluation"]["10bps"]["trades"] > 10


def metrics_sharpe(result):
    from regime_fusion.engine import metrics

    return metrics.sharpe(result.daily)
