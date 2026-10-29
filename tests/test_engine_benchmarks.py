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
