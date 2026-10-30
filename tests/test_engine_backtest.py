import time

import numpy as np
import pandas as pd
import pytest

from regime_fusion.engine import metrics
from regime_fusion.engine.backtest import backtest

DAYS = pd.bdate_range("2019-01-02", periods=8)


def bars(rows):
    """(open, high, low, close) per session."""
    frame = pd.DataFrame(rows, columns=["adj_open", "adj_high", "adj_low", "adj_close"], dtype=float)
    frame.index = DAYS[:len(rows)]
    return frame


FLAT = [(100, 100.5, 99.5, 100)] * 8


def signals(*days, n=8):
    out = np.zeros(n, dtype=bool)
    out[list(days)] = True
    return out


def test_take_profit_fills_at_the_target_on_the_first_high_above_it():
    rows = list(FLAT)
    rows[1] = (100, 101, 99, 100.5)     # bought at this open
    rows[2] = (101, 103, 100, 102.5)    # high reaches 102: sold at 102
    result = backtest(signals(0), bars(rows), days=5, rate=0.02, cost_bps=0)
    trade = result.trades.iloc[0]
    assert (trade["entry"], trade["exit"]) == (DAYS[1], DAYS[2])
    assert trade["entry_price"] == 100 and trade["exit_price"] == pytest.approx(102) and trade["reason"] == "target"
    assert result.daily.iloc[1] == pytest.approx(0.005) and result.daily.iloc[2] == pytest.approx(102 / 100.5 - 1)
    assert result.daily.drop(DAYS[[1, 2]]).eq(0).all()


def test_a_gap_above_the_target_sells_at_the_open():
    rows = list(FLAT)
    rows[2] = (104, 105, 103.5, 104.5)
    trade = backtest(signals(0), bars(rows), days=5, rate=0.02, cost_bps=0).trades.iloc[0]
    assert trade["exit_price"] == 104 and trade["reason"] == "gap"


def test_the_target_can_be_reached_on_the_entry_day():
    rows = list(FLAT)
    rows[1] = (100, 102.5, 99.5, 101)
    trade = backtest(signals(0), bars(rows), days=5, rate=0.02, cost_bps=0).trades.iloc[0]
    assert trade["exit"] == DAYS[1] and trade["exit_price"] == pytest.approx(102)


def test_time_exit_at_the_close_of_the_dth_day_and_signals_ignored_while_holding():
    rows = [(100, 100.5, 99.5, 100 + i) for i in range(8)]   # never 5% up within a day's high
    result = backtest(signals(0, 1, 2, 4), bars(rows), days=3, rate=0.05, cost_bps=0)
    first, second = result.trades.iloc[0], result.trades.iloc[1]
    assert (first["entry"], first["exit"], first["reason"]) == (DAYS[1], DAYS[3], "time")
    assert first["exit_price"] == 103
    assert second["entry"] == DAYS[5]                        # the signal of day 4, after the exit
    assert second["exit"] == DAYS[7] and second["reason"] == "time"


def test_costs_are_paid_on_both_sides_and_daily_returns_compound_to_the_trade():
    rows = [(100 + i, 100.5 + i, 99.5 + i, 100.2 + i) for i in range(8)]
    result = backtest(signals(0), bars(rows), days=4, rate=0.5, cost_bps=10)
    trade = result.trades.iloc[0]
    expected = trade["exit_price"] * 0.999 / (trade["entry_price"] * 1.001) - 1
    assert trade["net_return"] == pytest.approx(expected)
    assert np.prod(1 + result.daily) - 1 == pytest.approx(expected)


def test_an_open_position_is_closed_at_the_end_and_a_last_day_signal_is_ignored():
    result = backtest(signals(5, 7), bars(FLAT), days=10, rate=0.5, cost_bps=0)
    assert len(result.trades) == 1 and result.trades.iloc[0]["reason"] == "end"
    assert result.trades.iloc[0]["exit"] == DAYS[7]


def test_a_last_session_without_a_close_ends_the_trade_at_the_last_known_price():
    rows = FLAT[:7] + [(100, np.nan, np.nan, np.nan)]          # an open on the last session, but no close
    for signal_day in (4, 6):                                  # bought earlier, or on that last session
        result = backtest(signals(signal_day), bars(rows), 5, 0.05, 10)
        trade = result.trades.iloc[0]
        assert trade["exit_price"] == 100 and trade["reason"] == "end"
        assert trade["net_return"] == pytest.approx(0.999 / 1.001 - 1)          # both costs, nothing else
        assert (1 + result.daily).prod() - 1 == pytest.approx(trade["net_return"])


def test_no_signal_no_trade():
    result = backtest(np.zeros(8, dtype=bool), bars(FLAT), days=5, rate=0.02, cost_bps=10)
    assert result.trades.empty and result.daily.eq(0).all()
    assert metrics.summary(result)["sharpe"] == 0.0 and metrics.summary(result)["trades"] == 0


def test_later_prices_and_signals_never_change_earlier_returns():
    rng = np.random.default_rng(0)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    prices = pd.DataFrame({"adj_open": close * (1 + rng.normal(0, 0.003, n)), "adj_close": close},
                          index=pd.bdate_range("2019-01-02", periods=n))
    prices["adj_high"] = prices[["adj_open", "adj_close"]].max(axis=1) * 1.01
    prices["adj_low"] = prices[["adj_open", "adj_close"]].min(axis=1) * 0.99
    signal = rng.random(n) < 0.2
    full = backtest(signal, prices, days=10, rate=0.03, cost_bps=10).daily
    for cut in range(20, n, 37):
        changed = prices.copy()
        changed.iloc[cut:] *= 1.5
        other = signal.copy()
        other[cut:] = ~other[cut:]
        part = backtest(other, changed, days=10, rate=0.03, cost_bps=10).daily
        # Day cut-1 can differ only through a forced exit at the end, which the full run does not have.
        pd.testing.assert_series_equal(part.iloc[:cut - 1], full.iloc[:cut - 1])


def test_metrics_by_hand():
    daily = pd.Series([0.01, -0.02, 0.0, 0.03, -0.01])
    assert metrics.total_return(daily) == pytest.approx(1.01 * 0.98 * 1.03 * 0.99 - 1)
    assert metrics.sharpe(daily) == pytest.approx(daily.mean() / daily.std() * np.sqrt(252))
    assert metrics.max_drawdown(daily) == pytest.approx(0.02)
    assert metrics.cvar(pd.Series(np.arange(-10, 10) / 100)) == pytest.approx(0.10)  # the worst 1 of 20 days


def test_backtests_are_fast():
    n = 1000
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, n)))
    prices = pd.DataFrame({"adj_open": close, "adj_high": close * 1.01, "adj_low": close * 0.99,
                           "adj_close": close}, index=pd.bdate_range("2015-01-02", periods=n))
    signal = np.random.default_rng(2).random(n) < 0.1
    backtest(signal, prices, days=10, rate=0.03, cost_bps=10)  # compile
    from regime_fusion.engine.backtest import _simulate
    arrays = [prices[c].to_numpy() for c in ("adj_open", "adj_high", "adj_low", "adj_close")]
    started = time.perf_counter()
    for _ in range(1000):
        _simulate(signal, *arrays, 10, 0.03, 0.001)
    per_run = (time.perf_counter() - started) / 1000
    assert per_run < 1e-3  # well under the 0.25 ms per tree budget for the whole evaluation
