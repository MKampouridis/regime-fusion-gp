import numpy as np
import pytest

from regime_fusion.dev.designs import Design, PooledMarket, StockMarket, simulate_hold

OPEN = np.array([10.0, 10.0, 11.0, 12.0, 11.0, 10.0])
CLOSE = np.array([10.0, 10.5, 11.5, 11.8, 10.5, 10.2])


def test_hold_enters_at_the_next_open_and_leaves_at_the_open_after_the_signal_turns_off():
    signal = np.array([True, True, False, False, False, False])
    daily, trades = simulate_hold(signal, OPEN, CLOSE, np.inf, 0.0)
    # bought at the open of day 1 (10.0), marked at its close (10.5), sold at the open of day 3 (12.0)
    assert trades == 1
    assert daily[0] == 0.0
    assert daily[1] == pytest.approx(10.5 / 10.0 - 1)
    assert daily[2] == pytest.approx(11.5 / 10.5 - 1)
    assert daily[3] == pytest.approx(12.0 / 11.5 - 1)
    assert np.prod(1 + daily) == pytest.approx(12.0 / 10.0)
    assert np.all(daily[4:] == 0.0)


def test_costs_are_paid_on_entry_and_exit():
    signal = np.array([True, True, False, False, False, False])
    daily, _ = simulate_hold(signal, OPEN, CLOSE, np.inf, 0.001)
    assert np.prod(1 + daily) == pytest.approx(12.0 * 0.999 / (10.0 * 1.001))


def test_a_position_open_at_the_end_is_closed_at_the_last_close():
    daily, trades = simulate_hold(np.ones(6, dtype=bool), OPEN, CLOSE, np.inf, 0.001)
    assert trades == 1
    assert np.prod(1 + daily) == pytest.approx(10.2 * 0.999 / (10.0 * 1.001))


def test_the_trailing_stop_closes_and_waits_for_the_signal_to_reset():
    open_ = np.array([10.0, 10.0, 12.0, 10.0, 9.0, 9.0, 9.0])
    close = np.array([10.0, 12.0, 12.0, 10.0, 9.0, 9.0, 9.5])
    signal = np.array([True, True, True, True, False, True, True])
    daily, trades = simulate_hold(signal, open_, close, 0.10, 0.0)
    # peak 12, close of day 3 is 10 (more than 10% below): sold at the open of day 4 (9.0);
    # the signal is still on, so no new entry until it turns off (day 4) and on again (day 5)
    assert np.prod(1 + daily[:5]) == pytest.approx(9.0 / 10.0)
    assert daily[5] == 0.0 and daily[6] == pytest.approx(9.5 / 9.0 - 1)
    assert trades == 2


def test_long_design_without_a_signal_is_buy_and_hold():
    design = Design("long", 0.0)
    daily, trades = design.daily(np.zeros(6, dtype=bool), OPEN, CLOSE, CLOSE, CLOSE)
    assert trades == 1 and np.prod(1 + daily) == pytest.approx(10.2 / 10.0)


@pytest.mark.parametrize("name", ["acttrade", "hold", "long"])
def test_no_design_uses_the_signal_of_the_same_day(name):
    rng = np.random.default_rng(0)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.02, 200))
    open_, high, low = close * 0.999, close * 1.01, close * 0.99
    signal = rng.random(200) < 0.3
    design = Design(name, 0.001)
    base, _ = design.daily(signal, open_, high, low, close)
    for day in (50, 120):
        changed = signal.copy()
        changed[day] = not changed[day]
        moved, _ = design.daily(changed, open_, high, low, close)
        np.testing.assert_array_equal(base[:day + 1], moved[:day + 1])


def test_pooled_market_is_the_equal_weight_portfolio_of_its_stocks():
    rng = np.random.default_rng(1)
    design = Design("hold", 0.001)
    markets, signals = [], []
    for n in (30, 20):
        close = 10 * np.cumprod(1 + rng.normal(0, 0.02, n))
        markets.append(StockMarket(design, close, close, close, close))
        signals.append(rng.random(n) < 0.5)
    positions = [np.arange(30), np.arange(10, 30)]   # the second stock starts on day 10
    pooled = PooledMarket(markets, positions, 30)
    daily, trades = pooled.daily(np.concatenate(signals))
    first, n1 = markets[0].daily(signals[0])
    second, n2 = markets[1].daily(signals[1])
    assert trades == n1 + n2
    np.testing.assert_allclose(daily[:10], first[:10])
    np.testing.assert_allclose(daily[10:], (first[10:] + second) / 2)
