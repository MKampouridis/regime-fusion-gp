import numpy as np
import pytest

from regime_fusion.dev.designs import Design, PooledMarket, StockMarket, simulate_hold

OPEN = np.array([10.0, 10.0, 11.0, 12.0, 11.0, 10.0])
CLOSE = np.array([10.0, 10.5, 11.5, 11.8, 10.5, 10.2])


def test_hold_enters_at_the_next_open_and_leaves_at_the_open_after_the_signal_turns_off():
    signal = np.array([True, True, False, False, False, False])
    daily, trades, _ = simulate_hold(signal, OPEN, CLOSE, np.inf, 0.0)
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
    daily, _, _ = simulate_hold(signal, OPEN, CLOSE, np.inf, 0.001)
    assert np.prod(1 + daily) == pytest.approx(12.0 * 0.999 / (10.0 * 1.001))


def test_a_position_open_at_the_end_is_closed_at_the_last_close():
    daily, trades, _ = simulate_hold(np.ones(6, dtype=bool), OPEN, CLOSE, np.inf, 0.001)
    assert trades == 1
    assert np.prod(1 + daily) == pytest.approx(10.2 * 0.999 / (10.0 * 1.001))


def test_the_trailing_stop_closes_and_waits_for_the_signal_to_reset():
    open_ = np.array([10.0, 10.0, 12.0, 10.0, 9.0, 9.0, 9.0])
    close = np.array([10.0, 12.0, 12.0, 10.0, 9.0, 9.0, 9.5])
    signal = np.array([True, True, True, True, False, True, True])
    daily, trades, _ = simulate_hold(signal, open_, close, 0.10, 0.0)
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


def test_psr_lower_bound_needs_evidence():
    from regime_fusion.dev.fitness import FLOOR, psr_lower_bound
    rng = np.random.default_rng(2)
    trades = rng.normal(0.004, 0.02, 200)
    # the same kind of trades, fewer of them: a wider margin and a lower bound
    assert psr_lower_bound(trades) > psr_lower_bound(trades[:12])
    # the bound is below the plain Sharpe ratio
    assert psr_lower_bound(trades) < trades.mean() / trades.std(ddof=1)
    # fat tails with negative skew lower the bound at the same mean and volatility
    crash = np.full(200, 0.006)
    crash[:4] = 0.006 - 0.002 * 200 / 4
    crash = (crash - crash.mean()) / crash.std(ddof=1) * trades.std(ddof=1) + trades.mean()
    assert psr_lower_bound(crash) < psr_lower_bound(trades)
    assert psr_lower_bound(np.zeros(100)) == FLOOR


def test_trade_returns_match_the_daily_returns():
    rng = np.random.default_rng(4)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    open_, high, low = close * 0.999, close * 1.01, close * 0.99
    signal = rng.random(300) < 0.2
    for name in ("acttrade", "hold", "long"):
        daily, n, returns = Design(name, 0.001).simulate(signal, open_, high, low, close)
        assert len(returns) == n
        assert np.prod(1 + daily) == pytest.approx(np.prod(1 + returns))


def test_trailing_rank_uses_no_later_rows():
    import pandas as pd
    from regime_fusion.dev.features import trailing_rank
    from regime_fusion.testing import check_no_lookahead
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": np.cumsum(rng.normal(size=600)), "b": rng.normal(size=600)})
    frame.iloc[100:110, 1] = np.nan
    check_no_lookahead(trailing_rank, frame, [130, 200, 400, 599])
    ranked = trailing_rank(frame)
    assert ranked.iloc[:125].isna().all().all()
    steps = pd.DataFrame({"x": np.repeat([1.0, 2.0, 1.5], 200)})
    assert trailing_rank(steps)["x"].iloc[150] == pytest.approx(0.0, abs=0.01)   # a constant window sits in the middle
    assert ranked.stack().dropna().between(-1, 1).all()


def test_psr_lower_bound_never_drops_its_margin():
    from regime_fusion.dev.fitness import psr_lower_bound
    # four similar winning trades and one tiny one: strong positive skew, a negative variance term
    few = np.array([0.050, 0.051, 0.049, 0.050, 0.001])
    s = few.mean() / few.std(ddof=1)
    assert psr_lower_bound(few) < s - 1.0


def test_forward_returns_start_at_the_next_open():
    import pandas as pd
    from regime_fusion.dev.ic import forward_returns
    prices = pd.DataFrame({"adj_open": [10.0, 11.0, 12.0, 13.0, 14.0]})
    fwd = forward_returns(prices, 2)
    # decided at the close of day 0: bought at the open of day 1 (11), sold at the open of day 3 (13)
    assert fwd.iloc[0] == pytest.approx(13.0 / 11.0 - 1)
    assert fwd.iloc[2:].isna().all()


def test_long_short_market_holds_from_the_next_open_and_pays_costs_on_changes():
    from regime_fusion.dev.longshort import MIN_SIDE, LongShortMarket
    days, n = 12, 2 * MIN_SIDE
    opens = np.ones((days, n)) * 10.0
    opens[3:, :MIN_SIDE] = 11.0           # the first half rises by 10% at the open of day 3
    market = LongShortMarket(opens, horizon=4, cost=0.001)
    assert list(market.decisions) == [0, 4]
    signal = np.zeros((days, n), dtype=bool)
    signal[:, :MIN_SIDE] = True            # long the risers, short the rest
    net, active, _ = market.simulate(signal)
    # decision 0 holds from the open of day 1 to the open of day 5: long +10%, short 0%;
    # entering costs 2 (100% long + 100% short) x 0.1%; decision 4 keeps the same weights
    assert active == 2
    assert net[0] == pytest.approx(0.10 - 2 * 0.001)
    assert net[1] == pytest.approx(0.0)
    # too few stocks on one side: flat
    lonely = np.zeros((days, n), dtype=bool)
    lonely[:, 0] = True
    assert market.simulate(lonely)[1] == 0


def test_seeded_grammar_starts_with_every_median_rule():
    from regime_fusion.dev.longshort import SeededGrammar, median_rules
    groups = {"FA": ["fa:a", "fa:b"], "SA": ["sa:c"], "TA": ["ta:d"], "RG": []}
    rules = median_rules(groups, ("FA", "SA", "TA"))
    assert len(rules) == 2 * 4
    grammar = SeededGrammar(groups, fusion_functions=("AND", "OR"), seeds=rules)
    population = grammar.ramped_population(20, 2, np.random.default_rng(0))
    assert len(population) == 20
    assert [p.key() for p in population[:8]] == [r.key() for r in rules]
    data = {"fa:a": np.array([-0.5, 0.5]), "fa:b": np.zeros(2), "sa:c": np.zeros(2), "ta:d": np.zeros(2)}
    from regime_fusion.engine.tree import signals
    assert list(signals(rules[0], data)["C"]) == [False, True]   # fa:a above its median


def test_stock_factors_use_no_later_rows():
    import pandas as pd
    from regime_fusion.dev.factors import stock_factors
    from regime_fusion.testing import check_no_lookahead
    rng = np.random.default_rng(6)
    n = 600
    days = pd.bdate_range("2012-01-02", periods=n)
    close = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.015, n)), index=days)
    prices = pd.DataFrame({"adj_close": close, "close": close, "traded_close": close})
    steps = np.repeat(np.arange(n // 60 + 1), 60)[:n].astype(float)
    fundamentals = pd.DataFrame({c: 1e9 * (1 + 0.1 * steps) for c in (
        "shares_outstanding", "net_income_ttm", "net_income_ttm_prev", "equity", "revenue_ttm",
        "operating_cash_flow_ttm", "operating_income_ttm", "assets", "liabilities")}, index=days)
    market = pd.Series(rng.normal(0, 0.01, n), index=days)

    def compute(frame):
        return stock_factors(frame[["adj_close", "close", "traded_close"]], frame[fundamentals.columns],
                             frame["market"])

    data = pd.concat([prices, fundamentals, market.rename("market")], axis=1)
    check_no_lookahead(compute, data, [300, 450, 599])
    table = compute(data)
    assert table["mom_12_1"].iloc[300] == pytest.approx(close.iloc[279] / close.iloc[48] - 1)
