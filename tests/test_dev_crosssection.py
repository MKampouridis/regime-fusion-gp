import numpy as np
import pandas as pd
import pytest

from regime_fusion.dev import crosssection as C
from regime_fusion.dev import small
from regime_fusion.testing import check_no_lookahead


def frame(values, columns=None):
    values = np.asarray(values, dtype=float)
    index = pd.bdate_range("2015-01-05", periods=len(values))
    return pd.DataFrame(values, index=index, columns=columns or list("ABCDEFGHIJ")[:values.shape[1]])


def log_linear(market, betas):
    """Daily simple returns whose logs are exactly beta x the market's log return."""
    return frame(np.expm1(np.outer(market, betas)))


def test_betas_are_recovered_exactly_from_noise_free_returns():
    market = np.random.default_rng(1).normal(0, 0.01, 40)
    b = np.array([0.5, 1.0, 1.5])
    out = C.weekly_beta(log_linear(market, b), window=40, step=5)
    # every stock's 5-session log return is b_i times a common series; the market is their mean, which is
    # mean(b) = 1 times that series, so the beta of stock i is b_i / mean(b)
    assert out.iloc[-1].to_numpy() == pytest.approx(b / b.mean())
    assert out.iloc[:-1].isna().all().all()    # no full window before the last row


def test_a_stock_with_a_missing_return_in_the_window_gets_nan_and_rows_outside_it_do_not_matter():
    market = np.random.default_rng(2).normal(0, 0.01, 30)
    returns = log_linear(market, [0.5, 1.0, 1.5])
    returns.iloc[15, 1] = np.nan           # inside the window of the last row (rows 10 to 29)
    returns.iloc[3, 0] = np.nan            # outside it
    out = C.weekly_beta(returns, window=20, step=5)
    last = out.iloc[-1]
    assert np.isnan(last["B"])
    # without B the market is the mean of A and C (mean beta 1.0), so the betas stay 0.5 and 1.5
    assert last["A"] == pytest.approx(0.5) and last["C"] == pytest.approx(1.5)
    changed = returns.copy()
    changed.iloc[:10] = 0.5
    assert C.weekly_beta(changed, window=20, step=5).iloc[-1].equals(last)


def test_ineligible_stocks_are_left_out_of_the_market_and_get_nan():
    market = np.random.default_rng(3).normal(0, 0.01, 20)
    returns = log_linear(market, [0.5, 1.0, 3.0])
    eligible = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    eligible["C"] = False
    last = C.weekly_beta(returns, eligible=eligible, window=20, step=5).iloc[-1]
    assert np.isnan(last["C"])
    # the market is the mean of A and B (mean beta 0.75)
    assert last["A"] == pytest.approx(0.5 / 0.75) and last["B"] == pytest.approx(1.0 / 0.75)


def test_weekly_beta_and_value_traded_do_not_look_ahead():
    rng = np.random.default_rng(4)
    returns = frame(rng.normal(0, 0.02, (40, 4)))
    check_no_lookahead(lambda r: C.weekly_beta(r, window=10, step=5), returns, [12, 20, 39])
    close = frame(rng.uniform(5, 10, (40, 3)))
    volume = frame(rng.uniform(1e3, 2e3, (40, 3)))
    check_no_lookahead(lambda c: C.value_traded(c, volume.loc[c.index], window=10, min_periods=5), close, [6, 20, 39])


def test_value_traded_is_the_rolling_median_of_close_times_volume():
    close = frame([[2.0], [2.0], [2.0], [2.0]])
    volume = frame([[10.0], [30.0], [20.0], [100.0]])
    out = C.value_traded(close, volume, window=3, min_periods=2)
    # values 20, 60, 40, 200: needs two sessions; medians 40 (20, 60), 40, 60 (60, 40, 200)
    assert np.isnan(out.iloc[0, 0])
    assert out.iloc[1:, 0].tolist() == [40.0, 40.0, 60.0]


def test_decision_positions_start_on_or_after_start_and_leave_room_for_the_return():
    days = pd.bdate_range("2015-01-05", periods=60)
    pos = C.decision_positions(days, days[2] + pd.Timedelta(hours=1), days[40], horizon=5)
    # first session on or after the start is position 3; last session on or before end is 40;
    # t + 1 + 5 <= 40 means t <= 34: positions 3, 8, 13, 18, 23, 28, 33
    assert pos.tolist() == [3, 8, 13, 18, 23, 28, 33]


def test_forward_returns_run_from_the_open_after_the_decision_to_the_open_horizon_sessions_later():
    opens = frame([[1.0, 10], [2, 10], [4, 20], [8, 20], [16, 5], [32, 5]])
    out = C.forward_returns(opens, np.array([0, 1]), horizon=2)
    # t = 0: open[3] / open[1] - 1 = 8 / 2 - 1 = 3 and 20 / 10 - 1 = 1; t = 1: 16 / 4 - 1 = 3 and 5 / 20 - 1 = -0.75
    assert out.to_numpy().tolist() == [[3.0, 1.0], [3.0, -0.75]]
    assert list(out.index) == [opens.index[0], opens.index[1]]


def test_prices_after_the_end_change_nothing():
    days = pd.bdate_range("2015-01-05", periods=30)
    rng = np.random.default_rng(5)
    opens = pd.DataFrame(rng.uniform(5, 10, (30, 2)), index=days, columns=["A", "B"])
    end = days[20]
    pos = C.decision_positions(days, days[0], end, horizon=4)
    before = C.forward_returns(opens, pos, horizon=4)
    later = opens.copy()
    later.iloc[21:] = 1e6
    assert C.decision_positions(days, days[0], end, horizon=4).tolist() == pos.tolist()
    pd.testing.assert_frame_equal(C.forward_returns(later, pos, horizon=4), before)
    assert pos.max() + 1 + 4 <= 20


def test_hedged_subtracts_beta_times_market_and_treats_a_missing_beta_as_one():
    forward = frame([[0.10, 0.20, 5.0], [0.0, -0.5, -2.0]])
    beta = frame([[2.0, np.nan, 9.0], [0.5, 1.0, -3.0]])
    out = C.hedged(forward, beta)
    # clipped forward: row 0 0.1, 0.2, 3.0 (market 3.3 / 3 = 1.1); row 1 0.0, -0.5, -0.9 (market -1.4 / 3)
    # clipped beta: row 0 2, (missing -> 1), 4; row 1 0.5, 1, -1
    m0, m1 = 1.1, -1.4 / 3
    assert out.iloc[0].tolist() == pytest.approx([0.1 - 2 * m0, 0.2 - m0, 3.0 - 4 * m0])
    assert out.iloc[1].tolist() == pytest.approx([0.0 - 0.5 * m1, -0.5 - m1, -0.9 + m1])
    given = C.hedged(forward, beta, pd.Series([0.0, 0.0], index=forward.index))
    assert given.iloc[0].tolist() == pytest.approx([0.1, 0.2, 3.0])


def test_information_coefficient_is_one_or_minus_one_for_ordered_scores_and_drops_thin_dates():
    score = frame([[1, 2, 3, 4], [1, 2, 3, np.nan]])
    returns = frame([[10, 20, 30, 40], [-1, -2, -3, 5]])
    out = C.information_coefficient(score, returns, min_names=3)
    assert out.tolist() == [1.0, -1.0]      # row 1 has three names, perfectly reversed
    assert len(C.information_coefficient(score, returns, min_names=4)) == 1


def test_quantile_weights_sum_to_plus_and_minus_one_on_the_top_and_bottom_fifth():
    score = frame([list(range(10)), [np.nan] + list(range(9))])
    tradable = pd.DataFrame(True, index=score.index, columns=score.columns)
    tradable.iloc[1, 9] = False                  # the best score of row 1 cannot be traded
    w = C.quantile_weights(score, tradable, 0.2)
    # row 0: ranks pct 0.1..1.0; longs pct > 0.8: J and I; shorts pct <= 0.2: A and B
    assert w.iloc[0].tolist() == [-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5]
    # row 1: tradable scored stocks B..I (8 names, pct 1/8..1): longs pct > 0.8 = H, I (top 2); shorts pct <= 0.2 = B
    assert w.iloc[1]["B"] == -1.0 and w.iloc[1]["H"] == 0.5 and w.iloc[1]["I"] == 0.5
    assert w.iloc[1].sum() == pytest.approx(0.0) and w.iloc[1].abs().sum() == pytest.approx(2.0)
    only = C.quantile_weights(score, tradable, 0.2, long_only=True)
    assert only.iloc[0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]


def test_banded_weights_enter_in_the_top_share_stay_in_the_wider_band_and_re_enter_only_in_the_top_share():
    names = list("ABCDEFGHIJ")
    # stock A against nine others scoring 1 to 9. Row 0: A = 10, the top (pct 1.0 > 0.9): enters.
    # Row 1: A = 8.5, second (pct 0.9): not above 0.9 but above 0.6, so it stays.
    # Row 2: A = 5.5, sixth from the top (pct 0.6): not above 0.6, so it leaves.
    # Row 3: A = 8.5 again (pct 0.9): not in the top 10%, so it does not come back.
    rows = [[x, 1, 2, 3, 4, 5, 6, 7, 8, 9] for x in (10, 8.5, 5.5, 8.5)]
    score = frame(rows, names)
    tradable = pd.DataFrame(True, index=score.index, columns=names)
    w = C.banded_weights(score, tradable, enter=0.1, leave=0.4, long_only=True)
    held = (w["A"] > 0).tolist()
    assert held == [True, True, False, False]
    # tradability matters: an untradable stock leaves at once
    tradable.iloc[1, 0] = False
    assert (C.banded_weights(score, tradable, 0.1, 0.4, long_only=True)["A"] > 0).tolist()[:2] == [True, False]


def test_banded_weights_are_zero_where_a_needed_book_is_empty_and_short_allowed_filters_shorts():
    score = frame([[1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5]], list("ABCDE"))
    tradable = pd.DataFrame(True, index=score.index, columns=score.columns)
    allowed = pd.DataFrame(True, index=score.index, columns=score.columns)
    allowed["A"] = False
    w = C.banded_weights(score, tradable, 0.2, 0.2, short_allowed=allowed)
    assert (w == 0).all().all()                  # the only short, A, is not allowed: no shorts, so no weights
    assert C.banded_weights(score, tradable, 0.2, 0.2).iloc[0].tolist() == [-1, 0, 0, 0, 1]


def test_cost_per_side_at_tier_boundaries_and_for_a_missing_value():
    value = frame([[10e6, 9.99e6, 5e6, 2e6, 1e6, 0.5e6, 0.49e6, np.nan]])
    assert C.cost_per_side(value).iloc[0].tolist() == [0.0010, 0.0020, 0.0020, 0.0035, 0.0060, 0.0100, 0.0200, 0.0200]


def test_portfolio_returns_turnover_and_both_kinds_of_cost():
    weights = frame([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]])
    returns = frame([[0.10, 0.20, np.nan], [0.0, 0.0, 0.10], [0.3, 0.3, 0.3]])
    out = C.portfolio_returns(weights, returns, 0.01)
    assert out["gross"].tolist() == pytest.approx([0.15, 0.05, 0.0])
    # turnover: first date 1.0 from zero; second |0| + |-0.5| + |0.5| = 1.0; third 0.5 + 0 + 0.5 = 1.0
    assert out["turnover"].tolist() == [1.0, 1.0, 1.0]
    assert out["cost"].tolist() == pytest.approx([0.01, 0.01, 0.01])
    assert out["net"].tolist() == pytest.approx([0.14, 0.04, -0.01])
    per_stock = frame([[0.01, 0.02, 0.04]] * 3)
    out = C.portfolio_returns(weights, returns, per_stock)
    # date 0: 0.5 * 0.01 + 0.5 * 0.02 = 0.015; date 1: 0.5 * 0.02 + 0.5 * 0.04 = 0.03; date 2: 0.5 * 0.01 + 0.5 * 0.04 = 0.025
    assert out["cost"].tolist() == pytest.approx([0.015, 0.03, 0.025])
    assert out["net"].tolist() == pytest.approx([0.135, 0.02, -0.025])


def test_summarise_a_short_series():
    index = pd.to_datetime(["2015-03-01", "2015-09-01", "2016-03-01", "2016-09-01"])
    x = pd.Series([0.02, 0.02, -0.01, 0.01], index=index)
    out = C.summarise(x, periods_per_year=4)
    # mean 0.01, sd = sqrt(((0.01)^2 * 2 + 0.02^2 + 0) / 3) = sqrt(0.0006 / 3) = sqrt(0.0002)
    sd = np.sqrt(0.0002)
    assert out["return"] == pytest.approx(0.04)
    assert out["volatility"] == pytest.approx(sd * 2)
    assert out["sharpe"] == pytest.approx(0.01 / sd * 2)
    assert out["t"] == pytest.approx(0.01 / sd * 2)      # sqrt(4) = 2
    assert out["periods"] == 4
    assert (out["years_positive"], out["years"]) == (1, 2)    # 2015: +0.04; 2016: 0.00 is not positive


def test_read_until_cuts_a_table_at_the_end_of_the_development_period(tmp_path):
    index = pd.to_datetime(["2018-12-28", "2018-12-31", "2019-01-02", "2019-06-28"])
    path = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=index).to_parquet(path)
    assert small.read_until(path).index[-1] == pd.Timestamp("2018-12-31")
    assert small.read_until(path, columns=["a"]).columns.tolist() == ["a"]
    assert len(small.read_until(path, end="2019-01-02", start="2018-12-31")) == 2
