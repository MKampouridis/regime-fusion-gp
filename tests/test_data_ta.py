import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import ta
from regime_fusion.testing import check_no_lookahead

DAYS = pd.bdate_range("2016-01-04", periods=40)


def frame(close, adjust=None):
    close = pd.Series(np.asarray(close, dtype=float), index=DAYS[:len(close)])
    adjust = pd.Series(1.0, index=close.index) if adjust is None else adjust
    return pd.DataFrame({"traded_close": close, "adj_close": close * adjust,
                         "adj_high": (close + 1) * adjust, "adj_low": (close - 1) * adjust})


def test_hand_worked_values():
    out = ta.indicators(frame([10, 11, 12, 13, 14, 15]), windows=(5,)).iloc[5]
    assert out["moving_average_5"] == 13.0           # mean of 11..15
    assert out["momentum_5"] == 5.0 and out["roc_5"] == pytest.approx(50.0)
    assert out["volatility_5"] == pytest.approx(np.std([0.5, 0.4, 0.3, 0.2, 0.1], ddof=1))
    assert out["williams_r_5"] == pytest.approx(-100 * (16 - 15) / (16 - 10))
    assert out["midprice_5"] == 13.0                 # (16 + 10) / 2
    assert list(ta.indicators(frame(range(10, 30))).columns) == ta.columns()


def test_levels_do_not_depend_on_later_dividends():
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 40))
    before = frame(close)                             # downloaded before a dividend
    adjust = pd.Series(np.where(np.arange(40) < 30, 0.98, 1.0), index=DAYS)
    after = frame(close, adjust)                      # downloaded after a dividend on day 30
    early = slice(0, 30)
    pd.testing.assert_frame_equal(ta.indicators(after).iloc[early], ta.indicators(before).iloc[early])


def test_levels_run_on_smoothly_across_a_split():
    economic = 100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 40))  # in post-split units
    traded = np.where(np.arange(40) < 20, 2 * economic, economic)          # a 2-for-1 split on day 20
    prices = pd.DataFrame({"traded_close": traded, "adj_close": economic, "adj_high": economic + 1,
                           "adj_low": economic - 1}, index=DAYS)
    out = ta.indicators(prices, windows=(5,))
    level = out["moving_average_5"]
    assert level.iloc[19] == pytest.approx(2 * economic[15:20].mean())    # in the first session's prices
    assert level.iloc[20] == pytest.approx(2 * economic[16:21].mean())    # no fall by half on the split
    before = ta.indicators(prices.iloc[:20], windows=(5,))                  # known before the split: unchanged
    pd.testing.assert_frame_equal(before, out.iloc[:20])


def test_no_lookahead():
    close = 50 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 40)))
    check_no_lookahead(ta.indicators, frame(close), range(40))


def test_gaps_and_flat_windows_give_missing_values():
    prices = frame(np.linspace(20, 30, 20))
    prices.iloc[8] = np.nan                            # a session without a bar
    out = ta.indicators(prices, windows=(5,))
    assert out["moving_average_5"].iloc[8:13].isna().all() and out["moving_average_5"].iloc[13:].notna().all()
    flat = frame([10.0] * 12)
    flat["adj_high"] = flat["adj_low"] = flat["adj_close"]
    assert ta.indicators(flat, windows=(5,))["williams_r_5"].iloc[5:].isna().all()
