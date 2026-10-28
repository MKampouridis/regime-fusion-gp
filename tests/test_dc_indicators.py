import numpy as np
import pandas as pd
import pytest

from regime_fusion.dc import dc_indicator_frame, dc_indicators, detect_events
from regime_fusion.testing import check_no_lookahead


def random_walk(seed, n=400, sigma=0.02):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))


def test_hand_worked_values():
    x = [100, 104, 111, 115, 103, 102, 113, 112]
    frame = dc_indicators(x, 0.10, prefix="")

    assert frame["trend"].tolist() == [0, 0, 1, 1, -1, -1, 1, 1]
    assert frame.loc[:1, "osv"].isna().all()  # nothing confirmed yet

    last = frame.iloc[7]  # upturn confirmed at 113 (from 102) on day 6; price now 112
    assert last["osv"] == pytest.approx((112 / 113 - 1) / 0.1)
    assert last["tmv"] == pytest.approx((112 / 102 - 1) / 0.1)
    assert last["t_os"] == 1 and last["t_dc"] == 1
    assert last["prev_tmv"] == pytest.approx((102 / 115 - 1) / 0.1)  # the completed downtrend
    assert last["prev_t"] == 2
    assert frame["ndc"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("threshold", [0.02, 0.05])
def test_no_lookahead_on_prices(threshold):
    x = random_walk(1)
    check_no_lookahead(lambda s: dc_indicators(s, threshold, window=20), x, range(len(x)))


def test_no_lookahead_on_news_counts_in_absolute_mode():
    rng = np.random.default_rng(2)
    attention = pd.Series(
        np.log1p(rng.poisson(rng.gamma(2.0, 3.0, 300))),
        index=pd.bdate_range("2016-01-04", periods=300),
    )
    compute = lambda s: dc_indicators(s, 0.7, mode="absolute", window=20)  # noqa: E731
    check_no_lookahead(compute, attention, range(len(attention)))


def test_ndc_counts_confirmations_in_window():
    x = random_walk(3, n=500)
    window = 25
    frame = dc_indicators(x, 0.02, window=window, prefix="")
    confirmations = np.array([e.conf_idx for e in detect_events(x, 0.02)])
    for t in range(len(x)):
        expected = np.sum((confirmations > t - window) & (confirmations <= t))
        assert frame["ndc"].iloc[t] == expected


def test_dates_are_kept_and_columns_named_by_threshold():
    dates = pd.bdate_range("2019-01-02", periods=120)
    x = pd.Series(random_walk(4, n=120), index=dates)
    frame = dc_indicator_frame(x, [0.01, 0.05])
    assert frame.index.equals(dates)
    assert "dc0.01_osv" in frame.columns and "dc0.05_ndc" in frame.columns
    assert frame.shape[1] == 16
