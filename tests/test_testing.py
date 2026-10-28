import numpy as np
import pandas as pd
import pytest

from regime_fusion.testing import check_no_lookahead, check_point_in_time

SESSIONS = pd.bdate_range("2016-01-04", periods=30)
# Accounting values: each covers a period ending on "end" and is filed later.
FILED = pd.DataFrame(
    {
        "end": SESSIONS[[0, 3, 3, 9, 15]],
        "available": SESSIONS[[2, 5, 8, 11, 20]],  # rows 2 and 3: a value, then its restatement
        "value": [1.0, 2.0, 2.5, 4.0, 5.0],
    }
)


def trailing_mean(x):
    return pd.DataFrame({"m": pd.Series(np.asarray(x, dtype=float)).rolling(5, min_periods=1).mean()})


def centred_mean(x):  # peeks two rows ahead
    return pd.DataFrame(
        {"m": pd.Series(np.asarray(x, dtype=float)).rolling(5, center=True, min_periods=1).mean()}
    )


def test_causal_feature_passes():
    x = np.random.default_rng(0).standard_normal(60)
    check_no_lookahead(trailing_mean, x, range(60))


def test_feature_that_looks_ahead_fails():
    x = np.random.default_rng(0).standard_normal(60)
    with pytest.raises(AssertionError):
        check_no_lookahead(centred_mean, x, range(60))


def latest_by(column):
    """Daily value of the most recent item, items dated by ``column``."""
    def build(items):
        ordered = items.sort_values([column, "available"])
        daily = ordered.groupby(column)["value"].last().reindex(SESSIONS).ffill()
        return daily.to_frame()
    return build


def test_point_in_time_build_passes():
    check_point_in_time(latest_by("available"), FILED, "available", SESSIONS)


def test_build_that_dates_values_by_period_end_fails():
    # Using a value from the end of its period, before it was filed, is look-ahead.
    with pytest.raises(AssertionError):
        check_point_in_time(latest_by("end"), FILED, "available", SESSIONS)
