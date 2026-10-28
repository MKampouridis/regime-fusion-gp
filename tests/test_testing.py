import numpy as np
import pandas as pd
import pytest

from regime_fusion.testing import check_no_lookahead


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
