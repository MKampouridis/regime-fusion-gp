"""Checks shared by the test-suite and by feature code."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


def check_no_lookahead(
    compute: Callable[[Any], pd.DataFrame], data: Any, points: Iterable[int]
) -> None:
    """Assert that a feature computation never uses later rows.

    ``compute`` maps a sequence (array, Series or DataFrame) to a DataFrame with
    one row per input row. For each position t in ``points``, the last row of
    ``compute(data[: t + 1])`` must equal row t of ``compute(data)`` exactly.
    A computation that looks ahead changes its earlier rows when later data are
    added, and fails here.
    """
    full = compute(data)
    if len(full) != len(data):
        raise AssertionError("compute must return one row per input row")
    for t in points:
        partial = compute(_head(data, t + 1))
        pd.testing.assert_series_equal(
            partial.iloc[-1], full.iloc[t], check_names=False, check_exact=True
        )


def _head(data: Any, n: int) -> Any:
    return data.iloc[:n] if hasattr(data, "iloc") else data[:n]
