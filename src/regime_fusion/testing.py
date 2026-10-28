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


def check_point_in_time(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    items: pd.DataFrame,
    available: str,
    points: Iterable[Any],
) -> None:
    """Assert that a daily table built from dated items uses only items already public.

    The counterpart of ``check_no_lookahead`` for data that arrive as dated items
    (news articles, filed accounting values) rather than one row per day.
    Column ``available`` of ``items`` holds the first session on which each item
    may be used. ``build`` maps such items to a DataFrame indexed by session, with
    the same index whatever items it is given. For each session t in ``points``,
    row t of ``build(items[items[available] <= t])`` must equal row t of
    ``build(items)`` exactly. A build that uses an item before it is available
    changes when that item is withheld, and fails here.
    """
    full = build(items)
    for t in points:
        t = pd.Timestamp(t)
        partial = build(items[items[available] <= t])
        if not partial.index.equals(full.index):
            raise AssertionError("build must return the same index whatever items it is given")
        pd.testing.assert_series_equal(
            partial.loc[t], full.loc[t], check_names=False, check_exact=True
        )


def _head(data: Any, n: int) -> Any:
    return data.iloc[:n] if hasattr(data, "iloc") else data[:n]
