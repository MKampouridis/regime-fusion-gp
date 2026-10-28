"""Causal daily directional-change indicators.

Row t uses only events confirmed on or before t, so every value is known by the
close of day t. Moves are measured in threshold units: a move of 1.0 equals one
threshold (theta in relative mode, delta in absolute mode).

Columns (prefixed, by default with ``dc<threshold>_``):

trend     +1 after an upturn confirmation, -1 after a downturn, 0 before the first event
osv       overshoot so far: move from the last confirmation price to today
tmv       total move of the current trend so far: from its extreme to today
t_os      bars since the last confirmation
t_dc      bars from the current trend's extreme to its confirmation
prev_tmv  total move of the previous, completed trend (extreme to extreme)
prev_t    bars of the previous, completed trend (extreme to extreme)
ndc       number of confirmations in the trailing ``window`` bars, today included

osv, tmv and prev_tmv are signed: positive for rises, negative for falls.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from regime_fusion.dc.events import detect_events

COLUMNS = ("trend", "osv", "tmv", "t_os", "t_dc", "prev_tmv", "prev_t", "ndc")


def dc_indicators(
    x,
    threshold: float,
    *,
    mode: str = "relative",
    window: int = 60,
    initial_trend: str | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Daily DC indicators for one threshold; one row per input value."""
    if window < 1:
        raise ValueError("window must be at least 1")
    values = np.asarray(x, dtype=float)
    index = x.index if isinstance(x, pd.Series) else pd.RangeIndex(values.size)
    events = detect_events(values, threshold, mode=mode, initial_trend=initial_trend)

    if mode == "relative":
        def move(start, end):
            return (end / start - 1.0) / threshold
    else:
        def move(start, end):
            return (end - start) / threshold

    n = values.size
    out = {name: np.full(n, np.nan) for name in COLUMNS}
    out["trend"][:] = 0.0

    k = -1  # last event confirmed on or before t
    for t in range(n):
        while k + 1 < len(events) and events[k + 1].conf_idx <= t:
            k += 1
        if k < 0:
            continue
        event = events[k]
        out["trend"][t] = float(event.direction)
        out["osv"][t] = move(event.conf_value, values[t])
        out["tmv"][t] = move(event.ext_value, values[t])
        out["t_os"][t] = t - event.conf_idx
        out["t_dc"][t] = event.conf_idx - event.ext_idx
        if k >= 1:
            previous = events[k - 1]
            out["prev_tmv"][t] = move(previous.ext_value, event.ext_value)
            out["prev_t"][t] = event.ext_idx - previous.ext_idx

    confirmations = np.array([event.conf_idx for event in events], dtype=np.int64)
    positions = np.arange(n)
    out["ndc"] = (
        np.searchsorted(confirmations, positions, side="right")
        - np.searchsorted(confirmations, positions - window, side="right")
    ).astype(float)

    label = f"dc{threshold:g}_" if prefix is None else prefix
    return pd.DataFrame({f"{label}{name}": out[name] for name in COLUMNS}, index=index)


def dc_indicator_frame(
    x, thresholds: Iterable[float], *, mode: str = "relative", window: int = 60
) -> pd.DataFrame:
    """DC indicators for several thresholds side by side."""
    frames = [dc_indicators(x, theta, mode=mode, window=window) for theta in thresholds]
    return pd.concat(frames, axis=1)
