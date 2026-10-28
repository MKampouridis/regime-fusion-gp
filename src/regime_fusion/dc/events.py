"""Directional-change (DC) event detection.

With threshold theta, a DC event is confirmed when the series reverses by theta from
its most recent extreme:

* a downturn when x_t <= H * (1 - theta), H being the running high;
* an upturn when x_t >= L * (1 + theta), L being the running low.

H and L are tracked from the previous confirmation onwards, the confirmation
point itself included. In "absolute" mode the reversal is measured in the units
of the series (x_t <= H - delta, x_t >= L + delta). That mode suits series that
can be zero or negative, such as daily news counts or sentiment scores.

Each event records the extreme that starts it (the DC extreme, DCE) and the
point at which it is confirmed (the DC confirmation, DCC). Only confirmations
are observable in real time. The extreme is known at confirmation; the
overshoot of a trend (from its confirmation to the extreme that ends it) is
known only when the next event is confirmed.

This follows the algorithm of the group's Java implementation
(``misc/DCEventGenerator.java`` in adesolaadegboye/SymbolicRegression).
Confirmation points match it exactly. Start points differ in three edge cases
of the Java code; ``tests/reference/dc_java/README.md`` describes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

MODES = ("relative", "absolute")


class Direction(IntEnum):
    DOWN = -1
    UP = 1


@dataclass(frozen=True)
class DCEvent:
    """One directional-change event."""

    direction: Direction  # UP for an upturn event, DOWN for a downturn event
    ext_idx: int          # position of the extreme that starts the event (DCE)
    conf_idx: int         # position at which the event is confirmed (DCC)
    ext_value: float
    conf_value: float


def detect_events(
    x, threshold: float, mode: str = "relative", initial_trend: str | None = None
) -> list[DCEvent]:
    """Directional-change events of a series, in order of confirmation.

    Parameters
    ----------
    x
        Values in time order (prices in ``relative`` mode).
    threshold
        Theta as a fraction (0.02 = 2%) in ``relative`` mode; delta in the
        series' own units in ``absolute`` mode.
    mode
        ``"relative"`` or ``"absolute"``.
    initial_trend
        ``None`` (default) tracks both the high and the low until the first
        confirmation, so the first event is whichever reversal happens first.
        ``"up"`` or ``"down"`` start as if already in that trend with the first
        value as its extreme; ``"up"`` reproduces the Java implementation.
    """
    values = np.asarray(x, dtype=float)
    _validate(values, threshold, mode, initial_trend)
    if values.size == 0:
        return []

    relative = mode == "relative"
    down_factor = 1.0 - threshold
    up_factor = 1.0 + threshold

    def reverses_down(value: float, high: float) -> bool:
        return value <= (high * down_factor if relative else high - threshold)

    def reverses_up(value: float, low: float) -> bool:
        return value >= (low * up_factor if relative else low + threshold)

    trend = {None: 0, "up": 1, "down": -1}[initial_trend]
    high = low = float(values[0])
    high_idx = low_idx = 0
    events: list[DCEvent] = []

    for t in range(1, values.size):
        value = float(values[t])
        if trend == 1:
            if value > high:
                high, high_idx = value, t
            elif reverses_down(value, high):
                events.append(DCEvent(Direction.DOWN, high_idx, t, high, value))
                trend, low, low_idx = -1, value, t
        elif trend == -1:
            if value < low:
                low, low_idx = value, t
            elif reverses_up(value, low):
                events.append(DCEvent(Direction.UP, low_idx, t, low, value))
                trend, high, high_idx = 1, value, t
        else:  # no trend yet: watch both extremes
            if reverses_down(value, high):
                events.append(DCEvent(Direction.DOWN, high_idx, t, high, value))
                trend, low, low_idx = -1, value, t
            elif reverses_up(value, low):
                events.append(DCEvent(Direction.UP, low_idx, t, low, value))
                trend, high, high_idx = 1, value, t
            else:
                if value > high:
                    high, high_idx = value, t
                if value < low:
                    low, low_idx = value, t
    return events


def _validate(values: np.ndarray, threshold: float, mode: str, initial_trend: str | None) -> None:
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if initial_trend not in (None, "up", "down"):
        raise ValueError("initial_trend must be None, 'up' or 'down'")
    if not np.all(np.isfinite(values)):
        raise ValueError("x contains NaN or infinite values; clean the series first")
    if mode == "relative":
        if not 0 < threshold < 1:
            raise ValueError("a relative threshold must lie strictly between 0 and 1")
        if np.any(values <= 0):
            raise ValueError("relative mode needs strictly positive values")
    elif not threshold > 0:
        raise ValueError("an absolute threshold must be positive")
