import numpy as np
import pytest

from regime_fusion.dc import Direction, detect_events

UP, DOWN = Direction.UP, Direction.DOWN


def summary(events):
    return [(int(e.direction), e.ext_idx, e.conf_idx) for e in events]


def reference_events(x, threshold, mode="relative"):
    """Slow, independent re-statement of the DC definition (no trend at the start)."""
    if mode == "relative":
        def down(v, high): return v <= high * (1.0 - threshold)
        def up(v, low): return v >= low * (1.0 + threshold)
    else:
        def down(v, high): return v <= high - threshold
        def up(v, low): return v >= low + threshold
    events, start, trend = [], 0, 0
    for t in range(1, len(x)):
        past = x[start:t]  # from the last confirmation (inclusive) to yesterday
        high_idx = start + int(np.argmax(past))  # first occurrence, as in the detector
        low_idx = start + int(np.argmin(past))
        if trend in (0, 1) and down(x[t], x[high_idx]):
            events.append((-1, high_idx, t))
            start, trend = t, -1
        elif trend in (0, -1) and up(x[t], x[low_idx]):
            events.append((1, low_idx, t))
            start, trend = t, 1
    return events


def random_walk(seed, n=600, sigma=0.02, decimals=None):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    return np.round(prices, decimals) if decimals is not None else prices


def test_hand_worked_relative_example():
    x = [100, 104, 111, 115, 103, 102, 113, 112]
    events = detect_events(x, 0.10)
    # 111 >= 100 * 1.1 confirms an upturn from 100; 103 <= 115 * 0.9 a downturn
    # from 115; 113 >= 102 * 1.1 an upturn from 102.
    assert summary(events) == [(1, 0, 2), (-1, 3, 4), (1, 5, 6)]
    assert events[1].ext_value == 115 and events[1].conf_value == 103


def test_hand_worked_absolute_example_with_zeros():
    x = [0, 0, 3, 5, 1, 0, 4]  # e.g. daily news counts
    assert summary(detect_events(x, 3, mode="absolute")) == [(1, 0, 2), (-1, 3, 4), (1, 5, 6)]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("threshold", [0.01, 0.03, 0.08])
def test_matches_independent_reference(seed, threshold):
    x = random_walk(seed, decimals=2)  # rounding creates ties, as in real prices
    assert summary(detect_events(x, threshold)) == reference_events(x, threshold)


@pytest.mark.parametrize("seed", range(3))
def test_matches_reference_in_absolute_mode(seed):
    rng = np.random.default_rng(seed)
    counts = np.log1p(rng.poisson(rng.gamma(2.0, 3.0, 400)))
    assert summary(detect_events(counts, 0.7, mode="absolute")) == reference_events(
        counts, 0.7, mode="absolute"
    )


@pytest.mark.parametrize("seed", range(5))
def test_structural_properties(seed):
    x = random_walk(seed, n=1500)
    events = detect_events(x, 0.02)
    assert len(events) > 20
    for previous, event in zip(events, events[1:]):
        assert event.direction == -previous.direction  # directions alternate
        assert event.conf_idx > previous.conf_idx
        assert event.ext_idx >= previous.conf_idx  # each trend's extreme follows the last confirmation
    for event in events:
        assert event.ext_idx < event.conf_idx
        if event.direction is DOWN:
            assert event.conf_value <= event.ext_value * (1 - 0.02)
        else:
            assert event.conf_value >= event.ext_value * (1 + 0.02)


@pytest.mark.parametrize("seed", range(3))
def test_events_never_change_when_more_data_arrive(seed):
    x = random_walk(seed, n=800)
    full = summary(detect_events(x, 0.02))
    for m in range(2, len(x), 37):
        assert summary(detect_events(x[:m], 0.02)) == [e for e in full if e[2] < m]


def test_initial_trend_up_waits_for_a_downturn():
    x = [100, 112, 125, 110]
    assert summary(detect_events(x, 0.1)) == [(1, 0, 1), (-1, 2, 3)]
    assert summary(detect_events(x, 0.1, initial_trend="up")) == [(-1, 2, 3)]


def test_flat_and_short_series():
    assert detect_events([], 0.05) == []
    assert detect_events([100.0], 0.05) == []
    assert detect_events([100.0] * 50, 0.05) == []


@pytest.mark.parametrize(
    "x, threshold, mode",
    [
        ([100, 0, 101], 0.05, "relative"),        # non-positive price
        ([100, np.nan, 101], 0.05, "relative"),   # missing value
        ([100, 101], 0.0, "relative"),
        ([100, 101], 1.5, "relative"),
        ([1, 2], -1.0, "absolute"),
        ([1, 2], 1.0, "logarithmic"),
    ],
)
def test_invalid_input_is_rejected(x, threshold, mode):
    with pytest.raises(ValueError):
        detect_events(x, threshold, mode=mode)
