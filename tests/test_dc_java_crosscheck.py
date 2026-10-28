"""Cross-check against the group's earlier Java DC implementation.

The fixtures in tests/reference/dc_java hold five synthetic price series and the
events the Java code (DCReference.java, a faithful extraction) finds in them at
five thresholds. Confirmation points must match exactly. Start points (the
extremes) match except in three edge cases of the Java code, which are checked
precisely here so that any other difference fails the test.
"""

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from regime_fusion.dc import Direction, detect_events

HERE = Path(__file__).resolve().parent / "reference" / "dc_java"


def load_java_events():
    events = defaultdict(list)
    with open(HERE / "events.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            direction = Direction.UP if row["type"] == "Upturn" else Direction.DOWN
            # Java indices are 1-based; convert to 0-based positions.
            events[(int(row["series"]), row["threshold"])].append(
                (direction, int(row["start"]) - 1, int(row["end"]) - 1)
            )
    return events


JAVA = load_java_events()
CASES = sorted(JAVA)


def load_series(k):
    return [float(v) for v in (HERE / "series" / f"series_{k}.txt").read_text().split()]


def test_fixtures_cover_all_series_and_thresholds():
    assert len(CASES) == 25
    assert sum(len(v) for v in JAVA.values()) > 5000


@pytest.mark.parametrize("case", CASES, ids=[f"series{k}-theta{t}" for k, t in CASES])
def test_confirmation_points_match_java_exactly(case):
    k, threshold = case
    ours = detect_events(load_series(k), float(threshold), initial_trend="up")
    java = JAVA[case]
    assert [(e.direction, e.conf_idx) for e in ours] == [(d, end) for d, _, end in java]


@pytest.mark.parametrize("case", CASES, ids=[f"series{k}-theta{t}" for k, t in CASES])
def test_start_points_differ_only_in_known_java_edge_cases(case):
    k, threshold = case
    ours = detect_events(load_series(k), float(threshold), initial_trend="up")
    java = JAVA[case]
    for i, (event, (_, java_start, _)) in enumerate(zip(ours, java)):
        previous_conf = ours[i - 1].conf_idx if i else None
        if previous_conf is not None and event.ext_idx == previous_conf:
            # No new extreme after the previous confirmation, so the trend's
            # extreme is that confirmation point. Java instead starts an upturn
            # one bar later, and a downturn at the start of the previous upturn.
            if event.direction is Direction.UP:
                assert java_start == event.ext_idx + 1
            else:
                assert java_start == java[i - 1][1]
        elif event.direction is Direction.UP and previous_conf is not None and event.ext_idx == previous_conf + 2:
            # The low comes two bars after the downturn confirmation; Java's
            # adjust() moves the upturn's start one bar too early.
            assert java_start == event.ext_idx - 1
        else:
            assert java_start == event.ext_idx
