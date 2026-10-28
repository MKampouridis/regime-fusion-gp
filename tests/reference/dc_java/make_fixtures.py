"""Regenerate the Java reference fixtures used by tests/test_dc_java_crosscheck.py.

Needs a Java JDK (javac and java on the PATH). Run from the repository root:

    python tests/reference/dc_java/make_fixtures.py

Writes series/series_<k>.txt (one value per line, full precision) and
events.csv (series, threshold, type, start, end as printed by the Java code).
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
THRESHOLDS = ("0.005", "0.01", "0.02", "0.05", "0.1")
N = 1500


def make_series() -> list[np.ndarray]:
    rng = np.random.default_rng(20261028)

    def walk(sigma, jump_prob=0.0, flat_prob=0.0):
        steps = rng.normal(0.0, sigma, N)
        jumps = rng.random(N) < jump_prob
        steps[jumps] += rng.normal(0.0, 0.08, jumps.sum())
        steps[rng.random(N) < flat_prob] = 0.0
        return 100 * np.exp(np.cumsum(steps))

    return [
        np.round(walk(0.01), 2),                  # calm, cent prices (ties occur)
        np.round(walk(0.02), 2),
        walk(0.03),                               # volatile, unrounded
        np.round(walk(0.015, jump_prob=0.02), 2),  # occasional large jumps
        np.round(walk(0.02, flat_prob=0.4), 2),   # many unchanged days
    ]


def main() -> None:
    series_dir = HERE / "series"
    build_dir = HERE / "build"
    series_dir.mkdir(exist_ok=True)
    build_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["javac", "-d", str(build_dir), str(HERE / "DCReference.java")], check=True
    )

    rows = []
    for k, values in enumerate(make_series()):
        path = series_dir / f"series_{k}.txt"
        path.write_text("".join(f"{v!r}\n" for v in values.tolist()))
        for threshold in THRESHOLDS:
            result = subprocess.run(
                ["java", "-cp", str(build_dir), "DCReference", str(path), threshold],
                check=True, capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                kind, start, end = line.split(",")
                rows.append([k, threshold, kind, int(start), int(end)])

    with open(HERE / "events.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "threshold", "type", "start", "end"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} Java events for {k + 1} series")


if __name__ == "__main__":
    main()
