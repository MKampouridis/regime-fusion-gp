"""Walk-forward schedule: which dates train, validate and test each yearly model.

For every test year Y (rolling scheme, 3 + 1 years):

* stage 1 evolves strategies on Y-4 .. Y-2 and chooses the exit settings on Y-1;
* stage 2 re-evolves the strategies on Y-4 .. Y-1 with those settings;
* the stage-2 strategies are tested on Y.

The expanding scheme is the same except that training always starts at data_start.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from regime_fusion.config import StudyConfig


@dataclass(frozen=True)
class DateRange:
    """Inclusive range of calendar dates."""

    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"empty date range {self.start} .. {self.end}")

    def contains(self, day: dt.date) -> bool:
        return self.start <= day <= self.end

    def overlaps(self, other: DateRange) -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class Window:
    test_year: int
    train: DateRange        # stage 1: evolve strategies
    validation: DateRange   # stage 1: choose exit settings
    final_train: DateRange  # stage 2: re-evolve the strategies that are tested
    test: DateRange


def development_range(config: StudyConfig) -> DateRange:
    """The only data that may inform design decisions (the pilot, tuning)."""
    return DateRange(config.period.data_start, config.period.development_end)


def walk_forward(config: StudyConfig) -> list[Window]:
    """One window per test year, checked for overlap between training and testing."""
    period, scheme = config.period, config.windows
    windows = []
    for year in period.test_years:
        validation_start = year - scheme.validation_years
        if scheme.scheme == "rolling":
            train_start = max(dt.date(validation_start - scheme.train_years, 1, 1), period.data_start)
        else:
            train_start = period.data_start
        train = DateRange(train_start, dt.date(validation_start - 1, 12, 31))
        validation = DateRange(dt.date(validation_start, 1, 1), dt.date(year - 1, 12, 31))
        final_train = (
            DateRange(train.start, validation.end) if scheme.refit_on_train_and_validation else train
        )
        test = DateRange(dt.date(year, 1, 1), min(dt.date(year, 12, 31), period.data_end))
        window = Window(year, train, validation, final_train, test)
        _check(window, config)
        windows.append(window)
    return windows


def _check(window: Window, config: StudyConfig) -> None:
    for name in ("train", "validation", "final_train"):
        part = getattr(window, name)
        if part.overlaps(window.test) or part.end >= window.test.start:
            raise ValueError(f"{window.test_year}: {name} period reaches into the test year")
        if part.start < config.period.data_start:
            raise ValueError(f"{window.test_year}: {name} period starts before data_start")
    if window.train.overlaps(window.validation):
        raise ValueError(f"{window.test_year}: training and validation periods overlap")
