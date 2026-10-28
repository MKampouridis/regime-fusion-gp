import dataclasses
import datetime as dt

from regime_fusion.config import load_config
from regime_fusion.windows import DateRange, development_range, walk_forward

D = dt.date


def test_rolling_schedule_matches_the_plan():
    windows = {w.test_year: w for w in walk_forward(load_config())}
    assert sorted(windows) == list(range(2019, 2026))

    first = windows[2019]
    assert first.train == DateRange(D(2015, 2, 1), D(2017, 12, 31))  # clipped to data_start
    assert first.validation == DateRange(D(2018, 1, 1), D(2018, 12, 31))
    assert first.final_train == DateRange(D(2015, 2, 1), D(2018, 12, 31))
    assert first.test == DateRange(D(2019, 1, 1), D(2019, 12, 31))

    last = windows[2025]
    assert last.train == DateRange(D(2021, 1, 1), D(2023, 12, 31))
    assert last.validation == DateRange(D(2024, 1, 1), D(2024, 12, 31))
    assert last.final_train == DateRange(D(2021, 1, 1), D(2024, 12, 31))


def test_final_training_ends_the_day_before_each_test_year():
    for window in walk_forward(load_config()):
        assert window.final_train.end == window.test.start - dt.timedelta(days=1)
        assert not window.final_train.overlaps(window.test)


def test_expanding_scheme_always_starts_at_data_start():
    config = load_config()
    expanding = dataclasses.replace(
        config, windows=dataclasses.replace(config.windows, scheme="expanding")
    )
    for window in walk_forward(expanding):
        assert window.train.start == config.period.data_start
        assert window.final_train.end.year == window.test_year - 1


def test_no_refit_keeps_stage_one_training():
    config = load_config()
    no_refit = dataclasses.replace(
        config, windows=dataclasses.replace(config.windows, refit_on_train_and_validation=False)
    )
    for window in walk_forward(no_refit):
        assert window.final_train == window.train


def test_development_period_excludes_every_test_year():
    config = load_config()
    development = development_range(config)
    for window in walk_forward(config):
        assert not development.overlaps(window.test)
