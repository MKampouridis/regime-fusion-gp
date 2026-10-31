import datetime as dt
from pathlib import Path

import pytest
import yaml

from regime_fusion.config import DEFAULT_CONFIG, ConfigError, load_config


def write_variant(tmp_path: Path, **changes) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        raw[section][key] = value
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_repository_config_loads():
    config = load_config()
    assert config.period.data_start == dt.date(2015, 2, 1)
    assert config.period.development_end == dt.date(2018, 12, 31)
    assert config.period.test_years == (2019, 2020, 2021, 2022, 2023, 2024, 2025)
    assert config.windows.scheme == "rolling"
    assert config.runs_per_configuration == 50
    assert config.primary_cost_bps_per_side == 10


def test_digest_is_stable_and_sensitive(tmp_path):
    base = load_config()
    assert base.digest() == load_config().digest()
    changed = load_config(write_variant(tmp_path, runs__per_configuration=base.runs_per_configuration + 1))
    assert changed.digest() != base.digest()


@pytest.mark.parametrize(
    "changes",
    [
        {"period__test_years": [2018, 2019]},             # test year inside development period
        {"period__test_years": [2020, 2019]},             # not increasing
        {"period__test_years": [2026]},                   # beyond data_end
        {"windows__scheme": "sliding"},                   # unknown scheme
        {"windows__train_years": 5},                      # first window starts before the data
        {"costs__primary_bps_per_side": 7},               # primary cost not in the list
        {"runs__per_configuration": 0},
    ],
)
def test_inconsistent_configs_are_rejected(tmp_path, changes):
    with pytest.raises(ConfigError):
        load_config(write_variant(tmp_path, **changes))


def test_missing_key_is_reported(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    del raw["costs"]
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="costs"):
        load_config(path)
