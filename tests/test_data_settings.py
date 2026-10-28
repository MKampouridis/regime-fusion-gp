import datetime as dt
from pathlib import Path

import pytest
import yaml

from regime_fusion.config import ConfigError, load_config
from regime_fusion.data.settings import DEFAULT_SOURCES, load_settings


def write_variant(tmp_path: Path, section: str, key: str, value) -> Path:
    raw = yaml.safe_load(DEFAULT_SOURCES.read_text())
    raw[section][key] = value
    path = tmp_path / "data_sources.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_repository_settings_load_and_cover_the_study():
    settings = load_settings()
    assert settings.download_start == dt.date(2014, 1, 1)
    assert settings.download_end == dt.date(2025, 12, 31)
    assert "@" in settings.sec_user_agent
    settings.check_covers(load_config())


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("sec", "max_requests_per_second", 20),   # above the SEC's limit of 10
        ("sec", "user_agent", "anonymous"),       # no contact email
        ("download", "start", "2026-01-01"),      # after the end
        ("yahoo", "max_retries", 0),
    ],
)
def test_bad_settings_are_rejected(tmp_path, section, key, value):
    with pytest.raises(ConfigError):
        load_settings(write_variant(tmp_path, section, key, value))


def test_downloads_must_cover_the_study_period(tmp_path):
    late = load_settings(write_variant(tmp_path, "download", "start", "2016-01-01"))
    with pytest.raises(ConfigError, match="data_start"):
        late.check_covers(load_config())


def test_missing_setting_is_reported(tmp_path):
    raw = yaml.safe_load(DEFAULT_SOURCES.read_text())
    del raw["cboe"]
    path = tmp_path / "data_sources.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="cboe"):
        load_settings(path)
