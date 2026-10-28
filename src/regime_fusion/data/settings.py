"""Data-pipeline settings: load ``configs/data_sources.yaml``."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from regime_fusion.config import ConfigError, StudyConfig, _as_date

DEFAULT_SOURCES = Path(__file__).resolve().parents[3] / "configs" / "data_sources.yaml"
SEC_RATE_LIMIT = 10.0  # requests per second allowed by the SEC's fair-access policy


@dataclass(frozen=True)
class DataSettings:
    download_start: dt.date
    download_end: dt.date
    yahoo_pause_seconds: float
    yahoo_max_retries: int
    vix_url: str
    sec_user_agent: str
    sec_max_requests_per_second: float
    raw: dict[str, Any]  # the whole file, for sections not yet typed here

    def check_covers(self, config: StudyConfig) -> None:
        """Raise ConfigError unless the downloads cover the study period."""
        if self.download_start > config.period.data_start:
            raise ConfigError("download.start is later than period.data_start")
        if self.download_end < config.period.data_end:
            raise ConfigError("download.end is earlier than period.data_end")


def load_settings(path: str | Path = DEFAULT_SOURCES) -> DataSettings:
    """Read and validate the data-source settings."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    try:
        download, yahoo, sec = raw["download"], raw["yahoo"], raw["sec"]
        settings = DataSettings(
            download_start=_as_date(download["start"]),
            download_end=_as_date(download["end"]),
            yahoo_pause_seconds=float(yahoo["pause_seconds"]),
            yahoo_max_retries=int(yahoo["max_retries"]),
            vix_url=str(raw["cboe"]["vix_url"]),
            sec_user_agent=str(sec["user_agent"]),
            sec_max_requests_per_second=float(sec["max_requests_per_second"]),
            raw=raw,
        )
    except KeyError as missing:
        raise ConfigError(f"{path}: missing setting {missing}") from None
    if not settings.download_start < settings.download_end:
        raise ConfigError("need download.start < download.end")
    if not 0 < settings.sec_max_requests_per_second <= SEC_RATE_LIMIT:
        raise ConfigError(f"sec.max_requests_per_second must lie in (0, {SEC_RATE_LIMIT:g}]")
    if "@" not in settings.sec_user_agent:
        raise ConfigError("sec.user_agent must contain a contact email address")
    if settings.yahoo_pause_seconds < 0 or settings.yahoo_max_retries < 1:
        raise ConfigError("yahoo.pause_seconds must be >= 0 and yahoo.max_retries >= 1")
    return settings
