"""Study configuration: load ``configs/study.yaml`` into validated, immutable objects."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "study.yaml"

SCHEMES = ("rolling", "expanding")


class ConfigError(ValueError):
    """The study configuration is missing a value or is inconsistent."""


@dataclass(frozen=True)
class Period:
    data_start: dt.date
    data_end: dt.date
    development_end: dt.date
    test_years: tuple[int, ...]


@dataclass(frozen=True)
class Windows:
    scheme: str
    train_years: int
    validation_years: int
    refit_on_train_and_validation: bool


@dataclass(frozen=True)
class StudyConfig:
    name: str
    base_seed: int
    period: Period
    windows: Windows
    runs_per_configuration: int
    costs_bps_per_side: tuple[float, ...]
    primary_cost_bps_per_side: float

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible view (dates as ISO strings)."""
        return json.loads(json.dumps(asdict(self), default=str))

    def digest(self) -> str:
        """Short hash that identifies this configuration in result files."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def load_config(path: str | Path = DEFAULT_CONFIG) -> StudyConfig:
    """Read and validate a study configuration file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    try:
        study, period, windows = raw["study"], raw["period"], raw["windows"]
        runs, costs = raw["runs"], raw["costs"]
        config = StudyConfig(
            name=str(study["name"]),
            base_seed=int(study["base_seed"]),
            period=Period(
                data_start=_as_date(period["data_start"]),
                data_end=_as_date(period["data_end"]),
                development_end=_as_date(period["development_end"]),
                test_years=tuple(int(year) for year in period["test_years"]),
            ),
            windows=Windows(
                scheme=str(windows["scheme"]),
                train_years=int(windows["train_years"]),
                validation_years=int(windows["validation_years"]),
                refit_on_train_and_validation=bool(windows["refit_on_train_and_validation"]),
            ),
            runs_per_configuration=int(runs["per_configuration"]),
            costs_bps_per_side=tuple(float(cost) for cost in costs["bps_per_side"]),
            primary_cost_bps_per_side=float(costs["primary_bps_per_side"]),
        )
    except KeyError as missing:
        raise ConfigError(f"{path}: missing configuration key {missing}") from None
    validate(config)
    return config


def validate(config: StudyConfig) -> None:
    """Raise ConfigError if the configuration is internally inconsistent."""
    period, windows = config.period, config.windows
    if not period.data_start < period.development_end < period.data_end:
        raise ConfigError("need data_start < development_end < data_end")
    years = list(period.test_years)
    if not years:
        raise ConfigError("no test years given")
    if years != sorted(set(years)):
        raise ConfigError("test_years must be strictly increasing")
    if years[0] <= period.development_end.year:
        raise ConfigError("test years must come after the development period")
    if years[-1] > period.data_end.year:
        raise ConfigError("a test year lies beyond data_end")
    if windows.scheme not in SCHEMES:
        raise ConfigError(f"windows.scheme must be one of {SCHEMES}")
    if windows.train_years < 1 or windows.validation_years < 1:
        raise ConfigError("train_years and validation_years must be at least 1")
    first_train_year = years[0] - windows.validation_years - windows.train_years
    if first_train_year < period.data_start.year:
        raise ConfigError("the first walk-forward window starts before data_start")
    if config.runs_per_configuration < 1:
        raise ConfigError("runs.per_configuration must be at least 1")
    if any(cost < 0 for cost in config.costs_bps_per_side):
        raise ConfigError("transaction costs cannot be negative")
    if config.primary_cost_bps_per_side not in config.costs_bps_per_side:
        raise ConfigError("the primary cost level must be one of costs.bps_per_side")


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))
