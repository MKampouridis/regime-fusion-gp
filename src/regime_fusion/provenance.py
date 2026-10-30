"""Record what produced a result: code version, configuration, seed and environment."""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import os
import secrets
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from regime_fusion import __version__
from regime_fusion.config import StudyConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_PACKAGES = (
    "numpy", "pandas", "pyyaml", "pyarrow", "exchange_calendars", "scipy", "numba",
    "scikit-learn", "lightgbm", "torch", "requests", "yfinance",
)


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit and whether the working tree had uncommitted changes, when this process first asked:
    the code it runs, even if the files change while it works (and no git call per file written)."""
    return dict(_git_state(repo))


@functools.lru_cache(maxsize=None)
def _git_state(repo: Path) -> dict[str, Any]:

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def digest(value: Any) -> str:
    """A short, stable hash of a JSON-like value (a method's settings, say), to tell versions apart."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def package_versions() -> dict[str, str]:
    return dict(_package_versions())


@functools.lru_cache(maxsize=1)
def _package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def collect(config: StudyConfig | None = None, **extra: Any) -> dict[str, Any]:
    """Provenance record for one run; ``extra`` holds run details such as stock and seed."""
    record: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    if config is not None:
        record["config_digest"] = config.digest()
        record["config"] = config.to_dict()
    record.update(extra)
    return record


def write(path: str | Path, record: dict[str, Any]) -> None:
    """Write a record as JSON, whole or not at all (a stopped job leaves no partial file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temporary(path)
    try:
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def temporary(path: Path) -> Path:
    """A temporary name beside ``path``, its own to this process and write, so that two jobs writing
    the same file at once never mix their parts (the last whole file wins)."""
    return path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.part")

