"""Where data live, and how files are written.

All data sit under the folder named by the environment variable ``RFGP_DATA_DIR``
(default ``./data`` in the repository, which git ignores), in four layers:

raw/        files exactly as downloaded, never edited
interim/    cleaned tables aligned to the trading calendar
processed/  model-ready daily panels
reports/    coverage and quality reports

Files are written atomically (to a temporary name, then renamed), so an
interrupted download never leaves a half-written file under the final name.
Each file gets a provenance sidecar, ``<name>.provenance.json``, recording its
checksum, the code version and the details of how it was obtained.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd

from regime_fusion import provenance

DATA_DIR_ENV = "RFGP_DATA_DIR"
LAYERS = ("raw", "interim", "processed", "reports")


def data_dir() -> Path:
    """The data root: ``$RFGP_DATA_DIR``, or ``./data`` in the repository."""
    value = os.environ.get(DATA_DIR_ENV)
    root = Path(value).expanduser() if value else provenance.REPO_ROOT / "data"
    return root.resolve()


def layer(name: str, *parts: str, root: str | Path | None = None) -> Path:
    """Path inside one layer, e.g. ``layer("raw", "yahoo", "AAPL.parquet")``."""
    if name not in LAYERS:
        raise ValueError(f"unknown data layer {name!r}; expected one of {LAYERS}")
    base = Path(root) if root is not None else data_dir()
    return base.joinpath(name, *parts)


def sidecar(path: str | Path) -> Path:
    """Provenance file that accompanies a data file."""
    path = Path(path)
    return path.with_name(path.name + ".provenance.json")


def write_bytes(path: str | Path, content: bytes, **details: Any) -> Path:
    """Write bytes atomically and record their provenance."""
    path = Path(path)
    _atomic(path, lambda tmp: tmp.write_bytes(content))
    _record(path, details)
    return path


def write_parquet(path: str | Path, frame: pd.DataFrame, **details: Any) -> Path:
    """Write a table atomically as Parquet and record its provenance."""
    path = Path(path)
    _atomic(path, lambda tmp: frame.to_parquet(tmp))
    _record(path, {"rows": len(frame), "columns": [str(c) for c in frame.columns], **details})
    return path


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _record(path: Path, details: dict[str, Any]) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = provenance.collect(
        file=path.name, sha256=digest, size_bytes=path.stat().st_size, **details
    )
    provenance.write(sidecar(path), record)
