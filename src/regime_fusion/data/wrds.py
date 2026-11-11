"""Download CRSP, Compustat and Fama-French tables from WRDS to this machine.

WRDS forbids loading its data into AI tools, so this tool never prints, logs or reports a row or
a value of the data. It prints only table and column names, column types, row counts, the first and
last date of a date column, file names and sizes, and elapsed time. The one file meant to be read
by the research assistant is ``<data dir>/wrds/schema.json`` (from ``describe``).

  python -m regime_fusion.data.wrds describe
  python -m regime_fusion.data.wrds download [--tables a b ...] [--start YEAR] [--end YEAR]

The tables are listed in ``configs/wrds_tables.yaml``. The connection is the ``wrds`` package's own,
which asks for the username and password in the terminal; this module never reads, stores or
accepts credentials. ``wrds`` is imported only when a connection is opened.

Output: ``<data dir>/wrds/schema.json``, ``<data dir>/wrds/manifest.json`` and
``<data dir>/wrds/<name>/<name>[_<year>].parquet``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Protocol

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from regime_fusion import provenance
from regime_fusion.data import store

log = logging.getLogger(__name__)

SPEC_FILE = provenance.REPO_ROOT / "configs" / "wrds_tables.yaml"
CHUNK_ROWS = 500_000
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class WrdsConnection(Protocol):
    """What this module needs from a connection (tests supply a fake)."""

    def table_columns(self, schema: str, table: str) -> list[tuple[str, str]]:
        """(column, type) pairs of a table; an empty list if it does not exist."""

    def raw_sql(self, sql: str, chunksize: int | None = None) -> pd.DataFrame | Iterator[pd.DataFrame]:
        """Run a query; with ``chunksize``, an iterator of frames."""


class _WrdsPackageConnection:
    """The ``wrds`` package behind the small interface. Credentials are the package's business."""

    def __init__(self) -> None:
        import wrds  # lazily: only here, never at import of this module

        self._db = wrds.Connection()

    def table_columns(self, schema: str, table: str) -> list[tuple[str, str]]:
        frame = self._db.raw_sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema = '{schema}' AND table_name = '{table}' ORDER BY ordinal_position"
        )
        return [(str(a), str(b)) for a, b in zip(frame["column_name"], frame["data_type"])]

    def raw_sql(self, sql: str, chunksize: int | None = None):
        return self._db.raw_sql(sql, chunksize=chunksize)

    def close(self) -> None:
        self._db.close()


# --- the specification ---------------------------------------------------------------------------

def load_spec(path: str | Path = SPEC_FILE) -> dict[str, dict[str, Any]]:
    spec = yaml.safe_load(Path(path).read_text())["tables"]
    for name, entry in spec.items():
        for part in [name, *entry["table"].split("."), *(c for c in _cols(entry))]:
            if not _IDENT.match(part):
                raise ValueError(f"bad identifier {part!r} in the WRDS specification ({name})")
        if len(entry["table"].split(".")) != 2:
            raise ValueError(f"{name}: table must be schema.table")
        if entry.get("chunk_by") and entry["columns"] != "*" and entry["chunk_by"] not in entry["columns"]:
            raise ValueError(f"{name}: chunk_by column must be among the columns")
    return spec


def _cols(entry: dict[str, Any]) -> list[str]:
    return [] if entry["columns"] == "*" else list(entry["columns"])


def _candidates(entry: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for cand in entry.get("candidates", []):
        out.append({"table": cand} if isinstance(cand, str) else dict(cand))
    return out


def select_sql(entry: dict[str, Any], year: int | None = None) -> str:
    cols = "*" if entry["columns"] == "*" else ", ".join(entry["columns"])
    conditions = []
    if year is not None:
        col = entry["chunk_by"]
        conditions.append(f"{col} >= '{year}-01-01' AND {col} < '{year + 1}-01-01'")
    if entry.get("where"):
        conditions.append(f"({entry['where']})")
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return f"SELECT {cols} FROM {entry['table']}{where}"


# --- describe -----------------------------------------------------------------------------------

def _describe_table(db: WrdsConnection, table: str, date_column: str | None) -> dict[str, Any]:
    schema, name = table.split(".")
    columns = db.table_columns(schema, name)
    if not columns:
        return {"table": table, "exists": False}
    info: dict[str, Any] = {
        "table": table, "exists": True, "columns": [{"name": c, "type": t} for c, t in columns],
    }
    info["rows"] = int(db.raw_sql(f"SELECT COUNT(*) AS n FROM {table}").iloc[0, 0])
    if date_column and date_column in {c for c, _ in columns}:
        lim = db.raw_sql(f"SELECT MIN({date_column}) AS lo, MAX({date_column}) AS hi FROM {table}")
        info["date_column"] = date_column
        info["date_min"] = _date_str(lim.iloc[0, 0])
        info["date_max"] = _date_str(lim.iloc[0, 1])
    return info


def _date_str(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(pd.Timestamp(value).date())


def describe(db: WrdsConnection, spec: dict[str, dict[str, Any]], root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"tables": {}}
    for name, entry in spec.items():
        started = time.time()
        info = _describe_table(db, entry["table"], entry.get("date_column"))
        info["spec_name"] = name
        info["candidates"] = [
            _describe_table(db, c["table"], c.get("date_column")) for c in _candidates(entry)
        ]
        result["tables"][name] = info
        for item in (info, *info["candidates"]):
            _say(_describe_line(item) + f"  [{time.time() - started:.1f}s]")
    path = root / "wrds" / "schema.json"
    _write_json(path, result)
    _say(f"wrote {path} ({path.stat().st_size} bytes)")
    return result


def _describe_line(item: dict[str, Any]) -> str:
    if not item["exists"]:
        return f"{item['table']}: not found"
    line = f"{item['table']}: {len(item['columns'])} columns, {item['rows']} rows"
    if "date_min" in item:
        line += f", {item['date_column']} {item['date_min']} to {item['date_max']}"
    return line


# --- download -----------------------------------------------------------------------------------

def download(
    db: WrdsConnection, spec: dict[str, dict[str, Any]], root: Path,
    tables: list[str] | None = None, start: int | None = None, end: int | None = None,
    chunk_rows: int = CHUNK_ROWS,
) -> dict[str, Any]:
    chosen = tables if tables else [n for n, e in spec.items() if e.get("download")]
    unknown = [n for n in chosen if n not in spec]
    if unknown:
        raise SystemExit(f"not in the specification: {', '.join(unknown)}")
    manifest_path = root / "wrds" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    commit = provenance.git_state().get("commit")

    # Check every chosen table first, so a missing column stops the run before any download.
    for name in chosen:
        _check_columns(db, name, spec[name])

    for name in chosen:
        entry = spec[name]
        if entry.get("chunk_by"):
            first = max(int(entry.get("start", 0)), start or 0)
            last = min(end or dt.date.today().year, dt.date.today().year)
            jobs = [(y, root / "wrds" / name / f"{name}_{y}.parquet") for y in range(first, last + 1)]
        else:
            jobs = [(None, root / "wrds" / name / f"{name}.parquet")]
        for year, path in jobs:
            key = str(path.relative_to(root / "wrds"))
            sql = select_sql(entry, year)
            done = manifest["files"].get(key)
            if done and path.exists() and done["sql"] == sql and _file_rows(path) == done["rows"]:
                _say(f"{key}: already done, {done['rows']} rows")
                continue
            started = time.time()
            rows, lo, hi = _stream_to_parquet(db, sql, path, entry.get("chunk_by") or entry.get("date_column"), chunk_rows)
            manifest["files"][key] = {
                "table": entry["table"], "sql": sql, "rows": rows, "date_min": lo, "date_max": hi,
                "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "git_commit": commit,
            }
            _write_json(manifest_path, manifest)
            _say(f"{key}: {rows} rows, {path.stat().st_size} bytes, {time.time() - started:.1f}s")
    return manifest


def _check_columns(db: WrdsConnection, name: str, entry: dict[str, Any]) -> None:
    schema, table = entry["table"].split(".")
    live = db.table_columns(schema, table)
    if not live:
        raise SystemExit(f"{name}: table {entry['table']} not found on WRDS; run describe and correct the specification")
    missing = [c for c in _cols(entry) if c not in {n for n, _ in live}]
    if missing:
        raise SystemExit(
            f"{name}: {entry['table']} has no column(s) {', '.join(missing)}; "
            "nothing was downloaded. Run describe and correct configs/wrds_tables.yaml"
        )


def _file_rows(path: Path) -> int | None:
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def _stream_to_parquet(
    db: WrdsConnection, sql: str, path: Path, date_column: str | None, chunk_rows: int,
) -> tuple[int, str | None, str | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = provenance.temporary(path)
    rows, lo, hi = 0, None, None
    writer = None
    try:
        result = db.raw_sql(sql, chunksize=chunk_rows)
        frames = [result] if isinstance(result, pd.DataFrame) else result
        schema = None
        for frame in frames:
            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema)
            writer.write_table(table)
            rows += len(frame)
            if date_column and date_column in frame.columns and len(frame):
                col = pd.to_datetime(frame[date_column])
                a, b = _date_str(col.min()), _date_str(col.max())
                lo = a if lo is None or (a is not None and a < lo) else lo
                hi = b if hi is None or (b is not None and b > hi) else hi
        if writer is None:  # no rows at all: an empty file with the selected columns
            pq.write_table(pa.Table.from_pandas(pd.DataFrame(columns=_names(sql)), preserve_index=False), tmp)
        else:
            writer.close()
            writer = None
        tmp.replace(path)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
    return rows, lo, hi


def _names(sql: str) -> list[str]:
    head = sql.split(" FROM ")[0].removeprefix("SELECT ")
    return [] if head.strip() == "*" else [c.strip() for c in head.split(",")]


# --- helpers and command line ---------------------------------------------------------------------

def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance.write(path, record)


def _say(message: str) -> None:
    log.info(message)
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m regime_fusion.data.wrds",
        description="Describe and download WRDS tables. Prints only names, types, counts, date limits "
        "and file sizes, never data. The wrds package asks for your login in the terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("describe", help="write <data dir>/wrds/schema.json (columns, row counts, date limits)")
    d = sub.add_parser("download", help="download the tables to parquet, one file per year where chunked")
    d.add_argument("--tables", nargs="+", metavar="NAME", help="names from configs/wrds_tables.yaml (default: all marked download)")
    d.add_argument("--start", type=int, metavar="YEAR", help="first year of chunked tables")
    d.add_argument("--end", type=int, metavar="YEAR", help="last year of chunked tables")
    return parser


def main(argv: list[str] | None = None, connection: WrdsConnection | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec()
    root = store.data_dir()
    db = connection if connection is not None else _WrdsPackageConnection()
    started = time.time()
    try:
        if args.command == "describe":
            describe(db, spec, root)
        else:
            download(db, spec, root, args.tables, args.start, args.end)
    finally:
        if connection is None:
            db.close()
    _say(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
