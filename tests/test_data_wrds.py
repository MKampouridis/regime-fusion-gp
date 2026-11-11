"""The WRDS download tool, offline, with a fake connection backed by in-memory sqlite."""

import json
import logging
import sqlite3
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

from regime_fusion.data import wrds

SECRET_PRICE = 123456.789
SECRET_NAME = "Zyxwvut Quux Holdings"
SECRET_USER = "dr_secretuser"
SECRET_PASSWORD = "p4ssw0rd-hunter2"

SPEC = {
    "daily": {
        "table": "crsp.dsf", "candidates": [{"table": "crsp.dsf_v2", "date_column": "dlycaldt"}],
        "columns": ["permno", "date", "prc"], "date_column": "date", "chunk_by": "date",
        "start": 2000, "download": True,
    },
    "names": {
        "table": "crsp.msenames", "candidates": ["crsp.nothere"], "columns": ["permno", "comnam"],
        "download": True,
    },
    "annual": {
        "table": "comp.funda", "columns": ["gvkey", "datadate", "at"], "date_column": "datadate",
        "chunk_by": "datadate", "start": 2000, "where": "indfmt = 'INDL'", "download": True,
    },
    "off": {"table": "crsp.dsi", "columns": ["date"], "download": False},
}


class FakeConnection:
    """sqlite with attached databases named like the WRDS schemas."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        for schema in ("crsp", "comp"):
            self.db.execute(f"ATTACH ':memory:' AS {schema}")
        daily = pd.DataFrame({
            "permno": [1, 1, 2, 2, 2],
            "date": ["2000-03-01", "2000-12-29", "2001-01-02", "2001-06-01", "2002-05-05"],
            "prc": [SECRET_PRICE, 2.5, 3.5, 4.5, 5.5],
        })
        self._put(daily, "crsp.dsf")
        self._put(pd.DataFrame({"permno": [1, 2], "comnam": [SECRET_NAME, "Other Co"]}), "crsp.msenames")
        self._put(pd.DataFrame({
            "gvkey": ["000001", "000002"], "datadate": ["2000-12-31", "2001-12-31"],
            "at": [SECRET_PRICE, 7.0], "indfmt": ["INDL", "FS"],
        }), "comp.funda")
        self.queries = []
        self.logins = 0

    def _put(self, frame, name):
        cols = ", ".join(frame.columns)
        self.db.execute(f"CREATE TABLE {name} ({cols})")
        marks = ", ".join("?" * len(frame.columns))
        self.db.executemany(f"INSERT INTO {name} VALUES ({marks})", frame.astype(object).values.tolist())

    def table_columns(self, schema, table):
        rows = self.db.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
        return [(r[1], r[2] or "text") for r in rows]

    def raw_sql(self, sql, chunksize=None):
        self.queries.append(sql)
        return pd.read_sql_query(sql, self.db, chunksize=chunksize)


@pytest.fixture
def root(tmp_path):
    return tmp_path


def _manifest(root):
    return json.loads((root / "wrds" / "manifest.json").read_text())


def test_describe_writes_schema_without_data(root, capsys):
    out = wrds.describe(FakeConnection(), SPEC, root)
    schema = json.loads((root / "wrds" / "schema.json").read_text())
    assert schema == json.loads(json.dumps(out))
    daily = schema["tables"]["daily"]
    assert daily["exists"] and daily["rows"] == 5
    assert [c["name"] for c in daily["columns"]] == ["permno", "date", "prc"]
    assert (daily["date_min"], daily["date_max"]) == ("2000-03-01", "2002-05-05")
    assert daily["candidates"][0]["exists"] is False          # crsp.dsf_v2 is missing
    assert schema["tables"]["names"]["candidates"][0] == {"table": "crsp.nothere", "exists": False}
    assert "date_min" not in schema["tables"]["names"]
    assert "5 rows" in capsys.readouterr().out
    text = (root / "wrds" / "schema.json").read_text()
    assert str(SECRET_PRICE) not in text and SECRET_NAME not in text


def test_download_writes_one_file_per_year_and_manifest(root):
    db = FakeConnection()
    wrds.download(db, SPEC, root, tables=["daily", "names"], end=2002)
    folder = root / "wrds" / "daily"
    assert sorted(p.name for p in folder.iterdir()) == [f"daily_{y}.parquet" for y in (2000, 2001, 2002)]
    y2000 = pd.read_parquet(folder / "daily_2000.parquet")
    assert list(y2000["permno"]) == [1, 1] and list(y2000["prc"]) == [SECRET_PRICE, 2.5]
    assert len(pd.read_parquet(folder / "daily_2001.parquet")) == 2
    assert len(pd.read_parquet(root / "wrds" / "names" / "names.parquet")) == 2
    files = _manifest(root)["files"]
    entry = files["daily/daily_2000.parquet"]
    assert entry["rows"] == 2 and entry["table"] == "crsp.dsf"
    assert entry["date_min"] == "2000-03-01" and entry["date_max"] == "2000-12-29"
    assert entry["sql"] == (
        "SELECT permno, date, prc FROM crsp.dsf WHERE date >= '2000-01-01' AND date < '2001-01-01'")
    assert set(entry) >= {"downloaded_at", "git_commit"}
    assert not list((root / "wrds").rglob("*.part"))


def test_where_clause_is_applied(root):
    wrds.download(FakeConnection(), SPEC, root, tables=["annual"], end=2002)
    assert len(pd.read_parquet(root / "wrds" / "annual" / "annual_2000.parquet")) == 1
    assert len(pd.read_parquet(root / "wrds" / "annual" / "annual_2001.parquet")) == 0  # FS excluded
    assert "(indfmt = 'INDL')" in _manifest(root)["files"]["annual/annual_2000.parquet"]["sql"]


def test_default_tables_skip_those_not_marked(root):
    wrds.download(FakeConnection(), {k: v for k, v in SPEC.items() if k != "annual"}, root, end=2000)
    assert not (root / "wrds" / "off").exists()


def test_second_call_skips_done_chunks_and_redoes_missing(root):
    db = FakeConnection()
    wrds.download(db, SPEC, root, tables=["daily"], end=2002)
    n = len(db.queries)
    wrds.download(db, SPEC, root, tables=["daily"], end=2002)
    selects = [q for q in db.queries[n:] if q.startswith("SELECT permno")]
    assert selects == []
    (root / "wrds" / "daily" / "daily_2001.parquet").unlink()
    wrds.download(db, SPEC, root, tables=["daily"], end=2002)
    selects = [q for q in db.queries[n:] if q.startswith("SELECT permno")]
    assert len(selects) == 1 and "2001-01-01" in selects[0]
    assert len(pd.read_parquet(root / "wrds" / "daily" / "daily_2001.parquet")) == 2


def test_chunk_with_wrong_row_count_is_redone(root):
    db = FakeConnection()
    wrds.download(db, SPEC, root, tables=["daily"], end=2000)
    manifest = _manifest(root)
    manifest["files"]["daily/daily_2000.parquet"]["rows"] = 99
    (root / "wrds" / "manifest.json").write_text(json.dumps(manifest))
    n = len(db.queries)
    wrds.download(db, SPEC, root, tables=["daily"], end=2000)
    assert len(db.queries) > n


def test_missing_column_stops_before_any_download(root):
    spec = {"names": SPEC["names"], "daily": {**SPEC["daily"], "columns": ["permno", "date", "prc", "nosuch"]}}
    db = FakeConnection()
    with pytest.raises(SystemExit, match="nosuch"):
        wrds.download(db, spec, root, end=2002)
    assert not (root / "wrds").exists()                       # not even the other table
    assert not any(q.startswith("SELECT permno") for q in db.queries)


def test_missing_table_stops(root):
    with pytest.raises(SystemExit, match="not found"):
        wrds.download(FakeConnection(), {"x": {"table": "crsp.nothere", "columns": ["a"]}}, root, tables=["x"])


def test_streaming_in_small_chunks_equals_one_read(root, tmp_path):
    a, b = root / "a", tmp_path / "b"
    wrds.download(FakeConnection(), SPEC, a, tables=["daily"], end=2002, chunk_rows=1)
    wrds.download(FakeConnection(), SPEC, b, tables=["daily"], end=2002, chunk_rows=1000)
    for year in (2000, 2001, 2002):
        name = f"daily/daily_{year}.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(a / "wrds" / name), pd.read_parquet(b / "wrds" / name))
    assert pq.ParquetFile(a / "wrds" / "daily" / "daily_2000.parquet").metadata.num_rows == 2


def test_nothing_printed_logged_or_written_contains_data(root, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    db = FakeConnection()
    wrds.describe(db, SPEC, root)
    wrds.download(db, SPEC, root, end=2002)
    out = capsys.readouterr()
    reports = [out.out, out.err, caplog.text,
               (root / "wrds" / "schema.json").read_text(), (root / "wrds" / "manifest.json").read_text()]
    for text in reports:
        for secret in (str(SECRET_PRICE), "123456", SECRET_NAME, "Zyxwvut", "Other Co", "000002"):
            assert secret not in text
    assert "rows" in out.out                                  # counts are printed


def test_no_credentials_are_stored(root):
    db = FakeConnection()
    wrds.describe(db, SPEC, root)
    wrds.download(db, SPEC, root, end=2000)
    blob = "".join(p.read_text() for p in (root / "wrds").glob("*.json"))
    for word in (SECRET_USER, SECRET_PASSWORD, "password", "username"):
        assert word not in blob.lower()
    parser = wrds.build_parser()      # no option takes a password
    with pytest.raises(SystemExit):
        parser.parse_args(["download", "--password", "x"])


def test_real_specification_loads_and_is_well_formed():
    spec = wrds.load_spec()
    assert spec["crsp_dsf"]["table"] == "crsp.dsf" and spec["crsp_dsf"]["start"] == 1970
    assert "indfmt = 'INDL'" in spec["comp_funda"]["where"]
    assert spec["ff_factors_daily"]["columns"] == "*"


def test_main_uses_the_injected_connection(root, monkeypatch, capsys):
    monkeypatch.setenv("RFGP_DATA_DIR", str(root))
    monkeypatch.setattr(wrds, "load_spec", lambda: SPEC)
    assert wrds.main(["describe"], connection=FakeConnection()) == 0
    assert (root / "wrds" / "schema.json").exists()


def test_help_and_import_do_not_load_wrds_package():
    code = (
        "import sys; sys.modules['wrds'] = None\n"        # any 'import wrds' would now fail
        "import regime_fusion.data.wrds as m\n"
        "try:\n m.main(['--help'])\nexcept SystemExit as e:\n assert e.code == 0\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "describe" in done.stdout and "download" in done.stdout


def test_describe_records_a_table_without_permission_and_goes_on(root, capsys):
    class Denied(FakeConnection):
        def raw_sql(self, sql, chunksize=None):
            if "FROM crsp.msenames" in sql:
                raise RuntimeError("(psycopg2.errors.InsufficientPrivilege) permission denied for schema crsp_x")
            return super().raw_sql(sql, chunksize=chunksize)

    out = wrds.describe(Denied(), SPEC, root)
    names = out["tables"]["names"]
    assert names["exists"] and names["access"] is False and "rows" not in names
    assert out["tables"]["daily"]["rows"] == 5                     # the others are still described
    assert "no permission" in capsys.readouterr().out

    class Broken(FakeConnection):
        def raw_sql(self, sql, chunksize=None):
            raise RuntimeError("something else")

    with pytest.raises(RuntimeError):
        wrds.describe(Broken(), SPEC, root)
