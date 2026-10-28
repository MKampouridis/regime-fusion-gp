import hashlib
import json

import pandas as pd
import pytest

from regime_fusion.data.store import (
    DATA_DIR_ENV, data_dir, layer, sidecar, write_bytes, write_parquet,
)
from regime_fusion.provenance import REPO_ROOT


def test_data_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert data_dir() == tmp_path.resolve()
    assert layer("raw", "yahoo", "AAPL.parquet") == tmp_path.resolve() / "raw" / "yahoo" / "AAPL.parquet"
    monkeypatch.delenv(DATA_DIR_ENV)
    assert data_dir() == (REPO_ROOT / "data").resolve()


def test_unknown_layer_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="layer"):
        layer("bronze", root=tmp_path)


def test_tables_are_written_with_provenance(tmp_path):
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2015-02-02", "2015-02-03"]))
    path = write_parquet(tmp_path / "interim" / "x.parquet", frame, source="unit test")
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    record = json.loads(sidecar(path).read_text())
    assert record["rows"] == 2 and record["source"] == "unit test"
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "git" in record and "packages" in record


def test_raw_bytes_are_kept_exactly(tmp_path):
    content = b"DATE,CLOSE\n01/02/2015,17.79\n"
    path = write_bytes(tmp_path / "raw" / "vix.csv", content, url="https://example.org")
    assert path.read_bytes() == content
    assert json.loads(sidecar(path).read_text())["url"] == "https://example.org"


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    unwritable = pd.DataFrame({"mixed": [1, "x"]})  # Parquet cannot store mixed types
    with pytest.raises(Exception):
        write_parquet(tmp_path / "bad.parquet", unwritable)
    assert list(tmp_path.iterdir()) == []
