import hashlib
import io
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_fusion.data import cli, gdelt
from regime_fusion.data.store import DATA_DIR_ENV


def record(date, orgs, tone="2.5,4.0,1.5,5.5,20.0,1.0,200", gcam="wc:200,c6.4:4,c6.5:6,c6.6:2",
           title=None, url="https://example.org/a", source="example.org", all_names=""):
    """One GKG 2.1 record with the fields the extractor reads."""
    fields = [""] * gdelt.N_FIELDS
    fields[gdelt.ALL_NAMES] = all_names
    fields[gdelt.RECORD_ID], fields[gdelt.DATE] = f"{date}-1", str(date)
    fields[gdelt.SOURCE], fields[gdelt.URL] = source, url
    fields[gdelt.ORGANIZATIONS] = ";".join(name for name, _ in orgs)
    fields[gdelt.V2_ORGANIZATIONS] = ";".join(f"{name},{offset}" for name, offset in orgs)
    fields[gdelt.TONE], fields[gdelt.GCAM] = tone, gcam
    fields[gdelt.EXTRAS] = f"<PAGE_TITLE>{title}</PAGE_TITLE>" if title else ""
    return "\t".join(fields)


def zipped(lines):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("20240311143000.gkg.csv", "\n".join(lines))
    return buffer.getvalue()


@pytest.fixture
def matcher(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text(yaml.safe_dump({"stocks": {
        "AAPL": ["apple"], "F": ["ford"], "GE": ["general electric", "ge"], "KO": ["coca ?-?cola"]}}))
    return gdelt.load_matcher(path)


def test_anchors_match_whole_words(matcher):
    assert matcher.match("ford motor co;affordable housing trust") == ["F"]
    assert matcher.match("george washington university") == []
    assert matcher.match("ge healthcare;coca cola co") == ["GE", "KO"]
    assert matcher.match("Apple Inc") == ["AAPL"] and matcher.match("") == []
    with pytest.raises(ValueError, match="no anchors"):
        gdelt.load_matcher(stocks=["NOPE"])


def test_repository_anchors_cover_the_universe():
    universe = cli.load_universe()
    stocks = [s.id for s in universe.select("main", "holdout", include_holdout=True)]
    assert set(gdelt.load_matcher(stocks=stocks).stocks) == set(stocks)


def test_extraction_sums_by_hour_stock_and_variants(matcher):
    lines = [
        record(20240311143000, [("apple inc", 10), ("microsoft", 90)], title="Apple unveils a phone"),
        record(20240311144500, [("microsoft", 5), ("apple inc", 60)],
               tone="-1.5,1.0,2.5,3.5,10.0,0.0,100", gcam="wc:100,c6.4:5"),
        record(20240311144500, [("apple daily", 7)]),
        record(20240311150000, [("ibm", 3)]),               # no anchor: dropped
        "a malformed line",
    ]
    sums, titles = {}, []
    assert gdelt.extract(zipped(lines), matcher, sums, titles) == 3
    rows = {key: dict(zip(gdelt.SUM_COLUMNS, values)) for key, values in sums.items()}
    apple = rows[("2024031114", "AAPL", "apple inc")]
    assert apple["records"] == 2 and apple["first_mentions"] == 1 and apple["title_mentions"] == 1
    assert apple["tone_sum"] == pytest.approx(1.0) and apple["tone_sq_sum"] == pytest.approx(8.5)
    assert apple["lm_negative_sum"] == 9 and apple["gcam_words_sum"] == 300
    assert apple["lm_negative_share_sum"] == pytest.approx(4 / 200 + 5 / 100)
    assert rows[("2024031114", "AAPL", "apple daily")]["records"] == 1  # kept apart for later
    assert len(titles) == 1 and titles[0][1] == "AAPL" and titles[0][5] == "Apple unveils a phone"


def test_master_list_is_read_by_day():
    text = ("100 aaa http://data.gdeltproject.org/gdeltv2/20150218230000.gkg.csv.zip\n"
            "50 bbb http://data.gdeltproject.org/gdeltv2/20150218230000.export.CSV.zip\n"
            "200 ccc http://data.gdeltproject.org/gdeltv2/20150219000000.gkg.csv.zip\n"
            "300 ddd http://data.gdeltproject.org/gdeltv2/20150220000000.gkg.csv.zip\n")
    files = gdelt.master_list(text, "20150218", "20150219")
    assert files["day"].tolist() == ["20150218", "20150219"]
    assert files["url"].iloc[0].startswith("https://") and files["md5"].tolist() == ["aaa", "ccc"]


class FakeResponse:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)

    def get(self, url, timeout):
        return self.responses.pop(0)


def test_files_are_checked_against_their_md5():
    good = b"zip bytes"
    md5 = hashlib.md5(good).hexdigest()
    session = FakeSession([FakeResponse(200, b"corrupt"), FakeResponse(500), FakeResponse(200, good)])
    assert gdelt.fetch("u", md5, session, sleep=lambda s: None) == good
    with pytest.raises(gdelt.FetchError, match="404"):
        gdelt.fetch("u", md5, FakeSession([FakeResponse(404)]), sleep=lambda s: None)
    with pytest.raises(gdelt.FetchError, match="MD5"):
        gdelt.fetch("u", md5, FakeSession([FakeResponse(200, b"bad")] * 4), sleep=lambda s: None)


def test_a_day_is_processed_into_hourly_sums_and_titles(matcher):
    content = zipped([record(20240311143000, [("apple inc", 1)], title="Apple results")])
    md5 = hashlib.md5(content).hexdigest()
    session = FakeSession([FakeResponse(200, content), FakeResponse(404)])
    day, hourly, titled, summary = gdelt.process_day("20240311", [("u1", md5), ("u2", md5)], matcher, session)
    assert hourly["hour"].iloc[0] == pd.Timestamp("2024-03-11 14:00", tz="UTC")
    assert hourly["records"].sum() == 1 and len(titled) == 1
    assert summary["files"] == 2 and len(summary["failed"]) == 1


def test_gdelt_command_resumes_and_guards_the_disk(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    contents = {}
    lines = []
    for day in ("20150301", "20150302"):
        for hhmm in ("1400", "1415"):
            stamp = f"{day}{hhmm}00"
            content = zipped([record(int(stamp), [("apple inc", 1)], url=f"https://x/{stamp}")])
            url = f"https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
            contents[url] = content
            lines.append(f"{len(content)} {hashlib.md5(content).hexdigest()} {url}")
    monkeypatch.setattr(cli, "gdelt_master_text", lambda: "\n".join(lines))
    fetched = []

    def fake_fetch(url, md5, session, **_):
        fetched.append(url)
        return contents[url]

    monkeypatch.setattr(gdelt, "fetch", fake_fetch)
    monkeypatch.setattr(gdelt, "new_session", lambda: None)  # no network, no requests needed
    args = ["gdelt", "--start", "2015-03-01", "--end", "2015-03-02", "--workers", "0"]
    assert cli.main(args) == 0
    hourly = pd.read_parquet(cli.gdelt_path("20150302", "hourly"))
    assert hourly.loc[hourly["stock"] == "AAPL", "records"].sum() == 2
    assert len(fetched) == 4
    assert cli.main(args) == 0 and len(fetched) == 4  # nothing left to do

    for path in (tmp_path / "raw" / "gdelt" / "gkg").rglob("*.parquet"):
        path.unlink()
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: type("U", (), {"free": 1e9})())
    assert cli.main(args) == 1 and len(fetched) == 4  # stopped before fetching anything


def test_firms_named_only_in_the_all_names_field_are_found(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text(yaml.safe_dump({"stocks": {"GM": ["general motors", "gm"], "F": ["ford"]}}))
    matcher = gdelt.load_matcher(path)
    line = record(20150302143000, [("toyota", 400)], all_names="Tata Motors,1219;General Motors,90")
    sums, titles = {}, []
    assert gdelt.extract(zipped([line]), matcher, sums, titles) == 1
    rows = {key: dict(zip(gdelt.SUM_COLUMNS, values)) for key, values in sums.items()}
    gm = rows[("2015030214", "GM", "general motors")]
    assert gm["records"] == 1 and gm["first_mentions"] == 1  # offset 90 comes before toyota's 400


def test_stopping_ends_the_workers_at_once():
    pool = ProcessPoolExecutor(max_workers=2)
    for _ in range(3):
        pool.submit(time.sleep, 60)
    processes = list(pool._processes.values())
    started = time.monotonic()
    cli.stop_workers(pool)
    assert time.monotonic() - started < 20
    for process in processes:
        process.join(timeout=5)
        assert not process.is_alive()
