import gzip
import json

import pandas as pd
import pytest

from regime_fusion.data import sec

COMPANYFACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [
                    {"end": "2015-01-09", "val": 5824748000, "accn": "a1", "fy": 2015, "fp": "Q1",
                     "form": "10-Q", "filed": "2015-01-28"},
                ]},
            },
        },
        "us-gaap": {
            "NetIncomeLoss": {
                "units": {"USD": [
                    {"start": "2014-09-28", "end": "2014-12-27", "val": 18024000000, "accn": "a1",
                     "fy": 2015, "fp": "Q1", "form": "10-Q", "filed": "2015-01-28", "frame": "CY2014Q4"},
                    {"start": "2014-09-28", "end": "2014-12-27", "val": 18024000000, "accn": "e1",
                     "fy": 2015, "fp": "Q1", "form": "8-K", "filed": "2015-01-27"},
                ]},
            },
        },
    },
}


def test_facts_are_flattened_with_dates_and_values():
    facts = sec.facts_table(COMPANYFACTS)
    assert len(facts) == 3 and list(facts.columns) == list(sec.FACT_COLUMNS)
    shares = facts[facts["concept"] == "EntityCommonStockSharesOutstanding"].iloc[0]
    assert pd.isna(shares["start"]) and shares["end"] == pd.Timestamp("2015-01-09")
    assert shares["unit"] == "shares" and shares["value"] == 5824748000.0
    income = facts[(facts["concept"] == "NetIncomeLoss") & (facts["form"] == "10-Q")].iloc[0]
    assert income["filed"] == pd.Timestamp("2015-01-28") and income["frame"] == "CY2014Q4"


def test_only_annual_and_quarterly_reports_are_kept():
    assert sec.report_facts(COMPANYFACTS)["form"].tolist() == ["10-Q", "10-Q"]


def test_filer_names_include_former_names():
    submissions = {"name": "Meta Platforms, Inc.", "formerNames": [{"name": "Facebook Inc"}]}
    assert sec.filer_names(submissions) == ["Meta Platforms, Inc.", "Facebook Inc"]


def test_compression_is_deterministic_and_lossless(tmp_path):
    content = json.dumps(COMPANYFACTS).encode()
    assert sec.compress(content) == sec.compress(content)
    path = tmp_path / "facts.json.gz"
    path.write_bytes(sec.compress(content))
    assert sec.load_json(path) == COMPANYFACTS
    assert gzip.decompress(path.read_bytes()) == content


class FakeResponse:
    def __init__(self, status, content=b"{}", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses, self.headers, self.urls = list(responses), {}, []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.responses.pop(0)


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(responses, per_second=5.0):
    clock = FakeClock()
    session = FakeSession(responses)
    client = sec.Client("Test Person test@example.org", per_second, session=session,
                        clock=clock.time, sleep=clock.sleep)
    return client, session, clock


def test_requests_identify_the_contact_and_are_paced():
    client, session, clock = make_client([FakeResponse(200, b"a"), FakeResponse(200, b"b")])
    assert client.get("u1") == b"a" and client.get("u2") == b"b"
    assert session.headers["User-Agent"] == "Test Person test@example.org"
    assert clock.sleeps == [pytest.approx(0.2)]  # five requests per second at most


def test_throttling_is_retried_after_the_requested_pause():
    responses = [FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(503), FakeResponse(200, b"ok")]
    client, session, clock = make_client(responses)
    assert client.get("u") == b"ok"
    assert len(session.urls) == 3
    assert 3.0 in clock.sleeps and 4.0 in clock.sleeps  # Retry-After, then 2 ** attempt


def test_missing_documents_and_persistent_errors_raise():
    client, _, _ = make_client([FakeResponse(404)])
    with pytest.raises(sec.NotFound):
        client.get("u")
    client, _, _ = make_client([FakeResponse(500)] * 5)
    with pytest.raises(RuntimeError, match="500"):
        client.get("u")


@pytest.mark.parametrize(
    "name, filers, expected",
    [
        ("Apple Inc.", ["Apple Inc."], True),
        ("Google Inc Class C", ["GOOGLE LLC", "Google Inc."], True),
        ("3M Company", ["3M CO"], True),
        ("JP Morgan Chase & Co", ["JPMORGAN CHASE & CO"], True),
        ("Facebook", ["Meta Platforms, Inc.", "Facebook Inc"], True),
        ("Du Pont (E.I.)", ["EIDP, Inc.", "DUPONT E I DE NEMOURS & CO"], True),
        ("Exxon Mobil Corp", ["APPLE INC"], False),
        ("The Coca-Cola Company", ["PEPSICO INC"], False),
    ],
)
def test_stock_names_are_matched_to_filer_names(name, filers, expected):
    assert sec.names_match(name, filers) is expected


def test_filing_check_counts_reports_and_flags_gaps():
    filed = ["2015-01-28", "2015-04-28", "2015-07-22", "2015-10-28", "2016-01-27", "2016-01-27"]
    forms = ["10-Q", "10-Q", "10-Q", "10-K", "10-Q", "10-Q/A"]
    facts = pd.DataFrame({"accn": [f"a{i}" for i in range(6)], "form": forms,
                          "filed": pd.to_datetime(filed)})
    result = sec.check_filings("Apple Inc.", facts, ["Apple Inc."], range(2015, 2017))
    assert result.reports_per_year == {2015: 4, 2016: 1}
    assert result.first_filed == "2015-01-28" and result.last_filed == "2016-01-27"
    assert result.flags == ["fewer than four 10-K/10-Q filings in [2016]"]
    wrong = sec.check_filings("Exxon Mobil Corp", facts, ["Apple Inc."], range(2015, 2016))
    assert "does not match" in wrong.flags[0]
