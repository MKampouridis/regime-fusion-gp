"""SEC EDGAR: XBRL company facts and filer details, fetched within the fair-access rules.

The companyfacts API returns every XBRL fact a company has filed. For each
taxonomy (``us-gaap``, ``dei``), concept (``NetIncomeLoss``) and unit (``USD``) it
lists facts with the period (``start``, ``end``; instants have no start), the
value, the accession number, the form and the filing date. A period appears once
for every filing that reports it, so a restatement is simply a later fact for an
earlier period: this is what makes a point-in-time history possible. The
``fy`` and ``fp`` fields describe the filing, not the period, and are not used.

Only annual and quarterly reports (10-K, 10-Q, their amendments and transition
reports) are kept. Facts from 8-Ks (earnings releases) are dropped because few
firms file them in XBRL, and never consistently over 2014-2025.

Requests carry a User-Agent naming a contact (``configs/data_sources.yaml``) and
are paced below the SEC's limit of ten per second.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A")
FACT_COLUMNS = ("cik", "taxonomy", "concept", "unit", "start", "end", "value",
                "accn", "form", "filed", "frame")


class NotFound(LookupError):
    """EDGAR has no such document (HTTP 404)."""


class Client:
    """Fetches EDGAR documents, pacing requests and backing off when throttled."""

    def __init__(self, user_agent: str, max_per_second: float, *, session=None,
                 max_attempts: int = 5, clock=time.monotonic, sleep=time.sleep):
        if session is None:
            import requests

            session = requests.Session()
        session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.session = session
        self.min_interval = 1.0 / max_per_second
        self.max_attempts = max_attempts
        self._clock, self._sleep = clock, sleep
        self._last = -float("inf")

    def get(self, url: str) -> bytes:
        for attempt in range(1, self.max_attempts + 1):
            wait = self._last + self.min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()
            response = self.session.get(url, timeout=60)
            if response.status_code == 404:
                raise NotFound(url)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_attempts:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After", "")
                pause = float(retry_after) if retry_after.isdigit() else 2.0 ** attempt
                log.warning("%s: HTTP %d; retrying in %.0f s", url, response.status_code, pause)
                self._sleep(pause)
                continue
            response.raise_for_status()
            return response.content
        raise AssertionError("unreachable")


def compress(content: bytes) -> bytes:
    """Raw EDGAR JSON is stored gzip-compressed (mtime fixed, so identical input gives identical files)."""
    return gzip.compress(content, mtime=0)


def load_json(path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return json.load(handle)


def facts_table(companyfacts: dict[str, Any]) -> pd.DataFrame:
    """Every fact in a companyfacts document, one row each."""
    cik = int(companyfacts["cik"])
    records = [
        (cik, taxonomy, concept, unit, fact.get("start"), fact["end"], fact["val"],
         fact.get("accn"), fact.get("form"), fact.get("filed"), fact.get("frame"))
        for taxonomy, concepts in companyfacts.get("facts", {}).items()
        for concept, body in concepts.items()
        for unit, facts in body.get("units", {}).items()
        for fact in facts
    ]
    frame = pd.DataFrame.from_records(records, columns=list(FACT_COLUMNS))
    for column in ("start", "end", "filed"):
        frame[column] = pd.to_datetime(frame[column])
    frame["value"] = frame["value"].astype(float)
    return frame


def report_facts(companyfacts: dict[str, Any]) -> pd.DataFrame:
    """Facts from annual and quarterly reports only (see the module notes)."""
    frame = facts_table(companyfacts)
    return frame[frame["form"].isin(FORMS)].reset_index(drop=True)


def filer_names(submissions: dict[str, Any]) -> list[str]:
    """The filer's current name followed by its former names."""
    former = [entry["name"] for entry in submissions.get("formerNames", [])]
    return [submissions["name"], *former]


_GENERIC_WORDS = {
    "inc", "corp", "corporation", "co", "company", "companies", "the", "plc", "ltd",
    "group", "holdings", "holding", "incorporated", "llc", "class", "and", "of", "new",
}


def names_match(name: str, filer_names: list[str]) -> bool:
    """Whether a stock's name shares a distinctive word with any SEC filer name.

    Words are also compared with spaces removed, so "Du Pont" matches "DUPONT".
    """
    def words(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower().replace("&", " and "))
        return [token for token in tokens if token not in _GENERIC_WORDS and len(token) > 1]

    ours = words(name)
    for other in filer_names:
        theirs = words(other)
        if set(ours) & set(theirs) or (ours and "".join(ours) in "".join(theirs)):
            return True
    return False


@dataclass
class FilingCheck:
    """What EDGAR holds for one stock, with flags for anything needing a look."""

    filer_names: list[str]
    reports_per_year: dict[int, int]   # 10-K and 10-Q filings (amendments excluded)
    first_filed: str | None
    last_filed: str | None
    flags: list[str] = field(default_factory=list)


def check_filings(stock_name: str, facts: pd.DataFrame, filer_names: list[str],
                  years: range) -> FilingCheck:
    originals = facts[facts["form"].isin(["10-K", "10-Q", "10-KT", "10-QT"])]
    filings = originals.drop_duplicates("accn")
    per_year = filings["filed"].dt.year.value_counts()
    result = FilingCheck(
        filer_names=filer_names,
        reports_per_year={year: int(per_year.get(year, 0)) for year in years},
        first_filed=None if filings.empty else filings["filed"].min().date().isoformat(),
        last_filed=None if filings.empty else filings["filed"].max().date().isoformat(),
    )
    if not names_match(stock_name, filer_names):
        result.flags.append(f"name {stock_name!r} does not match the SEC filer names {filer_names}")
    thin = [year for year, count in result.reports_per_year.items() if count < 4]
    if thin:
        result.flags.append(f"fewer than four 10-K/10-Q filings in {thin}")
    return result
