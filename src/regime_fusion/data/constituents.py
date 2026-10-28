"""Index constituents in January 2015, from pinned Wikipedia revisions.

The universe is fixed as of January 2015, so firms that were later acquired or
delisted stay in it; using today's constituents would bias results upwards. The
lists come from the Wikipedia pages for the S&P 100 and the S&P 500 as they stood
at the end of January 2015. The revisions are pinned, so the lists can be rebuilt
exactly. Wikipedia is a secondary source; checks against an official list are
welcome where one is available.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass

import pandas as pd

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class Revision:
    page: str
    revid: int
    timestamp: str  # when the revision was saved (UTC)


SNAPSHOT = {
    "sp100": Revision("S&P 100", 642684730, "2015-01-16T00:58:45Z"),
    "sp500": Revision("List of S&P 500 companies", 644366768, "2015-01-27T06:34:47Z"),
}
COLUMNS = ("ticker", "name", "sector", "sub_industry", "cik", "sp100")

_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_SYMBOL = re.compile(r"\{\{\s*\w*Symbol\s*\|\s*([^}|]+?)\s*\}\}", re.IGNORECASE)


def parse_sp100(wikitext: str) -> pd.DataFrame:
    """Ticker and name of each row in the S&P 100 page's components table."""
    rows = [_cells(row) for row in _table_rows(wikitext, "== Components ==")]
    frame = pd.DataFrame([(_ticker(c[0]), _plain(c[1])) for c in rows], columns=["ticker", "name"])
    return _checked(frame, "S&P 100")


def parse_sp500(wikitext: str) -> pd.DataFrame:
    """Ticker, name, GICS sector and sub-industry, and CIK of each S&P 500 member."""
    records = []
    for cells in (_cells(row) for row in _table_rows(wikitext, "{|")):
        if len(cells) < 8:
            raise ValueError(f"unexpected S&P 500 row: {cells}")
        cik = re.sub(r"\D", "", cells[7])
        records.append(
            (_ticker(cells[0]), _plain(cells[1]), _plain(cells[3]), _plain(cells[4]),
             int(cik) if cik else None)
        )
    columns = ["ticker", "name", "sector", "sub_industry", "cik"]
    frame = pd.DataFrame(records, columns=columns).astype({"cik": "Int64"})
    return _checked(frame, "S&P 500")


def combine(sp100: pd.DataFrame, sp500: pd.DataFrame) -> pd.DataFrame:
    """S&P 500 members with a flag for S&P 100 membership."""
    missing = sorted(set(sp100["ticker"]) - set(sp500["ticker"]))
    if missing:
        raise ValueError(f"S&P 100 members missing from the S&P 500 list: {missing}")
    frame = sp500.assign(sp100=sp500["ticker"].isin(sp100["ticker"]))
    return frame.loc[:, list(COLUMNS)].sort_values("ticker").reset_index(drop=True)


def fetch(revision: Revision, user_agent: str) -> bytes:
    """Raw API response (JSON) holding one page revision."""
    import requests

    response = requests.get(
        WIKIPEDIA_API,
        params={
            "action": "query", "prop": "revisions", "revids": revision.revid,
            "rvprop": "ids|timestamp|content", "rvslots": "main",
            "format": "json", "formatversion": 2,
        },
        headers={"User-Agent": user_agent},
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def wikitext_of(payload: bytes, revision: Revision) -> str:
    """Page text from an API response, checked against the pinned revision."""
    page = json.loads(payload)["query"]["pages"][0]
    found = page["revisions"][0]
    if page["title"] != revision.page or found["revid"] != revision.revid:
        raise ValueError(f"expected {revision}, got {page['title']!r} revision {found['revid']}")
    return found["slots"]["main"]["content"]


def _table_rows(wikitext: str, marker: str) -> list[str]:
    """Data rows of the first table at or after ``marker``."""
    start = wikitext.index("{|", wikitext.index(marker))
    end = wikitext.index("\n|}", start)
    rows = wikitext[start:end].split("\n|-")[1:]
    return [row for row in rows if row.strip() and not row.strip().startswith("!")]


def _cells(row: str) -> list[str]:
    cells: list[str] = []
    for line in row.strip().splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells.extend(part.strip() for part in line[1:].split("||"))
        elif cells:  # a cell that continues on the next line
            cells[-1] += " " + line
    return cells


def _ticker(cell: str) -> str:
    match = _SYMBOL.search(cell)
    return (match.group(1) if match else _plain(cell)).strip().upper()


def _plain(cell: str) -> str:
    text = _LINK.sub(r"\1", cell)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _checked(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"no {label} members found")
    duplicated = frame["ticker"][frame["ticker"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{label}: duplicated tickers {duplicated}")
    if not frame["ticker"].str.fullmatch(r"[A-Z][A-Z.\-]{0,6}").all():
        raise ValueError(f"{label}: unexpected ticker format")
    return frame
