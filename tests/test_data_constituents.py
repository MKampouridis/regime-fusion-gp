import json

import pandas as pd
import pytest

from regime_fusion.data.constituents import (
    SNAPSHOT, Revision, combine, parse_sp100, parse_sp500, wikitext_of,
)

SP100_PAGE = """The '''S&P 100 Index''' is a [[stock market index]].

== Components ==
{| class="wikitable sortable"
|-
! Symbol
! Name
|-
| AAPL
| [[Apple Inc.]]
|-
| BRK.B
| [[Berkshire Hathaway]]
|-
| JNJ
| [[Johnson & Johnson|Johnson &amp; Johnson Inc]]
|}

==Investing==
{|
| 2013
| +27.40
|}
"""

SP500_PAGE = """{|  class="wikitable sortable"
|-
! [[Ticker symbol]] !! Security !! [[SEC filing]]s !! GICS Sector !! GICS Sub Industry !! Address !! Date first added !! [[Central Index Key|CIK]]
|-
|\t{{NasdaqSymbol|AAPL}}\t||\t[[Apple Inc.]]\t||\t[http://www.sec.gov reports]\t||\tInformation Technology ||\tComputer Hardware\t|| [[Cupertino, California]] || 1982-11-30 || 0000320193
|-
|\t{{NyseSymbol|BRK.B}}\t||\t[[Berkshire Hathaway]]\t||\t[http://www.sec.gov reports]\t||\tFinancials ||\tMulti-Sector Holdings\t|| [[Omaha, Nebraska]] || || 0001067983
|-
|\t{{NyseSymbol|JNJ}}\t||\t[[Johnson & Johnson|Johnson &amp; Johnson]]\t||\t[http://www.sec.gov reports]\t||\tHealth Care ||\tPharmaceuticals\t|| [[New Brunswick, New Jersey]] || 1973-06-30 || 0000200406
|-
|\t{{NyseSymbol|XRX}}\t||\t[[Xerox Corp.]]\t||\t[http://www.sec.gov reports]\t||\tInformation Technology ||\tTechnology Hardware\t|| [[Norwalk, Connecticut]] || || 0000108772
|}

{| class="wikitable sortable"
|-
| Date || Ticker || Security
|}
"""


def test_sp100_rows_are_read_with_plain_names():
    frame = parse_sp100(SP100_PAGE)
    assert frame["ticker"].tolist() == ["AAPL", "BRK.B", "JNJ"]
    assert frame["name"].tolist() == ["Apple Inc.", "Berkshire Hathaway", "Johnson & Johnson Inc"]


def test_sp500_rows_carry_sector_and_cik():
    frame = parse_sp500(SP500_PAGE)
    assert frame["ticker"].tolist() == ["AAPL", "BRK.B", "JNJ", "XRX"]
    apple = frame.iloc[0]
    assert apple["name"] == "Apple Inc." and apple["sector"] == "Information Technology"
    assert apple["sub_industry"] == "Computer Hardware" and apple["cik"] == 320193
    assert frame.loc[frame["ticker"] == "JNJ", "name"].item() == "Johnson & Johnson"


def test_combined_list_flags_sp100_members():
    combined = combine(parse_sp100(SP100_PAGE), parse_sp500(SP500_PAGE))
    assert combined.set_index("ticker")["sp100"].to_dict() == {
        "AAPL": True, "BRK.B": True, "JNJ": True, "XRX": False,
    }


def test_sp100_members_missing_from_the_sp500_are_reported():
    sp100 = pd.DataFrame({"ticker": ["AAPL", "ZZZ"], "name": ["Apple", "Unknown"]})
    with pytest.raises(ValueError, match="ZZZ"):
        combine(sp100, parse_sp500(SP500_PAGE))


def test_duplicated_tickers_are_rejected():
    doubled = SP100_PAGE.replace("| BRK.B", "| AAPL")
    with pytest.raises(ValueError, match="duplicated"):
        parse_sp100(doubled)


def api_payload(title, revid, content):
    page = {"title": title, "revisions": [{"revid": revid, "slots": {"main": {"content": content}}}]}
    return json.dumps({"query": {"pages": [page]}}).encode()


def test_api_response_must_match_the_pinned_revision():
    pinned = SNAPSHOT["sp100"]
    assert wikitext_of(api_payload(pinned.page, pinned.revid, "text"), pinned) == "text"
    other = Revision(pinned.page, pinned.revid + 1, pinned.timestamp)
    with pytest.raises(ValueError, match="expected"):
        wikitext_of(api_payload(pinned.page, pinned.revid, "text"), other)
