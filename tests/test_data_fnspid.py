import io

import pandas as pd
import pytest

from regime_fusion.data import fnspid
from regime_fusion.data.universe import Stock

CSV = b'''Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary
2020-06-05 06:30:54 UTC,Apple beats estimates,AAPL,https://x/1,Benzinga,,"A long text
over two lines, with a comma",,,,
2020-06-05 00:00:00 UTC,RTX wins contract,RTX,https://x/2,Benzinga,,,,,,
2020-06-05 07:00:00 UTC,Agilent moves,A,https://x/3,Benzinga,,,,,,
'''


def stock(stock_id, yahoo):
    return Stock(stock_id, stock_id, "Sector", (1,), yahoo, "main")


def test_tickers_include_later_names():
    mapping = fnspid.ticker_map([stock("UTX", "RTX"), stock("SYMC", "GEN"), stock("BRK.B", "BRK-B")])
    assert mapping["RTX"] == "UTX" and mapping["NLOK"] == "SYMC" and mapping["GEN"] == "SYMC"
    assert mapping["BRK.B"] == mapping["BRK-B"] == mapping["BRKB"] == "BRK.B"
    with pytest.raises(ValueError, match="both"):
        fnspid.ticker_map([stock("UTX", "RTX"), stock("RTX", "RTX2")])


def test_only_the_universes_rows_are_kept_without_text():
    tickers = fnspid.ticker_map([stock("AAPL", "AAPL"), stock("UTX", "RTX")])
    frame, seen = fnspid.filter_csv(io.BytesIO(CSV), tickers, chunk_rows=2)
    assert seen == 3
    assert frame["stock"].tolist() == ["AAPL", "UTX"]
    assert list(frame.columns) == [*fnspid.KEEP, "stock"]


class DroppingResponse:
    """Serves bytes from an offset in small chunks, failing once mid-way."""

    def __init__(self, data, start, fail_at):
        self.data, self.start, self.fail_at = data, start, fail_at
        self.status_code = 206 if start else 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        position = self.start
        while position < len(self.data):
            if self.fail_at is not None and position >= self.fail_at:
                raise ConnectionError("connection reset")
            yield self.data[position:position + 7]
            position += 7


class DroppingSession:
    def __init__(self, data):
        self.data, self.calls = data, []

    def get(self, url, headers, stream, timeout):
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        self.calls.append(start)
        return DroppingResponse(self.data, start, fail_at=30 if not headers else None)


def test_stream_resumes_where_the_connection_dropped():
    data = bytes(range(256)) * 3
    session = DroppingSession(data)
    stream = io.BufferedReader(fnspid.ResumableStream("u", session, sleep=lambda s: None), buffer_size=16)
    assert stream.read() == data
    assert session.calls[0] == 0 and session.calls[1] == 35  # reconnected at the byte offset
