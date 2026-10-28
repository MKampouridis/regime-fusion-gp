"""FNSPID headlines for the robustness check (CC BY-NC 4.0: research use, no redistribution).

FNSPID (Dong, Fan and Peng, 2024) is on Hugging Face as two CSV files of news
labelled with stock tickers: ``All_external.csv`` (5.7 GB) and
``nasdaq_exteral_data.csv`` (23 GB, with article text). Neither is stored: each
is streamed, parsed in chunks, and only the rows labelled with one of the
study's tickers are kept, without the article text (date, headline, ticker,
URL, publisher, author). The stream resumes from its byte offset if the
connection drops.

FNSPID labels news with the ticker of its time, so renamed firms appear under
later tickers too (Raytheon Technologies as RTX, Anthem as ELV); the stock's
January 2015 ticker, its Yahoo symbol and the tickers in ``LATER_TICKERS`` all
map to the stock. Dates are UTC; items stamped exactly at midnight carry only a
date (``regime_fusion.data.news``).
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterable

import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
FILES = ("Stock_news/All_external.csv", "Stock_news/nasdaq_exteral_data.csv")
KEEP = ("Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author")
# Tickers a stock traded under between January 2015 and 2023 other than its January
# 2015 ticker and today's Yahoo symbol.
LATER_TICKERS = {"SYMC": ["NLOK"], "GOOGL": ["GOOG"], "DD": ["DWDP"]}


class ResumableStream(io.RawIOBase):
    """A file-like view of a large HTTP download that reconnects at its current
    byte offset (with a Range request) when the connection drops."""

    def __init__(self, url: str, session, *, attempts: int = 8, pause: float = 5.0,
                 sleep=time.sleep):
        self.url, self.session = url, session
        self.attempts, self.pause, self._sleep = attempts, pause, sleep
        self.position = 0
        self._chunks = None
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        while not self._buffer:
            chunk = self._next_chunk()
            if chunk is None:
                return 0
            self._buffer = chunk
        n = min(len(target), len(self._buffer))
        target[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        self.position += n
        return n

    def _next_chunk(self) -> bytes | None:
        for attempt in range(1, self.attempts + 1):
            try:
                if self._chunks is None:
                    headers = {"Range": f"bytes={self.position}-"} if self.position else {}
                    response = self.session.get(self.url, headers=headers, stream=True, timeout=120)
                    if response.status_code == 416:  # nothing left from this offset
                        return None
                    response.raise_for_status()
                    if self.position and response.status_code != 206:
                        raise RuntimeError("the server ignored the Range request")
                    self._chunks = response.iter_content(chunk_size=1 << 20)
                return next(self._chunks)
            except StopIteration:
                return None
            except Exception as error:  # dropped connection: reconnect at the current offset
                self._chunks = None
                if attempt == self.attempts:
                    raise
                log.warning("%s: %s at byte %d; reconnecting", self.url, error, self.position)
                self._sleep(self.pause * attempt)
        return None


def ticker_map(stocks: Iterable) -> dict[str, str]:
    """FNSPID ticker -> stock id, for the given universe stocks."""
    mapping = {}
    for stock in stocks:
        names = {stock.id, stock.id.replace(".", "-"), stock.id.replace(".", ""),
                 *(LATER_TICKERS.get(stock.id, []))}
        if stock.yahoo:
            names.add(stock.yahoo)
        for name in names:
            if name in mapping and mapping[name] != stock.id:
                raise ValueError(f"ticker {name} maps to both {mapping[name]} and {stock.id}")
            mapping[name] = stock.id
    return mapping


def filter_csv(stream, tickers: dict[str, str], *, chunk_rows: int = 20_000) -> tuple[pd.DataFrame, int]:
    """Rows of an FNSPID CSV labelled with one of ``tickers``, and the number of rows read."""
    kept, seen = [], 0
    reader = pd.read_csv(stream, usecols=list(KEEP), dtype=str, chunksize=chunk_rows,
                         keep_default_na=False, on_bad_lines="warn")
    for chunk in reader:
        seen += len(chunk)
        hits = chunk[chunk["Stock_symbol"].isin(tickers)]
        if len(hits):
            kept.append(hits.assign(stock=hits["Stock_symbol"].map(tickers)))
    columns = [*KEEP, "stock"]
    frame = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=columns)
    return frame.loc[:, columns], seen
