"""GDELT's Global Knowledge Graph (GKG 2.1): news about the study's firms, summed by hour.

GDELT publishes one GKG file every 15 minutes, the first at 23:00 UTC on
18 February 2015, listed with sizes and MD5 checksums in its master file list
(English-language sources; translated sources have a separate list, not used).
The files are streamed one at a time into memory, checked against their MD5,
and discarded: about 2.75 TB pass over the network, but almost nothing is kept.

From each file only the records whose organisations match a firm's anchors
(``configs/news_aliases.yaml``) are used, and they are summed per UTC hour,
stock and *variant set*: the organisation names that matched the stock's
anchors in the record (for example "apple inc" or "apple inc|apple daily").
Keeping the variant sets apart lets the exact matching be decided later
(dropping "apple daily", say) without downloading GDELT again. The sums cover
article counts, first mentions (the stock is the first organisation named),
title mentions, GDELT's tone fields, and the GCAM word count with the six
Loughran-McDonald word counts (c6.1-c6.6: litigious, strong modal, weak modal,
negative, positive, uncertainty), plus per-article shares for averaging.

Records whose page title names the firm are also kept whole (time, stock,
source, URL, title, tone and LM counts): they serve hand checks of the matching
and give headlines for the FinBERT robustness check. GDELT records carry titles
in later years only.

Files are grouped by UTC day, and a day is saved once all its files are done,
so an interrupted run resumes where it stopped. Hourly sums lose nothing at the
close cut-off, because the NYSE closes on the hour (``regime_fusion.data.news``).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MASTER_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
DEFAULT_ALIASES = Path(__file__).resolve().parents[3] / "configs" / "news_aliases.yaml"

# Positions of the fields used, in a GKG 2.1 record of 27 tab-separated fields.
RECORD_ID, DATE, SOURCE, URL = 0, 1, 3, 4
ORGANIZATIONS, V2_ORGANIZATIONS, TONE, GCAM, EXTRAS = 13, 14, 15, 17, 26
N_FIELDS = 27
LM_CODES = {"c6.1": "lm_litigious", "c6.2": "lm_modal_strong", "c6.3": "lm_modal_weak",
            "c6.4": "lm_negative", "c6.5": "lm_positive", "c6.6": "lm_uncertainty"}
TONE_FIELDS = ("tone", "positive", "negative", "polarity", "words")  # first five of V1.5 tone
SHARE_CODES = ("lm_negative", "lm_positive", "lm_uncertainty")
# Hourly sums, one row per (hour, stock, variants).
SUM_COLUMNS = (
    "records", "first_mentions", "title_mentions",
    *(f"{name}_sum" for name in TONE_FIELDS), "tone_sq_sum",
    "gcam_words_sum", *(f"{name}_sum" for name in LM_CODES.values()),
    "gcam_records", *(f"{name}_share_sum" for name in SHARE_CODES),
)
TITLE_COLUMNS = ("date", "stock", "variants", "source", "url", "title", "tone", "gcam_words",
                 *SHARE_CODES, "first_mention")

_GCAM = re.compile(r"(?:^|,)(wc|c6\.[1-6]):(\d+)")
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


class FetchError(RuntimeError):
    """A GKG file could not be fetched intact."""


@dataclass(frozen=True)
class Matcher:
    """Anchors of the selected stocks, compiled."""

    any: re.Pattern
    stocks: dict[str, re.Pattern]
    digest: str

    def match(self, organizations: str) -> list[str]:
        text = organizations.lower()
        if not text or not self.any.search(text):
            return []
        return [stock for stock, pattern in self.stocks.items() if pattern.search(text)]


def load_matcher(path: str | Path = DEFAULT_ALIASES, stocks: Iterable[str] | None = None) -> Matcher:
    """Anchors for ``stocks`` (all in the file by default)."""
    raw = yaml.safe_load(Path(path).read_text())["stocks"]
    wanted = list(raw) if stocks is None else list(stocks)
    missing = sorted(set(wanted) - set(raw))
    if missing:
        raise ValueError(f"{path}: no anchors for {missing}")
    anchors = {stock: [str(a).lower() for a in raw[stock]] for stock in wanted}
    patterns = {stock: _words(items) for stock, items in anchors.items()}
    digest = hashlib.sha256(json.dumps(anchors, sort_keys=True).encode()).hexdigest()[:12]
    return Matcher(_words([a for items in anchors.values() for a in items]), patterns, digest)


def master_list(text: str, first_day: str, last_day: str) -> pd.DataFrame:
    """GKG files in GDELT's master file list between two UTC days (YYYYMMDD), inclusive."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[2].endswith(".gkg.csv.zip"):
            continue
        stamp = parts[2].rsplit("/", 1)[1][:14]
        if first_day <= stamp[:8] <= last_day:
            rows.append((stamp, stamp[:8], int(parts[0]), parts[1], parts[2].replace("http://", "https://")))
    frame = pd.DataFrame(rows, columns=["stamp", "day", "size", "md5", "url"])
    return frame.drop_duplicates("stamp").sort_values("stamp").reset_index(drop=True)


def extract(content: bytes, matcher: Matcher, sums: dict, titles: list) -> int:
    """Add the matching records of one zipped GKG file to ``sums`` and ``titles``.

    ``sums`` maps (hour, stock, variants) to a list of sums in ``SUM_COLUMNS`` order;
    ``titles`` collects tuples in ``TITLE_COLUMNS`` order. Returns the records used.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")
    used = 0
    for line in text.split("\n"):
        fields = line.split("\t")
        if len(fields) < N_FIELDS:
            continue
        stocks = matcher.match(fields[ORGANIZATIONS])
        if stocks:
            used += 1
            _add(fields, stocks, matcher, sums, titles)
    return used


def fetch(url: str, md5: str, session, *, attempts: int = 4, pause: float = 2.0,
          sleep=time.sleep) -> bytes:
    """A GKG file, checked against its MD5; retried on errors, not on 404."""
    problem = "no attempt"
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=120)
            if response.status_code == 404:
                raise FetchError(f"{url}: not on GDELT's server (404)")
            response.raise_for_status()
            if hashlib.md5(response.content).hexdigest() == md5:
                return response.content
            problem = "MD5 mismatch"
        except FetchError:
            raise
        except Exception as error:  # network errors: wait and retry
            problem = str(error)
        if attempt < attempts:
            sleep(pause * 2 ** attempt)
    raise FetchError(f"{url}: {problem} after {attempts} attempts")


def process_day(day: str, files: list[tuple[str, str]], matcher: Matcher,
                session=None) -> tuple[str, pd.DataFrame, pd.DataFrame, dict]:
    """Hourly sums and titled records of one UTC day, and a summary of what was fetched."""
    if session is None:
        import requests

        session = requests.Session()
    sums: dict = {}
    titles: list = []
    failed, fetched, used = [], 0, 0
    for url, md5 in files:
        try:
            content = fetch(url, md5, session)
        except FetchError as error:
            failed.append(str(error))
            continue
        fetched += len(content)
        used += extract(content, matcher, sums, titles)
    hourly = pd.DataFrame(
        [(hour, stock, variants, *values) for (hour, stock, variants), values in sums.items()],
        columns=["hour", "stock", "variants", *SUM_COLUMNS],
    )
    hourly["hour"] = pd.to_datetime(hourly["hour"], format="%Y%m%d%H", utc=True)
    hourly = hourly.astype({c: "float64" for c in SUM_COLUMNS}).sort_values(["hour", "stock", "variants"])
    titled = pd.DataFrame(titles, columns=list(TITLE_COLUMNS)).astype({"date": "int64"})
    summary = {"day": day, "files": len(files), "failed": failed, "records_used": used,
               "hourly_rows": len(hourly), "titled_records": len(titled), "bytes": fetched,
               "anchors_digest": matcher.digest}
    return day, hourly.reset_index(drop=True), titled, summary


def _words(alternatives: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b")


def _add(fields: list[str], stocks: list[str], matcher: Matcher, sums: dict, titles: list) -> None:
    hour = fields[DATE][:10]
    names = [name for name in fields[ORGANIZATIONS].lower().split(";") if name]
    offsets = []
    for item in fields[V2_ORGANIZATIONS].lower().split(";"):
        name, _, offset = item.rpartition(",")
        if name and offset.isdigit():
            offsets.append((int(offset), name))
    first = min(offsets)[1] if offsets else None
    tone = fields[TONE].split(",")
    tone_values = [_number(tone[i]) if i < len(tone) else np.nan for i in range(len(TONE_FIELDS))]
    gcam = dict(_GCAM.findall(fields[GCAM]))
    words = _number(gcam.get("wc"))
    lm = [_number(gcam.get(code, "0")) for code in LM_CODES]
    lm_by_name = dict(zip(LM_CODES.values(), lm))
    has_words = words > 0
    shares = [lm_by_name[name] / words if has_words else 0.0 for name in SHARE_CODES]
    match = _TITLE.search(fields[EXTRAS])
    title = match.group(1).strip() if match else None
    for stock in stocks:
        pattern = matcher.stocks[stock]
        matched = sorted({name for name in names if pattern.search(name)})
        variants = "|".join(matched)
        is_first = first is not None and first in matched
        in_title = bool(title) and bool(pattern.search(title.lower()))
        row = sums.setdefault((hour, stock, variants), [0.0] * len(SUM_COLUMNS))
        values = [1.0, float(is_first), float(in_title),
                  *(0.0 if np.isnan(v) else v for v in tone_values),
                  0.0 if np.isnan(tone_values[0]) else tone_values[0] ** 2,
                  words if has_words else 0.0, *lm, float(has_words), *shares]
        for i, value in enumerate(values):
            row[i] += value
        if in_title:
            titles.append((_integer(fields[DATE]), stock, variants, fields[SOURCE], fields[URL], title,
                           tone_values[0], words, *(lm_by_name[name] for name in SHARE_CODES), is_first))


def _number(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _integer(text) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return -1
