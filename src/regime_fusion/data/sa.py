"""Sentiment indicators from GDELT's news, daily.

Built from the hourly sums that ``regime_fusion.data.gdelt`` keeps, in three steps.

1. **Names.** The stored sums are split by *variant set*: the names in a record
   that matched a firm's broad anchors. ``configs/news_names.yaml`` decides which
   names really are the firm ("apple inc" but not "apple daily"); a record counts
   when at least one of its names does.
2. **Sessions.** Each hour's sums go to the first session whose close is strictly
   later than the hour (``regime_fusion.data.news``): news after the close of t
   counts for t+1. A session's news therefore runs from the previous close to its
   own close, so Monday's includes the weekend's.
3. **Coverage.** GDELT's archive lacks some of its 15-minute files, a few whole
   days among them (about 2% of files; most in late 2020 and June 2025), and
   some files could not be fetched. A session's coverage is the share of its
   15-minute files that were read. Article counts are scaled up by it, and a
   session with less than half of its files has no indicators.

The twelve indicators, each over the last w = 1 and w = 5 sessions:

=====================  ========================================================
attention_w            log(1 + articles), counts scaled up for missing files
abnormal_attention_w   attention_w minus its mean over the previous 60 sessions
tone_w                 mean GDELT tone of the articles (positive minus negative
                       words, % of all words)
lm_positive_w          mean share of Loughran-McDonald positive words (%)
lm_negative_w          mean share of Loughran-McDonald negative words (%)
lm_uncertainty_w       mean share of Loughran-McDonald uncertainty words (%)
=====================  ========================================================

Means are taken over articles (each article counts once), and are missing when
the window has no article. A five-session indicator needs at least three
sessions with enough coverage; its article count is scaled to five sessions.
The abnormal attention needs 20 of the previous 60 sessions. Values are raw;
scaling is fitted on training windows later (``regime_fusion.engine.panel``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regime_fusion.data import gdelt, news
from regime_fusion.data.calendar import TradingCalendar

DEFAULT_NAMES = Path(__file__).resolve().parents[3] / "configs" / "news_names.yaml"
WINDOWS = (1, 5)
MEASURES = ("attention", "abnormal_attention", "tone", "lm_positive", "lm_negative", "lm_uncertainty")
BASELINE, MIN_BASELINE = 60, 20
MIN_COVERAGE = 0.5
SLOT = pd.Timedelta(minutes=15)
SUFFIXES = (r"(?:,? (?:inc|incorporated|corp|corporation|co|cos|company|companies|ltd|limited|plc|llc|lp"
            r"|group|holdings?|intl))*")


def columns(windows=WINDOWS) -> list[str]:
    return [f"{name}_{w}" for w in windows for name in MEASURES]


@dataclass(frozen=True)
class Pattern:
    """A name pattern, possibly limited to UTC days from ``since`` to ``until``."""

    regex: re.Pattern
    since: pd.Timestamp | None = None
    until: pd.Timestamp | None = None

    @property
    def dated(self) -> bool:
        return self.since is not None or self.until is not None

    def active(self, day: pd.Timestamp) -> bool:
        return (self.since is None or day >= self.since) and (self.until is None or day <= self.until)

    def matches(self, name: str, day: pd.Timestamp) -> bool:
        return self.active(day) and self.regex.fullmatch(name) is not None


@dataclass
class NameRules:
    """Which matched names count as each firm (``configs/news_names.yaml``)."""

    prefix: re.Pattern
    accept: dict[str, tuple[Pattern, ...]]
    reject: dict[str, tuple[Pattern, ...]]
    digest: str
    veto: dict[str, tuple[Pattern, ...]] = field(default_factory=dict)  # names that rule out the whole record
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def stocks(self) -> list[str]:
        return list(self.accept)

    def clean(self, name: str) -> str:
        return self.prefix.sub("", name)

    def name_counts(self, stock: str, name: str, day: pd.Timestamp) -> bool:
        name = self.clean(name)
        return (any(p.matches(name, day) for p in self.accept[stock])
                and not any(p.matches(name, day) for p in self.reject.get(stock, ())))

    def counts(self, stock: str, variants: str, day) -> bool:
        """Whether a record whose matched names are ``variants`` ("a|b") is about ``stock``: one of its
        names counts, and none is vetoed (a veto, such as "kodak black", rules out a record whose other
        names, such as a bare "kodak", would count)."""
        day = pd.Timestamp(day)
        patterns = (*self.accept[stock], *self.reject.get(stock, ()), *self.veto.get(stock, ()))
        key = (stock, variants, tuple(p.active(day) for p in patterns if p.dated))
        if key not in self._cache:
            names = [self.clean(name) for name in variants.split("|") if name]
            vetoed = any(p.matches(name, day) for p in self.veto.get(stock, ()) for name in names)
            self._cache[key] = not vetoed and any(self.name_counts(stock, name, day) for name in names)
        return self._cache[key]


def load_rules(path: str | Path = DEFAULT_NAMES, stocks: Iterable[str] | None = None) -> NameRules:
    """Name rules for ``stocks`` (all in the file by default)."""
    raw = yaml.safe_load(Path(path).read_text())
    entries = raw["stocks"]
    wanted = list(entries) if stocks is None else list(stocks)
    missing = sorted(set(wanted) - set(entries))
    if missing:
        raise ValueError(f"{path}: no name rules for {missing}")
    accept, reject, veto, normalised = {}, {}, {}, {}
    for stock in wanted:
        entry = entries[stock]
        if entry == "all":
            entry = {"accept": [".*"]}
        if not isinstance(entry, dict) or not entry.get("accept"):
            raise ValueError(f"{path}: {stock} needs 'all' or a list of accept patterns")
        accept[stock] = tuple(_pattern(item) for item in entry["accept"])
        reject[stock] = tuple(_pattern(item) for item in entry.get("reject", []))
        veto[stock] = tuple(_pattern(item) for item in entry.get("veto", []))
        normalised[stock] = {"accept": entry["accept"], "reject": entry.get("reject", []),
                             **({"veto": entry["veto"]} if entry.get("veto") else {})}
    prefixes = [str(p).lower() for p in raw.get("prefixes", [])]
    prefix = re.compile(r"^(?:(?:" + "|".join(map(re.escape, prefixes)) + r") )+") if prefixes else re.compile(r"(?!)")
    digest = hashlib.sha256(json.dumps({"prefixes": prefixes, "stocks": normalised}, sort_keys=True,
                                       default=str).encode()).hexdigest()[:12]
    return NameRules(prefix, accept, reject, digest, veto)


def keep(hourly: pd.DataFrame, rules: NameRules, day) -> np.ndarray:
    """Mask of the hourly rows (of one UTC day) whose names count for their stock."""
    known = hourly["stock"].isin(rules.stocks).to_numpy()
    decisions = {}
    for stock, variants in hourly.loc[known, ["stock", "variants"]].drop_duplicates().itertuples(index=False):
        decisions[(stock, variants)] = rules.counts(stock, variants, day)
    mask = np.zeros(len(hourly), dtype=bool)
    mask[known] = [decisions[key] for key in zip(hourly.loc[known, "stock"], hourly.loc[known, "variants"])]
    return mask


def tally(counter: Counter, hourly: pd.DataFrame, mask: np.ndarray) -> None:
    """Add the records naming each (stock, counted, name) to ``counter``.

    ``counted`` says whether the records counted for the stock, so the names of
    records left out show what the rules may have missed.
    """
    grouped = hourly.assign(counted=mask).groupby(["stock", "variants", "counted"])["records"].sum()
    for (stock, variants, counted), records in grouped.items():
        for name in variants.split("|"):
            counter[(stock, bool(counted), name)] += records


def session_sums(hourly: pd.DataFrame, calendar: TradingCalendar) -> pd.DataFrame:
    """Sums of hourly rows per (session, stock); hours after the calendar's last close are dropped."""
    values = hourly.loc[:, ["hour", "stock", *gdelt.SUM_COLUMNS]]
    if values.empty:
        index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["session", "stock"])
        return pd.DataFrame(columns=list(gdelt.SUM_COLUMNS), index=index, dtype="float64")
    sessions = calendar.session_of_timestamp(pd.DatetimeIndex(values["hour"]))
    present = sessions[~sessions.isna()].unique()
    totals = news.daily_sums(values, calendar, present)
    return totals[(totals != 0).any(axis=1)]


def fetched_stamps(day_files: pd.DataFrame, failed: Iterable[str]) -> pd.DatetimeIndex:
    """Times of the 15-minute files read for a day: its files in the master list minus failed ones."""
    lost = set(re.findall(r"/(\d{14})\.gkg\.csv\.zip", " ".join(failed)))
    stamps = day_files.loc[~day_files["stamp"].isin(lost), "stamp"]
    return pd.DatetimeIndex(pd.to_datetime(stamps, format="%Y%m%d%H%M%S", utc=True))


def coverage(stamps: pd.DatetimeIndex, calendar: TradingCalendar) -> pd.Series:
    """Share of each session's 15-minute files that were read (missing for the first session).

    A session's news runs from the previous close (included) to its own close
    (excluded), so it spans (close - previous close) / 15 minutes files.
    """
    sessions = calendar.session_of_timestamp(stamps)
    counts = pd.Series(1.0, index=sessions[~sessions.isna()]).groupby(level=0).sum()
    counts = counts.reindex(calendar.sessions, fill_value=0.0)
    slots = pd.Series(calendar.closes, index=calendar.sessions).diff() / SLOT
    return (counts / slots).clip(upper=1.0).rename("coverage")


def fnspid_agreement(articles: pd.Series, headlines: pd.Series, min_weeks: int = 20) -> dict[str, Any]:
    """How well a stock's GDELT articles follow its FNSPID headlines, over the sessions given (those
    both sources cover): the rank correlation of their weekly counts, and the share of the sessions
    with an FNSPID headline on which GDELT has news too. A check of GDELT's company matching."""
    both = pd.DataFrame({"gdelt": articles, "fnspid": headlines}).dropna()
    weeks = both.groupby(both.index.to_period("W-FRI")).sum()
    correlation = (weeks["gdelt"].corr(weeks["fnspid"], method="spearman")
                   if len(weeks) >= min_weeks and weeks.nunique().min() > 1 else np.nan)
    with_headline = both["fnspid"] > 0
    share = (both.loc[with_headline, "gdelt"] > 0).mean() if with_headline.any() else np.nan
    return {"weeks": len(weeks), "weekly_rank_correlation": None if np.isnan(correlation) else round(float(correlation), 3),
            "fnspid_days_with_gdelt_news": None if np.isnan(share) else round(float(share), 3)}


def scaled_articles(sums: pd.DataFrame, cover: pd.Series) -> pd.Series:
    """Articles per session scaled up for missing files; missing when coverage is too low."""
    records = sums["records"].reindex(cover.index, fill_value=0.0)
    return (records / cover).where(cover >= MIN_COVERAGE)


def indicators(sums: pd.DataFrame, cover: pd.Series, windows=WINDOWS) -> pd.DataFrame:
    """The indicators of one stock for every session of ``cover``.

    ``sums`` holds the stock's session sums (``gdelt.SUM_COLUMNS``) indexed by
    session; sessions without news may be absent.
    """
    sums = sums.reindex(cover.index, fill_value=0.0)
    valid = (cover >= MIN_COVERAGE).astype(float)

    def window_sum(values: pd.Series, w: int) -> pd.Series:
        return values.where(valid > 0, 0.0).rolling(w, min_periods=1).sum()

    articles = (sums["records"] / cover).where(valid > 0, 0.0)
    out = {}
    for w in windows:
        sessions = valid.rolling(w, min_periods=1).sum()
        enough = sessions >= (w + 1) // 2
        attention = np.log1p(window_sum(articles, w) * w / sessions.where(enough)).where(enough)
        baseline = attention.shift(1).rolling(BASELINE, min_periods=MIN_BASELINE).mean()
        records, gcam = window_sum(sums["records"], w), window_sum(sums["gcam_records"], w)
        out[f"attention_{w}"] = attention
        out[f"abnormal_attention_{w}"] = attention - baseline
        out[f"tone_{w}"] = (window_sum(sums["tone_sum"], w) / records).where(enough & (records > 0))
        for measure in ("lm_positive", "lm_negative", "lm_uncertainty"):
            share = 100.0 * window_sum(sums[f"{measure}_share_sum"], w) / gcam
            out[f"{measure}_{w}"] = share.where(enough & (gcam > 0))
    return pd.DataFrame(out, index=cover.index).loc[:, columns(windows)]


def _pattern(item) -> Pattern:
    if isinstance(item, dict):
        text, since, until = item["pattern"], item.get("since"), item.get("until")
    else:
        text, since, until = item, None, None
    regex = re.compile(str(text).replace("{S}", SUFFIXES))
    return Pattern(regex, None if since is None else pd.Timestamp(since),
                   None if until is None else pd.Timestamp(until))
