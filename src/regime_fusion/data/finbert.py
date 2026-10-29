"""FinBERT headline sentiment, for the robustness check (option B, docs/DECISIONS.md).

Headlines come from two sources (MK, 29 October 2026), and each walk-forward
window uses one of them throughout, chosen by its test year
(``finbert_sources`` in ``configs/rungs.yaml``):

* ``fnspid_headlines``: the universe's FNSPID headlines
  (``regime_fusion.data.fnspid``), for test years up to 2023 (FNSPID ends in
  December 2023). All of them count as items with only a date, so each counts
  for the first session after its date (``regime_fusion.data.news``). FNSPID
  labels its times UTC, but 99.5% are exactly midnight (a date only), and the
  rest cluster between 03:00 and 09:00, the New York pre-market hours, which
  suggests local times labelled UTC. Ignoring the time of day rules out any
  look-ahead from that doubt and delays few headlines.
* ``gdelt_headlines``: the page titles of GDELT records that name the firm (kept
  by ``regime_fusion.data.gdelt``), for 2024 and 2025. GDELT records carry titles
  from 23 September 2019, so they cover the whole window of those test years.
  The caller keeps only records that count for their firm under the sentiment
  indicators' name rules (``regime_fusion.data.sa``); each counts from the
  15-minute update that first carried it.

A headline counts once per stock and session, so copies of a story on several
sites, or the same item twice, count once.

* ``score_texts``: FinBERT's probabilities that a headline is positive,
  negative or neutral (ProsusAI/finbert, fine-tuned on the Financial PhraseBank,
  text written before 2014). Each distinct headline is scored once and the
  scores are kept, so the step can be run again cheaply.
* ``covered_years``: the years in which a source follows a stock, those in which
  at most two of its months have no headline for it. Coverage differs by stock
  and year: FNSPID's ``All_external.csv`` ends in June 2020 and 18 of the stocks
  have no later source, and some have whole years without a headline (Apple in
  2021). Coverage describes the dataset, not the market, so it is fixed once from
  the whole source; each day's values still use only the headlines up to it.
* ``indicators``: four sentiment indicators per session, over the last w = 1
  and w = 5 sessions:

  ====================  ===============================================
  finbert_attention_w   log(1 + headlines)
  finbert_sentiment_w   mean of P(positive) - P(negative) over headlines
  ====================  ===============================================

  The sentiment is missing when the window has no headline. In years the source
  does not cover, and on sessions whose GDELT files were mostly not read, the
  indicators are missing rather than zero news.

The indicators join the sentiment branch (as ``sa:finbert_...``) in the
robustness rungs (``finbert: true`` in ``configs/rungs.yaml``), for the
stock-years whose whole window their source covers (``covered_test_years``).
"""

from __future__ import annotations

import hashlib
import html
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from regime_fusion.data import news
from regime_fusion.data.calendar import TradingCalendar

MODEL = "ProsusAI/finbert"
SOURCES = ("fnspid", "gdelt")
WINDOWS = (1, 5)
COLUMNS = [f"finbert_{name}_{w}" for w in WINDOWS for name in ("attention", "sentiment")]
MAX_EMPTY_MONTHS = 2


def text_key(texts: pd.Series) -> pd.Series:
    """A short, stable key for each headline text (to keep scores between runs)."""
    return texts.map(lambda t: hashlib.sha256(str(t).encode()).hexdigest()[:16])


def fnspid_headlines(frames: Iterable[pd.DataFrame], calendar: TradingCalendar) -> pd.DataFrame:
    """Distinct FNSPID headlines per stock with their time and session (``stock`` is the study's id)."""
    raw = pd.concat(list(frames), ignore_index=True)
    raw = raw.dropna(subset=["Article_title", "Date"])
    stamps, _ = news.fnspid_time(raw["Date"])
    out = pd.DataFrame({"stock": raw["stock"].to_numpy(), "time": stamps,
                        "title": raw["Article_title"].astype(str).str.strip().to_numpy()})
    out["session"] = news.assign_sessions(calendar, stamps, np.ones(len(stamps), dtype=bool))  # dates only
    before = stamps.tz_convert("UTC").tz_localize(None).normalize() < calendar.sessions[0]
    return _distinct(out[~before])


def gdelt_headlines(frames: Iterable[pd.DataFrame], calendar: TradingCalendar) -> pd.DataFrame:
    """Distinct headlines from GDELT records whose page title names their firm (``gdelt.TITLE_COLUMNS``,
    already kept by the name rules), each from the 15-minute update that first carried it."""
    raw = pd.concat(list(frames), ignore_index=True).dropna(subset=["title", "date"])
    stamps = news.gdelt_time(raw["date"])
    out = pd.DataFrame({"stock": raw["stock"].to_numpy(), "time": stamps,
                        "title": raw["title"].astype(str).map(html.unescape).str.strip().to_numpy()})
    out["session"] = news.assign_sessions(calendar, stamps)
    return _distinct(out)


def _distinct(items: pd.DataFrame) -> pd.DataFrame:
    """One row per stock, session and headline (its first appearance), with the headline's key."""
    items = items.dropna(subset=["session"])
    items = items[items["title"].str.len() > 0].sort_values(["stock", "time"], kind="stable")
    items = items.drop_duplicates(["stock", "session", "title"])
    return items.assign(key=text_key(items["title"])).reset_index(drop=True)


def finbert_scorer(batch: int = 64, device: str | None = None) -> tuple[Callable[[list[str]], np.ndarray], str]:
    """FinBERT as a function from texts to (positive, negative, neutral) probabilities, and its revision.

    By default it runs on a GPU when there is one (Apple's, through MPS, or CUDA), else on the CPU.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    order = [next(i for i, name in model.config.id2label.items() if name.lower() == label)
             for label in ("positive", "negative", "neutral")]

    def score(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for first in range(0, len(texts), batch):
                encoded = tokenizer(texts[first:first + batch], padding=True, truncation=True, max_length=64,
                                    return_tensors="pt").to(device)
                out.append(torch.softmax(model(**encoded).logits, dim=-1)[:, order].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    return score, str(getattr(model.config, "_commit_hash", "unknown"))


def score_texts(items: pd.DataFrame, known: pd.DataFrame, scorer: Callable[[list[str]], np.ndarray]) -> pd.DataFrame:
    """Scores of every distinct headline in ``items``, reusing ``known`` (key, positive, negative, neutral)."""
    distinct = items.drop_duplicates("key")[["key", "title"]]
    todo = distinct[~distinct["key"].isin(known["key"])] if len(known) else distinct
    probabilities = scorer(todo["title"].tolist()) if len(todo) else np.zeros((0, 3))
    fresh = pd.DataFrame(probabilities, columns=["positive", "negative", "neutral"]).assign(
        key=todo["key"].to_numpy())
    return pd.concat([known, fresh], ignore_index=True)[["key", "positive", "negative", "neutral"]]


def covered_years(items: pd.DataFrame, sessions: pd.DatetimeIndex, max_empty: int = MAX_EMPTY_MONTHS) -> list[int]:
    """Years in which a source follows one stock: at most ``max_empty`` of the year's months (those
    with sessions) have no headline for it.

    This describes the dataset (where it follows a stock), not the market, and is
    fixed once from the whole source; the indicators of a day still use only the
    headlines up to it.
    """
    months = pd.Series(sessions.to_period("M").unique())
    with_news = set(pd.DatetimeIndex(items["session"]).to_period("M"))
    empty = months.map(lambda month: month not in with_news).groupby(months.map(lambda m: m.year)).sum()
    return [int(year) for year, count in empty.items() if count <= max_empty]


def covered_test_years(years: Iterable[int], windows) -> list[int]:
    """Test years whose whole window, from its first training year to the test year, lies in ``years``
    (``windows`` from ``regime_fusion.windows.walk_forward``)."""
    years = set(years)
    return [w.test_year for w in windows if set(range(w.train.start.year, w.test_year + 1)) <= years]


def indicators(items: pd.DataFrame, scores: pd.DataFrame, sessions: pd.DatetimeIndex, years: Iterable[int],
               unread: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """The four indicators of one stock for every session in ``sessions``; missing outside the covered
    ``years`` (``covered_years``) and on the ``unread`` sessions (GDELT files mostly not read)."""
    scored = items.merge(scores, on="key", how="left")
    scored["tone"] = scored["positive"] - scored["negative"]
    daily = scored.groupby("session").agg(count=("tone", "size"), tone=("tone", "sum"))
    daily = daily.reindex(sessions, fill_value=0.0)
    out = {}
    for w in WINDOWS:
        count = daily["count"].rolling(w, min_periods=1).sum()
        tone = daily["tone"].rolling(w, min_periods=1).sum()
        out[f"finbert_attention_{w}"] = np.log1p(count)
        out[f"finbert_sentiment_{w}"] = (tone / count).where(count > 0)
    table = pd.DataFrame(out, index=sessions)[COLUMNS]
    table.loc[~table.index.year.isin(list(years))] = np.nan
    if unread is not None:
        table.loc[table.index.isin(unread)] = np.nan
    return table
