"""FinBERT headline sentiment from FNSPID, for the robustness check (option B, docs/DECISIONS.md).

* ``headlines``: the universe's FNSPID headlines (``regime_fusion.data.fnspid``),
  one row per stock, headline and time, with the session each counts for. All
  of them count as items with only a date, so each counts for the first session
  after its date (``regime_fusion.data.news``). FNSPID labels its times UTC, but
  99.5% are exactly midnight (a date only), and the rest cluster between 03:00
  and 09:00, the New York pre-market hours, which suggests local times labelled
  UTC. Ignoring the time of day rules out any look-ahead from that doubt and
  delays few headlines.
* ``score_texts``: FinBERT's probabilities that a headline is positive,
  negative or neutral (ProsusAI/finbert, fine-tuned on the Financial PhraseBank,
  text written before 2014). Each distinct headline is scored once and the
  scores are kept, so the step can be run again cheaply.
* ``indicators``: four sentiment indicators per session, over the last w = 1
  and w = 5 sessions:

  ====================  ===============================================
  finbert_attention_w   log(1 + headlines)
  finbert_sentiment_w   mean of P(positive) - P(negative) over headlines
  ====================  ===============================================

  The sentiment is missing when the window has no headline. FNSPID's coverage
  differs by stock (``All_external.csv`` ends in June 2020, and 18 of the stocks
  have no later source), so sessions outside a stock's first and last headline
  have no value at all, rather than zero news.

The indicators join the sentiment branch (as ``sa:finbert_...``) in the
robustness rungs (``finbert: true`` in ``configs/rungs.yaml``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from regime_fusion.data import news
from regime_fusion.data.calendar import TradingCalendar

MODEL = "ProsusAI/finbert"
WINDOWS = (1, 5)
COLUMNS = [f"finbert_{name}_{w}" for w in WINDOWS for name in ("attention", "sentiment")]


def text_key(texts: pd.Series) -> pd.Series:
    """A short, stable key for each headline text (to keep scores between runs)."""
    return texts.map(lambda t: hashlib.sha256(str(t).encode()).hexdigest()[:16])


def headlines(frames: Iterable[pd.DataFrame], calendar: TradingCalendar) -> pd.DataFrame:
    """Distinct headlines per stock with their time and session (``stock`` is the study's id)."""
    raw = pd.concat(list(frames), ignore_index=True)
    raw = raw.dropna(subset=["Article_title", "Date"])
    stamps, _ = news.fnspid_time(raw["Date"])
    out = pd.DataFrame({"stock": raw["stock"].to_numpy(), "time": stamps,
                        "title": raw["Article_title"].astype(str).str.strip().to_numpy()})
    out["session"] = news.assign_sessions(calendar, stamps, np.ones(len(stamps), dtype=bool))  # dates only
    before = stamps.tz_convert("UTC").tz_localize(None).normalize() < calendar.sessions[0]
    out = out[~before].dropna(subset=["session"]).drop_duplicates(["stock", "time", "title"])
    out["key"] = text_key(out["title"])
    return out.sort_values(["stock", "time"]).reset_index(drop=True)


def finbert_scorer(batch: int = 64, device: str = "cpu") -> tuple[Callable[[list[str]], np.ndarray], str]:
    """FinBERT as a function from texts to (positive, negative, neutral) probabilities, and its revision."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def coverage(items: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """FNSPID's coverage of one stock: the sessions of its first and last headline.

    This describes the dataset (where it stops following a stock), not the
    market, and is fixed once from the whole dataset; the indicators of a day
    still use only the headlines up to it.
    """
    return (items["session"].min(), items["session"].max()) if len(items) else None


def indicators(items: pd.DataFrame, scores: pd.DataFrame, sessions: pd.DatetimeIndex,
               covered: tuple[pd.Timestamp, pd.Timestamp] | None) -> pd.DataFrame:
    """The four indicators of one stock for every session in ``sessions``; missing outside
    ``covered`` (``coverage`` of the stock's headlines)."""
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
    if covered is None:
        return table * np.nan
    inside = (table.index >= covered[0]) & (table.index <= covered[1])
    table.loc[~inside] = np.nan
    return table
