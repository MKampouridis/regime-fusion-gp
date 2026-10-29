"""FinBERT headline sentiment from FNSPID, for the robustness check (option B, docs/DECISIONS.md).

* ``headlines``: the universe's FNSPID headlines (``regime_fusion.data.fnspid``),
  one row per stock, headline and time, with the session each counts for
  (``regime_fusion.data.news``: after the close of t counts for t+1; items with
  only a date count for the first session after it).
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

  The sentiment is missing when the window has no headline. Sessions after the
  last day FNSPID covers have no value at all, rather than zero news.

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
    stamps, date_only = news.fnspid_time(raw["Date"])
    out = pd.DataFrame({"stock": raw["stock"].to_numpy(), "time": stamps,
                        "title": raw["Article_title"].astype(str).str.strip().to_numpy()})
    out["session"] = news.assign_sessions(calendar, stamps, date_only)
    out = out.dropna(subset=["session"]).drop_duplicates(["stock", "time", "title"])
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


def indicators(items: pd.DataFrame, scores: pd.DataFrame, sessions: pd.DatetimeIndex,
               covered_until: pd.Timestamp) -> pd.DataFrame:
    """The four indicators of one stock for every session in ``sessions``."""
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
    table.loc[table.index > covered_until] = np.nan
    return table
