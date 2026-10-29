"""News-attention inputs for the regimes of rung A4: directional changes of news attention.

The attention level of a session is 1 + the articles of the last ``sessions``
sessions, from ``interim/news`` (``regime_fusion.data.sa``; counts scaled up for
missing GDELT files). Sessions without enough GDELT files are left out of the
window's mean, and when none in the window has enough, the last level is kept.

Directional changes are taken on the logarithm of the level with an absolute
threshold, so that a threshold of 0.25 means a rise or fall of about 28% and
rises and falls weigh the same. As for prices (``regime_fusion.regimes.market``),
the inputs are the log number of events confirmed in the last ``window``
sessions and the move of the current trend in threshold units. Every value uses
the sessions up to its own only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_fusion.dc import dc_indicators


def attention_level(articles: pd.Series, sessions: int) -> pd.Series:
    """1 + articles over the last ``sessions`` sessions (sessions without coverage left out)."""
    mean = articles.rolling(sessions, min_periods=1).mean()
    return (1.0 + sessions * mean).ffill().rename("attention_level")


def attention_inputs(articles: pd.Series, sessions: int, threshold: float, window: int) -> pd.DataFrame:
    """Event frequency and the current trend's move, from DC of the log attention level."""
    level = np.log(attention_level(articles, sessions)).dropna()
    frame = dc_indicators(level, threshold, mode="absolute", window=window, prefix="")
    frame = frame.reindex(articles.index)
    return pd.DataFrame({"log_events": np.log1p(frame["ndc"]), "trend_move": frame["tmv"]})


def market_articles(per_stock: dict[str, pd.Series]) -> pd.Series:
    """Articles about all the given stocks together; missing on sessions without coverage."""
    return pd.DataFrame(per_stock).sum(axis=1, min_count=1).rename("articles")
