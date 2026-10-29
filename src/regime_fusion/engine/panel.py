"""Inputs of one GP run: a stock's indicators and prices over a window, scaled on its training part.

Indicators come from the processed data (``regime_fusion.data``): ActTrade's
fundamental (``processed/fa``) and technical (``processed/ta``) indicators,
sentiment indicators (``processed/sa``, once built) and, for the regime rungs,
the filtered regime probabilities of the window (``processed/regimes``). Names
carry their type: ``fa:``, ``sa:``, ``ta:`` and ``rg:``.

Scaling is fitted on the training rows only and then applied to every row:

* ``minmax`` maps the training minimum and maximum to -1 and 1, as ActTrade
  scaled its indicators (later values may fall outside [-1, 1]);
* ``rank`` maps each value to 2 F(x) - 1, F being the training rows' empirical
  distribution, which outliers (a price-earnings ratio of 1,000) cannot squash.

Regime probabilities p need no fitting: they enter as 2p - 1. An indicator with
values on fewer than half of the training rows is dropped for the run.

For the robustness check, FinBERT headline sentiment from one source
(``processed/finbert/<source>``) can join the sentiment branch: its indicators
are then named ``sa:finbert_...``, and the source must cover every year of the
window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_fusion.data import store
from regime_fusion.engine.fitness import Market
from regime_fusion.engine.tree import SOURCES
from regime_fusion.windows import DateRange

SCALINGS = ("minmax", "rank")
MIN_COVERAGE = 0.5
PRICE_COLUMNS = ("adj_open", "adj_high", "adj_low", "adj_close")
PROCESSED = {"FA": "fa", "SA": "sa", "TA": "ta"}


@dataclass(frozen=True)
class Panel:
    """Scaled indicators and prices of one stock over [first training day, last evaluation day]."""

    indicators: pd.DataFrame   # columns named "fa:...", "sa:...", "ta:...", "rg:..."
    prices: pd.DataFrame       # adjusted open, high, low, close
    train: pd.DatetimeIndex
    evaluate: pd.DatetimeIndex

    def names(self) -> dict[str, list[str]]:
        groups = {s: [] for s in (*SOURCES, "RG")}
        for column in self.indicators.columns:
            groups[column.split(":", 1)[0].upper()].append(column)
        return groups

    def arrays(self, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        frame = self.indicators.loc[days]
        return {column: frame[column].to_numpy(dtype=np.float64) for column in frame.columns}

    def market(self, days: pd.DatetimeIndex, exit_days: int, rate: float, cost_bps: float) -> Market:
        frame = self.prices.loc[days]
        return Market(*(frame[c].to_numpy(dtype=np.float64) for c in PRICE_COLUMNS),
                      days=exit_days, rate=rate, cost=cost_bps / 1e4)


def finbert_years(source: str, stock: str, root: Path | None = None) -> set[int]:
    """Years in which ``source``'s headlines cover ``stock`` (``regime_fusion.data.finbert``)."""
    path = store.layer("processed", "finbert", source, f"{stock}.parquet", root=root)
    if not path.exists():
        raise FileNotFoundError(f"{stock}: no FinBERT indicators from {source} at {path}")
    return set(json.loads(store.sidecar(path).read_text()).get("covered_years", []))


def load_panel(stock: str, train: DateRange, evaluate: DateRange, sources: tuple[str, ...],
               regimes: Path | None = None, scaling: str = "minmax", root: Path | None = None,
               finbert: str | None = None) -> Panel:
    """Indicators and prices of ``stock``; ``regimes`` is a processed regime-probability file;
    ``finbert`` (a headline source) adds its FinBERT indicators to the sentiment branch."""
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}")
    prices = pd.read_parquet(store.layer("interim", "prices", f"{stock}.parquet", root=root))
    sessions = prices.index[(prices.index >= pd.Timestamp(train.start)) & (prices.index <= pd.Timestamp(evaluate.end))]
    sessions = sessions[prices.loc[sessions, "adj_close"].notna()]
    train_days = sessions[sessions <= pd.Timestamp(train.end)]
    evaluate_days = sessions[sessions >= pd.Timestamp(evaluate.start)]

    frames = []
    for source in sources:
        path = store.layer("processed", PROCESSED[source], f"{stock}.parquet", root=root)
        if not path.exists():
            raise FileNotFoundError(f"{stock}: no {source} indicators at {path}")
        table = pd.read_parquet(path)
        frames.append(table.add_prefix(f"{source.lower()}:"))
    if finbert and "SA" in sources:
        missing = sorted(set(range(sessions[0].year, sessions[-1].year + 1)) - finbert_years(finbert, stock, root))
        if missing:
            raise ValueError(f"{stock}: the {finbert} headlines do not cover {missing}, so this window "
                             f"is not part of the FinBERT check")
        path = store.layer("processed", "finbert", finbert, f"{stock}.parquet", root=root)
        frames.append(pd.read_parquet(path).add_prefix("sa:"))
    raw = pd.concat(frames, axis=1).reindex(sessions)
    raw = raw.loc[:, [c for c in raw.columns if c.split(":", 1)[0] in {s.lower() for s in sources}
                      and not c.endswith(("market_value", "diluted_shares"))]]
    scaled = scale(raw, train_days, scaling)
    empty = [s for s in sources if not any(c.startswith(f"{s.lower()}:") for c in scaled.columns)]
    if empty:
        raise ValueError(f"{stock}: no {' or '.join(empty)} indicator has values on at least "
                         f"{MIN_COVERAGE:.0%} of the training days {train.start} to {train.end}")
    if regimes is not None:
        probabilities = pd.read_parquet(regimes).reindex(sessions)
        scaled = pd.concat([scaled, (2.0 * probabilities - 1.0).add_prefix("rg:")], axis=1)
    return Panel(scaled, prices.loc[sessions, list(PRICE_COLUMNS)], train_days, evaluate_days)


def scale(raw: pd.DataFrame, train_days: pd.DatetimeIndex, scaling: str) -> pd.DataFrame:
    """Scale every column with a map fitted on ``train_days``; thin columns are dropped."""
    fit = raw.loc[train_days]
    keep = [c for c in raw.columns if fit[c].notna().mean() >= MIN_COVERAGE and fit[c].nunique() > 1]
    out = {}
    for column in keep:
        known = fit[column].dropna().to_numpy()
        values = raw[column].to_numpy(dtype=np.float64)
        if scaling == "minmax":
            low, high = known.min(), known.max()
            out[column] = 2.0 * (values - low) / (high - low) - 1.0
        else:
            ordered = np.sort(known)
            share = np.searchsorted(ordered, values, side="right") / len(ordered)
            out[column] = np.where(np.isnan(values), np.nan, 2.0 * share - 1.0)
    return pd.DataFrame(out, index=raw.index)
