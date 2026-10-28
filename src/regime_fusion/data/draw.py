"""Random draws of the study's stocks.

Main universe: ActTrade's US stocks plus a draw from the other usable January
2015 S&P 100 members. The draw brings the sector mix of the whole universe
close to the S&P 100's: each GICS sector's target is its share of the S&P 100
times the universe size, and the draws go to the sectors short of their
targets, in proportion to the shortfall (largest remainders, ties broken at
random). Within a sector, stocks are taken in a random order.

Hold-out stocks: a random order of the January 2015 S&P 500 members outside the
S&P 100 and the main universe, one share class per firm.

In both draws a stock that turns out to be unusable (no verified Yahoo prices
from the start of the study) is skipped and the next one in the random order
is taken. Random generators come from ``regime_fusion.seeds``, keyed by the
draw's name only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable

import numpy as np

MAIN_SIZE = 50      # docs/DECISIONS.md: about 50 stocks, ActTrade's US stocks first
HOLDOUT_SIZE = 50


def shuffled(ids: Iterable[str], rng: np.random.Generator) -> list[str]:
    """A random order of ``ids`` that depends only on the generator and the set of ids."""
    ordered = sorted(set(ids))
    return [ordered[i] for i in rng.permutation(len(ordered))]


def allocate(shares: dict[str, float], fixed: dict[str, int], size: int,
             available: dict[str, int], rng: np.random.Generator) -> dict[str, int]:
    """Number of stocks to draw per sector so that, with ``fixed`` stocks already in,
    the universe of ``size`` stocks comes close to the sector ``shares``."""
    sectors = sorted(shares)
    needed = size - sum(fixed.values())
    if needed < 0 or needed > sum(available.get(s, 0) for s in sectors):
        raise ValueError(f"cannot draw {needed} stocks from what is available")
    shortfall = {s: max(0.0, size * shares[s] - fixed.get(s, 0)) for s in sectors}
    weights = shortfall if sum(shortfall.values()) > 0 else dict(shares)
    total = sum(weights.values())
    quota = {s: needed * weights[s] / total for s in sectors}
    counts = {s: min(math.floor(quota[s]), available.get(s, 0)) for s in sectors}
    tiebreak = dict(zip(shuffled(sectors, rng), range(len(sectors))))
    order = sorted(sectors, key=lambda s: (-(quota[s] - math.floor(quota[s])), tiebreak[s]))
    while sum(counts.values()) < needed:  # hand out what is left, never beyond availability
        for sector in order:
            if sum(counts.values()) < needed and counts[sector] < available.get(sector, 0):
                counts[sector] += 1
    return counts


def take_usable(order: list[str], count: int, usable: Callable[[str], bool]) -> tuple[list[str], list[str]]:
    """The first ``count`` usable ids in ``order``, and the unusable ids skipped on the way."""
    taken, skipped = [], []
    for stock_id in order:
        if len(taken) == count:
            break
        (taken if usable(stock_id) else skipped).append(stock_id)
    if len(taken) < count:
        raise ValueError(f"only {len(taken)} usable stocks for {count} places")
    return taken, skipped


def draw_main(candidates: dict[str, str], fixed: dict[str, str], shares: dict[str, float],
              size: int, rng: np.random.Generator,
              usable: Callable[[str], bool]) -> tuple[list[str], list[str]]:
    """Main-universe draw. ``candidates`` and ``fixed`` map stock ids to sectors."""
    fixed_counts = {s: sum(v == s for v in fixed.values()) for s in shares}
    available = {s: sum(v == s for v in candidates.values()) for s in shares}
    counts = allocate(shares, fixed_counts, size, available, rng)
    drawn, skipped = [], []
    for sector in sorted(counts):
        order = shuffled([i for i, s in candidates.items() if s == sector], rng)
        taken, passed = take_usable(order, counts[sector], usable)
        drawn += taken
        skipped += passed
    return drawn, skipped


def ticker_candidates(ticker_2015: str, current: list[str]) -> list[str]:
    """Yahoo symbols to try for a firm, from its current SEC tickers: the January 2015
    ticker first, then shorter tickers (common shares rather than warrants); preferred
    shares (``-P`` suffixes such as CTA-PA) are left out."""
    yahoo_2015 = ticker_2015.replace(".", "-")
    common = {t for t in current if not re.search(r"-P[A-Z]?$", t)}
    return sorted(common, key=lambda t: (t != yahoo_2015, len(t), t))
