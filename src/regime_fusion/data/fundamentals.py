"""Point-in-time accounting values from filed XBRL facts.

Rules (see also docs/DATA.md):

* A fact may be used from the first session after its filing date.
* On day t, a period's value is the one in the latest filing available by t. A
  later filing that restates the period replaces it only from its own
  availability on.
* Balance-sheet items (instants) take the value at the latest period end known on t;
  so do period averages such as the diluted share count (kind "latest"), which
  take the shortest period ending there (the quarter rather than the year).
  Both also report the filing date of the value used, so that later stock splits
  can be applied to share counts.
* Income and cash-flow items (flows) take the trailing twelve months (TTM) ending
  at the latest period end known on t, and the TTM one year earlier, for growth
  rates. 10-Qs report the year to date and 10-Ks the fiscal year, so

      TTM = the fiscal-year value, if one is filed for that period end;
          = previous fiscal year + year to date - the same span a year earlier;
          = the sum of four filed quarters, as a last resort.

* A concept is a list of tags in order of preference, because firms change tags
  (Apple reported revenue as SalesRevenueNet until 2018, then as
  RevenueFromContractWithCustomerExcludingAssessedTax). Within one period, the
  latest filing wins and the preference order breaks ties.

Span lengths allow for 52/53-week years and for firms with 12-week quarters and a
16-week fourth quarter (PepsiCo, Costco).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_fusion.config import ConfigError

DEFAULT_CONCEPTS = Path(__file__).resolve().parents[3] / "configs" / "fundamentals.yaml"
KINDS = ("flow", "instant", "latest")
QUARTER = (80, 115)        # days, both inclusive
YEAR_TO_DATE = (80, 290)   # a quarter, a half year or nine months
YEAR = (350, 380)
TOLERANCE = pd.Timedelta(days=7)  # for matching period ends across years
ONE_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Concept:
    name: str
    kind: str                 # "flow", "instant" or "latest"
    tags: tuple[str, ...]     # "taxonomy:concept", in order of preference
    unit: str = "USD"

    @property
    def columns(self) -> list[str]:
        if self.kind == "flow":
            return [f"{self.name}_ttm", f"{self.name}_ttm_prev", f"{self.name}_end"]
        return [self.name, f"{self.name}_end", f"{self.name}_filed"]


def load_concepts(path: str | Path = DEFAULT_CONCEPTS) -> list[Concept]:
    raw = yaml.safe_load(Path(path).read_text())
    try:
        concepts = [
            Concept(name, str(spec["kind"]), tuple(str(tag) for tag in spec["tags"]),
                    str(spec.get("unit", "USD")))
            for name, spec in raw["concepts"].items()
        ]
    except (KeyError, TypeError, AttributeError) as problem:
        raise ConfigError(f"{path}: bad concept list ({problem})") from None
    for concept in concepts:
        if concept.kind not in KINDS or not concept.tags:
            raise ConfigError(f"{path}: {concept.name} needs a kind in {KINDS} and tags")
        if any(tag.count(":") != 1 for tag in concept.tags):
            raise ConfigError(f"{path}: {concept.name}: tags look like 'us-gaap:Assets'")
    return concepts


def asof(facts: pd.DataFrame, concept: Concept, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily values of one concept as known at the close of each session.

    ``facts`` holds filed facts with an ``available`` column (first session of use).
    """
    rows = _candidates(facts, concept)
    dates = pd.DatetimeIndex(sorted(rows["available"].dropna().unique()))
    value = {"flow": _flow_values, "instant": _instant_values, "latest": _latest_values}[concept.kind]
    records = [value(rows[rows["available"] <= day]) for day in dates]
    known = pd.DataFrame(records, index=dates, columns=concept.columns)
    dates_columns = [c for c in concept.columns if c.endswith(("_end", "_filed"))]
    numbers = [column for column in concept.columns if column not in dates_columns]
    known[numbers] = known[numbers].astype(float)
    for column in dates_columns:
        known[column] = pd.to_datetime(known[column])
    daily = known.reindex(known.index.union(sessions)).ffill().reindex(sessions)
    daily.index.name = "session"
    return daily


def daily_panel(facts: pd.DataFrame, concepts: list[Concept], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """All concepts side by side, one row per session."""
    return pd.concat([asof(facts, concept, sessions) for concept in concepts], axis=1)


def ttm(periods: pd.DataFrame, end: pd.Timestamp) -> float:
    """Trailing twelve months ending at ``end``, from one value per (start, end) period."""
    at_end = periods[periods["end"] == end]
    year = at_end[at_end["length"].between(*YEAR)]
    if len(year):
        return float(year["value"].iloc[0])

    to_date = at_end[at_end["length"].between(*YEAR_TO_DATE)]
    if len(to_date):
        current = to_date.loc[to_date["length"].idxmax()]
        fiscal_start = current["start"]
        previous_year = periods[
            periods["length"].between(*YEAR)
            & ((periods["end"] - (fiscal_start - ONE_DAY)).abs() <= pd.Timedelta(days=3))
        ]
        a_year_earlier = periods[
            ((periods["end"] - (end - pd.DateOffset(years=1))).abs() <= TOLERANCE)
            & ((periods["length"] - current["length"]).abs() <= 7)
        ]
        if len(previous_year) and len(a_year_earlier):
            return float(previous_year["value"].iloc[0] + current["value"]
                         - a_year_earlier["value"].iloc[0])

    quarters = periods[periods["length"].between(*QUARTER)]
    total, quarter_end = 0.0, end
    for _ in range(4):
        match = quarters[(quarters["end"] - quarter_end).abs() <= pd.Timedelta(days=3)]
        if match.empty:
            return np.nan
        total += float(match["value"].iloc[0])
        quarter_end = match["start"].iloc[0] - ONE_DAY
    return total


def latest_values(known: pd.DataFrame) -> pd.DataFrame:
    """One row per period: the latest available filing's value, tag preference breaking ties."""
    ordered = known.sort_values(
        ["available", "rank", "accn", "value"], ascending=[False, True, False, True], kind="mergesort"
    )
    return ordered.drop_duplicates(["start", "end"], keep="first")


def _candidates(facts: pd.DataFrame, concept: Concept) -> pd.DataFrame:
    tags = facts["taxonomy"] + ":" + facts["concept"]
    rank = tags.map({tag: i for i, tag in enumerate(concept.tags)})
    rows = facts.assign(rank=rank)[rank.notna() & (facts["unit"] == concept.unit)]
    if concept.kind == "instant":
        return rows[rows["start"].isna()]
    rows = rows[rows["start"].notna()]
    return rows.assign(length=(rows["end"] - rows["start"]).dt.days + 1)


def _instant_values(known: pd.DataFrame) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    periods = latest_values(known)
    if periods.empty:
        return np.nan, pd.NaT, pd.NaT
    latest = periods.loc[periods["end"].idxmax()]
    return float(latest["value"]), latest["end"], latest["filed"]


def _latest_values(known: pd.DataFrame) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    periods = latest_values(known)
    periods = periods[periods["length"].between(QUARTER[0], YEAR[1])]
    if periods.empty:
        return np.nan, pd.NaT, pd.NaT
    at_end = periods[periods["end"] == periods["end"].max()]
    shortest = at_end.loc[at_end["length"].idxmin()]
    return float(shortest["value"]), shortest["end"], shortest["filed"]


def _flow_values(known: pd.DataFrame) -> tuple[float, float, pd.Timestamp]:
    periods = latest_values(known)
    periods = periods[periods["length"].between(QUARTER[0], YEAR[1])]
    for end in sorted(periods["end"].unique(), reverse=True)[:4]:
        end = pd.Timestamp(end)
        value = ttm(periods, end)
        if np.isnan(value):
            continue
        earlier = periods["end"][(periods["end"] - (end - pd.DateOffset(years=1))).abs() <= TOLERANCE]
        previous = ttm(periods, pd.Timestamp(earlier.max())) if len(earlier) else np.nan
        return value, previous, end
    return np.nan, np.nan, pd.NaT


def coverage(panel: pd.DataFrame, concepts: list[Concept], sessions: pd.DatetimeIndex,
             *, min_share: float = 0.9, max_age_days: int = 200) -> tuple[dict, list[str]]:
    """Per concept over ``sessions``: the share of days with a value and the age of the
    latest period (days since its end), with flags for thin or stale concepts."""
    rows = panel.reindex(sessions)
    summary, flags = {}, []
    for concept in concepts:
        value = rows[concept.columns[0]]
        age = (pd.Series(rows.index, index=rows.index) - rows[f"{concept.name}_end"]).dt.days
        summary[concept.name] = {
            "share": round(float(value.notna().mean()), 4),
            "median_age_days": None if age.isna().all() else float(age.median()),
            "max_age_days": None if age.isna().all() else int(age.max()),
        }
        if summary[concept.name]["share"] < min_share:
            flags.append(f"{concept.name}: values on {summary[concept.name]['share']:.0%} of days")
        elif summary[concept.name]["max_age_days"] > max_age_days:
            flags.append(f"{concept.name}: latest period up to {summary[concept.name]['max_age_days']} days old")
    return summary, flags
