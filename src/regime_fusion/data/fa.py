"""ActTrade's twelve fundamental indicators, daily and point in time.

Built from the daily accounting values (``regime_fusion.data.fundamentals``) and
the day's traded price (``regime_fusion.data.prices``); on day t every input is
the one known at the close of t.

====================  ===================================================
net_profit_ratio      net income / revenue (trailing twelve months)
return_on_equity      net income / shareholders' equity
quick_ratio           (current assets - inventory) / current liabilities
price_earnings        market value / net income
price_book            market value / equity
price_sales           market value / revenue
debt_equity           (long-term debt + current debt) / equity
revenue               revenue
ebitda                operating income + depreciation and amortisation
levered_fcf           operating cash flow - capital expenditure
eps_diluted           net income / diluted shares
research_development  research and development expenses
====================  ===================================================

Fallbacks, each using only what is known on the day: a bank's revenue is its net
interest income plus non-interest income; a firm without an operating-income line
(oil majors, large drug makers, IBM) gets EBITDA from pre-tax income plus interest
expense; and R&D is zero for a firm that has filed no R&D figure so far.

Market value is the traded close times the diluted share count, which covers
every share class (firms such as Estee Lauder file their cover-page share counts
per class). Reports filed after a stock split restate their share counts, so a
count is multiplied only by the splits after its filing date. Only true splits
count: not the spin-offs that Yahoo records as fractional splits, but DowDuPont's
1-for-3 reverse split, which Yahoo merged with the Corteva spin-off, comes from
``configs/price_corrections.yaml``.

Values are raw: normalisation is fitted on training windows later. Ratios with a
zero denominator are missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_fusion.data.prices import Correction

INDICATORS = (
    "net_profit_ratio", "return_on_equity", "quick_ratio", "price_earnings", "price_book",
    "price_sales", "debt_equity", "revenue", "ebitda", "levered_fcf", "eps_diluted",
    "research_development",
)


def is_true_split(ratio: float) -> bool:
    """Whether a Yahoo split ratio is a whole-share split such as 4, 3/2 or 1/8."""
    return any(abs(ratio * b - round(ratio * b)) < 1e-6 and round(ratio * b) <= 100
               for b in range(1, 21))


def true_splits(split: pd.Series, corrections: list[Correction]) -> pd.Series:
    """Ratios that change the share count, on their ex-dates.

    ``split`` is the aligned price table's split column (1 when there is none).
    """
    corrected = {c.date: c.kept for c in corrections}
    ratios = {day: ratio for day, ratio in split[split != 1.0].items()
              if day not in corrected and is_true_split(ratio)}
    ratios.update({day: kept for day, kept in corrected.items()
                   if kept != 1.0 and day in split.index})
    return pd.Series(ratios, dtype=float).sort_index()


def split_adjusted(shares: pd.Series, filed: pd.Series, splits: pd.Series) -> pd.Series:
    """Share counts on each day, times every split after the filing of the count."""
    days = shares.index.to_numpy()
    if splits.empty:
        return shares.astype(float)
    cumulative = np.concatenate([[1.0], splits.cumprod().to_numpy()])
    ex_dates = splits.index.to_numpy()
    upto_day = cumulative[np.searchsorted(ex_dates, days, side="right")]
    filed_dates = pd.to_datetime(filed).to_numpy()
    upto_filed = cumulative[np.searchsorted(ex_dates, filed_dates, side="right")]
    factor = np.where(pd.isna(filed_dates), np.nan, upto_day / upto_filed)
    return shares.astype(float) * factor


def indicators(panel: pd.DataFrame, traded_close: pd.Series, splits: pd.Series) -> pd.DataFrame:
    """The twelve indicators, plus market value and the share count used, per session."""
    shares = split_adjusted(panel["diluted_shares"], panel["diluted_shares_filed"], splits)
    value = traded_close.reindex(panel.index) * shares
    income, equity = panel["net_income_ttm"], panel["equity"]
    revenue = panel["revenue_ttm"].fillna(panel["net_interest_income_ttm"] + panel["noninterest_income_ttm"])
    operating = panel["operating_income_ttm"].fillna(panel["pretax_income_ttm"] + panel["interest_expense_ttm"])
    research = panel["research_development_ttm"].where(panel["research_development_end"].notna(), 0.0)
    debt = panel[["long_term_debt", "current_debt"]].sum(axis=1, min_count=1)
    out = pd.DataFrame(
        {
            "net_profit_ratio": income / revenue,
            "return_on_equity": income / equity,
            "quick_ratio": (panel["current_assets"] - panel["inventory"].fillna(0.0))
            / panel["current_liabilities"],
            "price_earnings": value / income,
            "price_book": value / equity,
            "price_sales": value / revenue,
            "debt_equity": debt / equity,
            "revenue": revenue,
            "ebitda": operating + panel["depreciation_amortization_ttm"],
            "levered_fcf": panel["operating_cash_flow_ttm"] - panel["capital_expenditure_ttm"],
            "eps_diluted": income / shares,
            "research_development": research,
            "market_value": value,
            "diluted_shares": shares,
        },
        index=panel.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)
