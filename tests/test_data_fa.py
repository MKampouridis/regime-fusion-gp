import numpy as np
import pandas as pd
import pytest

from regime_fusion.data import fa
from regime_fusion.data.calendar import TradingCalendar
from regime_fusion.data.prices import Correction
from regime_fusion.testing import check_no_lookahead

CAL = TradingCalendar.nyse("2020-07-01", "2020-12-31")
DAYS = CAL.sessions


def test_true_splits_leave_out_spin_offs():
    assert fa.is_true_split(4.0) and fa.is_true_split(1.5) and fa.is_true_split(0.125)
    assert not fa.is_true_split(2.376) and not fa.is_true_split(1.04)
    assert not fa.is_true_split(0.4725) and not fa.is_true_split(1.998)
    split = pd.Series(1.0, index=DAYS)
    split[pd.Timestamp("2020-08-31")] = 4.0          # a split
    split[pd.Timestamp("2020-10-01")] = 1.589        # a spin-off recorded as a split
    split[pd.Timestamp("2020-11-02")] = 0.4725       # a reverse split merged with a spin-off
    fix = Correction("X", pd.Timestamp("2020-11-02"), "spin-off", 1 / 3, {"Y": 1 / 3})
    ratios = fa.true_splits(split, [fix])
    assert ratios.to_dict() == {pd.Timestamp("2020-08-31"): 4.0, pd.Timestamp("2020-11-02"): 1 / 3}


def test_share_counts_follow_splits_after_their_filing_only():
    splits = pd.Series({pd.Timestamp("2020-08-31"): 4.0})
    shares = pd.Series(100.0, index=DAYS)
    filed = pd.Series(pd.Timestamp("2020-07-30"), index=DAYS)       # filed before the split
    filed[DAYS >= pd.Timestamp("2020-10-30")] = pd.Timestamp("2020-10-29")  # restated report
    shares[DAYS >= pd.Timestamp("2020-10-30")] = 400.0
    adjusted = fa.split_adjusted(shares, filed, splits)
    assert adjusted[pd.Timestamp("2020-08-28")] == 100.0
    assert adjusted[pd.Timestamp("2020-08-31")] == 400.0  # the old count, split-adjusted
    assert adjusted[pd.Timestamp("2020-11-02")] == 400.0  # the restated count, not split again


def make_panel():
    n = len(DAYS)
    panel = pd.DataFrame({
        "net_income_ttm": 50.0, "revenue_ttm": 200.0, "equity": 100.0, "current_assets": 80.0,
        "inventory": 20.0, "current_liabilities": 40.0, "long_term_debt": 30.0, "current_debt": np.nan,
        "operating_income_ttm": 60.0, "depreciation_amortization_ttm": 10.0,
        "operating_cash_flow_ttm": 70.0, "capital_expenditure_ttm": 25.0,
        "research_development_ttm": np.nan, "research_development_end": pd.NaT,
        "pretax_income_ttm": 55.0, "interest_expense_ttm": 5.0,
        "net_interest_income_ttm": np.nan, "noninterest_income_ttm": np.nan,
        "diluted_shares": 10.0, "diluted_shares_filed": pd.Timestamp("2020-06-30"),
    }, index=DAYS)
    panel.loc[DAYS[n // 2]:, "net_income_ttm"] = 60.0
    return panel


def test_indicators_follow_their_definitions():
    panel = make_panel()
    price = pd.Series(30.0, index=DAYS)
    table = fa.indicators(panel, price, pd.Series(dtype=float))
    first = table.iloc[0]
    assert first["net_profit_ratio"] == 0.25 and first["return_on_equity"] == 0.5
    assert first["quick_ratio"] == 1.5 and first["debt_equity"] == 0.3
    assert first["market_value"] == 300.0 and first["price_earnings"] == 6.0
    assert first["price_book"] == 3.0 and first["price_sales"] == 1.5
    assert first["ebitda"] == 70.0 and first["levered_fcf"] == 45.0 and first["eps_diluted"] == 5.0
    assert first["research_development"] == 0.0  # no R&D filed so far
    panel["equity"] = 0.0
    assert table.columns.tolist()[:12] == list(fa.INDICATORS)
    assert fa.indicators(panel, price, pd.Series(dtype=float))["price_book"].isna().all()


def test_indicators_never_look_ahead():
    panel = make_panel()
    panel["traded_close"] = 30.0 + np.arange(len(DAYS))
    panel["split"] = 1.0
    panel.loc[pd.Timestamp("2020-08-31"), "split"] = 4.0

    def compute(frame):
        splits = fa.true_splits(frame["split"], [])
        return fa.indicators(frame, frame["traded_close"], splits)

    check_no_lookahead(compute, panel, range(0, len(DAYS), 7))


def test_fallbacks_for_banks_and_firms_without_operating_income():
    panel = make_panel()
    panel["revenue_ttm"] = np.nan
    panel["net_interest_income_ttm"], panel["noninterest_income_ttm"] = 120.0, 80.0
    panel["operating_income_ttm"] = np.nan
    panel["research_development_ttm"] = np.nan
    panel["research_development_end"] = pd.Timestamp("2019-12-31")  # filed before, now unknown
    table = fa.indicators(panel, pd.Series(30.0, index=DAYS), pd.Series(dtype=float))
    first = table.iloc[0]
    assert first["revenue"] == 200.0 and first["net_profit_ratio"] == 0.25
    assert first["ebitda"] == 55.0 + 5.0 + 10.0
    assert np.isnan(first["research_development"])
