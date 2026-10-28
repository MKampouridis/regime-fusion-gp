import pandas as pd
import pytest

from regime_fusion.data import market
from regime_fusion.data.calendar import TradingCalendar

CAL = TradingCalendar.nyse("2015-03-02", "2015-03-09")  # Monday to Monday

VIX_CSV = b"""DATE,OPEN,HIGH,LOW,CLOSE
02/27/2015,13.81,14.04,13.21,13.34
03/02/2015,13.60,13.78,12.90,13.04
03/03/2015,13.31,14.18,13.30,13.86
03/05/2015,14.25,14.43,13.47,14.04
03/06/2015,14.22,15.68,13.98,15.20
03/07/2015,15.20,15.20,15.20,15.20
03/09/2015,15.32,15.67,14.73,15.06
"""


def test_vix_file_is_parsed_with_dates():
    vix = market.parse_vix(VIX_CSV)
    assert list(vix.columns) == ["open", "high", "low", "close"]
    assert vix.loc[pd.Timestamp("2015-03-02"), "close"] == 13.04


def test_vix_is_aligned_to_sessions_with_counts():
    aligned, counts = market.align_vix(market.parse_vix(VIX_CSV), CAL)
    assert aligned.index.equals(CAL.sessions)
    assert counts == {"off_calendar_rows": 1, "missing_sessions": 1, "non_positive": 0}
    assert pd.isna(aligned.loc[pd.Timestamp("2015-03-04"), "close"])


def test_unexpected_vix_layout_is_rejected():
    with pytest.raises(ValueError, match="columns"):
        market.parse_vix(b"Date,Close\n03/02/2015,13.04\n")
