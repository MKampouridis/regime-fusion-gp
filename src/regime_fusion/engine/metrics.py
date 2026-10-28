"""Performance measures on daily returns; years have 252 sessions.

The primary metric is the annualised Sharpe ratio of daily returns, days without
a position counting as zero (docs/RESEARCH_PLAN.md). No risk-free rate is
subtracted. A strategy that never trades has a Sharpe ratio of zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_fusion.engine.backtest import Result

DAYS_PER_YEAR = 252


def sharpe(daily: pd.Series) -> float:
    returns = np.asarray(daily, dtype=float)
    if returns.size < 2:
        return 0.0
    spread = returns.std(ddof=1)
    return 0.0 if spread == 0 else float(returns.mean() / spread * np.sqrt(DAYS_PER_YEAR))


def total_return(daily: pd.Series) -> float:
    return float(np.prod(1.0 + np.asarray(daily, dtype=float)) - 1.0)


def annual_return(daily: pd.Series) -> float:
    n = len(daily)
    return 0.0 if n == 0 else float((1.0 + total_return(daily)) ** (DAYS_PER_YEAR / n) - 1.0)


def max_drawdown(daily: pd.Series) -> float:
    """Largest fall of the equity curve from its previous peak, as a positive fraction."""
    equity = np.cumprod(1.0 + np.asarray(daily, dtype=float))
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float(np.max(1.0 - equity / peak))


def cvar(daily: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: the mean loss on the worst (1 - level) share of days, as a positive fraction."""
    returns = np.sort(np.asarray(daily, dtype=float))  # worst first
    if returns.size == 0:
        return 0.0
    worst = max(1, int(np.ceil(round((1.0 - level) * returns.size, 9))))  # 0.05 * 20 is not exactly 1
    return float(-returns[:worst].mean())


def summary(result: Result) -> dict[str, float]:
    """The measures reported for one backtest."""
    daily, trades = result.daily, result.trades
    sessions = len(daily)
    years = sessions / DAYS_PER_YEAR if sessions else np.nan
    days_in = sum(daily.index.get_loc(e) - daily.index.get_loc(s) + 1
                  for s, e in zip(trades["entry"], trades["exit"]))
    return {
        "sharpe": sharpe(daily),
        "total_return": total_return(daily),
        "annual_return": annual_return(daily),
        "max_drawdown": max_drawdown(daily),
        "cvar_95": cvar(daily),
        "trades": float(len(trades)),
        "trades_per_year": len(trades) / years if sessions else 0.0,
        "exposure": days_in / sessions if sessions else 0.0,
        "hit_rate": float((trades["net_return"] > 0).mean()) if len(trades) else np.nan,
    }
