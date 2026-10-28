"""Market regimes: inputs from the S&P 500 index, fitted per walk-forward window.

Inputs (``configs/regimes.yaml``, a draft until the pilot):

volatility  rung A2: log annualised realised volatility of the index's daily log
            returns over the last ``window`` sessions
dc          rung A3: directional changes of the index at ``threshold``: the
            number of events confirmed in the last ``window`` sessions, as
            log(1 + count), and the total move of the current trend, in
            threshold units

For each test year, an HMM is fitted on each training range of the year's
window (``regime_fusion.windows``) and then run forward with fixed parameters:
stage 1 fits on the three training years and runs through the validation year;
stage 2 fits on all four years and runs through the test year. Test-day
probabilities thus come from a model that never saw the test year, and every
day's probabilities use only the days up to it (filtering).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regime_fusion.dc import dc_indicators
from regime_fusion.regimes import hmm
from regime_fusion.seeds import make_rng
from regime_fusion.windows import DateRange, Window

DEFAULT_SETTINGS = Path(__file__).resolve().parents[3] / "configs" / "regimes.yaml"
TRADING_DAYS = 252


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def realised_volatility(close: pd.Series, window: int) -> pd.Series:
    """Log annualised standard deviation of daily log returns over the last ``window`` sessions."""
    returns = np.log(close).diff()
    return np.log(returns.rolling(window).std() * np.sqrt(TRADING_DAYS)).rename("log_volatility")


def dc_inputs(close: pd.Series, threshold: float, window: int) -> pd.DataFrame:
    """Event frequency and the current trend's move, from causal DC indicators."""
    frame = dc_indicators(close.dropna(), threshold, window=window, prefix="").reindex(close.index)
    return pd.DataFrame({"log_events": np.log1p(frame["ndc"]), "trend_move": frame["tmv"]})


def inputs(close: pd.Series, name: str, settings: dict[str, Any]) -> pd.DataFrame:
    spec = settings["inputs"][name]
    if name == "volatility":
        return realised_volatility(close, int(spec["window"])).to_frame()
    if name == "dc":
        return dc_inputs(close, float(spec["threshold"]), int(spec["window"]))
    raise ValueError(f"unknown regime inputs {name!r}")


@dataclass(frozen=True)
class Fit:
    test_year: int
    stage: int                  # 1: fitted on the training years; 2: on all four
    fitted_on: DateRange
    model: hmm.GaussianHMM
    probabilities: pd.DataFrame  # filtered, from the start of the fit range to the end of the next period


def fit_window(features: pd.DataFrame, fitted_on: DateRange, through: pd.Timestamp,
               n_states: int, rng: np.random.Generator, restarts: int) -> tuple[hmm.GaussianHMM, pd.DataFrame]:
    """Fit on one range, then filter from its start to ``through`` with fixed parameters."""
    start, end = pd.Timestamp(fitted_on.start), pd.Timestamp(fitted_on.end)
    model = hmm.fit(features.loc[start:end].to_numpy(), n_states, rng, restarts=restarts)
    span = features.loc[start:through]
    probabilities = pd.DataFrame(hmm.filter(model, span.to_numpy()), index=span.index,
                                 columns=[f"p_state{k}" for k in range(n_states)])
    return model, probabilities


def walk_forward(features: pd.DataFrame, windows: list[Window], settings: dict[str, Any],
                 base_seed: int) -> list[Fit]:
    """Both stages of every window. Seeds depend on the test year and stage only."""
    fits = []
    for window in windows:
        stages = ((1, window.train, window.validation.end), (2, window.final_train, window.test.end))
        for stage, fitted_on, through in stages:
            rng = make_rng(base_seed, "regimes", window.test_year, stage)
            model, probabilities = fit_window(features, fitted_on, pd.Timestamp(through),
                                              int(settings["n_states"]), rng, int(settings["restarts"]))
            fits.append(Fit(window.test_year, stage, fitted_on, model, probabilities))
    return fits
