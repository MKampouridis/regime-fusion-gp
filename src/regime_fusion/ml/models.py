"""Models of the machine-learning benchmarks, and the settings of all benchmarks (``configs/benchmarks.yaml``).

Each model predicts the probability that the trade a buy signal would open
gains (``regime_fusion.ml.labels``):

* ``logistic``: L2-regularised logistic regression; missing values enter as 0,
  the middle of the scaled range;
* ``lightgbm``: gradient-boosted trees (LightGBM), which handle missing values;
* ``regime_lightgbm``: one LightGBM per regime (the most probable state of the
  filtered regime probabilities), trained on that regime's training days; a
  regime with too few days uses the model trained on all of them;
* ``gated``: neural gated fusion (``regime_fusion.ml.neural``);
* ``tft``: the Temporal Fusion Transformer, one model across all stocks
  (``regime_fusion.ml.tft``, with its own runner).

A training set with a single class gives a constant prediction.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from regime_fusion import provenance

DEFAULT_SETTINGS = Path(__file__).resolve().parents[3] / "configs" / "benchmarks.yaml"
KINDS = ("logistic", "lightgbm", "regime_lightgbm", "gated", "tft")
POOLED = ("tft",)  # kinds trained once across all stocks
RULE_KINDS = ("buy_and_hold", "always", "moving_average", "momentum")
BOOSTING = {"min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
            "reg_lambda": 1.0, "deterministic": True, "force_row_wise": True, "n_jobs": 1, "verbose": -1}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    runs: int
    grid: dict[str, list]
    min_rows: int = 100

    @property
    def pooled(self) -> bool:
        return self.kind in POOLED

    def candidates(self) -> list[dict[str, Any]]:
        """Every combination of the grid, in a fixed order."""
        keys = sorted(self.grid)
        return [dict(zip(keys, values)) for values in itertools.product(*(self.grid[k] for k in keys))]


@dataclass(frozen=True)
class Settings:
    sources: tuple[str, ...]
    regimes: str | None
    scaling: str
    quantiles: tuple[float, ...]
    min_trades: int
    models: dict[str, ModelSpec]
    rules: dict[str, ModelSpec]
    raw: dict[str, Any]

    def methods(self) -> dict[str, ModelSpec]:
        """Every benchmark that runs like a rung: rules and models."""
        return {**self.rules, **self.models}

    def digest(self) -> str:
        return provenance.digest(self.raw)

    def method_digest(self, name: str) -> str:
        """Digest of one benchmark's settings: its own entry and those every benchmark shares, so
        that changing another benchmark leaves it alone."""
        shared = {k: v for k, v in self.raw.items() if k not in ("models", "rules")}
        own = {**self.raw.get("rules", {}), **self.raw["models"]}.get(name)
        if own is None:
            raise ValueError(f"no benchmark {name!r}")
        return provenance.digest({"name": name, "own": own, "shared": shared})


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> Settings:
    raw = yaml.safe_load(Path(path).read_text())
    models, rules = {}, {}
    for name, entry in raw["models"].items():
        if entry["kind"] not in KINDS:
            raise ValueError(f"{name}: kind must be one of {KINDS}")
        models[name] = ModelSpec(name, entry["kind"], int(entry["runs"]), dict(entry["grid"]),
                                 int(entry.get("min_rows", 100)))
    for name, entry in raw.get("rules", {}).items():
        if entry["kind"] not in RULE_KINDS:
            raise ValueError(f"{name}: kind must be one of {RULE_KINDS}")
        rules[name] = ModelSpec(name, entry["kind"], 1, dict(entry.get("grid", {})))
    features = raw["features"]
    return Settings(tuple(features["sources"]), features.get("regimes"), features["scaling"],
                    tuple(float(q) for q in raw["quantiles"]), int(raw["min_trades"]), models, rules, raw)


def build(spec: ModelSpec, params: dict[str, Any], seed: int, columns: list[str] | None = None):
    """A learner; ``columns`` (the feature names) tells the gated network which inputs are which."""
    if spec.kind == "logistic":
        return Logistic(**params)
    if spec.kind == "lightgbm":
        return Boosting(seed, **params)
    if spec.kind == "gated":
        from regime_fusion.ml.neural import Gated

        return Gated(seed, list(columns or []), **params)
    if spec.pooled:
        raise ValueError(f"{spec.name} is trained across all stocks; use regime_fusion.ml.tft.run_tft")
    return RegimeBoosting(seed, spec.min_rows, **params)


class Constant:
    """The training share of gaining trades, for a training set with one class."""

    def __init__(self, y: np.ndarray):
        self.rate = float(np.mean(y)) if len(y) else 0.0

    def predict(self, x: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
        return np.full(len(x), self.rate)


class Logistic:
    def __init__(self, C: float):
        self.C = C

    def fit(self, x: np.ndarray, y: np.ndarray, states: np.ndarray | None = None):
        from sklearn.linear_model import LogisticRegression

        if len(np.unique(y)) < 2:
            self.model = Constant(y)
            return self
        self.model = LogisticRegression(C=self.C, max_iter=5000).fit(np.nan_to_num(x), y)
        return self

    def predict(self, x: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
        if isinstance(self.model, Constant):
            return self.model.predict(x)
        return self.model.predict_proba(np.nan_to_num(x))[:, 1]


class Boosting:
    def __init__(self, seed: int, **params):
        self.seed, self.params = seed, params

    def fit(self, x: np.ndarray, y: np.ndarray, states: np.ndarray | None = None):
        from lightgbm import LGBMClassifier

        if len(np.unique(y)) < 2:
            self.model = Constant(y)
            return self
        self.model = LGBMClassifier(**BOOSTING, **self.params, random_state=self.seed).fit(x, y)
        return self

    def predict(self, x: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
        if isinstance(self.model, Constant):
            return self.model.predict(x)
        return self.model.predict_proba(x)[:, 1]


class RegimeBoosting:
    def __init__(self, seed: int, min_rows: int, **params):
        self.seed, self.min_rows, self.params = seed, min_rows, params

    def fit(self, x: np.ndarray, y: np.ndarray, states: np.ndarray | None = None):
        if states is None:
            raise ValueError("a regime-switching model needs the regime of every row")
        self.pooled = Boosting(self.seed, **self.params).fit(x, y)
        self.by_state = {}
        for state in np.unique(states[states >= 0]):
            rows = states == state
            if rows.sum() >= self.min_rows and len(np.unique(y[rows])) == 2:
                self.by_state[int(state)] = Boosting(self.seed, **self.params).fit(x[rows], y[rows])
        return self

    def predict(self, x: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
        out = self.pooled.predict(x)
        for state, model in self.by_state.items():
            rows = states == state
            if rows.any():
                out[rows] = model.predict(x[rows])
        return out
