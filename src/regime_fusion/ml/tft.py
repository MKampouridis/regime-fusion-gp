"""Temporal Fusion Transformer (ML-TFT): one model across all stocks.

After Lim, Arık, Loeff and Pfister (2021), cut down to one-step classification.
For each stock and day, the model reads the last ``sequence`` sessions of the
stock's indicators (the GP's, scaled per stock on its training rows) and regime
probabilities, plus the stock's identity. It predicts the log-odds that the
trade a buy signal would open gains (labels as for the other benchmarks,
``regime_fusion.ml.labels``, each stock with its own exit settings).

Kept from the TFT:

* static covariate encoders, from a learned embedding of the stock, that
  condition variable selection, the LSTM's initial state and static enrichment;
* variable selection networks and gated residual networks (GRNs) with gated
  skip connections;
* an LSTM encoder;
* interpretable multi-head attention (the heads share the value projection),
  masked so that a day sees only itself and earlier days.

Simplified: inside variable selection one GRN serves all variables (the paper
gives each its own), and there are no known future inputs and no quantile
outputs, only one classification output for the last day. The
variable-selection weights, summed by analysis type, show which type the model
trusts and when: the neural counterpart of the GP's trust map.

Protocol, per test year, stage and run, pooled over the stocks: stage 1 trains
each candidate of the grid once on all stocks' training years, gives each stock
the threshold quantile with its best validation Sharpe ratio, and keeps the
candidate with the highest median of those ratios over stocks. Stage 2 retrains
the winner on the four years before the test year; each stock keeps its
quantile. Missing indicator values enter as 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from regime_fusion import provenance
from regime_fusion.config import StudyConfig
from regime_fusion.engine import metrics
from regime_fusion.engine.backtest import backtest
from regime_fusion.engine.panel import Panel, load_panel
from regime_fusion.engine.run import RunSpec, pack_signal, regimes_file, result_path
from regime_fusion.ml.labels import trade_returns
from regime_fusion.ml.models import ModelSpec, Settings
from regime_fusion.seeds import make_rng
from regime_fusion.windows import walk_forward

SOURCES = ("FA", "SA", "TA", "RG")
DEFAULTS = {"d_model": 16, "dropout": 0.1, "heads": 4, "sequence": 20, "epochs": 10, "batch": 256,
            "learning_rate": 1e-3}


def _torch():
    import torch

    torch.set_num_threads(1)
    return torch


# Data -------------------------------------------------------------------------------------------
@dataclass
class Pooled:
    """All stocks' scaled indicators over one window, in a common column order."""

    stocks: list[str]
    columns: list[str]
    panels: dict[str, Panel]
    features: np.ndarray            # all stocks' rows, one after another, missing values as 0
    offsets: dict[str, int]         # first row of each stock in ``features``
    exits: dict[str, tuple[int, float]]

    def rows(self, stock: str, days: pd.DatetimeIndex) -> np.ndarray:
        """Positions in ``features`` of a stock's days."""
        sessions = self.panels[stock].indicators.index
        return self.offsets[stock] + sessions.get_indexer(days)


def assemble(stocks: list[str], train, evaluate, settings: Settings, test_year: int, stage: int,
             exits: dict[str, tuple[int, float]], scheme: str = "rolling") -> Pooled:
    panels = {s: load_panel(s, train, evaluate, settings.sources,
                            regimes_file(settings.regimes, s, test_year, stage, scheme), settings.scaling)
              for s in stocks}
    columns = sorted({c for panel in panels.values() for c in panel.indicators.columns})
    blocks, offsets, first = [], {}, 0
    for stock in stocks:
        frame = panels[stock].indicators.reindex(columns=columns)
        offsets[stock] = first
        first += len(frame)
        blocks.append(np.nan_to_num(frame.to_numpy(dtype=np.float32)))
    return Pooled(stocks, columns, panels, np.concatenate(blocks), offsets, exits)


def windows(pooled: Pooled, positions: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Sequences of the last ``length`` rows up to each position, zero before each stock's first row."""
    steps = positions[:, None] - np.arange(length - 1, -1, -1)[None, :]
    inside = steps >= starts[:, None]
    out = pooled.features[np.where(inside, steps, 0)]
    out[~inside] = 0.0
    return out


def training_samples(pooled: Pooled, cost_bps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positions, stock indices, first rows and labels of every labelled training day."""
    positions, owners, starts, labels = [], [], [], []
    for k, stock in enumerate(pooled.stocks):
        panel = pooled.panels[stock]
        days, rate = pooled.exits[stock]
        outcome = trade_returns(panel.prices.loc[panel.train], days, rate, cost_bps)
        known = outcome.notna().to_numpy()
        rows = pooled.rows(stock, panel.train)[known]
        positions.append(rows)
        owners.append(np.full(len(rows), k))
        starts.append(np.full(len(rows), pooled.offsets[stock]))
        labels.append((outcome.to_numpy()[known] > 0).astype(np.float32))
    return tuple(np.concatenate(part) for part in (positions, owners, starts, labels))


# Model ------------------------------------------------------------------------------------------
def build_network(n_variables: int, n_stocks: int, d_model: int, heads: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class GRN(nn.Module):
        """Gated residual network: ELU layer, optional context, GLU gate, skip and layer norm."""

        def __init__(self, d_in, d_out, d_context=None):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_model)
            self.context = nn.Linear(d_context, d_model, bias=False) if d_context else None
            self.fc2 = nn.Linear(d_model, d_model)
            self.drop = nn.Dropout(dropout)
            self.gate = nn.Linear(d_model, 2 * d_out)
            self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            self.norm = nn.LayerNorm(d_out)

        def forward(self, x, context=None):
            hidden = self.fc1(x)
            if self.context is not None and context is not None:
                hidden = hidden + self.context(context)
            hidden = self.drop(self.fc2(nn.functional.elu(hidden)))
            return self.norm(self.skip(x) + nn.functional.glu(self.gate(hidden), dim=-1))

    class GatedSkip(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(d_model, 2 * d_model)
            self.norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, skip):
            return self.norm(skip + nn.functional.glu(self.gate(self.drop(x)), dim=-1))

    class TFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.stock = nn.Embedding(n_stocks, d_model)
            self.static = nn.ModuleList([GRN(d_model, d_model) for _ in range(4)])
            self.embed_weight = nn.Parameter(torch.randn(n_variables, d_model) * 0.1)
            self.embed_bias = nn.Parameter(torch.zeros(n_variables, d_model))
            self.select = GRN(n_variables * d_model, n_variables, d_context=d_model)
            self.variable = GRN(d_model, d_model)
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
            self.after_lstm = GatedSkip()
            self.enrich = GRN(d_model, d_model, d_context=d_model)
            self.head_size = max(d_model // heads, 1)
            self.queries = nn.Linear(d_model, heads * self.head_size)
            self.keys = nn.Linear(d_model, heads * self.head_size)
            self.values = nn.Linear(d_model, self.head_size)
            self.attention_out = nn.Linear(self.head_size, d_model)
            self.heads = heads
            self.after_attention = GatedSkip()
            self.positionwise = GRN(d_model, d_model)
            self.final = GatedSkip()
            self.output = nn.Linear(d_model, 1)

        def forward(self, x, stock):
            batch, steps, _ = x.shape
            static = self.stock(stock)
            c_select, c_enrich, c_h, c_c = (encoder(static) for encoder in self.static)
            embedded = x.unsqueeze(-1) * self.embed_weight + self.embed_bias       # (B, T, V, d)
            context = c_select.unsqueeze(1).expand(batch, steps, -1)
            weights = torch.softmax(self.select(embedded.flatten(-2), context), dim=-1)  # (B, T, V)
            selected = (weights.unsqueeze(-1) * self.variable(embedded)).sum(dim=-2)     # (B, T, d)
            encoded, _ = self.lstm(selected, (c_h.unsqueeze(0).contiguous(), c_c.unsqueeze(0).contiguous()))
            temporal = self.after_lstm(encoded, selected)
            enriched = self.enrich(temporal, c_enrich.unsqueeze(1).expand(batch, steps, -1))
            q = self.queries(enriched).view(batch, steps, self.heads, self.head_size).transpose(1, 2)
            k = self.keys(enriched).view(batch, steps, self.heads, self.head_size).transpose(1, 2)
            v = self.values(enriched).unsqueeze(1)                                    # shared by the heads
            scores = q @ k.transpose(-1, -2) / self.head_size ** 0.5
            future = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)
            attention = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
            attended = self.attention_out((attention @ v).mean(dim=1))
            mixed = self.after_attention(attended, enriched)
            out = self.final(self.positionwise(mixed), temporal)
            return self.output(out[:, -1]).squeeze(-1), weights[:, -1]

    return TFT()


class Trained:
    """A trained TFT and what it needs to predict."""

    def __init__(self, network, sequence: int):
        self.network, self.sequence = network, sequence

    def predict(self, pooled: Pooled, stock_index: int, positions: np.ndarray, start: int,
                batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """Probabilities and variable-selection weights for one stock's positions."""
        torch = _torch()
        probabilities, weights = [], []
        with torch.no_grad():
            for first in range(0, len(positions), batch):
                rows = positions[first:first + batch]
                x = torch.tensor(windows(pooled, rows, np.full(len(rows), start), self.sequence))
                logits, w = self.network(x, torch.full((len(rows),), stock_index, dtype=torch.long))
                probabilities.append(torch.sigmoid(logits).numpy())
                weights.append(w.numpy())
        if not probabilities:
            return np.zeros(0), np.zeros((0, pooled.features.shape[1]))
        return np.concatenate(probabilities).astype(np.float64), np.concatenate(weights)


def train(pooled: Pooled, params: dict[str, Any], seed: int, cost_bps: float) -> Trained:
    torch = _torch()
    settings = {**DEFAULTS, **params}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = build_network(len(pooled.columns), len(pooled.stocks), int(settings["d_model"]),
                            int(settings["heads"]), float(settings["dropout"]))
    positions, owners, starts, labels = training_samples(pooled, cost_bps)
    optimiser = torch.optim.Adam(network.parameters(), lr=float(settings["learning_rate"]))
    loss = torch.nn.BCEWithLogitsLoss()
    length, batch = int(settings["sequence"]), int(settings["batch"])
    network.train()
    for _ in range(int(settings["epochs"])):
        order = rng.permutation(len(positions))
        for first in range(0, len(order), batch):
            chosen = order[first:first + batch]
            x = torch.tensor(windows(pooled, positions[chosen], starts[chosen], length))
            logits, _ = network(x, torch.tensor(owners[chosen], dtype=torch.long))
            optimiser.zero_grad()
            loss(logits, torch.tensor(labels[chosen])).backward()
            optimiser.step()
    network.eval()
    return Trained(network, length)


# Protocol ---------------------------------------------------------------------------------------
def seed(config: StudyConfig, test_year: int, run: int, stage: int) -> int:
    """One model serves every stock, so its seed is keyed by "pooled" instead of a stock."""
    rng = make_rng(config.base_seed, "run", "pooled", test_year, run, stage)
    return int(rng.integers(2**31 - 1))


def predictions(trained: Trained, pooled: Pooled) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for k, stock in enumerate(pooled.stocks):
        panel = pooled.panels[stock]
        start = pooled.offsets[stock]
        train_p, _ = trained.predict(pooled, k, pooled.rows(stock, panel.train), start)
        later_p, later_w = trained.predict(pooled, k, pooled.rows(stock, panel.evaluate), start)
        out[stock] = {"train": train_p, "evaluate": later_p, "weights": later_w}
    return out


def best_quantile(stock_prediction: dict[str, np.ndarray], panel: Panel, exits: tuple[int, float],
                  settings: Settings, cost_bps: float) -> tuple[float, float]:
    """The quantile with the best validation Sharpe ratio for one stock, and that ratio."""
    best = (settings.quantiles[0], -np.inf)
    prices = panel.prices.loc[panel.evaluate]
    for quantile in settings.quantiles:
        threshold = float(np.quantile(stock_prediction["train"], quantile))
        result = backtest(stock_prediction["evaluate"] >= threshold, prices, exits[0], exits[1], cost_bps)
        score = metrics.sharpe(result.daily) if len(result.trades) >= settings.min_trades else 0.0
        if score > best[1]:
            best = (quantile, score)
    return best


def run_tft(test_year: int, stage: int, run: int, stocks: list[str], exits: dict[str, tuple[int, float]],
            model: ModelSpec, settings: Settings, config: StudyConfig, *, campaign: str = "dev",
            write: bool = True) -> dict[str, dict[str, Any]]:
    """Tune, train and evaluate the pooled model; one record per stock (written like other runs)."""
    started = time.perf_counter()
    window = next(w for w in walk_forward(config) if w.test_year == test_year)
    primary = config.primary_cost_bps_per_side
    scheme = config.windows.scheme
    tuning = assemble(stocks, window.train, window.validation, settings, test_year, 1, exits, scheme)
    candidates = []
    for params in model.candidates():
        predicted = predictions(train(tuning, params, seed(config, test_year, run, 1), primary), tuning)
        chosen = {s: best_quantile(predicted[s], tuning.panels[s], exits[s], settings, primary) for s in stocks}
        candidates.append({"params": params, "chosen": chosen, "predicted": predicted,
                           "median_sharpe": float(np.median([v[1] for v in chosen.values()]))})
    best = max(candidates, key=lambda c: c["median_sharpe"])  # first of equals wins
    if stage == 1:
        final, predicted, evaluated_on, fitted_on = tuning, best["predicted"], window.validation, window.train
    else:
        final = assemble(stocks, window.final_train, window.test, settings, test_year, 2, exits, scheme)
        predicted = predictions(train(final, best["params"], seed(config, test_year, run, 2), primary), final)
        evaluated_on, fitted_on = window.test, window.final_train
    seconds = round(time.perf_counter() - started, 2)
    records = {}
    for stock in stocks:
        panel, (days, rate) = final.panels[stock], exits[stock]
        quantile = best["chosen"][stock][0]
        threshold = float(np.quantile(predicted[stock]["train"], quantile))
        signal = predicted[stock]["evaluate"] >= threshold
        prices = panel.prices.loc[panel.evaluate]
        spec = RunSpec(stock, test_year, run, stage, days, rate)
        train_result = backtest(predicted[stock]["train"] >= threshold, panel.prices.loc[panel.train], days, rate,
                                primary)
        record = {
            **spec.__dict__,
            "rung": model.name,
            "campaign": campaign,
            "sources": list(settings.sources),
            "train": [str(fitted_on.start), str(fitted_on.end)],
            "evaluate": [str(evaluated_on.start), str(evaluated_on.end)],
            "indicators": len(final.columns),
            "model": f"{model.name} {best['params']} (pooled over {len(stocks)} stocks), "
                     f"buy at or above the {quantile:g} quantile",
            "fitness": metrics.sharpe(train_result.daily),
            "validation_sharpe": best["chosen"][stock][1],
            "candidates": [{**c["params"], "median_sharpe": c["median_sharpe"]} for c in candidates],
            "evaluation": {f"{cost:g}bps": metrics.summary(backtest(signal, prices, days, rate, cost))
                           for cost in config.costs_bps_per_side},
            "signal": pack_signal(signal),
            "selection_weights": selection_weights(predicted[stock]["weights"], final.columns),
            "branches": {},
            "seconds": seconds,
        }
        if write:
            record["provenance"] = provenance.collect(config, ml_settings_digest=settings.digest(),
                                                      ml_model=model.__dict__, pooled_stocks=stocks)
            provenance.write(result_path(campaign, model.name, spec), record)
        records[stock] = record
    return records


def selection_weights(weights: np.ndarray, columns: list[str]) -> dict[str, float]:
    """Mean variable-selection weight of each analysis type over the evaluation days."""
    if len(weights) == 0:
        return {}
    means = weights.mean(axis=0)
    return {source: float(sum(m for m, c in zip(means, columns) if c.startswith(f"{source.lower()}:")))
            for source in SOURCES}
