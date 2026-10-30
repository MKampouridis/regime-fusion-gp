"""Neural gated fusion (ML-GATED), a benchmark for regime-gated fusion that is learned, not evolved.

One small network per analysis type (FA, SA, TA) scores the day from that
type's indicators. A gate turns the regime probabilities into weights over the
types (a softmax), and the weighted sum of the scores is the log-odds that the
trade a buy signal would open gains. Without regime inputs the gate is a fixed
learned weighting. Trained with binary cross-entropy and Adam; missing
indicator values enter as 0, the middle of the scaled range.

Runs are reproducible: PyTorch and the batch order are seeded from
``regime_fusion.seeds`` (through the run's seed), and PyTorch uses one thread.
A training set with a single class gives a constant prediction, as for the
other models (``regime_fusion.ml.models``).
"""

from __future__ import annotations

import numpy as np

SOURCES = ("FA", "SA", "TA")


def _torch():
    import torch

    torch.set_num_threads(1)
    return torch


def groups(columns: list[str]) -> tuple[dict[str, list[int]], list[int]]:
    """Column positions of each analysis type, and of the regime inputs (``rg:``)."""
    by_source = {s: [i for i, c in enumerate(columns) if c.startswith(f"{s.lower()}:")] for s in SOURCES}
    return {s: idx for s, idx in by_source.items() if idx}, [i for i, c in enumerate(columns) if c.startswith("rg:")]


def network(sources: dict[str, list[int]], gate_columns: list[int], hidden: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class GatedFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.sources = sources
            self.experts = nn.ModuleDict({
                s: nn.Sequential(nn.Linear(len(idx), hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
                for s, idx in sources.items()})
            self.gate_columns = gate_columns
            self.gate = nn.Linear(max(len(gate_columns), 1), len(sources))

        def weights(self, x):
            g = x[:, self.gate_columns] if self.gate_columns else torch.ones(len(x), 1)
            return torch.softmax(self.gate(g), dim=1)

        def forward(self, x):
            scores = torch.cat([self.experts[s](x[:, idx]) for s, idx in self.sources.items()], dim=1)
            return (self.weights(x) * scores).sum(dim=1)

    return GatedFusion()


class Gated:
    """Learner with the interface of ``regime_fusion.ml.models`` (fit, predict)."""

    def __init__(self, seed: int, columns: list[str], hidden: int = 16, weight_decay: float = 1e-4,
                 epochs: int = 40, dropout: float = 0.1, learning_rate: float = 1e-3, batch: int = 128):
        self.seed, self.columns = seed, columns
        self.hidden, self.weight_decay, self.epochs = int(hidden), float(weight_decay), int(epochs)
        self.dropout, self.learning_rate, self.batch = float(dropout), float(learning_rate), int(batch)

    def fit(self, x: np.ndarray, y: np.ndarray, states: np.ndarray | None = None):
        from regime_fusion.ml.models import Constant

        self.constant = Constant(y) if len(np.unique(y)) < 2 else None
        if self.constant is not None:
            return self
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        sources, gate_columns = groups(self.columns)
        self.model = network(sources, gate_columns, self.hidden, self.dropout)
        inputs = torch.tensor(np.nan_to_num(x), dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(inputs))
            for first in range(0, len(order), self.batch):
                rows = torch.as_tensor(order[first:first + self.batch])
                optimiser.zero_grad()
                loss(self.model(inputs[rows]), targets[rows]).backward()
                optimiser.step()
        self.model.eval()
        return self

    def predict(self, x: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
        if self.constant is not None:
            return self.constant.predict(x)
        torch = _torch()
        with torch.no_grad():
            logits = self.model(torch.tensor(np.nan_to_num(x), dtype=torch.float32))
        return torch.sigmoid(logits).numpy().astype(np.float64)

    def gate_weights(self, x: np.ndarray) -> np.ndarray:
        """Weight of each analysis type on each row (interpretability)."""
        torch = _torch()
        with torch.no_grad():
            return self.model.weights(torch.tensor(np.nan_to_num(x), dtype=torch.float32)).numpy()
