"""Deterministic random seeds.

A seed depends on the base seed and on what identifies a run (for example stock,
test year and run number), never on the configuration being run. Every
configuration therefore sees the same 30 seeds for a given stock and year, which
makes paired comparisons between configurations fairer.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def derive_seed(base_seed: int, *keys: object) -> int:
    """64-bit seed from a base seed and identifying keys."""
    payload = json.dumps([int(base_seed), *(str(key) for key in keys)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def make_rng(base_seed: int, *keys: object) -> np.random.Generator:
    """Independent numpy generator for the given keys."""
    return np.random.default_rng(derive_seed(base_seed, *keys))


def run_seeds(base_seed: int, stock: str, test_year: int, n_runs: int) -> list[int]:
    """Seeds for runs 0 .. n_runs-1 of one stock and test year."""
    return [derive_seed(base_seed, "run", stock, test_year, run) for run in range(n_runs)]
