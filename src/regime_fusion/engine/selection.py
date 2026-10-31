"""Parent selection: tournament, epsilon-lexicase and DALex.

* Tournament: the fittest of ``size`` individuals drawn at random (ActTrade: 4).
* Epsilon-lexicase (La Cava et al., 2016): cases are taken in a random order and
  the candidates kept are those within epsilon of the best on each case, epsilon
  being the case's median absolute deviation over the population, until one
  remains (or a random one of the survivors when the cases run out).
* DALex (Ni, Ding and Spector, 2024): a lexicase-like selection by weighted sums;
  each selection draws case weights as the softmax of normal draws with standard
  deviation ``particularity`` and takes the individual with the best weighted
  sum of normalised case scores (normalised once per generation, ``dalex_scores``).
  Large values approach lexicase.

Case matrices hold one row per individual and one column per case, higher being
better.
"""

from __future__ import annotations

import numpy as np


def tournament(fitness: np.ndarray, size: int, rng: np.random.Generator) -> int:
    entrants = rng.integers(len(fitness), size=size)
    return int(entrants[np.argmax(fitness[entrants])])


def mad_epsilon(cases: np.ndarray) -> np.ndarray:
    median = np.median(cases, axis=0)
    return np.median(np.abs(cases - median), axis=0)


def epsilon_lexicase(cases: np.ndarray, epsilon: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.arange(len(cases))
    for case in rng.permutation(cases.shape[1]):
        values = cases[candidates, case]
        candidates = candidates[values >= values.max() - epsilon[case]]
        if len(candidates) == 1:
            return int(candidates[0])
    return int(candidates[rng.integers(len(candidates))])


def dalex_scores(cases: np.ndarray) -> np.ndarray:
    """Each case's scores scaled to [0, 1] over the population."""
    spread = cases.max(axis=0) - cases.min(axis=0)
    return (cases - cases.min(axis=0)) / np.where(spread > 0, spread, 1.0)


def dalex(scores: np.ndarray, particularity: float, rng: np.random.Generator) -> int:
    raw = rng.normal(0.0, particularity, size=scores.shape[1])
    weights = np.exp(raw - raw.max())
    weights /= weights.sum()
    return int(np.argmax(scores @ weights))
