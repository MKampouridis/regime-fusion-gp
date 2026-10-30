"""Gaussian hidden Markov model with diagonal covariances, fitted by EM.

Only filtered probabilities are used in the study: the probability of each
state on day t given the observations up to day t (``filter``), never smoothed
probabilities, which use later days. ``fit`` estimates the parameters with the
Baum-Welch algorithm (scaled forward-backward), from several random starts
drawn from a generator of ``regime_fusion.seeds``, keeping the best likelihood.
States are then ordered by the mean of the first feature, so that state 0 is,
for example, the calmest when the first feature is a volatility.

A row's missing values count for nothing: with diagonal covariances the density
of the values observed is exact, so a day without news still counts through its
price inputs, and a row with no value carries no information (the filter only
moves the probabilities forward through the transition matrix that day). A day
so unlikely under every reachable state that its likelihoods underflow is
treated in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

VARIANCE_FLOOR = 1e-6  # relative to each feature's overall variance
LOST_DAY_LIKELIHOOD = 1e-300  # a day no reachable state explains (see _forward_pass)


@dataclass(frozen=True)
class GaussianHMM:
    start: np.ndarray        # (K,) initial state probabilities
    transition: np.ndarray   # (K, K) row-stochastic
    means: np.ndarray        # (K, D)
    variances: np.ndarray    # (K, D)
    log_likelihood: float

    @property
    def n_states(self) -> int:
        return len(self.start)

    def expected_durations(self) -> np.ndarray:
        """Mean number of days a state lasts once entered."""
        return 1.0 / (1.0 - np.diag(self.transition))


def fit(x: np.ndarray, n_states: int, rng: np.random.Generator, *, restarts: int = 10,
        max_iter: int = 500, tol: float = 1e-6) -> GaussianHMM:
    """Maximum-likelihood fit on complete rows of ``x`` (days by features)."""
    x = _as_2d(x)
    x = x[~np.isnan(x).any(axis=1)]
    if len(x) < 10 * n_states:
        raise ValueError(f"too few complete rows ({len(x)}) for {n_states} states")
    floor = VARIANCE_FLOOR * x.var(axis=0) + 1e-12
    best = None
    for _ in range(restarts):
        model = _em(x, _initial(x, n_states, rng), floor, max_iter, tol)
        if _finite(model) and (best is None or model.log_likelihood > best.log_likelihood):
            best = model
    if best is None:
        raise ValueError("no EM start reached a finite likelihood with finite parameters")
    return _ordered(best)


def _finite(model: GaussianHMM) -> bool:
    """Whether a fitted model's likelihood and every one of its parameters are finite."""
    arrays = (model.start, model.transition, model.means, model.variances)
    return bool(np.isfinite(model.log_likelihood) and all(np.isfinite(a).all() for a in arrays))


def filter(model: GaussianHMM, x: np.ndarray) -> np.ndarray:  # noqa: A001 (the standard name)
    """Filtered state probabilities P(state on day t | days up to t), one row per day."""
    x = _as_2d(x)
    alpha, _, _ = _forward_pass(model.start, model.transition, _emission(model, x))
    return alpha


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def _initial(x: np.ndarray, k: int, rng: np.random.Generator) -> GaussianHMM:
    means = x[rng.choice(len(x), size=k, replace=False)]
    variances = np.tile(x.var(axis=0), (k, 1))
    transition = np.full((k, k), 0.1 / max(k - 1, 1)) + np.eye(k) * (0.9 - 0.1 / max(k - 1, 1))
    transition /= transition.sum(axis=1, keepdims=True)
    return GaussianHMM(np.full(k, 1.0 / k), transition, means, variances, -np.inf)


def _log_emission(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - model.means[None, :, :]
    terms = np.log(2 * np.pi * model.variances)[None] + diff ** 2 / model.variances[None]
    # A missing feature contributes nothing; a model with missing parameters gives NaN, never a likelihood.
    return -0.5 * np.where(np.isnan(x)[:, None, :], 0.0, terms).sum(axis=2)


def _emission(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """Likelihoods of each row under each state, scaled by the row's largest (see the module notes)."""
    log_b = _log_emission(model, x)
    return np.exp(log_b - log_b.max(axis=1, keepdims=True))


def _forward(model: GaussianHMM, x: np.ndarray):
    """Scaled forward pass: normalised alphas, scales, emissions and the log-likelihood. A day that no
    reachable state can explain counts at the floor ``LOST_DAY_LIKELIHOOD`` times its best emission."""
    log_b = _log_emission(model, x)
    shift = log_b.max(axis=1, keepdims=True)
    b = np.exp(log_b - shift)
    alpha, scale, lost = _forward_pass(model.start, model.transition, b)
    return alpha, scale, b, float(np.log(scale).sum() + shift.sum() + lost.sum() * np.log(LOST_DAY_LIKELIHOOD))


def _em(x, model, floor, max_iter, tol) -> GaussianHMM:
    previous = -np.inf
    for _ in range(max_iter):
        alpha, scale, b, log_likelihood = _forward(model, x)
        if log_likelihood - previous < tol * abs(log_likelihood):
            break
        previous = log_likelihood
        beta = _backward_pass(model.transition, b, scale)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * model.transition[None] * (b[1:] * beta[1:])[:, None, :]
              / scale[1:, None, None]).sum(axis=0)
        weights = np.maximum(gamma.sum(axis=0), 1e-300)  # a state no day belongs to: no division by zero
        means = (gamma.T @ x) / weights[:, None]
        variances = (gamma.T @ x ** 2) / weights[:, None] - means ** 2
        leaving = xi.sum(axis=1, keepdims=True)
        transition = np.where(leaving > 0, xi / np.where(leaving > 0, leaving, 1.0), 1.0 / len(weights))
        model = GaussianHMM(gamma[0], transition, means, np.maximum(variances, floor), log_likelihood)
    return GaussianHMM(model.start, model.transition, model.means, model.variances,
                       _forward(model, x)[3])


@njit(cache=True)
def _forward_pass(start, transition, b):
    """Normalised forward probabilities (the filter), the scale of each step, and the days whose
    likelihood underflowed under every reachable state. Such a day carries no information: its
    emissions become 1, in ``b`` itself so that the backward pass agrees, and the filter moves on
    through the transition matrix."""
    n, k = b.shape
    alpha = np.empty((n, k))
    scale = np.empty(n)
    lost = np.zeros(n, dtype=np.bool_)
    prior = np.empty(k)
    for t in range(n):
        for j in range(k):
            if t == 0:
                prior[j] = start[j]
            else:
                prior[j] = 0.0
                for i in range(k):
                    prior[j] += alpha[t - 1, i] * transition[i, j]
            alpha[t, j] = prior[j] * b[t, j]
        total = alpha[t].sum()
        if total <= 0.0:
            lost[t] = True
            b[t, :] = 1.0
            alpha[t] = prior
            total = prior.sum()
        scale[t] = total
        alpha[t] /= total
    return alpha, scale, lost


@njit(cache=True)
def _backward_pass(transition, b, scale):
    n, k = b.shape
    beta = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        for i in range(k):
            total = 0.0
            for j in range(k):
                total += transition[i, j] * b[t + 1, j] * beta[t + 1, j]
            beta[t, i] = total / scale[t + 1]
    return beta


def _ordered(model: GaussianHMM) -> GaussianHMM:
    order = np.argsort(model.means[:, 0], kind="stable")
    return GaussianHMM(
        start=model.start[order],
        transition=model.transition[np.ix_(order, order)],
        means=model.means[order],
        variances=model.variances[order],
        log_likelihood=model.log_likelihood,
    )
