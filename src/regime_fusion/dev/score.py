"""E10, stage S1a: a GP whose rules give each stock a score, traded with banded holdings (docs/E10_PLAN.md, part 1).

A model has one arithmetic tree per source (``FA`` over the ``fa:`` inputs, ``TA`` over the ``ta:`` inputs) and
a fusion tree. Inputs are date x stock cross-sectional ranks in [-1, 1], NaN where unknown; ``rg:`` inputs are
conditions (stock context, or market state with the same value for every stock of a date).

Branch nodes: ADD, SUB, MUL, MIN, MAX (two arguments), NEG (one), VAR (an input of the branch's source), CONST
(in [-1, 1]) and IFS(condition, a, b), which takes ``a`` where the condition holds and ``b`` elsewhere. A condition
(COND) is one comparison, GT or LT, of one allowed ``rg:`` input with a constant in [-1, 1]. IFS exists only when
the caller passes ``if_inputs``. A missing input counts as 0 inside a branch; a missing condition input counts as
the condition not holding. Each branch's output is ranked across the tradable stocks of the date to [-1, 1]
(ties share their average rank) before the fusion sees it. Fusion nodes: LEAF (a ranked branch output), WSUM(w, a, b)
= w a + (1 - w) b with w in [0, 1], MIN, MAX and IFS. The model's score is the fusion output ranked again across
the tradable stocks; NaN for a stock that cannot be traded. A date's scores depend on that date's row only.

Node counting rule: every entry of a tree's prefix list counts as one node, so a COND counts one (its input and
its constant are part of it), a WSUM counts one (its weight is part of it), and IFS(cond, a, b) counts
1 + 1 + size(a) + size(b). At most ``BRANCH_NODES`` = 15 per branch and ``FUSION_NODES`` = 7 in the fusion, kept at
creation and after every crossover and mutation.

Evolution is a small generational GP of its own (the engine's ``evolve`` is tied to Boolean ``Individual`` trees,
the three-branch active trading operator and ``engine.operators``): the ``GPConfig`` fields population,
generations, crossover, tournament and elitism mean what they mean there; ramped half-and-half initial
population (branch depths 1 to ``initial_depth`` + 1, fusion depths 1 to ``initial_depth``); one-child typed
crossover (a branch swaps a subtree only with the same source's branch, the fusion with the fusion; a swap that
would break a size limit is retried and then the child is a copy of the first parent); mutation is subtree
replacement or constant perturbation (Gaussian, sd 0.2, kept in range). All randomness comes from the ``rng``
handed in.

Two fitnesses, both the PSR lower bound (``dev.fitness``) and both scored on the training dates: ``book`` on the
per-date net result of the banded book of the score on ``e9.fitness_returns(hedged, "ranks")`` with costs
``e9.fitness_costs(hedged, cost, "ranks")``; ``rank`` on the per-date Spearman correlation between the score and
the hedged returns over the tradable stocks (dates with fewer than 10 tradable stocks skipped; a date on which
the correlation is undefined, a constant score or constant returns, counts as 0). A model whose score is the same
for all stocks on more than 10% of the dates with at least two tradable stocks scores ``FLOOR`` under either.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from regime_fusion.dev import crosssection as C
from regime_fusion.dev import e9
from regime_fusion.dev.fitness import FLOOR, psr_lower_bound
from regime_fusion.engine.gp import GPConfig

BRANCH_NODES = 15
FUSION_NODES = 7
FITNESSES = ("book", "rank")
CONSTANT_SHARE = 0.10               # more than this share of dates with one score for every stock: the floor
MIN_RANK_STOCKS = 10                # fewer tradable stocks than this and a date is skipped by the rank fitness
PERTURB_SD = 0.2
CROSSOVER_TRIES = 5

BINARY = ("ADD", "SUB", "MUL", "MIN", "MAX")
ARITY = {"ADD": 2, "SUB": 2, "MUL": 2, "MIN": 2, "MAX": 2, "WSUM": 2, "NEG": 1, "IFS": 3,
         "VAR": 0, "CONST": 0, "COND": 0, "LEAF": 0}
FUSION_BINARY = ("MIN", "MAX")
Node = tuple
Tree = list


# ------------------------------------------------------------------------------------------ ranks
def _average_ranks(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per date, the 0-based rank of each valid value among the valid values (ties share their average).
    The entries of invalid stocks are meaningless."""
    v = np.where(valid, values, np.inf)
    order = np.argsort(v, axis=1, kind="stable")
    sv = np.take_along_axis(v, order, axis=1)
    t, n = v.shape
    pos = np.broadcast_to(np.arange(n), (t, n))
    first = np.ones((t, n), dtype=bool)
    first[:, 1:] = sv[:, 1:] != sv[:, :-1]
    last = np.ones((t, n), dtype=bool)
    last[:, :-1] = first[:, 1:]
    start = np.maximum.accumulate(np.where(first, pos, 0), axis=1)
    end = np.minimum.accumulate(np.where(last, pos, n)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty((t, n))
    np.put_along_axis(ranks, order, (start + end) / 2.0, axis=1)
    return ranks


def rank_scale(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Each valid value replaced by its rank among its date's valid values, from -1 to 1 (average ranks for
    ties; 0 if a date has one valid stock); NaN where not valid."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    ranks = _average_ranks(values, valid)
    n = valid.sum(axis=1, keepdims=True)
    scaled = np.where(n > 1, 2.0 * ranks / np.maximum(n - 1, 1) - 1.0, 0.0)
    return np.where(valid, scaled, np.nan)


# ------------------------------------------------------------------------------------------ models
@dataclass
class ScoreModel:
    """The two branches (source -> prefix list of nodes), the fusion and, once evaluated, ``fitness``.
    ``final`` (set on the model an evolution returns) is the last population, each model with its fitness."""

    fusion: Tree
    branches: dict[str, Tree]
    fitness: float = float("nan")
    final: tuple = field(default=(), repr=False, compare=False)

    def key(self) -> tuple:
        return (tuple(self.fusion), *((s, tuple(t)) for s, t in self.branches.items()))

    def copy(self) -> "ScoreModel":
        return ScoreModel(list(self.fusion), {s: list(t) for s, t in self.branches.items()})

    def score(self, inputs: Mapping[str, np.ndarray], tradable: np.ndarray) -> np.ndarray:
        """The date x stock score, in [-1, 1], NaN where ``tradable`` is False. Works on any dates."""
        return _score(self, _Data(inputs, np.asarray(tradable, dtype=bool)))

    def describe(self) -> str:
        lines = [f"{s}: {_text(tree)[0]}" for s, tree in self.branches.items()]
        return "\n".join(lines + [f"fusion: {_text(self.fusion)[0]}"])

    def inputs_used(self) -> list[str]:
        """Sorted names of the inputs that can affect the score: those of the branches the fusion names and the
        conditions of the fusion. An input inside a branch multiplied by 0 still counts."""
        named = {n[1] for n in self.fusion if n[0] == "LEAF"}
        used = {n[1][1] for n in self.fusion if n[0] == "COND"}
        for source in named:
            for n in self.branches[source]:
                if n[0] == "VAR":
                    used.add(n[1])
                elif n[0] == "COND":
                    used.add(n[1][1])
        return sorted(used)

    def to_record(self) -> dict[str, Any]:
        """JSON-ready, constants at full precision; ``from_record`` reads it back."""
        def nodes(tree):
            return [[n[0], list(n[1]) if n[0] == "COND" else n[1]] for n in tree]

        return {"fusion": nodes(self.fusion), "branches": {s: nodes(t) for s, t in self.branches.items()},
                "fitness": None if math.isnan(self.fitness) else float(self.fitness)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ScoreModel":
        def nodes(items):
            return [(k, tuple(p) if k == "COND" else p) for k, p in items]

        fitness = record.get("fitness")
        return cls(nodes(record["fusion"]), {s: nodes(t) for s, t in record["branches"].items()},
                   float("nan") if fitness is None else float(fitness))


def within_limits(model: ScoreModel) -> bool:
    return len(model.fusion) <= FUSION_NODES and all(len(t) <= BRANCH_NODES for t in model.branches.values())


def _text(tree: Sequence[Node], i: int = 0) -> tuple[str, int]:
    kind, payload = tree[i]
    if kind in ("VAR", "LEAF"):
        return str(payload), i + 1
    if kind == "CONST":
        return f"{payload:.3f}", i + 1
    if kind == "COND":
        direction, name, c = payload
        return f"{name} {'>' if direction == 'GT' else '<'} {c:.3f}", i + 1
    parts, i = [], i + 1
    if kind == "WSUM":
        parts.append(f"{payload:.3f}")
    for _ in range(ARITY[kind]):
        part, i = _text(tree, i)
        parts.append(part)
    return f"{kind}({', '.join(parts)})", i


def _end(tree: Sequence[Node], start: int) -> int:
    slots, i = 1, start
    while slots:
        slots += ARITY[tree[i][0]] - 1
        i += 1
    return i


# ------------------------------------------------------------------------------------------ evaluation
class _Data:
    """Inputs on some dates, with missing values filled by 0 on demand; ``tradable`` is date x stock."""

    def __init__(self, inputs: Mapping[str, np.ndarray], tradable: np.ndarray):
        self.inputs, self.tradable, self.shape = inputs, tradable, tradable.shape
        self._filled: dict[str, np.ndarray] = {}

    def raw(self, name: str) -> np.ndarray:
        return np.asarray(self.inputs[name], dtype=float)

    def filled(self, name: str) -> np.ndarray:
        if name not in self._filled:
            a = self.raw(name)
            self._filled[name] = np.where(np.isnan(a), 0.0, a)
        return self._filled[name]


def _condition(tree, i, data: _Data) -> tuple[np.ndarray, int]:
    direction, name, c = tree[i][1]
    values = data.raw(name)
    with np.errstate(invalid="ignore"):
        return (values > c if direction == "GT" else values < c), i + 1


def _eval(tree, i, data: _Data, leaves: Mapping[str, np.ndarray] | None):
    kind, payload = tree[i]
    if kind == "VAR":
        return data.filled(payload), i + 1
    if kind == "CONST":
        return payload, i + 1
    if kind == "LEAF":
        return leaves[payload], i + 1
    if kind == "IFS":
        cond, i = _condition(tree, i + 1, data)
        a, i = _eval(tree, i, data, leaves)
        b, i = _eval(tree, i, data, leaves)
        return np.where(cond, a, b), i
    a, i = _eval(tree, i + 1, data, leaves)
    if kind == "NEG":
        return -a, i
    b, i = _eval(tree, i, data, leaves)
    if kind == "ADD":
        return a + b, i
    if kind == "SUB":
        return a - b, i
    if kind == "MUL":
        return a * b, i
    if kind == "MIN":
        return np.minimum(a, b), i
    if kind == "MAX":
        return np.maximum(a, b), i
    return payload * a + (1.0 - payload) * b, i          # WSUM


def _score(model: ScoreModel, data: _Data) -> np.ndarray:
    leaves = {}
    for source, tree in model.branches.items():
        out = np.broadcast_to(_eval(tree, 0, data, None)[0], data.shape)
        leaves[source] = rank_scale(out, data.tradable)
    out = np.broadcast_to(_eval(model.fusion, 0, data, leaves)[0], data.shape)
    return rank_scale(out, data.tradable)


def ensemble_score(models: Sequence[ScoreModel], inputs: Mapping[str, np.ndarray], tradable: np.ndarray) -> np.ndarray:
    """The mean of the models' scores per stock and date, ranked again across the tradable stocks to [-1, 1]
    (NaN where not tradable). For one model this is its own score."""
    tradable = np.asarray(tradable, dtype=bool)
    data = _Data(inputs, tradable)
    mean = np.mean([_score(m, data) for m in models], axis=0)
    return rank_scale(mean, tradable)


# ------------------------------------------------------------------------------------------ the market
class ScoreMarket:
    """Hedged returns and costs of a date x stock grid; ``simulate`` scores a date x stock score by the banded,
    long-only, equal-weight book it implies.

    A stock enters when its score is in the top ``enter`` share of the date's tradable stocks (percentile
    rank, ties averaged, strictly above 1 - ``enter``) and stays until it is no longer strictly above
    1 - ``leave``, has no score or stops being tradable. This is ``crosssection.banded_weights(...,
    long_only=True)`` exactly. ``initial`` (one weight per stock, as e9 carries ``final_weights`` into the
    next year) both sets the trades of the first date and seeds the band: a stock with a positive initial weight
    counts as already held before the first date, so it stays if it is in the top ``leave`` share and a stock
    outside it, or not tradable, is sold at the first date's cost. Result per date: sum of weight x hedged return,
    less the mean hedged return of the tradable stocks, less |weight change| x cost (an empty book scores 0 before
    costs). ``active`` counts the dates with at least one tradable stock; if ``low`` or ``high`` is given, only
    those on which the share held lies within them."""

    def __init__(self, hedged: np.ndarray, cost: np.ndarray, enter: float = 0.2, leave: float = 0.4,
                 low: float | None = None, high: float | None = None, initial: np.ndarray | None = None):
        if not 0.0 < enter <= leave <= 1.0:
            raise ValueError("need 0 < enter <= leave <= 1")
        self.returns = np.asarray(hedged, dtype=float)
        self.tradable = np.isfinite(self.returns)
        self.cost = np.where(np.isfinite(cost), cost, C.COST_BELOW)
        self.enter, self.leave, self.low, self.high = enter, leave, low, high
        self.initial = np.zeros(self.returns.shape[1]) if initial is None else np.asarray(initial, dtype=float)
        self.count = self.tradable.sum(axis=1)
        self._r0 = np.where(self.tradable, self.returns, 0.0)
        self.base = np.divide(self._r0.sum(axis=1), self.count, out=np.zeros(len(self.count)), where=self.count > 0)

    def held(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        known = self.tradable & np.isfinite(score)
        m = known.sum(axis=1, keepdims=True)
        pct = np.where(known, (_average_ranks(score, known) + 1.0) / np.maximum(m, 1), np.nan)
        with np.errstate(invalid="ignore"):
            top, stay = pct > 1 - self.enter, pct > 1 - self.leave
        out = np.zeros(pct.shape, dtype=bool)
        now = self.initial > 0
        for k in range(len(pct)):
            now = top[k] | (now & stay[k])
            out[k] = now
        return out

    def weights(self, score: np.ndarray) -> np.ndarray:
        held = self.held(score)
        n = held.sum(axis=1, keepdims=True)
        return np.where(n > 0, held / np.maximum(n, 1), 0.0)

    def share(self, score: np.ndarray) -> np.ndarray:
        n = self.held(score).sum(axis=1)
        return np.divide(n, self.count, out=np.zeros(len(n)), where=self.count > 0)

    def _gross(self, w: np.ndarray, any_held: np.ndarray) -> np.ndarray:
        return np.where(any_held, (w * self._r0).sum(axis=1) - self.base, 0.0)

    def _change(self, w: np.ndarray) -> np.ndarray:
        return np.abs(np.diff(np.vstack([self.initial, w]), axis=0))

    def gross(self, score: np.ndarray) -> np.ndarray:
        held = self.held(score)
        return self._gross(self.weights(score), held.any(axis=1))

    def turnover(self, score: np.ndarray) -> np.ndarray:
        return self._change(self.weights(score)).sum(axis=1)

    def costs(self, score: np.ndarray) -> np.ndarray:
        return (self._change(self.weights(score)) * self.cost).sum(axis=1)

    def contributions(self, score: np.ndarray) -> np.ndarray:
        """Date x stock split of the net result; each row sums to ``simulate(score)[0]``."""
        held = self.held(score)
        n = held.sum(axis=1, keepdims=True)
        w = np.where(n > 0, held / np.maximum(n, 1), 0.0)
        base = np.where(held.any(axis=1), self.base, 0.0)
        return e9._decompose(w, self._r0, base, self.initial, self.cost)

    def daily(self, score: np.ndarray) -> tuple[np.ndarray, int]:
        net, active, _ = self.simulate(score)
        return net, active

    def simulate(self, score: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """Per-date results net of costs, the number of active dates, and the results again."""
        held = self.held(score)
        n = held.sum(axis=1, keepdims=True)
        w = np.where(n > 0, held / np.maximum(n, 1), 0.0)
        net = self._gross(w, held.any(axis=1)) - (self._change(w) * self.cost).sum(axis=1)
        inside = self.count > 0
        eps = 1e-9
        if self.low is not None:
            inside &= n[:, 0] >= self.low * self.count - eps
        if self.high is not None:
            inside &= n[:, 0] <= self.high * self.count + eps
        return net, int(inside.sum()), net


# ------------------------------------------------------------------------------------------ fitness
class ScoreEvaluator:
    """Fitness of models on the training dates, cached by tree key; ``final`` is the last population shown to
    ``retain``. ``fitness`` is "book" or "rank" (module docstring)."""

    def __init__(self, inputs: Mapping[str, np.ndarray], hedged: np.ndarray, cost: np.ndarray,
                 fitness: str = "book", enter: float = 0.2, leave: float = 0.4):
        if fitness not in FITNESSES:
            raise ValueError(f"fitness must be one of {FITNESSES}")
        self.kind = fitness
        hedged = np.asarray(hedged, dtype=float)
        self.tradable = np.isfinite(hedged)
        self.data = _Data(inputs, self.tradable)
        self.final: list = []
        self._cache: dict[tuple, float] = {}
        if fitness == "book":
            self.market = ScoreMarket(e9.fitness_returns(hedged, "ranks"),
                                      e9.fitness_costs(hedged, cost, "ranks"), enter, leave)
        else:
            self._returns_rank = rank_scale(hedged, self.tradable)
            self._enough = self.tradable.sum(axis=1) >= MIN_RANK_STOCKS

    def evaluate(self, model: ScoreModel) -> float:
        key = model.key()
        if key not in self._cache:
            self._cache[key] = self._fitness(_score(model, self.data))
        model.fitness = self._cache[key]
        return model.fitness

    def retain(self, population) -> None:
        self.final = list(population)

    def _fitness(self, score: np.ndarray) -> float:
        two = self.tradable.sum(axis=1) >= 2
        if two.any():
            constant = (np.where(self.tradable, score, -np.inf).max(axis=1)
                        == np.where(self.tradable, score, np.inf).min(axis=1)) & two
            if constant.sum() > CONSTANT_SHARE * two.sum():
                return FLOOR
        if self.kind == "book":
            return float(psr_lower_bound(self.market.simulate(score)[0]))
        a = np.where(self.tradable, score, 0.0)
        b = np.where(self.tradable, self._returns_rank, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        corr = np.where(np.isfinite(corr), corr, 0.0)
        return float(psr_lower_bound(corr[self._enough]))


# ------------------------------------------------------------------------------------------ random trees
@dataclass(frozen=True)
class _Space:
    groups: Mapping[str, Sequence[str]]
    if_inputs: tuple[str, ...] = ()

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self.groups)


def _condition_node(space: _Space, rng: np.random.Generator) -> Node:
    return ("COND", (("GT", "LT")[rng.integers(2)], space.if_inputs[rng.integers(len(space.if_inputs))],
                     float(rng.uniform(-1.0, 1.0))))


def _grow(space: _Space, names: Sequence[str] | None, budget: int, depth: int, method: str,
          rng: np.random.Generator, out: Tree) -> None:
    """Append a random subtree of at most ``budget`` nodes and ``depth`` levels below it. ``names`` are the
    branch's inputs, or None for the fusion."""
    fusion = names is None
    options = []
    if depth > 0 and budget >= 3:
        options += list(FUSION_BINARY + ("WSUM",) if fusion else BINARY)
    if depth > 0 and budget >= 2 and not fusion:
        options.append("NEG")
    if depth > 0 and budget >= 4 and space.if_inputs:
        options.append("IFS")
    stop = not options or (method == "grow" and rng.random() < 0.3)
    if stop:
        if fusion:
            out.append(("LEAF", space.sources[rng.integers(len(space.sources))]))
        elif rng.random() < 0.75:
            out.append(("VAR", names[rng.integers(len(names))]))
        else:
            out.append(("CONST", float(rng.uniform(-1.0, 1.0))))
        return
    kind = options[rng.integers(len(options))]
    rest = budget - 1
    if kind == "NEG":
        out.append((kind, None))
        _grow(space, names, rest, depth - 1, method, rng, out)
        return
    if kind == "IFS":
        out.append((kind, None))
        out.append(_condition_node(space, rng))
        rest -= 1
    else:
        out.append((kind, float(rng.uniform(0.0, 1.0)) if kind == "WSUM" else None))
    first = int(rng.integers(1, rest))                       # leaves at least 1 node for the second child
    start = len(out)
    _grow(space, names, first, depth - 1, method, rng, out)
    _grow(space, names, rest - (len(out) - start), depth - 1, method, rng, out)


def _random_model(space: _Space, depth: int, method: str, rng: np.random.Generator,
                  fusion_depth: int) -> ScoreModel:
    fusion: Tree = []
    _grow(space, None, FUSION_NODES, fusion_depth, method, rng, fusion)
    branches = {}
    for source in space.sources:
        tree: Tree = []
        _grow(space, space.groups[source], BRANCH_NODES, depth, method, rng, tree)
        branches[source] = tree
    return ScoreModel(fusion, branches)


def _ramped(space: _Space, size: int, initial_depth: int, rng: np.random.Generator) -> list[ScoreModel]:
    """Ramped half-and-half: depths 1..initial_depth + 1 (branches) in turn, alternating full and grow."""
    levels = initial_depth + 1
    population = []
    for i in range(size):
        depth = 1 + i % levels
        method = "full" if (i // levels) % 2 == 0 else "grow"
        population.append(_random_model(space, depth, method, rng, min(depth, initial_depth)))
    return population


# ------------------------------------------------------------------------------------------ operators
def _component(model: ScoreModel, space: _Space, rng: np.random.Generator) -> tuple[str, int]:
    parts = ("fusion",) + space.sources
    name = parts[rng.integers(len(parts))]
    return name, (FUSION_NODES if name == "fusion" else BRANCH_NODES)


def _tree_of(model: ScoreModel, name: str) -> Tree:
    return model.fusion if name == "fusion" else model.branches[name]


def _with(model: ScoreModel, name: str, tree: Tree) -> ScoreModel:
    child = model.copy()
    if name == "fusion":
        child.fusion = tree
    else:
        child.branches[name] = tree
    return child


def _points(tree: Sequence[Node]) -> list[int]:
    """Nodes that may root a swapped or replaced subtree (a condition only ever changes as a whole by mutation
    of its constant, since an IFS needs one as its first child)."""
    return [i for i, n in enumerate(tree) if n[0] != "COND"]


def crossover(a: ScoreModel, b: ScoreModel, space: _Space, rng: np.random.Generator) -> ScoreModel:
    """Typed one-child crossover: a subtree of ``a`` is replaced by one of ``b``'s same component."""
    name, limit = _component(a, space, rng)
    ta, tb = _tree_of(a, name), _tree_of(b, name)
    pa, pb = _points(ta), _points(tb)
    for _ in range(CROSSOVER_TRIES):
        i, j = pa[rng.integers(len(pa))], pb[rng.integers(len(pb))]
        child = ta[:i] + tb[j:_end(tb, j)] + ta[_end(ta, i):]
        if len(child) <= limit:
            return _with(a, name, child)
    return a.copy()


def mutate(model: ScoreModel, space: _Space, rng: np.random.Generator) -> ScoreModel:
    """Subtree replacement or (half the time, if the component has a constant) constant perturbation."""
    name, limit = _component(model, space, rng)
    tree = _tree_of(model, name)
    constants = [i for i, n in enumerate(tree) if n[0] in ("CONST", "COND", "WSUM")]
    if constants and rng.random() < 0.5:
        i = constants[rng.integers(len(constants))]
        kind, payload = tree[i]
        low = 0.0 if kind == "WSUM" else -1.0
        if kind == "COND":
            direction, input_name, c = payload
            node = (kind, (direction, input_name, float(np.clip(c + rng.normal(0, PERTURB_SD), -1.0, 1.0))))
        else:
            node = (kind, float(np.clip(payload + rng.normal(0, PERTURB_SD), low, 1.0)))
        return _with(model, name, tree[:i] + [node] + tree[i + 1:])
    points = _points(tree)
    i = points[rng.integers(len(points))]
    end = _end(tree, i)
    room = limit - (len(tree) - (end - i))
    new: Tree = []
    _grow(space, None if name == "fusion" else space.groups[name], room, 2, "grow", rng, new)
    return _with(model, name, tree[:i] + new + tree[end:])


# ------------------------------------------------------------------------------------------ evolution
def evolve_score(inputs: Mapping[str, np.ndarray], groups: Mapping[str, Sequence[str]], hedged: np.ndarray,
                 cost: np.ndarray, *, fitness: str = "book", if_inputs: Sequence[str] = (),
                 budget: Mapping[str, int] | None = None, rng: np.random.Generator,
                 enter: float = 0.2, leave: float = 0.4) -> ScoreModel:
    """Evolve score models on the training dates (``inputs`` hold the ``groups`` and ``if_inputs`` names, all
    arrays over the training dates only). Returns the fittest model of the last generation with ``final`` set
    to the whole last population. ``budget`` is {"population", "generations"} (default ``e9.BUDGET``)."""
    budget = dict(e9.BUDGET if budget is None else budget)
    config = GPConfig(population=int(budget["population"]), generations=int(budget["generations"]),
                      active_trading=False)
    space = _Space({s: list(names) for s, names in groups.items()}, tuple(if_inputs))
    evaluator = ScoreEvaluator(inputs, hedged, cost, fitness, enter, leave)

    def evaluate(population):
        for model in population:
            evaluator.evaluate(model)

    population = _ramped(space, config.population, config.initial_depth, rng)
    evaluate(population)
    evaluator.retain(population)
    for _ in range(config.generations):
        fit = np.array([m.fitness for m in population])
        order = np.argsort(-fit, kind="stable")
        following = [population[i] for i in order[:config.elitism]]

        def pick() -> ScoreModel:
            entrants = rng.integers(len(fit), size=config.tournament)
            return population[int(entrants[np.argmax(fit[entrants])])]

        while len(following) < config.population:
            if rng.random() < config.crossover:
                following.append(crossover(pick(), pick(), space, rng))
            else:
                following.append(mutate(pick(), space, rng))
        population = following
        evaluate(population)
        evaluator.retain(population)
    best = max(population, key=lambda m: m.fitness)
    result = best.copy()
    result.fitness = best.fitness
    result.final = tuple(evaluator.final)
    return result
