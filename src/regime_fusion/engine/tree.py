"""Strongly-typed GP trees: grammar, random generation, evaluation, printing.

An individual has one *branch* per analysis type (FA, SA, TA) and a *fusion*
tree that combines the branches' signals:

* A branch of type t is a Boolean tree over indicators of type t: AND_t and OR_t
  join two branch nodes; GT_t and LT_t compare an indicator of type t with an
  ephemeral random constant (ERC) in [-1, 1], as in ActTrade.
* The fusion tree's leaves name a branch (LEAF FA, SA or TA). Its functions are
  AND and OR (two children), ALL, ANY and VOTE (three children: all of them,
  any of them, at least two of them) and IF (a regime condition, then two fusion
  trees). A regime condition is a Boolean tree of type RG over regime inputs.
  ActTrade's 3-AND root is the fixed fusion tree ALL(FA, SA, TA).

Trees are lists of nodes in prefix order. A node is a tuple (kind, type,
payload): the payload is the indicator name of a VAR, the constant of an ERC,
and the branch of a LEAF. Indicators are looked up by name in a dict of arrays
whose values are scaled to [-1, 1] on the training window; a comparison with a
missing value is false.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

SOURCES = ("FA", "SA", "TA")
REGIME = "RG"
FUSION = "F"
ERC_RANGE = (-1.0, 1.0)

AND, OR, GT, LT, VAR, ERC = "AND", "OR", "GT", "LT", "VAR", "ERC"
F_AND, F_OR, F_ALL, F_ANY, F_VOTE, F_IF, LEAF = "AND", "OR", "ALL", "ANY", "VOTE", "IF", "LEAF"
ARITY = {AND: 2, OR: 2, GT: 2, LT: 2, VAR: 0, ERC: 0, F_ALL: 3, F_ANY: 3, F_VOTE: 3, F_IF: 3, LEAF: 0}
COMPARISONS = (GT, LT)
LOGIC = (AND, OR)

Node = tuple
Tree = list


def arity(node: Node) -> int:
    return ARITY[node[0]]


def end_of(tree: Sequence[Node], start: int) -> int:
    """Index just past the subtree that starts at ``start``."""
    open_slots, i = 1, start
    while open_slots:
        open_slots += arity(tree[i]) - 1
        i += 1
    return i


def depth(tree: Sequence[Node]) -> int:
    """Edges on the longest path from the root to a leaf (a lone leaf has depth 0)."""
    deepest, stack = 0, []
    for node in tree:
        level = stack.pop() if stack else 0
        deepest = max(deepest, level)
        stack.extend([level + 1] * arity(node))
    return deepest


@dataclass
class Individual:
    fusion: Tree
    branches: dict[str, Tree]
    fitness: float = float("nan")
    stats: dict = field(default_factory=dict)  # filled in by evaluation

    def key(self) -> tuple:
        return (tuple(self.fusion), *((s, tuple(b)) for s, b in self.branches.items()))

    def size(self) -> int:
        return len(self.fusion) + sum(len(b) for b in self.branches.values())

    def copy(self) -> Individual:
        return Individual(list(self.fusion), {s: list(b) for s, b in self.branches.items()})


@dataclass(frozen=True)
class Grammar:
    """What trees may contain: the indicators of each type and the fusion functions allowed."""

    indicators: Mapping[str, Sequence[str]]     # type (FA, SA, TA, RG) -> indicator names
    fusion_functions: tuple[str, ...] = ()      # () fixes the fusion (ActTrade: ALL(FA, SA, TA))
    sources: tuple[str, ...] = SOURCES          # the branches an individual has
    max_branch_depth: int = 5                   # ActTrade: depth 6 counting its 3-AND root
    max_fusion_depth: int = 3
    max_regime_depth: int = 2

    @property
    def fixed_fusion(self) -> bool:
        return not self.fusion_functions

    def leaf_probability(self, kind: str) -> float:
        """Grow method: chance of stopping at a comparison (ActTrade's terminal share)."""
        names = len(self.indicators[kind]) + 1  # indicators and the ERC
        return names / (names + 4)

    # Random generation ---------------------------------------------------------------
    def random_branch(self, kind: str, max_depth: int, method: str, rng: np.random.Generator) -> Tree:
        """A Boolean tree of type ``kind`` (a source or RG) of depth at most ``max_depth``."""
        if max_depth < 1:
            raise ValueError("a branch needs depth 1 at least (a comparison)")
        tree: Tree = []
        self._grow(kind, max_depth, method, rng, tree)
        return tree

    def _grow(self, kind, room, method, rng, tree):
        stop = room == 1 or (method == "grow" and rng.random() < self.leaf_probability(kind))
        if stop:
            tree.append((COMPARISONS[rng.integers(2)], kind, None))
            names = self.indicators[kind]
            tree.append((VAR, kind, names[rng.integers(len(names))]))
            tree.append((ERC, None, float(rng.uniform(*ERC_RANGE))))
            return
        tree.append((LOGIC[rng.integers(2)], kind, None))
        self._grow(kind, room - 1, method, rng, tree)
        self._grow(kind, room - 1, method, rng, tree)

    def random_fusion(self, max_depth: int, method: str, rng: np.random.Generator) -> Tree:
        if self.fixed_fusion:
            return fixed_fusion(self.sources)
        tree: Tree = []
        self._grow_fusion(max_depth, method, rng, tree)
        return tree

    def _grow_fusion(self, room, method, rng, tree):
        functions = self.fusion_functions
        stop = room == 0 or (method == "grow" and rng.random() < 3 / (3 + len(functions)))
        if stop:
            tree.append((LEAF, FUSION, self.sources[rng.integers(len(self.sources))]))
            return
        kind = functions[rng.integers(len(functions))]
        tree.append((kind, FUSION, None))
        if kind == F_IF:
            regime_room = max(1, min(self.max_regime_depth, room))
            self._grow(REGIME, regime_room, method, rng, tree)
            self._grow_fusion(room - 1, method, rng, tree)
            self._grow_fusion(room - 1, method, rng, tree)
            return
        for _ in range(ARITY[kind]):
            self._grow_fusion(room - 1, method, rng, tree)

    def random_individual(self, branch_depth: int, method: str, rng: np.random.Generator) -> Individual:
        fusion = self.random_fusion(min(branch_depth, self.max_fusion_depth), method, rng)
        branches = {s: self.random_branch(s, branch_depth, method, rng) for s in self.sources}
        return Individual(fusion, branches)

    def ramped_population(self, size: int, initial_depth: int, rng: np.random.Generator) -> list[Individual]:
        """Ramped half-and-half: depths 1..initial_depth in turn, alternating full and grow."""
        population = []
        for i in range(size):
            depth_now = 1 + i % initial_depth
            method = "full" if (i // initial_depth) % 2 == 0 else "grow"
            population.append(self.random_individual(depth_now, method, rng))
        return population


def acttrade_fusion() -> Tree:
    """ActTrade's root: all three branches must agree to buy."""
    return fixed_fusion(SOURCES)


def fixed_fusion(sources: Sequence[str]) -> Tree:
    """All branches must agree: ALL for three, AND for two, the branch itself for one."""
    leaves = [(LEAF, FUSION, s) for s in sources]
    if len(sources) == 1:
        return leaves
    return [((F_ALL if len(sources) == 3 else F_AND), FUSION, None), *leaves]


# Evaluation ----------------------------------------------------------------------------
def boolean(tree: Sequence[Node], data: Mapping[str, np.ndarray], start: int = 0) -> tuple[np.ndarray, int]:
    """Signal of the Boolean subtree at ``start`` (branch or regime), and the index after it."""
    kind, _, payload = tree[start]
    if kind in COMPARISONS:
        values = data[tree[start + 1][2]]
        threshold = tree[start + 2][2]
        with np.errstate(invalid="ignore"):
            signal = values > threshold if kind == GT else values < threshold
        return signal, start + 3
    left, i = boolean(tree, data, start + 1)
    right, i = boolean(tree, data, i)
    return (left & right if kind == AND else left | right), i


def fused(tree: Sequence[Node], signals: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray],
          start: int = 0) -> tuple[np.ndarray, int]:
    """Signal of the fusion subtree at ``start`` given the branches' ``signals``."""
    kind, _, payload = tree[start]
    if kind == LEAF:
        return signals[payload], start + 1
    if kind == F_IF:
        condition, i = boolean(tree, data, start + 1)
        then, i = fused(tree, signals, data, i)
        otherwise, i = fused(tree, signals, data, i)
        return np.where(condition, then, otherwise), i
    children, i = [], start + 1
    for _ in range(ARITY[kind]):
        child, i = fused(tree, signals, data, i)
        children.append(child)
    if kind == F_AND:
        return children[0] & children[1], i
    if kind == F_OR:
        return children[0] | children[1], i
    if kind == F_ALL:
        return children[0] & children[1] & children[2], i
    if kind == F_ANY:
        return children[0] | children[1] | children[2], i
    votes = children[0].astype(np.int8) + children[1] + children[2]
    return votes >= 2, i


def signals(individual: Individual, data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Buy signals of each branch and of the whole individual ("C")."""
    out = {s: boolean(tree, data)[0] for s, tree in individual.branches.items()}
    out["C"] = fused(individual.fusion, out, data)[0]
    return out


def active_sources(individual: Individual, data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """For each branch, the days on which the whole signal depends on it.

    A branch is active on a day when its leaf lies on the path that the IF nodes
    of the fusion tree choose that day; without IF nodes every branch in the
    fusion tree is always active. This is the study's measure of which analysis
    type a strategy listens to, and when (research question 4).
    """
    n = len(next(iter(data.values()))) if data else 0
    out = {s: np.zeros(n, dtype=bool) for s in individual.branches}
    _active(individual.fusion, data, 0, np.ones(n, dtype=bool), out)
    return out


def _active(tree, data, start, mask, out) -> int:
    kind, _, payload = tree[start]
    if kind == LEAF:
        out[payload] |= mask
        return start + 1
    if kind == F_IF:
        condition, i = boolean(tree, data, start + 1)
        i = _active(tree, data, i, mask & condition, out)
        return _active(tree, data, i, mask & ~condition, out)
    i = start + 1
    for _ in range(ARITY[kind]):
        i = _active(tree, data, i, mask, out)
    return i


# Printing ------------------------------------------------------------------------------
def text(tree: Sequence[Node], start: int = 0) -> str:
    return _text(tree, start)[0]


def _text(tree, start):
    kind, _, payload = tree[start]
    if kind == VAR:
        return str(payload), start + 1
    if kind == ERC:
        return f"{payload:.3f}", start + 1
    if kind == LEAF:
        return str(payload), start + 1
    parts, i = [], start + 1
    for _ in range(ARITY[kind]):
        part, i = _text(tree, i)
        parts.append(part)
    return f"{kind}({', '.join(parts)})", i


def describe(individual: Individual) -> str:
    lines = [f"fusion: {text(individual.fusion)}"]
    lines += [f"{s}: {text(tree)}" for s, tree in individual.branches.items()]
    return "\n".join(lines)
