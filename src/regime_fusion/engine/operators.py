"""Typed variation operators, as in ActTrade.

* Crossover recombines each branch in turn (FA, then SA, then TA, and the fusion
  tree when it evolves): a random node of the first parent's tree is replaced by
  a subtree of the second parent's tree of the same class. Function nodes are
  picked 90% of the time and terminals 10% (Koza's bias, as in ActTrade). A child
  deeper than allowed keeps the first parent's tree: a branch deeper than its
  maximum, or a fusion tree whose fusion nodes or IF conditions are deeper than
  theirs (``Grammar.max_fusion_depth``, ``Grammar.max_regime_depth``).
* Point mutation changes one tree, picked at random (ActTrade's paper: "point
  mutation occurs in one of the three branches"; its code mutates the whole
  tree). Each node of that tree changes with probability ``flip``: AND<->OR,
  GT<->LT, an indicator for another of its type, an ERC for a new one; in the
  fusion tree AND<->OR, ALL/ANY/VOTE among themselves, a leaf for another branch.

Node classes keep trees well typed: a Boolean node of type t is exchanged only
with a Boolean node of type t, an indicator with an indicator of its type, an ERC
with an ERC, and a fusion subtree with a fusion subtree.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from regime_fusion.engine.tree import (
    AND, COMPARISONS, ERC, ERC_RANGE, F_ALL, F_AND, F_ANY, F_OR, F_VOTE, FUSION, GT, LEAF, LOGIC,
    LT, OR, VAR, Grammar, Individual, Node, Tree, arity, depth, end_of, fusion_shape,
)

FUNCTION_BIAS = 0.9
_TRIPLES = (F_ALL, F_ANY, F_VOTE)


def node_class(node: Node) -> tuple:
    kind, kind_type, _ = node
    if kind_type == FUSION:
        return ("fusion",)
    if kind == VAR:
        return ("var", kind_type)
    if kind == ERC:
        return ("erc",)
    return ("bool", kind_type)


def subtree_crossover(first: Tree, second: Tree, rng: np.random.Generator, max_depth: int,
                      fits=None) -> Tree:
    """``fits`` (the child is allowed) replaces the depth limit when given."""
    functions = [i for i, node in enumerate(first) if arity(node) > 0]
    terminals = [i for i, node in enumerate(first) if arity(node) == 0]
    pool = functions if functions and rng.random() < FUNCTION_BIAS else terminals
    i = pool[rng.integers(len(pool))]
    wanted = node_class(first[i])
    matches = [j for j, node in enumerate(second) if node_class(node) == wanted]
    if not matches:
        return list(first)
    j = matches[rng.integers(len(matches))]
    child = first[:i] + second[j:end_of(second, j)] + first[end_of(first, i):]
    allowed = fits(child) if fits is not None else depth(child) <= max_depth
    return child if allowed else list(first)


def crossover(first: Individual, second: Individual, grammar: Grammar,
              rng: np.random.Generator) -> Individual:
    branches = {s: subtree_crossover(first.branches[s], second.branches[s], rng, grammar.max_branch_depth)
                for s in grammar.sources}
    fusion = list(first.fusion)
    if not grammar.fixed_fusion:
        def fits(child: Tree) -> bool:
            fusion_depth, condition_depth = fusion_shape(child)
            return fusion_depth <= grammar.max_fusion_depth and condition_depth <= grammar.max_regime_depth

        fusion = subtree_crossover(first.fusion, second.fusion, rng, 0, fits)
    return Individual(fusion, branches)


def point_mutation(tree: Sequence[Node], grammar: Grammar, rng: np.random.Generator, flip: float) -> Tree:
    out = list(tree)
    for i, (kind, kind_type, payload) in enumerate(out):
        if rng.random() >= flip:
            continue
        if kind_type == FUSION:
            out[i] = _mutate_fusion_node(out[i], grammar, rng)
        elif kind in LOGIC:
            out[i] = (OR if kind == AND else AND, kind_type, None)
        elif kind in COMPARISONS:
            out[i] = (LT if kind == GT else GT, kind_type, None)
        elif kind == VAR:
            names = [n for n in grammar.indicators[kind_type] if n != payload]
            if names:
                out[i] = (VAR, kind_type, names[rng.integers(len(names))])
        elif kind == ERC:
            out[i] = (ERC, None, float(rng.uniform(*ERC_RANGE)))
    return out


def mutate(individual: Individual, grammar: Grammar, rng: np.random.Generator, flip: float) -> Individual:
    child = individual.copy()
    targets = list(grammar.sources) + ([] if grammar.fixed_fusion else ["fusion"])
    target = targets[rng.integers(len(targets))]
    if target == "fusion":
        child.fusion = point_mutation(child.fusion, grammar, rng, flip)
    else:
        child.branches[target] = point_mutation(child.branches[target], grammar, rng, flip)
    return child


def _mutate_fusion_node(node: Node, grammar: Grammar, rng: np.random.Generator) -> Node:
    kind, _, payload = node
    allowed = grammar.fusion_functions
    if kind in (F_AND, F_OR):
        other = F_OR if kind == F_AND else F_AND
        return (other, FUSION, None) if other in allowed else node
    if kind in _TRIPLES:
        others = [k for k in _TRIPLES if k != kind and k in allowed]
        return (others[rng.integers(len(others))], FUSION, None) if others else node
    if kind == LEAF:
        others = [s for s in grammar.sources if s != payload]
        return (LEAF, FUSION, others[rng.integers(len(others))]) if others else node
    return node  # IF keeps its shape; its condition mutates through its own nodes
