"""The GP loop shared by every rung of the ablation ladder (docs/ENGINE.md).

One generation: the best individuals are copied (elitism); with the active
trading operator on, its three offspring are added (ActTrade, Algorithm 1);
the rest of the population is bred by crossover (probability ``crossover``) or
point mutation. The first parent of a crossover, or the parent of a mutation,
is selected by the selection method with probability p1 and is one of the three
offspring with probabilities p2, p3 and p4 (ActTrade, Algorithm 2); the second
parent of a crossover always comes from the selection method.

The run's model is the individual with the best weighted-sum fitness in the
last generation, whatever the selection method, so that every rung picks its
model the same way. An optional checkpoint sees the best individual every few
generations (the GP-parameter pilot evaluates it on the validation year, to
see what stopping there would give).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from regime_fusion.engine import operators, selection
from regime_fusion.engine.fitness import Evaluator
from regime_fusion.engine.tree import Grammar, Individual

SELECTIONS = ("tournament", "lexicase", "dalex")


@dataclass(frozen=True)
class GPConfig:
    population: int = 1000
    generations: int = 50
    crossover: float = 0.95
    tournament: int = 4
    initial_depth: int = 2           # branch depth; ActTrade's initial depth 3 counts its root
    flip: float = 0.5                # point-mutation probability per node (ActTrade)
    elitism: int = 1
    active_trading: bool = True
    first_parent: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
    selection: str = "tournament"
    particularity: float = 20.0      # DALex

    def __post_init__(self) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        if abs(sum(self.first_parent) - 1.0) > 1e-9:
            raise ValueError("first-parent probabilities must sum to 1")


@dataclass
class Evolution:
    best: Individual
    history: list[dict] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)


def evolve(grammar: Grammar, evaluator: Evaluator, config: GPConfig, rng: np.random.Generator,
           checkpoint: Callable[[Individual], dict] | None = None, every: int = 10) -> Evolution:
    """Evolve a population; ``checkpoint`` (if given) sees the best individual of generation 0,
    of every ``every``-th generation and of the last one, and its results are kept."""
    if config.active_trading and len(grammar.sources) != 3:
        raise ValueError("the active trading operator needs ActTrade's three branches")
    population = grammar.ramped_population(config.population, config.initial_depth, rng)
    _evaluate(population, evaluator)
    history = [_summary(0, population)]
    checkpoints = []
    for generation in range(0, config.generations + 1):
        if generation:
            population = _next_generation(population, grammar, evaluator, config, rng)
            _evaluate(population, evaluator)
            evaluator.retain(population)
            history.append(_summary(generation, population))
        if checkpoint and (generation % every == 0 or generation == config.generations):
            best = max(population, key=lambda individual: individual.fitness)
            checkpoints.append({"generation": generation, "best_fitness": best.fitness, **checkpoint(best)})
    best = max(population, key=lambda individual: individual.fitness)
    return Evolution(best, history, checkpoints)


def active_trading_offspring(population: list[Individual], fusion: list, min_trades: int = 0) -> list[Individual]:
    """ActTrade's three offspring: each takes one source's highest-Sharpe branch and the
    other sources' most-trading branches; ties go to the smaller branch. A branch with fewer
    than ``min_trades`` trades has a Sharpe ratio of zero here, as in the fitness."""
    sources = list(population[0].branches)
    most_trades, best_sharpe = {}, {}
    for s in sources:
        branches = [(ind.stats[s]["trades"],
                     ind.stats[s]["sharpe"] if ind.stats[s]["trades"] >= min_trades else 0.0,
                     -len(ind.branches[s]), ind.branches[s])
                    for ind in population]
        most_trades[s] = max(branches, key=lambda b: (b[0], b[2]))[3]
        best_sharpe[s] = max(branches, key=lambda b: (b[1], b[2]))[3]
    offspring = []
    for chosen in sources:
        branches = {s: list(best_sharpe[s] if s == chosen else most_trades[s]) for s in sources}
        offspring.append(Individual(list(fusion), branches))
    return offspring


def _next_generation(population, grammar, evaluator, config, rng):
    fitness = np.array([individual.fitness for individual in population])
    order = np.argsort(-fitness, kind="stable")
    following = [population[i] for i in order[:config.elitism]]
    offspring = []
    if config.active_trading:
        offspring = active_trading_offspring(population, population[order[0]].fusion,
                                             evaluator.settings.min_trades)
        following += offspring

    cases = epsilon = scores = None
    if config.selection != "tournament":
        cases = np.array([individual.stats["cases"] for individual in population])
    if config.selection == "lexicase":
        epsilon = selection.mad_epsilon(cases)
    elif config.selection == "dalex":
        scores = selection.dalex_scores(cases)  # once per generation rather than once per pick

    def pick() -> Individual:
        if config.selection == "tournament":
            return population[selection.tournament(fitness, config.tournament, rng)]
        if config.selection == "lexicase":
            return population[selection.epsilon_lexicase(cases, epsilon, rng)]
        return population[selection.dalex(scores, config.particularity, rng)]

    def first_parent() -> Individual:
        if not offspring:
            return pick()
        choice = rng.choice(4, p=config.first_parent)
        return pick() if choice == 0 else offspring[choice - 1]

    while len(following) < config.population:
        if rng.random() < config.crossover:
            child = operators.crossover(first_parent(), pick(), grammar, rng)
        else:
            child = operators.mutate(first_parent(), grammar, rng, config.flip)
        following.append(child)
    return following


def _evaluate(population: list[Individual], evaluator: Evaluator) -> None:
    for individual in population:
        if not individual.stats:
            evaluator.evaluate(individual)


def _summary(generation: int, population: list[Individual]) -> dict:
    fitness = np.array([individual.fitness for individual in population])
    best = population[int(np.argmax(fitness))]
    trades = np.array([individual.stats["C"]["trades"] for individual in population])
    return {
        "generation": generation,
        "best_fitness": float(fitness.max()),
        "mean_fitness": float(fitness.mean()),
        "best_sharpe": best.stats["C"]["sharpe"],
        "best_trades": best.stats["C"]["trades"],
        "no_trade_share": float((trades == 0).mean()),
        "distinct": len({individual.key() for individual in population}),
        "mean_size": float(np.mean([individual.size() for individual in population])),
    }
