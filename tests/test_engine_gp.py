import numpy as np
import pytest

from regime_fusion.engine import gp, operators, selection
from regime_fusion.engine.fitness import COMPONENTS, Evaluator, Market, Settings, block_means, sharpe
from regime_fusion.engine.metrics import sharpe as metrics_sharpe
from regime_fusion.engine.tree import (
    ARITY, ERC, FUSION, GT, LEAF, REGIME, SOURCES, VAR, Grammar, Individual, acttrade_fusion,
    boolean, depth, end_of, fused, fusion_shape, signals, text,
)

INDICATORS = {"FA": ["fa:a", "fa:b"], "SA": ["sa:a", "sa:b"], "TA": ["ta:a", "ta:b", "ta:edge"],
              "RG": ["rg:p1"]}
EVOLVING = ("AND", "OR", "ALL", "ANY", "VOTE", "IF")


def check_tree(tree, kind, start=0):
    """Return the index after a well-typed tree of ``kind`` (a source, RG or F)."""
    node_kind, node_type, payload = tree[start]
    if kind == "F":
        assert node_type == FUSION
        if node_kind == LEAF:
            assert payload in SOURCES
            return start + 1
        i = start + 1
        if node_kind == "IF":
            i = check_tree(tree, REGIME, i)
            return check_tree(tree, "F", check_tree(tree, "F", i))
        for _ in range(ARITY[node_kind]):
            i = check_tree(tree, "F", i)
        return i
    assert node_type == kind and node_kind in ("AND", "OR", "GT", "LT")
    if node_kind in ("GT", "LT"):
        var, erc = tree[start + 1], tree[start + 2]
        assert var[0] == VAR and var[1] == kind and var[2] in INDICATORS[kind]
        assert erc[0] == ERC and -1 <= erc[2] <= 1
        return start + 3
    return check_tree(tree, kind, check_tree(tree, kind, start + 1))


def check_individual(individual, grammar):
    for s in SOURCES:
        assert check_tree(individual.branches[s], s) == len(individual.branches[s])
        assert depth(individual.branches[s]) <= grammar.max_branch_depth
    assert check_tree(individual.fusion, "F") == len(individual.fusion)
    fusion_depth, condition_depth = fusion_shape(individual.fusion)
    assert fusion_depth <= grammar.max_fusion_depth and condition_depth <= grammar.max_regime_depth


def test_fusion_depth_and_condition_depth_are_measured_apart():
    leaf = lambda s: ("LEAF", FUSION, s)                                   # noqa: E731
    compare = [("GT", REGIME, None), (VAR, REGIME, "rg:p1"), (ERC, None, 0.1)]
    condition = [("AND", REGIME, None), *compare, *compare]                 # depth 2
    tree = [("IF", FUSION, None), *condition, ("AND", FUSION, None), leaf("FA"), ("OR", FUSION, None),
            leaf("SA"), leaf("TA"), leaf("TA")]
    assert fusion_shape(tree) == (3, 2) and depth(tree) == 3
    chain = [("AND", FUSION, None)] * 4 + [leaf("FA")] * 5                  # AND(AND(AND(AND(FA, FA), ...
    assert fusion_shape(chain) == (4, 0)                                   # too deep for A1 (3)


def test_tree_shape_helpers():
    tree = [("AND", "TA", None), ("GT", "TA", None), (VAR, "TA", "ta:a"), (ERC, None, 0.5),
            ("LT", "TA", None), (VAR, "TA", "ta:b"), (ERC, None, -0.2)]
    assert end_of(tree, 0) == 7 and end_of(tree, 1) == 4 and depth(tree) == 2
    assert text(tree) == "AND(GT(ta:a, 0.500), LT(ta:b, -0.200))"


def test_evaluation_by_hand():
    data = {"ta:a": np.array([0.9, 0.1, np.nan, 0.7]), "ta:b": np.array([-0.5, -0.1, -0.9, 0.0]),
            "rg:p1": np.array([1.0, -1.0, 1.0, -1.0])}
    tree = [("AND", "TA", None), ("GT", "TA", None), (VAR, "TA", "ta:a"), (ERC, None, 0.5),
            ("LT", "TA", None), (VAR, "TA", "ta:b"), (ERC, None, -0.2)]
    np.testing.assert_array_equal(boolean(tree, data)[0], [True, False, False, False])  # NaN is false
    sig = {"FA": np.array([1, 1, 0, 0], bool), "SA": np.array([1, 0, 1, 0], bool),
           "TA": np.array([1, 0, 0, 1], bool)}
    leaves = [(LEAF, FUSION, s) for s in SOURCES]
    for kind, expected in (("ALL", [1, 0, 0, 0]), ("ANY", [1, 1, 1, 1]), ("VOTE", [1, 0, 0, 0])):
        np.testing.assert_array_equal(fused([(kind, FUSION, None), *leaves], sig, data)[0], expected)
    gate = [("IF", FUSION, None), ("GT", REGIME, None), (VAR, REGIME, "rg:p1"), (ERC, None, 0.0),
            (LEAF, FUSION, "FA"), (LEAF, FUSION, "TA")]
    np.testing.assert_array_equal(fused(gate, sig, data)[0], [True, False, False, True])


def test_ramped_population_is_well_typed():
    rng = np.random.default_rng(0)
    for functions in ((), EVOLVING):
        grammar = Grammar(INDICATORS, fusion_functions=functions)
        for individual in grammar.ramped_population(60, 3, rng):
            check_individual(individual, grammar)
            assert all(depth(individual.branches[s]) <= 3 for s in SOURCES)
            if not functions:
                assert individual.fusion == acttrade_fusion()


def test_operators_keep_trees_typed_and_within_depth():
    rng = np.random.default_rng(1)
    for functions in ((), EVOLVING):
        grammar = Grammar(INDICATORS, fusion_functions=functions, max_branch_depth=4)
        population = grammar.ramped_population(40, 3, rng)
        for _ in range(400):
            a, b = population[rng.integers(40)], population[rng.integers(40)]
            child = operators.crossover(a, b, grammar, rng) if rng.random() < 0.5 else \
                operators.mutate(a, grammar, rng, 0.5)
            check_individual(child, grammar)
            population[rng.integers(40)] = child
        if not functions:
            assert all(individual.fusion == acttrade_fusion() for individual in population)


def test_mutation_changes_one_tree_only():
    rng = np.random.default_rng(2)
    grammar = Grammar(INDICATORS)
    parent = grammar.random_individual(3, "full", rng)
    for _ in range(50):
        child = operators.mutate(parent, grammar, rng, 1.0)
        changed = [s for s in SOURCES if child.branches[s] != parent.branches[s]]
        assert len(changed) <= 1 and child.fusion == parent.fusion


def market_and_data(n=600, seed=3):
    """A random walk, and a TA indicator that knows the next five days' return."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    ahead = np.roll(close, -5) / close - 1
    ahead[-5:] = 0
    data = {name: rng.uniform(-1, 1, n) for names in INDICATORS.values() for name in names}
    data["ta:edge"] = np.clip(ahead / np.abs(ahead).max(), -1, 1)
    market = Market(close, close * 1.01, close * 0.99, close, days=5, rate=0.02, cost=0.001)
    return market, data


def test_fitness_is_the_weighted_sum_and_trees_are_cached():
    market, data = market_and_data()
    settings = Settings(weights={"C": 0.25, "FA": 0.25, "SA": 0.25, "TA": 0.25}, min_trades=3, blocks=4)
    evaluator = Evaluator(data, market, settings)
    individual = Grammar(INDICATORS).random_individual(2, "full", np.random.default_rng(4))
    evaluator.evaluate(individual)
    sig = signals(individual, data)
    expected = 0.0
    for component in COMPONENTS:
        daily, trades = market.daily(sig[component])
        value = metrics_sharpe(daily) if trades >= 3 else 0.0
        assert sharpe(daily) == pytest.approx(metrics_sharpe(daily))
        expected += 0.25 * value
    assert individual.fitness == pytest.approx(expected)
    assert individual.stats["cases"].shape == (4 * 4,)
    for k, component in enumerate(("C", *SOURCES)):              # cases of a signal with < 3 trades are zero
        daily, trades = market.daily(sig[component])
        cases = individual.stats["cases"][4 * k:4 * k + 4]
        np.testing.assert_allclose(cases, block_means(daily, 4) if trades >= 3 else 0.0)
    few = Evaluator(data, market, Settings(weights=settings.weights, min_trades=10**6, blocks=4))
    again = individual.copy()
    few.evaluate(again)
    assert again.fitness == 0.0 and not again.stats["cases"].any()
    calls = []
    original = market.daily
    object.__setattr__(market, "daily", lambda s: calls.append(1) or original(s))
    evaluator.evaluate(individual.copy())
    assert calls == []  # every tree came from the cache


def test_selection_methods():
    rng = np.random.default_rng(5)
    fitness = np.array([0.1, 0.9, 0.3])
    assert selection.tournament(fitness, 50, rng) == 1
    cases = np.array([[1.0, 1.0], [0.0, 0.5], [0.9, 0.9]])
    assert selection.epsilon_lexicase(cases, np.zeros(2), rng) == 0
    specialists = np.array([[1.0, 0.0], [0.0, 1.0]])
    picks = {selection.epsilon_lexicase(specialists, np.zeros(2), rng) for _ in range(50)}
    assert picks == {0, 1}
    assert selection.dalex(np.array([[0.2], [0.8], [0.5]]), 20.0, rng) == 1


def test_active_trading_offspring_follow_algorithm_1():
    def individual(trades, sharpes):
        ind = Individual(acttrade_fusion(), {s: [("GT", s, None), (VAR, s, INDICATORS[s][0]), (ERC, None, v)]
                                              for s, v in zip(SOURCES, (0.1, 0.2, 0.3))})
        ind.stats = {s: {"trades": t, "sharpe": h} for s, t, h in zip(SOURCES, trades, sharpes)}
        return ind
    busy = individual((50, 50, 50), (0.1, 0.1, 0.1))
    sharp = individual((5, 5, 5), (2.0, 2.0, 2.0))
    sharp.branches = {s: [("LT", s, None), (VAR, s, INDICATORS[s][1]), (ERC, None, 0.0)] for s in SOURCES}
    offspring = gp.active_trading_offspring([busy, sharp], acttrade_fusion())
    for chosen, child in zip(SOURCES, offspring):
        for s in SOURCES:
            source = sharp if s == chosen else busy
            assert child.branches[s] == source.branches[s]
    lucky = individual((2, 2, 2), (5.0, 5.0, 5.0))                 # two lucky trades: zero, as in the fitness
    lucky.branches = {s: [("GT", s, None), (VAR, s, INDICATORS[s][-1]), (ERC, None, 0.9)] for s in SOURCES}
    offspring = gp.active_trading_offspring([busy, sharp, lucky], acttrade_fusion(), min_trades=3)
    assert [child.branches[s] for child, s in zip(offspring, SOURCES)] == [sharp.branches[s] for s in SOURCES]
    unfloored = gp.active_trading_offspring([busy, sharp, lucky], acttrade_fusion())
    assert unfloored[0].branches["FA"] == lucky.branches["FA"]


@pytest.mark.parametrize("selection_method, functions", [
    ("tournament", ()), ("tournament", EVOLVING), ("lexicase", EVOLVING), ("dalex", EVOLVING)])
def test_short_runs_are_reproducible_and_find_the_planted_edge(selection_method, functions):
    market, data = market_and_data()
    settings = Settings(weights={"C": 0.25, "FA": 0.25, "SA": 0.25, "TA": 0.25}, min_trades=3, blocks=4)
    grammar = Grammar(INDICATORS, fusion_functions=functions)
    config = gp.GPConfig(population=60, generations=8, selection=selection_method)
    first = gp.evolve(grammar, Evaluator(data, market, settings), config, np.random.default_rng(6))
    again = gp.evolve(grammar, Evaluator(data, market, settings), config, np.random.default_rng(6))
    assert first.best.key() == again.best.key()
    check_individual(first.best, grammar)
    assert len(first.history) == 9
    assert first.best.stats["TA"]["sharpe"] > 1.0  # the TA branch found the indicator that sees ahead
    assert "ta:edge" in text(first.best.branches["TA"])


def test_active_sources_follow_the_regime_gates():
    from regime_fusion.engine.tree import active_sources

    branch = {s: [("GT", s, None), (VAR, s, f"{s.lower()}:x"), (ERC, None, 0.0)] for s in SOURCES}
    data = {"rg:p1": np.array([0.5, -0.5, np.nan, 0.9]),
            **{f"{s.lower()}:x": np.zeros(4) for s in SOURCES}}
    gate = [("IF", FUSION, None), ("GT", REGIME, None), (VAR, REGIME, "rg:p1"), (ERC, None, 0.0),
            (LEAF, FUSION, "FA"), ("OR", FUSION, None), (LEAF, FUSION, "SA"), (LEAF, FUSION, "TA")]
    active = active_sources(Individual(gate, branch), data)
    assert active["FA"].tolist() == [True, False, False, True]   # where the regime condition holds
    assert active["SA"].tolist() == [False, True, True, False]   # otherwise (a missing value is false)
    assert active["TA"].tolist() == active["SA"].tolist()
    fixed = active_sources(Individual(acttrade_fusion(), branch), data)
    assert all(v.all() for v in fixed.values())


def test_saved_models_give_the_same_signals_again():
    import json

    from regime_fusion.engine.tree import Individual, describe, from_record, signals, to_record

    def model(threshold):
        return Individual([("LEAF", "F", "FA")], {"FA": [("GT", "FA", None), ("VAR", "FA", "fa:x"),
                                                         ("ERC", None, threshold)]})

    low, high = model(0.12340), model(0.12349)
    data = {"fa:x": np.array([0.12345, 0.2, -0.5])}
    assert describe(low) == describe(high)                     # the text rounds both to 0.123
    assert signals(low, data)["C"][0] and not signals(high, data)["C"][0]
    for individual in (low, high):
        again = from_record(json.loads(json.dumps(to_record(individual))))   # as a result file keeps it
        assert again.key() == individual.key()
        np.testing.assert_array_equal(signals(again, data)["C"], signals(individual, data)["C"])

