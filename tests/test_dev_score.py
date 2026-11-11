import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.dev import crosssection as C
from regime_fusion.dev import score as S
from regime_fusion.dev.fitness import FLOOR, psr_lower_bound
from regime_fusion.testing import check_no_lookahead

NAN = np.nan


def approx(x, **kw):
    return pytest.approx(np.asarray(x, dtype=float), **kw)
GROUPS = {"FA": ["fa:a", "fa:b", "fa:c"], "TA": ["ta:a", "ta:b", "ta:c"]}
IF = ("rg:s", "rg:m")


def make_space(if_inputs=IF):
    return S._Space(GROUPS, tuple(if_inputs))


def rank_inputs(rng, names, t, n, missing=0.1):
    out = {}
    for name in names:
        values = S.rank_scale(rng.normal(size=(t, n)), np.ones((t, n), dtype=bool))
        values[rng.random((t, n)) < missing] = NAN
        out[name] = values
    return out


def valid_tree(tree, space_names, conds_ok):
    """Parses completely; an IFS has a COND first; VAR names are the branch's; COND names are allowed."""
    if S._end(tree, 0) != len(tree):
        return False
    for i, (kind, payload) in enumerate(tree):
        if kind == "IFS" and tree[i + 1][0] != "COND":
            return False
        if kind == "VAR" and payload not in space_names:
            return False
        if kind == "COND" and (payload[1] not in conds_ok or tree[i - 1][0] != "IFS"):
            return False
    return True


# ------------------------------------------------------------------------------------------ nodes by hand
X = np.array([[0.5, -1.0, NAN], [0.2, 0.4, 1.0]])
Y = np.array([[-0.5, 0.5, 1.0], [NAN, 0.0, -1.0]])
Z = np.array([[0.3, NAN, -0.7], [0.9, -0.9, 0.0]])
DATA = {"fa:x": X, "fa:y": Y, "rg:z": Z}


def ev(*nodes, leaves=None):
    data = S._Data(DATA, np.ones((2, 3), dtype=bool))
    out, end = S._eval(list(nodes), 0, data, leaves)
    assert end == len(nodes)
    return np.broadcast_to(out, (2, 3))


VX, VY = ("VAR", "fa:x"), ("VAR", "fa:y")


def test_arithmetic_nodes_with_missing_values_counting_as_zero():
    # x filled: [[.5, -1, 0], [.2, .4, 1]]; y filled: [[-.5, .5, 1], [0, 0, -1]]
    assert ev(("ADD", None), VX, VY) == approx([[0.0, -0.5, 1.0], [0.2, 0.4, 0.0]])
    assert ev(("SUB", None), VX, VY) == approx([[1.0, -1.5, -1.0], [0.2, 0.4, 2.0]])
    assert ev(("MUL", None), VX, VY) == approx([[-0.25, -0.5, 0.0], [0.0, 0.0, -1.0]])
    assert ev(("MIN", None), VX, VY) == approx([[-0.5, -1.0, 0.0], [0.0, 0.0, -1.0]])
    assert ev(("MAX", None), VX, VY) == approx([[0.5, 0.5, 1.0], [0.2, 0.4, 1.0]])
    assert ev(("NEG", None), VX) == approx([[-0.5, 1.0, 0.0], [-0.2, -0.4, -1.0]])
    assert ev(("CONST", 0.7)) == approx(np.full((2, 3), 0.7))
    assert ev(("ADD", None), ("CONST", 0.25), VY) == approx([[-0.25, 0.75, 1.25], [0.25, 0.25, -0.75]])


def test_ifs_in_both_directions_and_missing_condition_inputs():
    # z > 0: [[T, F (missing), F], [T, F, F]]  -> a = x where true, b = y elsewhere
    gt = ev(("IFS", None), ("COND", ("GT", "rg:z", 0.0)), VX, VY)
    assert gt == approx([[0.5, 0.5, 1.0], [0.2, 0.0, -1.0]])
    # z < 0: [[F, F (missing), T], [F, T, F]]
    lt = ev(("IFS", None), ("COND", ("LT", "rg:z", 0.0)), VX, VY)
    assert lt == approx([[-0.5, 0.5, 0.0], [0.0, 0.4, -1.0]])
    # a comparison is strict: z = 0 holds neither way
    assert ev(("IFS", None), ("COND", ("GT", "rg:z", 0.0)), ("CONST", 1.0), ("CONST", -1.0))[1, 2] == -1.0
    assert ev(("IFS", None), ("COND", ("LT", "rg:z", 0.0)), ("CONST", 1.0), ("CONST", -1.0))[1, 2] == -1.0


def test_fusion_nodes():
    fa = np.array([[1.0, -1.0, 0.0], [0.5, 0.5, -0.5]])
    ta = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    leaves = {"FA": fa, "TA": ta}
    leaf = lambda s: ("LEAF", s)                                              # noqa: E731
    assert ev(("WSUM", 0.25), leaf("FA"), leaf("TA"), leaves=leaves) == approx(0.25 * fa + 0.75 * ta)
    assert ev(("WSUM", 1.0), leaf("FA"), leaf("TA"), leaves=leaves) == approx(fa)
    assert ev(("MIN", None), leaf("FA"), leaf("TA"), leaves=leaves) == approx(np.minimum(fa, ta))
    assert ev(("MAX", None), leaf("FA"), leaf("TA"), leaves=leaves) == approx(np.maximum(fa, ta))
    # z > 0 holds only at [0,0] and [1,0]: FA there, TA elsewhere
    got = ev(("IFS", None), ("COND", ("GT", "rg:z", 0.0)), leaf("FA"), leaf("TA"), leaves=leaves)
    assert got == approx([[1.0, 1.0, 0.0], [0.5, -1.0, 1.0]])


def test_rank_scale_by_hand_and_against_pandas():
    valid = np.array([[True, True, True, False]])
    assert S.rank_scale(np.array([[3.0, 1.0, 2.0, 9.0]]), valid)[0] == approx([1.0, -1.0, 0.0, NAN], nan_ok=True)
    # a tie of the two top values shares ranks 1 and 2 (0-based) of three: (1.5 x 2 / 2) - 1 = 0.5
    assert S.rank_scale(np.array([[5.0, 5.0, 1.0]]), np.ones((1, 3), bool))[0] == approx([0.5, 0.5, -1.0])
    assert S.rank_scale(np.array([[5.0, 4.0]]), np.array([[True, False]]))[0] == approx([0.0, NAN], nan_ok=True)
    rng = np.random.default_rng(3)
    values = np.round(rng.normal(size=(20, 30)), 1)
    valid = rng.random(values.shape) < 0.8
    got = S.rank_scale(values, valid)
    for k in range(20):
        v = pd.Series(np.where(valid[k], values[k], NAN))
        r = v.rank().to_numpy() - 1.0
        n = valid[k].sum()
        expect = np.where(valid[k], 2.0 * r / (n - 1) - 1.0, NAN)
        assert got[k] == approx(expect, nan_ok=True, abs=1e-12)


def test_model_score_by_hand():
    # FA ranks among the three tradable stocks: a = [.9, -.9, 0] -> [1, -1, 0]; TA: b = [.1, .3, .2] -> [-1, 1, 0]
    # fusion 0.75 FA + 0.25 TA = [.5, -1, 0] -> ranked [1, -1, 0]; the untradable fourth stock is NaN
    inputs = {"fa:a": np.array([[0.9, -0.9, 0.0, 0.5]]), "ta:b": np.array([[0.1, 0.3, 0.2, -1.0]])}
    model = S.ScoreModel([("WSUM", 0.75), ("LEAF", "FA"), ("LEAF", "TA")],
                         {"FA": [("VAR", "fa:a")], "TA": [("VAR", "ta:b")]})
    tradable = np.array([[True, True, True, False]])
    assert model.score(inputs, tradable)[0] == approx([1.0, -1.0, 0.0, NAN], nan_ok=True)
    assert model.inputs_used() == ["fa:a", "ta:b"]


def test_inputs_used_ignores_branches_the_fusion_never_reads():
    fusion = [("MAX", None), ("LEAF", "FA"), ("LEAF", "FA")]
    model = S.ScoreModel(fusion, {"FA": [("MUL", None), ("VAR", "fa:a"), ("CONST", 0.0)], "TA": [("VAR", "ta:a")]})
    assert model.inputs_used() == ["fa:a"]
    both = S.ScoreModel([("IFS", None), ("COND", ("GT", "rg:s", 0.1)), ("LEAF", "FA"), ("LEAF", "TA")],
                        {"FA": [("IFS", None), ("COND", ("LT", "rg:m", 0.0)), ("VAR", "fa:b"), ("CONST", 1.0)],
                         "TA": [("VAR", "ta:c")]})
    assert both.inputs_used() == ["fa:b", "rg:m", "rg:s", "ta:c"]
    assert both.describe().splitlines() == [
        "FA: IFS(rg:m < 0.000, fa:b, 1.000)", "TA: ta:c", "fusion: IFS(rg:s > 0.100, FA, TA)"]


# ------------------------------------------------------------------------------------------ size limits
@pytest.mark.parametrize("if_inputs", [(), IF])
def test_size_limits_and_types_hold_for_random_models_and_operators(if_inputs):
    space = make_space(if_inputs)
    rng = np.random.default_rng(11)
    models = S._ramped(space, 200, 2, rng)
    allowed = set(if_inputs)

    def check(model):
        assert S.within_limits(model)
        assert valid_tree(model.fusion, (), allowed)
        for source, tree in model.branches.items():
            assert valid_tree(tree, GROUPS[source], allowed), (source, tree)
        assert all(n[0] in ("WSUM", "MIN", "MAX", "IFS", "LEAF", "COND") for n in model.fusion)
        if not if_inputs:
            assert not any(n[0] in ("IFS", "COND") for t in [*model.branches.values(), model.fusion] for n in t)
        for n in model.fusion:
            if n[0] == "WSUM":
                assert 0.0 <= n[1] <= 1.0
            if n[0] == "COND":
                assert -1.0 <= n[1][2] <= 1.0

    for model in models:
        check(model)
    for _ in range(200):
        a, b = models[rng.integers(200)], models[rng.integers(200)]
        child = S.crossover(a, b, space, rng)
        check(child)
        check(S.mutate(child, space, rng))
    # the sizes actually vary and reach the limits
    assert max(len(m.fusion) for m in models) == S.FUSION_NODES
    assert max(len(m.branches["FA"]) for m in models) > 7


def test_typed_crossover_never_mixes_sources():
    space = make_space()
    rng = np.random.default_rng(5)
    models = S._ramped(space, 60, 2, rng)
    for _ in range(300):
        child = S.crossover(models[rng.integers(60)], models[rng.integers(60)], space, rng)
        for source, tree in child.branches.items():
            assert {n[1] for n in tree if n[0] == "VAR"} <= set(GROUPS[source])
        assert {n[1] for n in child.fusion if n[0] == "LEAF"} <= {"FA", "TA"}
        assert not any(n[0] in ("VAR", "CONST") for n in child.fusion)


# ------------------------------------------------------------------------------------------ the market
def random_market_case(seed, t=25, n=40, ties=True):
    rng = np.random.default_rng(seed)
    hedged = rng.normal(0, 0.05, (t, n))
    hedged[rng.random((t, n)) < 0.2] = NAN
    cost = rng.choice([0.001, 0.003, 0.01], size=(t, n))
    cost[rng.random((t, n)) < 0.1] = NAN
    score = rng.normal(size=(t, n))
    if ties:
        score = np.round(score, 1)
    score[rng.random((t, n)) < 0.1] = NAN
    return hedged, cost, score


@pytest.mark.parametrize("enter,leave", [(0.2, 0.4), (0.1, 0.5), (0.25, 0.25)])
def test_score_market_weights_equal_banded_weights(enter, leave):
    for seed in range(5):
        hedged, cost, score = random_market_case(seed)
        index = pd.RangeIndex(len(hedged))
        expect = C.banded_weights(pd.DataFrame(score, index=index), pd.DataFrame(np.isfinite(hedged), index=index),
                                  enter, leave, long_only=True).to_numpy()
        market = S.ScoreMarket(hedged, cost, enter, leave)
        np.testing.assert_array_equal(market.weights(score), expect)
        # and the money: portfolio_returns agrees with the market's gross and costs
        result = C.portfolio_returns(pd.DataFrame(expect, index=index), pd.DataFrame(hedged, index=index),
                                     pd.DataFrame(cost, index=index))
        base = np.nanmean(hedged, axis=1)
        held = expect.sum(axis=1) > 0
        assert market.gross(score) == approx(np.where(held, result["gross"] - base, 0.0))
        assert market.costs(score) == approx(result["cost"].to_numpy())
        assert market.turnover(score) == approx(result["turnover"].to_numpy())


def reference_held(score, tradable, enter, leave, initial):
    """A plain loop: percentile among the tradable stocks with a score, then enter / stay."""
    now = initial > 0
    out = []
    for k in range(len(score)):
        s = pd.Series(np.where(tradable[k], score[k], NAN))
        pct = s.rank(pct=True).to_numpy()
        ok = np.isfinite(pct)
        with np.errstate(invalid="ignore"):
            now = ok & ((pct > 1 - enter) | (now & (pct > 1 - leave)))
        out.append(now.copy())
    return np.array(out)


def test_initial_holdings_seed_the_band():
    for seed in range(4):
        hedged, cost, score = random_market_case(seed + 10)
        rng = np.random.default_rng(seed)
        initial = np.where(rng.random(hedged.shape[1]) < 0.3, 0.1, 0.0)
        market = S.ScoreMarket(hedged, cost, initial=initial)
        np.testing.assert_array_equal(market.held(score),
                                      reference_held(score, np.isfinite(hedged), 0.2, 0.4, initial))
        # zero initial holdings are the plain banded book
        plain = S.ScoreMarket(hedged, cost)
        np.testing.assert_array_equal(plain.held(score), reference_held(score, np.isfinite(hedged), 0.2, 0.4,
                                                                         np.zeros(hedged.shape[1])))
    # by hand: 5 stocks, one date, scores rising with the stock number. Percentiles .2 .4 .6 .8 1.
    # Stock 5 enters (above .8); a held stock 4 (pct .8 > .6) stays; a held stock 2 (pct .4) leaves, and pays.
    m = S.ScoreMarket(np.full((1, 5), 0.01), np.full((1, 5), 0.01), initial=np.array([0, 0.5, 0, 0.5, 0]))
    score = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert m.held(score)[0].tolist() == [False, False, False, True, True]
    # weights go 0, .5, 0, .5, 0 -> 0, 0, 0, .5, .5: changes .5 (stock 2) + 0 (stock 4) + .5 (stock 5) = 1.0
    assert m.turnover(score) == approx([1.0])


def test_contributions_sum_to_the_net_result_and_an_empty_book_scores_zero():
    for seed in range(3):
        hedged, cost, score = random_market_case(seed + 20)
        initial = np.zeros(hedged.shape[1])
        initial[:4] = 0.25
        market = S.ScoreMarket(hedged, cost, initial=initial)
        net, active, evidence = market.simulate(score)
        assert market.contributions(score).sum(axis=1) == approx(net)
        assert evidence is net
        assert active == int((np.isfinite(hedged).sum(axis=1) > 0).sum())
        assert market.daily(score)[1] == active
    hedged, cost, _ = random_market_case(1)
    market = S.ScoreMarket(hedged, cost)
    nothing = np.full(hedged.shape, NAN)
    assert not market.gross(nothing).any()
    assert not market.simulate(nothing)[0].any()
    assert not market.contributions(nothing).any()
    assert market.share(nothing).sum() == 0.0


def test_bounds_on_active_dates_when_given():
    hedged = np.full((3, 10), 0.01)
    score = np.tile(np.arange(10.0), (3, 1))
    # band 20/40 of 10 stocks holds 2 (the top two: pct .9 and 1.0), a share of 0.2 each date
    assert S.ScoreMarket(hedged, hedged).simulate(score)[1] == 3
    assert S.ScoreMarket(hedged, hedged, low=0.1, high=0.3).simulate(score)[1] == 3
    assert S.ScoreMarket(hedged, hedged, low=0.25).simulate(score)[1] == 0
    assert S.ScoreMarket(hedged, hedged, high=0.1).simulate(score)[1] == 0


def test_market_dates_use_earlier_rows_only():
    hedged, cost, score = random_market_case(7, ties=False)
    market = S.ScoreMarket(hedged, cost)
    full = market.simulate(score)[0]
    for k in (3, 10, 20):
        part = S.ScoreMarket(hedged[:k], cost[:k]).simulate(score[:k])[0]
        np.testing.assert_array_equal(part, full[:k])


# ------------------------------------------------------------------------------------------ fitnesses
def fitness_case():
    names = {"fa:a": None, "ta:b": None}
    a = np.tile(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), (3, 1))
    inputs = {"fa:a": a, "ta:b": a.copy()}
    model = S.ScoreModel([("LEAF", "FA")], {"FA": [("VAR", "fa:a")], "TA": [("VAR", "ta:b")]})
    return names, inputs, model


def test_book_fitness_by_hand():
    _, inputs, model = fitness_case()
    # score = [-1, -.5, 0, .5, 1] on three dates; band 20/40 of five stocks holds only the top one
    # (percentiles .2 .4 .6 .8 1.0: only 1.0 is strictly above .8; .8 is above the leave line .6 but never enters).
    # hedged returns rise with the stock number, so their ranks are [-1, -.5, 0, .5, 1], their mean 0.
    # Gross on the ranks = 1 each date. The interquartile range of hedged .01..05 is .02 (25th pct .02, 75th .04),
    # so a cost of .002 per side is .1 in rank units: the entry on date 0 costs 1 x .1; nothing after.
    hedged = np.tile(np.array([0.01, 0.02, 0.03, 0.04, 0.05]), (3, 1))
    cost = np.full((3, 5), 0.002)
    ev_ = S.ScoreEvaluator(inputs, hedged, cost, "book")
    assert ev_.evaluate(model) == approx(psr_lower_bound(np.array([0.9, 1.0, 1.0])))
    assert ev_.market.simulate(_score := model.score(inputs, np.isfinite(hedged)))[0] == approx([0.9, 1.0, 1.0])
    assert model.fitness == ev_._cache[model.key()]


def test_rank_fitness_by_hand_and_short_dates_are_skipped():
    _, _, model = fitness_case()
    order = np.arange(10) / 9.0 * 2 - 1
    a = np.tile(order, (5, 1))
    inputs = {"fa:a": a, "ta:b": a.copy()}
    hedged = np.tile(np.arange(10) * 0.01, (5, 1))
    hedged[1] = hedged[1][::-1]                 # date 1: returns in the reverse order of the score, correlation -1
    hedged[4, 5:] = NAN                         # date 4: only five tradable stocks, skipped
    hedged[4, :5] = np.arange(5)[::-1] * 0.01   # (it would have been -1 as well)
    ev_ = S.ScoreEvaluator(inputs, hedged, np.full(hedged.shape, 0.001), "rank")
    # correlations: date 0 +1, date 1 -1, dates 2 and 3 +1 -> [1, -1, 1, 1]
    assert ev_.evaluate(model) == approx(psr_lower_bound(np.array([1.0, -1.0, 1.0, 1.0])))
    assert ev_.evaluate(model) != approx(psr_lower_bound(np.array([1.0, -1.0, 1.0, 1.0, -1.0])))


@pytest.mark.parametrize("kind", ["book", "rank"])
def test_a_constant_score_on_more_than_a_tenth_of_dates_scores_the_floor(kind):
    rng = np.random.default_rng(2)
    t, n = 20, 30
    hedged = rng.normal(0, 0.05, (t, n))
    inputs = rank_inputs(rng, ["fa:a", "ta:b"], t, n, missing=0.0)
    ev_ = S.ScoreEvaluator(inputs, hedged, np.full((t, n), 0.001), kind)
    constant = S.ScoreModel([("LEAF", "FA")], {"FA": [("CONST", 0.3)], "TA": [("VAR", "ta:b")]})
    assert ev_.evaluate(constant) == FLOOR
    # a score that is constant on 2 of 20 dates only (10%: not more) is not floored: input 0 on two dates
    flat = inputs["fa:a"].copy()
    flat[:2] = 0.5
    ev2 = S.ScoreEvaluator({**inputs, "fa:a": flat}, hedged, np.full((t, n), 0.001), kind)
    live = S.ScoreModel([("LEAF", "FA")], {"FA": [("VAR", "fa:a")], "TA": [("VAR", "ta:b")]})
    assert ev2.evaluate(live) > FLOOR
    flat[:3] = 0.5
    ev3 = S.ScoreEvaluator({**inputs, "fa:a": flat}, hedged, np.full((t, n), 0.001), kind)
    assert ev3.evaluate(live) == FLOOR


# ------------------------------------------------------------------------------------------ look-ahead
def test_scores_do_not_look_ahead():
    rng = np.random.default_rng(4)
    t, n = 12, 25
    names = [*GROUPS["FA"], *GROUPS["TA"], *IF]
    inputs = rank_inputs(rng, names, t, n)
    tradable = rng.random((t, n)) < 0.85
    space = make_space()
    model = S._ramped(space, 30, 2, rng)[-1]

    def compute(idx):
        return pd.DataFrame(model.score({k: v[idx] for k, v in inputs.items()}, tradable[idx]))

    check_no_lookahead(compute, np.arange(t), range(t))

    def ensemble(idx):
        models = S._ramped(space, 5, 2, np.random.default_rng(9))
        return pd.DataFrame(S.ensemble_score(models, {k: v[idx] for k, v in inputs.items()}, tradable[idx]))

    check_no_lookahead(ensemble, np.arange(t), range(t))
    # changing later inputs leaves earlier scores alone
    changed = {k: np.vstack([v[:6], rng.normal(size=(t - 6, n))]) for k, v in inputs.items()}
    np.testing.assert_array_equal(model.score(changed, tradable)[:6], model.score(inputs, tradable)[:6])


# ------------------------------------------------------------------------------------------ evolution
def planted_case(seed, t=40, n=300, noise=6, planted_ic=0.10, planted="fa:signal"):
    rng = np.random.default_rng(seed)
    names = {"FA": [f"fa:n{i}" for i in range(noise)] + ["fa:signal"], "TA": [f"ta:n{i}" for i in range(noise)]}
    ones = np.ones((t, n), dtype=bool)
    inputs = {name: S.rank_scale(rng.normal(size=(t, n)), ones) for group in names.values() for name in group}
    z = rng.normal(size=(t, n))
    signal = inputs[planted]
    # returns correlate with the planted rank at about planted_ic: the rank is nearly uniform, so use its own value
    r = planted_ic * np.sqrt(3.0) * signal + np.sqrt(1 - planted_ic ** 2) * z
    hedged = 0.05 * r
    return names, inputs, hedged, np.full((t, n), 0.001)


def spearman_mean(score, hedged):
    out = []
    for s, h in zip(score, hedged):
        ok = np.isfinite(s) & np.isfinite(h)
        out.append(pd.Series(s[ok]).corr(pd.Series(h[ok]), method="spearman"))
    return float(np.mean(out))


@pytest.mark.parametrize("kind", ["book", "rank"])
def test_the_planted_input_is_found(kind):
    names, inputs, hedged, cost = planted_case(100)
    model = S.evolve_score(inputs, names, hedged, cost, fitness=kind, budget={"population": 100, "generations": 8},
                           rng=np.random.default_rng(0))
    assert "fa:signal" in model.inputs_used()
    # fresh test dates drawn the same way (different seed: the noise inputs are new too)
    _, t_inputs, t_hedged, _ = planted_case(101)
    score = model.score(t_inputs, np.isfinite(t_hedged))
    assert spearman_mean(score, t_hedged) > 0.0
    assert len(model.final) == 100 and all(np.isfinite(m.fitness) for m in model.final)
    assert model.fitness == max(m.fitness for m in model.final)


def test_two_runs_with_the_same_generator_state_agree_exactly():
    names, inputs, hedged, cost = planted_case(7, t=15, n=60, noise=3)
    inputs["rg:s"] = np.tile(np.linspace(-1, 1, 15)[:, None], (1, 60))
    runs = [S.evolve_score(inputs, names, hedged, cost, fitness="book", if_inputs=("rg:s",),
                           budget={"population": 30, "generations": 4}, rng=np.random.default_rng(42))
            for _ in range(2)]
    assert runs[0].to_record() == runs[1].to_record()
    assert [m.to_record() for m in runs[0].final] == [m.to_record() for m in runs[1].final]
    other = S.evolve_score(inputs, names, hedged, cost, fitness="book", if_inputs=("rg:s",),
                           budget={"population": 30, "generations": 4}, rng=np.random.default_rng(43))
    assert other.to_record() != runs[0].to_record()


def test_record_round_trip_gives_identical_scores():
    rng = np.random.default_rng(8)
    t, n = 10, 30
    inputs = rank_inputs(rng, [*GROUPS["FA"], *GROUPS["TA"], *IF], t, n)
    tradable = rng.random((t, n)) < 0.9
    for model in S._ramped(make_space(), 40, 2, rng):
        model.fitness = float(rng.normal())
        back = S.ScoreModel.from_record(json.loads(json.dumps(model.to_record())))
        assert back.key() == model.key() and back.fitness == model.fitness
        np.testing.assert_array_equal(back.score(inputs, tradable), model.score(inputs, tradable))
    assert S.ScoreModel.from_record(S._ramped(make_space(), 1, 2, rng)[0].to_record()).fitness != 0.0


def test_ensemble_of_one_model_is_its_score_and_of_several_is_ranked():
    rng = np.random.default_rng(6)
    t, n = 8, 40
    inputs = rank_inputs(rng, [*GROUPS["FA"], *GROUPS["TA"], *IF], t, n)
    tradable = rng.random((t, n)) < 0.9
    models = S._ramped(make_space(), 3, 2, rng)
    np.testing.assert_array_equal(S.ensemble_score(models[:1], inputs, tradable), models[0].score(inputs, tradable))
    both = S.ensemble_score(models, inputs, tradable)
    assert np.array_equal(np.isnan(both), ~tradable)
    assert both[tradable].min() >= -1.0 and both[tradable].max() <= 1.0
    mean = np.mean([m.score(inputs, tradable) for m in models], axis=0)
    np.testing.assert_array_equal(both, S.rank_scale(mean, tradable))
