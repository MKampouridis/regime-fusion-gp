import numpy as np
import pandas as pd
import pytest

from regime_fusion.dev import crosssection as C
from regime_fusion.dev import e9
from regime_fusion.dev import factors as F
from regime_fusion.engine.tree import Grammar, from_record, signals
from regime_fusion.seeds import make_rng
from regime_fusion.testing import check_no_lookahead

NAN = np.nan


def tiny_market(**bounds):
    hedged = np.array([[0.02, 0.04, -0.02, 0.00],
                       [0.01, NAN, 0.03, 0.05],
                       [0.00, 0.02, NAN, 0.04]])
    cost = np.array([[0.01, 0.01, 0.01, 0.01],
                     [0.01, 0.03, 0.01, NAN],
                     [0.01, 0.01, 0.01, 0.01]])
    signal = np.array([[1, 1, 0, 0], [1, 1, 0, 1], [0, 1, 1, 1]], dtype=bool)
    return e9.BookMarket(hedged, cost, **bounds), signal


def test_book_market_on_a_hand_checked_example():
    market, signal = tiny_market(low=0.25, high=0.5)
    # date 0: holds A, B (weights 1/2); mean of all four = 0.01; gross 0.03 - 0.01; entry costs 0.01 per side
    # date 1: B cannot be traded, so the book is A and D; mean of the tradable A, C, D = 0.03;
    #         gross 0.5 x 0.01 + 0.5 x 0.05 - 0.03 = 0; B's exit costs its 0.03, D's entry its missing cost (0.02)
    # date 2: C cannot be traded; the book is B and D; mean of A, B, D = 0.02; gross 0.03 - 0.02;
    #         A's exit and B's entry cost 0.01 each
    assert market.gross(signal) == pytest.approx([0.02, 0.0, 0.01])
    assert market.costs(signal) == pytest.approx([0.01, 0.5 * 0.03 + 0.5 * C.COST_BELOW, 0.01])
    net, active, evidence = market.simulate(signal)
    assert net == pytest.approx([0.01, -0.025, 0.0])
    assert evidence == pytest.approx(net)
    assert market.turnover(signal) == pytest.approx([1.0, 1.0, 1.0])
    assert market.share(signal) == pytest.approx([0.5, 2 / 3, 2 / 3])
    # only the first date holds between 25% and 50% of the tradable stocks
    assert active == 1
    assert market.daily(signal)[1] == 1


def test_active_dates_follow_the_bounds_inclusively():
    market, signal = tiny_market(low=0.5, high=0.7)
    assert market.simulate(signal)[1] == 3          # shares 1/2, 2/3, 2/3
    market, signal = tiny_market(low=0.1, high=0.4)
    assert market.simulate(signal)[1] == 0
    market, signal = tiny_market(low=0.6, high=0.7)
    assert market.simulate(signal)[1] == 2


def test_an_empty_book_has_zero_weights_and_trades_nothing():
    market, _ = tiny_market()
    none = np.zeros((3, 4), dtype=bool)
    assert market.weights(none).sum() == 0
    assert market.costs(none).sum() == 0
    assert market.simulate(none)[1] == 0
    # a held stock that is switched off pays its exit
    some = none.copy()
    some[0, 0] = True
    assert market.costs(some) == pytest.approx([0.01, 0.01, 0.0])


def test_book_market_agrees_with_portfolio_returns():
    rng = np.random.default_rng(7)
    hedged = rng.normal(0, 0.05, (15, 30))
    hedged[rng.random(hedged.shape) < 0.15] = NAN
    cost = rng.choice([0.001, 0.002, 0.01], size=hedged.shape)
    cost[rng.random(cost.shape) < 0.1] = NAN
    signal = rng.random(hedged.shape) < 0.3
    market = e9.BookMarket(hedged, cost)
    index = pd.bdate_range("2015-01-05", periods=15)
    w = pd.DataFrame(market.weights(signal), index=index)
    result = C.portfolio_returns(w, pd.DataFrame(hedged, index=index), pd.DataFrame(cost, index=index))
    tradable_mean = np.nanmean(hedged, axis=1)
    assert market.gross(signal) + tradable_mean == pytest.approx(result["gross"].to_numpy())
    assert market.costs(signal) == pytest.approx(result["cost"].to_numpy())
    assert market.turnover(signal) == pytest.approx(result["turnover"].to_numpy())
    assert market.simulate(signal)[0] + tradable_mean == pytest.approx(result["net"].to_numpy())


def test_a_signal_known_only_from_the_same_dates_inputs_gives_results_that_ignore_later_rows():
    rng = np.random.default_rng(3)
    names = {"FA": ["a", "b"], "TA": ["c", "d"], "RG": ["e"]}
    grammar = Grammar(names, fusion_functions=("AND", "OR", "IF"), sources=("FA", "TA"), max_branch_depth=3)
    hedged = rng.normal(0, 0.05, (12, 10))
    cost = np.full((12, 10), 0.002)
    data = {n: rng.uniform(-1, 1, (12, 10)) for n in "abcde"}
    later = {n: np.vstack([v[:6], rng.uniform(-1, 1, (6, 10))]) for n, v in data.items()}
    later_hedged = np.vstack([hedged[:6], rng.normal(0, 0.05, (6, 10))])
    for _ in range(10):
        individual = grammar.random_individual(3, "grow", rng)
        full = e9.BookMarket(hedged, cost).simulate(signals(individual, data)["C"])[0]
        other = e9.BookMarket(later_hedged, cost).simulate(signals(individual, later)["C"])[0]
        assert other[:6] == pytest.approx(full[:6])


def test_folds_drop_the_last_date_before_the_test_year_and_never_overlap():
    dates = pd.bdate_range("2012-07-02", "2018-12-31")[::20]
    out = e9.folds(dates)
    assert list(out) == [2014, 2015, 2016, 2017, 2018]
    for year, (train, test) in out.items():
        assert set(dates[test].year) == {year}
        assert (dates.year[test] == year).sum() == len(test)
        assert train[0] == 0 and np.array_equal(train, np.arange(len(train)))
        dropped = test[0] - 1
        assert dropped not in train and train[-1] == dropped - 1
        assert not set(train) & set(test)
        assert dates[train].max() < pd.Timestamp(f"{year}-01-01")
    assert set(e9.MAIN_YEARS) == set(out) - {2014}


def test_market_state_by_hand():
    r = pd.Series([0.1, 0.1, -0.1, 0.0], index=pd.bdate_range("2015-01-05", periods=4))
    out = e9.market_state(r, vol_window=2, return_window=2, min_history=2)
    assert list(out.columns) == ["market_volatility", "market_return", "market_drawdown"]
    # volatility of two days: 0, 0.1414, 0.0707 -> percentiles among those to date: 1 (top of two), 2/3 of three
    assert out["market_volatility"].isna().tolist() == [True, True, False, False]
    assert out["market_volatility"].iloc[2:].tolist() == pytest.approx([1.0, 2 * 2 / 3 - 1])
    # drawdown: 0, -0.1, 0 (level over its two-day high); percentiles to date: 1/2 of two, then the top of three
    assert out["market_drawdown"].isna().tolist() == [True, True, False, False]
    assert out["market_drawdown"].iloc[2:].tolist() == pytest.approx([0.0, 1.0])
    # two-day return: -0.01 then -0.1; the first value alone is below the minimum history
    assert out["market_return"].isna().tolist() == [True, True, True, False]
    assert out["market_return"].iloc[3] == pytest.approx(0.0)


def test_market_state_does_not_look_ahead_and_stays_in_range():
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0, 0.01, 80), index=pd.bdate_range("2015-01-05", periods=80))
    check_no_lookahead(lambda s: e9.market_state(s, 5, 8, 6), series, [10, 30, 60, 79])
    out = e9.market_state(series, 5, 8, 6)
    values = out.to_numpy().ravel()
    values = values[np.isfinite(values)]
    assert len(values) > 100 and values.min() >= -1 and values.max() <= 1


def synthetic_panel(stocks=8, dates=12, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2013-01-02", periods=dates * 21)[::21][:dates]
    columns = [f"S{i}" for i in range(stocks)]

    def frame(values):
        return pd.DataFrame(values, index=index, columns=columns)

    member = frame(np.ones((dates, stocks), dtype=bool))
    member.iloc[:3, 0] = False
    factors = {}
    for name in F.SIGNS:
        values = rng.normal(0, 1, (dates, stocks))
        values[rng.random(values.shape) < 0.1] = NAN
        factors[name] = frame(values).where(member)
    forward = frame(rng.normal(0.01, 0.1, (dates, stocks))).where(member)
    value = frame(rng.lognormal(15, 1, (dates, stocks))).where(member)
    beta = frame(rng.uniform(0.5, 1.5, (dates, stocks)))
    return C.Panel(index, factors, forward, member, beta, value, forward.mean(axis=1))


def test_build_inputs_names_ranges_and_unsigned_factors():
    panel = synthetic_panel()
    rng = np.random.default_rng(1)
    sessions = pd.bdate_range(end=panel.dates[-1], periods=700)
    market = pd.Series(rng.normal(0, 0.01, len(sessions)), index=sessions)
    inputs = e9.build_inputs(panel, market)
    expected = ([f"fa:{f}" for f in e9.FA_FACTORS] + [f"ta:{f}" for f in e9.TA_FACTORS]
                + list(e9.CONTEXT) + list(e9.STATE))
    assert sorted(inputs) == sorted(expected) and len(expected) == 22
    assert set(e9.FA_FACTORS) | set(e9.TA_FACTORS) == set(F.SIGNS)
    for name, values in inputs.items():
        assert values.shape == panel.member.shape
        finite = values[np.isfinite(values)]
        assert len(finite) > 0 and finite.min() >= -1 and finite.max() <= 1, name
    # unsigned: a higher raw value gives a higher rank, even for factors the literature signs negatively
    for factor, key in (("rev_1m", "ta:rev_1m"), ("leverage", "fa:leverage"), ("size", "ta:size")):
        assert F.SIGNS[factor] == -1
        row = 5
        raw = panel.factors[factor].iloc[row].to_numpy()
        ok = np.isfinite(raw)
        assert np.array_equal(np.argsort(raw[ok]), np.argsort(inputs[key][row][ok]))
    assert np.array_equal(inputs["rg:size"], inputs["ta:size"], equal_nan=True)
    # non-members have no value; market-state inputs are the same for every stock on a date
    assert np.isnan(inputs["ta:mom_12_1"][:3, 0]).all()
    for name in e9.STATE:
        assert np.array_equal(inputs[name], np.repeat(inputs[name][:, [0]], 8, axis=1), equal_nan=True)
        assert np.isfinite(inputs[name][-1]).all()


def test_bootstrap_mean_p():
    rng = np.random.default_rng(0)
    positive = 0.01 + rng.normal(0, 0.01, 60)
    assert e9.bootstrap_mean_p(positive, make_rng(1, "a"), resamples=2000) < 0.01
    centred = rng.normal(0, 0.01, 60)
    centred -= centred.mean()
    assert e9.bootstrap_mean_p(centred, make_rng(1, "b"), resamples=2000) == pytest.approx(0.5, abs=0.06)
    assert e9.bootstrap_mean_p(-positive, make_rng(1, "c"), resamples=2000) > 0.99


def synthetic_context(seed=0):
    panel = synthetic_panel(stocks=20, dates=30, seed=seed)
    rng = np.random.default_rng(seed + 1)
    shape = panel.member.shape
    inputs = {n: rng.uniform(-1, 1, shape) for n in
              [f"fa:{f}" for f in e9.FA_FACTORS] + [f"ta:{f}" for f in e9.TA_FACTORS] + list(e9.CONTEXT) + list(e9.STATE)}
    trend = rng.normal(0, 0.2, shape[0])
    return panel, e9.Context(panel, inputs, rng.normal(0, 0.05, shape), np.full(shape, 0.002), trend)


def test_a_tiny_end_to_end_run_returns_a_record_and_repeats_exactly():
    panel, ctx = synthetic_context()
    assert 2014 in e9.folds(ctx.dates)
    budget = {"population": 20, "generations": 2}
    a = e9.run("small", "C3", 2014, 0, write=False, ctx=ctx, budget=budget)
    b = e9.run("small", "C3", 2014, 0, write=False, ctx=ctx, budget=budget)
    expected = {"universe", "rung", "year", "run", "model", "tree", "fitness", "test_dates", "test_net", "test_gross",
                "test_share", "test_turnover", "inputs_used", "seconds", "provenance"}
    assert expected <= set(a)
    n = len(e9.folds(ctx.dates)[2014][1])
    assert len(a["test_dates"]) == len(a["test_net"]) == len(a["test_gross"]) == len(a["test_share"]) == n
    assert all(name in ctx.inputs for name in a["inputs_used"])
    for key in ("model", "tree", "fitness", "test_net", "test_gross", "test_share", "test_turnover", "inputs_used"):
        assert a[key] == b[key] or (key == "fitness" and a[key] == pytest.approx(b[key]))
    c = e9.run("small", "C3", 2014, 1, write=False, ctx=ctx, budget=budget)
    assert c["tree"] != a["tree"]
    # the plain fusion rung has no regime inputs
    d = e9.run("small", "C1", 2014, 0, write=False, ctx=ctx, budget=budget)
    assert not any(n.startswith("rg:") for n in d["inputs_used"])


def test_seeds_do_not_depend_on_the_rung():
    # the seed keys are the universe, year and run only; the rung never enters them
    panel, ctx = synthetic_context()
    budget = {"population": 10, "generations": 1}
    a = e9.run("small", "C1", 2014, 3, write=False, ctx=ctx, budget=budget)
    b = e9.run("small", "C2", 2014, 3, write=False, ctx=ctx, budget=budget)
    assert a["run"] == b["run"] == 3 and a["rung"] != b["rung"]


def test_jobs_skip_finished_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(e9.small, "SMALL_DIR", tmp_path)
    todo = e9.jobs("small", ["C1"], [2015], runs=3)
    assert todo == [("small", "C1", 2015, k) for k in range(3)]
    target = e9.path("small", "C1", 2015, 1)
    assert target == tmp_path / "results" / "dev-e9" / "C1" / "2015" / "run01.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    assert [j[3] for j in e9.jobs("small", ["C1"], [2015], runs=3)] == [0, 2]


def test_initial_weights_change_only_the_first_dates_cost():
    market, signal = tiny_market(low=0.25, high=0.5)
    start = np.array([0.0, 0.0, 0.5, 0.5])
    held = e9.BookMarket(market.returns, market.cost, 0.25, 0.5, initial=start)
    # the first date's book is A, B at 1/2 each: from C, D at 1/2 each, all four weights change by 1/2
    assert held.turnover(signal) == pytest.approx([2.0, 1.0, 1.0])
    assert held.costs(signal)[0] == pytest.approx(0.5 * (0.01 + 0.01 + 0.01 + 0.01))
    assert held.costs(signal)[1:] == pytest.approx(market.costs(signal)[1:])
    assert held.gross(signal) == pytest.approx(market.gross(signal))
    # starting from the first date's own book, nothing is traded on that date
    same = e9.BookMarket(market.returns, market.cost, 0.25, 0.5, initial=market.weights(signal)[0])
    assert same.costs(signal)[0] == 0.0 and same.turnover(signal)[0] == 0.0


def test_an_empty_book_scores_zero_before_costs_and_still_pays_to_leave():
    market, signal = tiny_market()
    signal = signal.copy()
    signal[1] = False                       # date 1 holds nothing
    gross = market.gross(signal)
    assert gross[1] == 0.0 and gross[0] == pytest.approx(0.02)
    # date 1: A and B (1/2 each) are sold at their date-1 costs, 0.01 and 0.03
    assert market.costs(signal)[1] == pytest.approx(0.5 * 0.01 + 0.5 * 0.03)
    assert market.simulate(signal)[0][1] == pytest.approx(-0.02)
    assert market.simulate(np.zeros((3, 4), dtype=bool))[0] == pytest.approx([0.0, 0.0, 0.0])


def synthetic_context_in(tmp_path, monkeypatch):
    panel, ctx = synthetic_context()
    monkeypatch.setattr(e9.small, "SMALL_DIR", tmp_path)
    monkeypatch.setitem(e9._CONTEXTS, "small", ctx)
    return ctx


def test_books_carry_across_test_years(tmp_path, monkeypatch):
    ctx = synthetic_context_in(tmp_path, monkeypatch)
    budget = {"population": 20, "generations": 2}
    assert {2014, 2015} <= set(e9.folds(ctx.dates))
    with pytest.raises(FileNotFoundError, match="2014"):
        e9.run("small", "C1", 2015, 0, write=False, ctx=ctx, budget=budget)
    first = e9.run("small", "C1", 2014, 0, ctx=ctx, budget=budget)
    assert e9.path("small", "C1", 2014, 0).exists()
    assert first["final_weights"] == {} or sum(first["final_weights"].values()) == pytest.approx(1.0)
    second = e9.run("small", "C1", 2015, 0, write=False, ctx=ctx, budget=budget)
    zero = e9.run("small", "C1", 2015, 0, write=False, ctx=ctx, budget=budget, initial=None)
    # a different run number has no record of the year before
    with pytest.raises(FileNotFoundError):
        e9.run("small", "C1", 2015, 1, write=False, ctx=ctx, budget=budget)
    # the same model; only the first date's trading differs, by the distance from the carried weights
    assert second["tree"] == zero["tree"] and second["test_gross"] == zero["test_gross"]
    columns = list(ctx.panel.forward.columns)
    carried = np.zeros(len(columns))
    for stock, w in first["final_weights"].items():
        carried[columns.index(stock)] = w
    test = e9.folds(ctx.dates)[2015][1]
    names = [n for g in e9._names("C1").values() for n in g]
    signal = signals(from_record(second["tree"]), {n: ctx.inputs[n][test] for n in names})["C"]
    w0 = e9.BookMarket(ctx.hedged[test], ctx.cost[test]).weights(signal)[0]
    assert zero["test_turnover"][0] == pytest.approx(np.abs(w0).sum(), abs=1e-5)
    assert second["test_turnover"][0] == pytest.approx(np.abs(w0 - carried).sum(), abs=1e-5)
    assert second["test_turnover"][1:] == zero["test_turnover"][1:]
    assert second["test_net"][1:] == zero["test_net"][1:]


def test_benchmarks_run_continuously_over_the_test_years(tmp_path, monkeypatch):
    ctx = synthetic_context_in(tmp_path, monkeypatch)
    out = e9.benchmarks("small")
    expected = ctx.dates[np.concatenate([t for _, t in e9.folds(ctx.dates).values()])]
    assert len(out) == 10 and e9.BANDED in out
    for name, frame in out.items():
        assert frame.index.equals(expected), name
    # (with 20 stocks the ICs are all missing, so the IC-based books are empty: cash, scoring zero)
    assert out["past-IC weights, rebuilt"]["gross"].eq(0.0).all()
    for name in ("composite (literature signs), rebuilt", e9.BANDED):
        assert out[name]["turnover"].iloc[0] == pytest.approx(1.0)
    # one continuous series: a later year's first date is scored from the holdings of the date before it
    resid = pd.DataFrame(ctx.hedged, index=ctx.dates)
    costs = pd.DataFrame(ctx.cost, index=ctx.dates)
    score = C.composite(ctx.panel).set_axis(ctx.dates).set_axis(resid.columns, axis=1)
    again = e9._evaluate_book(score.loc[expected], resid.loc[expected], costs.loc[expected], True)
    pd.testing.assert_frame_equal(out[e9.BANDED], again, check_names=False, check_exact=False)


def test_report_prints_and_writes_on_a_few_tiny_runs(tmp_path, monkeypatch, capsys):
    ctx = synthetic_context_in(tmp_path, monkeypatch)
    budget = {"population": 20, "generations": 2}
    for rung in e9.RUNGS:
        for year in (2014, 2015):
            e9.run("small", rung, year, 0, ctx=ctx, budget=budget)
    original = e9.bootstrap_mean_p
    monkeypatch.setattr(e9, "bootstrap_mean_p", lambda x, rng, mean_block=3, resamples=10_000: original(x, rng, mean_block, 300))
    table = e9.report("small")
    out = capsys.readouterr().out
    assert (tmp_path / "results" / "e9_report.csv").exists()
    assert (tmp_path / "results" / "e9_conditions.csv").exists()
    saved = pd.read_csv(tmp_path / "results" / "e9_conditions.csv")
    assert list(saved.columns) == e9.CONDITION_COLUMNS
    assert len(saved) == 3 * 4 * 2 and set(saved["rung"]) == set(e9.RUNGS)
    assert "Conditions named in advance (GP less the banded composite, after costs)" in out
    for text in ("Rule 1", "Rule 2", "Rule 3", "Rule 4 (C2 against C1)", "Rule 4 (C3 against C2)", "GP C1", "inputs most often used",
                 "Three C1 rules", "Random books"):
        assert text in out, text
    assert e9.BANDED in set(table["line"]) and any(l.startswith("GP C1") for l in table["line"])


def test_report_works_with_no_gp_runs(tmp_path, monkeypatch, capsys):
    synthetic_context_in(tmp_path, monkeypatch)
    e9.report("small")
    out = capsys.readouterr().out
    assert "Rule 4 needs runs" in out and "Conditions named in advance" not in out
    assert not (tmp_path / "results" / "e9_conditions.csv").exists()


# ------------------------------------------------------------------------ ranked evidence for the training fitness
def test_fitness_costs_by_hand():
    hedged = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],          # 25th and 75th percentiles 1 and 3: range 2
                       [NAN, 0.0, 10.0, NAN, 20.0],        # finite 0, 10, 20: percentiles 5 and 15: range 10
                       [1.0, 1.0, 1.0, 1.0, 1.0],          # zero range
                       [NAN, NAN, NAN, NAN, NAN]])         # nothing finite
    cost = np.array([[0.02, 0.04, NAN, 0.01, 0.01],
                     [0.10, 0.10, 0.20, NAN, 0.05],
                     [0.01, 0.02, 0.03, 0.04, NAN],
                     [0.01, NAN, 0.01, 0.01, 0.01]])
    filled = np.where(np.isnan(cost), C.COST_BELOW, cost)
    for how in ("raw", "clipped"):
        assert e9.fitness_costs(hedged, cost, how) == pytest.approx(filled)
    default = e9.fitness_costs(hedged, cost)
    assert default == pytest.approx(filled)
    ranks = e9.fitness_costs(hedged, cost, "ranks")
    assert ranks[0] == pytest.approx(filled[0] / 2.0)
    assert ranks[1] == pytest.approx(filled[1] / 10.0)
    assert ranks[2] == pytest.approx(filled[2]) and ranks[3] == pytest.approx(filled[3])
    assert ranks[0, 2] == pytest.approx(C.COST_BELOW / 2.0)
    # a row's result does not depend on the other rows
    for i in range(4):
        alone = e9.fitness_costs(hedged[[i]], cost[[i]], "ranks")
        assert alone[0] == pytest.approx(ranks[i])
    with pytest.raises(ValueError, match="unknown"):
        e9.fitness_costs(hedged, cost, "sideways")
    # the interquartile range of the ranks themselves is 1, so these costs keep their meaning
    rng = np.random.default_rng(0)
    big = rng.normal(0, 0.1, (3, 101))
    r = e9.fitness_returns(big, "ranks")
    assert np.percentile(r, 75, axis=1) - np.percentile(r, 25, axis=1) == pytest.approx([1.0] * 3)


def test_run_scores_training_on_ranks_by_default_and_tests_on_real_returns():
    panel, ctx = synthetic_context()
    budget = {"population": 20, "generations": 2}
    default = e9.run("small", "C1", 2014, 0, write=False, ctx=ctx, budget=budget)
    ranked = e9.run("small", "C1", 2014, 0, write=False, ctx=ctx, budget=budget, evidence="ranks")
    raw = e9.run("small", "C1", 2014, 0, write=False, ctx=ctx, budget=budget, evidence="raw")
    again = e9.run("small", "C1", 2014, 0, write=False, ctx=ctx, budget=budget, evidence="raw")
    assert default["evidence"] == "ranks" and raw["evidence"] == "raw"
    assert default["provenance"]["evidence"] == "ranks" and raw["provenance"]["evidence"] == "raw"
    for key in ("tree", "test_net", "test_gross", "fitness"):
        assert default[key] == ranked[key] and raw[key] == again[key]
    # the training score is on another scale: the same seed, another fitness
    assert raw["fitness"] != default["fitness"]
    # whatever the evidence, a tree's test fields are on the real returns and costs
    test = e9.folds(ctx.dates)[2014][1]
    names = [n for g in e9._names("C1").values() for n in g]
    for rec in (default, raw):
        signal = signals(from_record(rec["tree"]), {n: ctx.inputs[n][test] for n in names})["C"]
        net, active, _ = e9.BookMarket(ctx.hedged[test], ctx.cost[test], e9.LOW, e9.HIGH).simulate(signal)
        assert rec["test_net"] == pytest.approx(net, abs=1e-7) and rec["test_active"] == active


def test_raw_evidence_trains_on_the_real_returns_and_costs():
    # evidence="raw" builds exactly the market the run used before ranks were added
    panel, ctx = synthetic_context()
    train = e9.folds(ctx.dates)[2014][0]
    market = e9.BookMarket(e9.fitness_returns(ctx.hedged[train], "raw"), e9.fitness_costs(ctx.hedged[train], ctx.cost[train], "raw"))
    plain = e9.BookMarket(ctx.hedged[train], ctx.cost[train])
    assert np.array_equal(market.returns, plain.returns, equal_nan=True) and np.array_equal(market.cost, plain.cost)


# ------------------------------------------------------------------------------------ contributions and conditions
def test_contributions_sum_to_the_net_result():
    rng = np.random.default_rng(11)
    hedged = rng.normal(0, 0.05, (20, 12))
    hedged[rng.random(hedged.shape) < 0.2] = NAN
    cost = rng.choice([0.001, 0.01], size=hedged.shape)
    cost[rng.random(cost.shape) < 0.2] = NAN
    signal = rng.random(hedged.shape) < 0.4
    signal[5] = False                                          # an empty book on one date
    start = rng.random(12)
    start /= start.sum()
    for initial in (None, start):
        market = e9.BookMarket(hedged, cost, initial=initial)
        contrib = market.contributions(signal)
        assert contrib.shape == hedged.shape
        assert contrib.sum(axis=1) == pytest.approx(market.simulate(signal)[0], abs=1e-12)
    market, signal = tiny_market()
    assert market.contributions(signal).sum(axis=1) == pytest.approx(market.simulate(signal)[0])
    # date 0 of the hand-checked example: A and B each 0.5 x (return - 0.01) less 0.5 x 0.01
    assert market.contributions(signal)[0] == pytest.approx([0.5 * 0.01 - 0.005, 0.5 * 0.03 - 0.005, 0.0, 0.0])


def conditions_context(**kwargs):
    panel, ctx = synthetic_context(**kwargs)
    return ctx


def test_banded_composite_parts_add_up_to_its_net_column_and_the_groups_to_the_total(tmp_path, monkeypatch):
    ctx = synthetic_context_in(tmp_path, monkeypatch)
    bench = e9.benchmarks("small")
    stock, dated = e9._stock_masks(ctx), e9._date_masks(ctx)
    parts = e9._banded_composite_parts(ctx, stock, dated, bench)       # raises if the net column is not reproduced
    assert parts["net"].to_numpy() == pytest.approx(bench[e9.BANDED]["net"].reindex(parts.index).to_numpy(), abs=1e-9)
    assert (parts["thin"] + parts["liquid"]).to_numpy() == pytest.approx(parts["net"].to_numpy(), abs=1e-12)
    assert (parts["small"] + parts["larger"]).to_numpy() == pytest.approx(parts["net"].to_numpy(), abs=1e-12)
    for a, b in (("high_volatility", "low_volatility"), ("rising", "falling")):
        assert parts[[a, b]].sum(axis=1, min_count=1).to_numpy() == pytest.approx(parts["net"].to_numpy())
        assert (parts[a].notna() ^ parts[b].notna()).all()


def test_gp_parts_reproduce_each_runs_stored_net_and_split_it_by_group(tmp_path, monkeypatch):
    ctx = synthetic_context_in(tmp_path, monkeypatch)
    budget = {"population": 20, "generations": 2}
    records = [e9.run("small", "C2", year, k, ctx=ctx, budget=budget) for year in (2014, 2015) for k in (0, 1)]
    stock, dated = e9._stock_masks(ctx), e9._date_masks(ctx)
    parts = e9._gp_parts("small", "C2", records, ctx, stock, dated)    # raises if a run's net is not reproduced
    mean = pd.concat([pd.Series(r["test_net"], index=pd.DatetimeIndex(r["test_dates"])) for r in records]
                     ).groupby(level=0).mean().sort_index()
    assert parts["net"].to_numpy() == pytest.approx(mean.to_numpy(), abs=1e-7)
    assert (parts["thin"] + parts["liquid"]).to_numpy() == pytest.approx(parts["net"].to_numpy(), abs=1e-12)
    # a record that does not match is refused
    bad = [dict(records[0], test_net=[x + 0.01 for x in records[0]["test_net"]])]
    with pytest.raises(ValueError, match="test_net"):
        e9._gp_parts("small", "C2", bad, ctx, stock, dated)


def test_a_book_better_in_one_half_differs_from_the_benchmark_only_there():
    # 8 stocks; value traded rank rises with the column. Columns 0-3 are thin, 4-7 liquid.
    hedged = np.tile(np.array([0.00, 0.01, 0.02, 0.03, 0.02, 0.01, 0.00, -0.01]), (6, 1))
    market = e9.BookMarket(hedged, np.zeros_like(hedged), low=0.0, high=1.0)
    gp = np.zeros(hedged.shape, dtype=bool)
    gp[:, 3] = gp[:, 5] = True                     # the best thin stock and a liquid one
    other = np.zeros(hedged.shape, dtype=bool)
    other[:, 0] = other[:, 5] = True               # the worst thin stock and the same liquid one
    value = np.tile(np.linspace(-1, 1, 8), (6, 1))
    ctx = e9.Context(None, {"rg:value_traded": value, "rg:size": -value, "rg:market_volatility": np.zeros((6, 8))},
                     hedged, np.zeros_like(hedged))
    stock = e9._stock_masks(ctx)
    assert stock["thin"][0].tolist() == [True] * 4 + [False] * 4
    assert stock["small"][0].tolist() == [False] * 4 + [True] * 4
    rows = np.arange(6)
    index = pd.bdate_range("2015-01-05", periods=6)
    a = e9._parts(market.contributions(gp), rows, stock, {}, index)
    b = e9._parts(market.contributions(other), rows, stock, {}, index)
    # thin: 0.5 x (0.03 - base) against 0.5 x (0.00 - base), the base cancelling; liquid: the same stock, so the same
    assert (a["thin"] - b["thin"]).to_numpy() == pytest.approx([0.5 * 0.03] * 6, abs=1e-12)
    assert (a["liquid"] - b["liquid"]).to_numpy() == pytest.approx(np.zeros(6), abs=1e-12)
    assert e9.bootstrap_mean_p((a["thin"] - b["thin"]).to_numpy() + np.array([0, .001, 0, .001, 0, .001]),
                               make_rng(1, "t"), resamples=500) < 0.05


def test_date_conditions_use_only_the_decision_dates_information():
    rng = np.random.default_rng(8)
    series = pd.Series(rng.normal(0, 0.01, 80), index=pd.bdate_range("2015-01-05", periods=80))
    check_no_lookahead(lambda s: e9.trailing_return(s, 8).to_frame("trend"), series, [8, 20, 50, 79])
    r = e9.trailing_return(series, 8)
    assert r.iloc[:8].isna().all()
    assert r.iloc[10] == pytest.approx((1 + series.iloc[3:11]).prod() - 1)       # sessions 3 to 10: eight returns
    # the groups themselves: the volatility percentile above its middle and the trend above zero
    def groups(s):
        vol = e9.market_state(s, 5, 8, 6)["market_volatility"]
        return pd.DataFrame({"high_volatility": vol > 0, "rising": e9.trailing_return(s, 8) > 0}).astype(float)
    check_no_lookahead(groups, series, [10, 30, 60, 79])
    # market_state still uses the same trailing return
    assert e9.market_state(series, 5, 8, 6)["market_return"].notna().sum() > 0
