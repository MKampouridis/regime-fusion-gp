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
    return panel, e9.Context(panel, inputs, rng.normal(0, 0.05, shape), np.full(shape, 0.002))


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
    for text in ("Rule 1", "Rule 2", "Rule 3", "Rule 4 (C2 against C1)", "Rule 4 (C3 against C2)", "GP C1", "inputs most often used",
                 "Three C1 rules", "Random books"):
        assert text in out, text
    assert e9.BANDED in set(table["line"]) and any(l.startswith("GP C1") for l in table["line"])
