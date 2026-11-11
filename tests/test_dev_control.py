import json

import numpy as np
import pandas as pd
import pytest

from regime_fusion.dev import control
from regime_fusion.dev import crosssection as C
from regime_fusion.dev import e9
from regime_fusion.dev import factors as F
from regime_fusion.engine.panel import Panel
from regime_fusion.engine.tree import Grammar, from_record, signals

NAN = np.nan


def synthetic_context(stocks=60, dates=30, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2013-01-02", periods=dates * 21)[::21][:dates]
    columns = [f"S{i}" for i in range(stocks)]

    def frame(values):
        return pd.DataFrame(values, index=index, columns=columns)

    member = frame(np.ones((dates, stocks), dtype=bool))
    factors = {name: frame(rng.normal(0, 1, (dates, stocks))) for name in F.SIGNS}
    forward = frame(rng.normal(0.01, 0.1, (dates, stocks)))
    value = frame(rng.lognormal(15, 1, (dates, stocks)))
    panel = C.Panel(index, factors, forward, member, frame(rng.uniform(0.5, 1.5, (dates, stocks))), value,
                    forward.mean(axis=1))
    hedged = rng.normal(0, 0.08, (dates, stocks))
    hedged[rng.random(hedged.shape) < 0.1] = NAN
    inputs = {n: rng.uniform(-1, 1, (dates, stocks)) for n in
              [f"fa:{f}" for f in e9.FA_FACTORS] + [f"ta:{f}" for f in e9.TA_FACTORS] + list(e9.CONTEXT) + list(e9.STATE)}
    return e9.Context(panel, inputs, hedged, np.full((dates, stocks), 0.002))


@pytest.fixture(scope="module")
def ctx():
    return synthetic_context()


def test_the_calibrated_ic_matches_the_target(ctx):
    for plant in control.PLANTS:
        for ic in (0.03, 0.10):
            data = control.planted_data(ctx, plant, ic, 700, 0)
            realised = control.mean_spearman(data["g"], data["returns"])
            assert realised == pytest.approx(ic, abs=control.IC_TOLERANCE_1A)
            assert data["lambda"] > 0


def test_ic_zero_leaves_the_shuffled_returns_untouched(ctx):
    zero = control.planted_data(ctx, "P1", 0.0, 700, 0)
    assert zero["lambda"] == 0.0
    other = control.planted_data(ctx, "P3", 0.0, 700, 0)
    assert np.array_equal(zero["returns"], other["returns"], equal_nan=True)
    planted = control.planted_data(ctx, "P1", 0.05, 700, 0)
    # the shuffle is the returns less lambda g
    assert np.allclose(planted["returns"] - planted["lambda"] * planted["g"], zero["returns"], equal_nan=True)


def test_the_shuffle_keeps_each_dates_returns_among_tradable_stocks(ctx):
    data = control.planted_data(ctx, "P1", 0.0, 300, 1)
    tradable = data["tradable"]
    assert np.array_equal(np.isfinite(data["returns"]), tradable)
    assert tradable.sum() < np.isfinite(ctx.hedged).sum()          # a subset
    for d in range(len(ctx.dates)):
        kept = np.sort(data["returns"][d][tradable[d]])
        # the date's set of returns is some subset's own returns, permuted: compare with the unsubsetted source
        assert np.isin(kept, ctx.hedged[d]).all()
    full = control.planted_data(ctx, "P1", 0.0, 700, 1)
    for d in range(len(ctx.dates)):
        assert np.sort(full["returns"][d][full["tradable"][d]]) == pytest.approx(np.sort(ctx.hedged[d][np.isfinite(ctx.hedged[d])]))
    assert full["tradable"].sum() == np.isfinite(ctx.hedged).sum()
    assert not np.array_equal(full["returns"], ctx.hedged, equal_nan=True)


def test_the_same_size_and_run_give_the_same_sample_for_every_plant_and_ic(ctx):
    base = control.planted_data(ctx, "P1", 0.0, 300, 2)
    for plant, ic in (("P1", 0.05), ("P2", 0.05), ("P3", 0.10), ("P2", 0.0)):
        other = control.planted_data(ctx, plant, ic, 300, 2)
        assert np.array_equal(base["tradable"], other["tradable"])
        for name in control.PLANTED:
            assert np.array_equal(base["inputs"][name], other["inputs"][name], equal_nan=True), (plant, ic, name)
        shuffled = other["returns"] - other["lambda"] * other["g"]
        assert shuffled == pytest.approx(base["returns"], nan_ok=True)
    different = control.planted_data(ctx, "P1", 0.0, 300, 3)
    assert not np.array_equal(base["tradable"], different["tradable"])
    # the inputs: 17 decoys, the two context ranks and three planted ones, restricted to the subset
    inputs = base["inputs"]
    assert len(inputs) == 17 + 2 + 3
    out = ~base["tradable"].any(axis=0)
    assert np.isnan(inputs["fa:earnings_price"][:, out]).all()


def test_p2_g_is_zero_mean_and_p3_switches_with_the_state(ctx):
    p2 = control.planted_data(ctx, "P2", 0.05, 700, 0)
    means = np.nanmean(p2["g"], axis=1)
    assert means == pytest.approx(0.0, abs=1e-12)
    u1, u2 = p2["inputs"]["fa:planted"], p2["inputs"]["ta:planted"]
    both = (u1 > 0.2) & (u2 > 0.2) & p2["tradable"]
    top = p2["g"][both]
    assert (top > 0).all()
    p3 = control.planted_data(ctx, "P3", 0.05, 700, 0)
    state = p3["inputs"]["rg:planted_state"][:, 0]
    assert set(state) <= {-1.0, 1.0}
    for d in range(len(state)):
        u = p3["inputs"]["fa:planted"][d] if state[d] > 0 else p3["inputs"]["ta:planted"][d]
        assert np.array_equal(p3["g"][d], u, equal_nan=True)
    p1 = control.planted_data(ctx, "P1", 0.05, 700, 0)
    assert np.array_equal(p1["g"], p1["inputs"]["fa:planted"], equal_nan=True)
    assert abs(state.sum()) < len(state)          # both states occur with 30 dates and these seeds


def test_planted_characteristics_are_slow_and_ranked(ctx):
    u1 = control.planted_data(ctx, "P1", 0.0, 700, 0)["inputs"]["fa:planted"]
    finite = u1[np.isfinite(u1)]
    assert finite.min() > -1 and finite.max() <= 1
    both = np.isfinite(u1[1:]) & np.isfinite(u1[:-1])
    assert np.corrcoef(u1[1:][both], u1[:-1][both])[0, 1] > 0.6


def test_the_oracle_book_earns_more_than_a_random_book(ctx):
    rng = np.random.default_rng(0)
    for plant in control.PLANTS:
        data = control.planted_data(ctx, plant, 0.10, 700, 0)
        market = e9.BookMarket(data["returns"], np.zeros_like(data["returns"]))
        oracle = market.gross(control.oracle_signal(plant, data["g"], data["tradable"])).mean()
        random = market.gross(control.top_book(rng.random(data["g"].shape), data["tradable"])).mean()
        assert oracle > 0.01 and oracle > random + 0.01


def test_top_book_picks_a_fifth_of_the_tradable_stocks():
    score = np.arange(20, dtype=float)[None, :].repeat(2, axis=0)
    tradable = np.ones((2, 20), dtype=bool)
    tradable[1, :10] = False
    book = control.top_book(score, tradable)
    assert book[0].sum() == 4 and book[0][-4:].all()
    assert book[1].sum() == 2 and book[1][-2:].all()
    score[0, -1] = NAN
    assert not control.top_book(score, tradable)[0][-1]


def make_individual(grammar, rng, wanted, tries=3000):
    for _ in range(tries):
        ind = grammar.random_individual(3, "grow", rng)
        if wanted(ind):
            return ind
    raise AssertionError("no individual found")


def test_detected_logic_for_each_plant_on_hand_made_individuals():
    assert control.detected("P1", ["fa:planted", "ta:x"])
    assert not control.detected("P1", ["ta:planted"])
    assert control.detected("P2", ["ta:planted", "fa:planted"])
    assert not control.detected("P2", ["fa:planted"])
    assert control.detected("P3", ["fa:planted", "ta:planted", "rg:planted_state", "rg:size"])
    assert not control.detected("P3", ["fa:planted", "ta:planted"])
    assert control.uses_planted(["rg:planted_state"]) and not control.uses_planted(["fa:a", "rg:size"])
    # and from real individuals: inputs_used decides, so an input in an unnamed branch does not count
    names = {"FA": ["fa:planted", "fa:b"], "TA": ["ta:planted", "ta:c"], "RG": ["rg:planted_state"]}
    grammar = Grammar(names, fusion_functions=("AND", "OR", "IF"), sources=("FA", "TA"), max_branch_depth=3)
    rng = np.random.default_rng(1)
    ind = make_individual(grammar, rng, lambda i: "fa:planted" in e9.inputs_used(i))
    assert control.detected("P1", e9.inputs_used(ind))
    ind = make_individual(grammar, rng, lambda i: "fa:planted" not in e9.inputs_used(i))
    assert not control.detected("P1", e9.inputs_used(ind))


def test_cells_and_paths(tmp_path, monkeypatch):
    assert len(control.CELLS_1A) == 60 and len(set(control.CELLS_1A)) == 60
    assert sum(1 for c in control.CELLS_1A if c[3] == 36) == 15
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path)
    target = control.path_1a("P2", 0.02, 300, 57, 4)
    assert target == tmp_path / "results" / "dev-control" / "1a" / "P2" / "ic002" / "n300" / "t57" / "run04.json"
    assert control.path_1a("P1", 0.10, 700, 36, 0).parts[-4] == "ic010"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    assert len(control.jobs_1a(runs=2)) == 120
    target.with_name("run01.json").write_text("{}")
    assert ("P2", 0.02, 300, 57, 1) not in control.jobs_1a(runs=2)
    assert control.jobs_1a(runs=5).__len__() == 60 * 5 - 2 and ("P2", 0.02, 300, 57, 4) not in control.jobs_1a(runs=5)
    monkeypatch.setenv("RFGP_DATA_DIR", str(tmp_path / "d"))
    assert control.path_1b("ABT", 0.3, 1).parts[-3:] == ("ABT", "ic030", "run01.json")
    assert len(control.jobs_1b()) == 500


def test_a_tiny_end_to_end_run_1a_repeats_exactly(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path)
    budget = {"population": 20, "generations": 2}
    kwargs = dict(ctx=ctx, budget=budget, test_start=20)
    a = control.run_1a("P1", 0.05, 300, 12, 0, write=False, **kwargs)
    b = control.run_1a("P1", 0.05, 300, 12, 0, write=False, **kwargs)
    for key in ("model", "tree", "fitness", "inputs_used", "detected", "uses_planted", "gp_test_mean", "oracle_test_mean",
                "captured", "references", "lambda"):
        assert a[key] == b[key], key
    assert set(a["references"]) == set(control.REFERENCES)
    assert a["references"]["best_single"]["chosen"] in control.planted_data(ctx, "P1", 0.05, 300, 0)["inputs"]
    assert all(a["references"][k]["detected"] in (True, False) for k in control.REFERENCES)
    assert np.isfinite(a["oracle_test_mean"])
    expected = a["gp_test_mean"] / a["oracle_test_mean"] if a["oracle_test_mean"] > 0 else None
    assert a["captured"] == (None if expected is None else pytest.approx(expected))
    assert len(a["test_dates"]) == 10 and "provenance" in a
    zero = control.run_1a("P2", 0.0, 300, 12, 1, write=False, **kwargs)
    assert zero["captured"] is None and zero["lambda"] == 0.0
    assert all(zero["references"][k]["detected"] is None for k in control.REFERENCES)
    c = control.run_1a("P3", 0.05, 300, 12, 0, **kwargs)
    assert control.path_1a("P3", 0.05, 300, 12, 0).exists() and c["detected"] in (True, False)
    # the GP seed does not depend on the plant or the strength: same stocks and run, same ic 0 GP
    same = control.run_1a("P1", 0.0, 300, 12, 1, write=False, **kwargs)
    assert same["tree"] == zero["tree"]


# ---------------------------------------------------------------------------------------------------- 1B
def synthetic_prices(n=150, seed=0):
    rng = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + rng.normal(0, 0.01, n))
    frame = pd.DataFrame({"adj_open": close * 0.999, "adj_high": close * 1.01, "adj_low": close * 0.99, "adj_close": close},
                         index=pd.bdate_range("2015-02-02", periods=n))
    return frame


def test_planted_prices_share_a_factor_that_uses_only_earlier_indicator_values():
    prices = synthetic_prices()
    s = control.ar1(np.random.default_rng(3), len(prices))
    lam = 0.004
    planted = control.plant_prices(prices, s, lam)
    ratio = planted / prices
    assert ratio.iloc[:, 0].to_numpy() == pytest.approx(ratio.iloc[:, 3].to_numpy())
    for c in prices.columns:
        assert ratio[c].to_numpy() == pytest.approx(ratio["adj_close"].to_numpy())
    assert ratio["adj_open"].iloc[0] == 1.0
    # each day's return gains lam x yesterday's indicator
    assert (ratio["adj_close"].iloc[1:].to_numpy() / ratio["adj_close"].iloc[:-1].to_numpy()) == pytest.approx(1 + lam * s[:-1])
    # changing s[t] changes nothing up to and including day t
    t = 60
    other = s.copy()
    other[t] += 2.0
    changed = control.plant_prices(prices, other, lam)
    assert changed.iloc[:t + 1].to_numpy() == pytest.approx(planted.iloc[:t + 1].to_numpy())
    assert not np.allclose(changed.iloc[t + 1:].to_numpy(), planted.iloc[t + 1:].to_numpy())
    # lambda 0 leaves the prices unchanged
    assert control.plant_prices(prices, s, 0.0).to_numpy() == pytest.approx(prices.to_numpy())
    with pytest.raises(ValueError):
        control.plant_prices(prices, s, 10.0)


def test_the_1b_calibration_hits_the_target_and_zero_gives_zero():
    prices = synthetic_prices(600, seed=1)
    s = control.ar1(np.random.default_rng(4), len(prices))
    assert control.calibrate_1b(prices, s, 0.0, 400) == 0.0
    for ic in (0.05, 0.20):
        lam = control.calibrate_1b(prices, s, ic, 400)
        assert lam > 0
        assert control.planted_ic_1b(prices, s, lam, 400) == pytest.approx(ic, abs=control.IC_TOLERANCE_1B)


def test_an_ar1_series_is_stationary_and_persistent():
    s = control.ar1(np.random.default_rng(0), 20000)
    assert s.std() == pytest.approx(1.0, abs=0.05) and abs(s.mean()) < 0.1
    assert np.corrcoef(s[1:], s[:-1])[0, 1] == pytest.approx(0.9, abs=0.03)


def synthetic_panel_1b(n_train=120, n_eval=40, seed=0):
    prices = synthetic_prices(n_train + n_eval, seed)
    rng = np.random.default_rng(seed + 5)
    indicators = pd.DataFrame({"fa:x": rng.uniform(-1, 1, len(prices)), "ta:y": rng.uniform(-1, 1, len(prices)),
                               "sa:z": rng.uniform(-1, 1, len(prices))}, index=prices.index)
    return Panel(indicators, prices, prices.index[:n_train], prices.index[n_train:])


def test_plant_panel_adds_a_scaled_indicator_and_keeps_the_real_ones():
    panel = synthetic_panel_1b()
    s = control.ar1(np.random.default_rng(2), len(panel.prices))
    planted = control.plant_panel(panel, s, 0.003)
    col = planted.indicators["ta:planted"]
    train = col.iloc[:len(panel.train)]
    assert train.min() == pytest.approx(-1.0) and train.max() == pytest.approx(1.0)
    assert planted.indicators[["fa:x", "ta:y", "sa:z"]].equals(panel.indicators)
    assert "ta:planted" in planted.names()["TA"]
    assert not planted.prices.equals(panel.prices)


def test_a_tiny_end_to_end_run_1b_repeats_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("RFGP_DATA_DIR", str(tmp_path))
    panel = synthetic_panel_1b()
    budget = {"population": 20, "generations": 2}
    a = control.run_1b("ABT", 0.2, 0, write=False, panel=panel, budget=budget)
    b = control.run_1b("ABT", 0.2, 0, write=False, panel=panel, budget=budget)
    for key in ("tree", "fitness", "lambda", "gp_sharpe", "rule_sharpe", "hold_sharpe", "detected", "gp_trades"):
        assert a[key] == b[key], key
    assert a["lambda"] > 0 and control.run_1b("ABT", 0.0, 0, write=False, panel=panel, budget=budget)["lambda"] == 0.0
    # the buy-and-hold of the planted prices is the long design that never trades out
    assert np.isfinite(a["hold_sharpe"])
    control.run_1b("ABT", 0.2, 0, panel=panel, budget=budget)
    assert control.path_1b("ABT", 0.2, 0).exists()


# ------------------------------------------------------------------------------------------------ report
def fake_1a(plant, ic, stocks, train, run, detected, gp_test=0.01, refs=(0.01, 0.02, 0.0), ref_detected=True):
    refs_out = {k: {"test_mean": v, "captured": None if ic == 0 else v / 0.05, "chosen": "x",
                    "detected": ref_detected if plant == "P1" else None} for k, v in zip(control.REFERENCES, refs)}
    return {"plant": plant, "ic": ic, "stocks": stocks, "train": train, "run": run, "detected": detected,
            "uses_planted": detected, "fitness": 0.5 + run / 10, "gp_test_mean": gp_test,
            "captured": None if ic == 0 else gp_test / 0.05, "references": refs_out}


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_the_report_on_a_few_tiny_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path / "small")
    monkeypatch.setenv("RFGP_DATA_DIR", str(tmp_path / "data"))
    # with no runs at all
    assert control.report() == {}
    assert "0 runs found" in capsys.readouterr().out
    for k in range(4):
        for plant in control.PLANTS:
            write(control.path_1a(plant, 0.0, 700, 57, k), fake_1a(plant, 0.0, 700, 57, k, False, 0.001 * (-1) ** k))
        write(control.path_1a("P1", 0.05, 700, 57, k), fake_1a("P1", 0.05, 700, 57, k, k < 3, 0.02, ref_detected=True))
        write(control.path_1a("P1", 0.10, 300, 57, k), fake_1a("P1", 0.10, 300, 57, k, k < 1, 0.03, ref_detected=True))
    write(control.path_1a("P2", 0.05, 700, 57, 0), fake_1a("P2", 0.05, 700, 57, 0, True))
    write(control.path_1a("P2", 0.05, 700, 57, 0).with_name("run00.json.provenance.json"), {"junk": 1})
    for ic in (0.0, 0.2):
        for k in range(3):
            write(control.path_1b("ABT", ic, k), {"ic": ic, "detected": k == 0 and ic > 0, "gp_sharpe": 0.1 * k,
                                                   "rule_sharpe": 1.0, "hold_sharpe": 0.1, "gp_trades": k})
    out = control.report()
    text = capsys.readouterr().out
    assert set(out) == {"1a", "1b"}
    table = out["1a"]
    row = table[(table["plant"] == "P1") & (table["ic"] == 0.05)].iloc[0]
    assert row["runs"] == 4 and row["gp_detection"] == pytest.approx(0.75) and row["ridge_detection"] == 1.0
    assert row["gp_captured_median"] == pytest.approx(0.4) and row["gp_train_fitness_mean"] == pytest.approx(0.65)
    assert (table["plant"] == "P2").sum() == 2
    for line in ("(1) nothing planted", "does not hold", "(2) P1, IC 0.05, 700 stocks", "(3) in no P1 cell",
                 "breaks it: IC 0.1, 300 stocks", "mean training fitness"):
        assert line in text, line
    assert "(2) P1, IC 0.05, 700 stocks, training 57 dates: the GP's detection rate is 75%" in text and "does not hold" in text
    # ic 0 is counted once per sample, not once per plant: 4 runs
    assert "over 4 runs" in text
    assert (tmp_path / "small" / "results" / "control_1a.csv").exists()
    one = out["1b"]
    assert list(one["ic"]) == [0.0, 0.2] and one["runs"].tolist() == [3, 3]
    assert one.loc[one["ic"] == 0.2, "detection"].iloc[0] == pytest.approx(1 / 3)
    assert one["share_never_trade"].iloc[0] == pytest.approx(1 / 3) and one["share_above_hold"].iloc[0] == pytest.approx(1 / 3)
    assert (tmp_path / "data" / "results" / "control_1b.csv").exists()


def test_a_negative_lambda_only_when_chance_already_exceeds_the_target():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(40, 30))
    noise = rng.normal(size=(40, 30))
    chancy = noise + 0.5 * g                       # already correlated with g by chance
    assert control.mean_spearman(g, chancy) > 0.1
    lam = control.calibrate_1a(chancy, g, 0.02)
    assert lam < 0
    assert control.mean_spearman(g, chancy + lam * g) == pytest.approx(0.02, abs=control.IC_TOLERANCE_1A)
    assert control.calibrate_1a(noise - 0.5 * g, g, 0.02) > 0
    # the same for a single stock's indicator, whatever the draw
    prices = synthetic_prices(600, seed=2)
    for seed in range(6):
        s = control.ar1(np.random.default_rng(seed), len(prices))
        lam = control.calibrate_1b(prices, s, 0.05, 400)
        assert control.planted_ic_1b(prices, s, lam, 400) == pytest.approx(0.05, abs=control.IC_TOLERANCE_1B)
        assert abs(lam) * np.abs(s).max() < 1


# ------------------------------------------------------------------- the model chosen on held-back dates
def test_holdback_split_keeps_the_latest_dates_out():
    fit, held = e9.holdback_split(np.arange(57))
    assert list(fit) == list(range(40)) and list(held) == list(range(40, 57))
    fit, held = e9.holdback_split(np.arange(3))
    assert len(fit) == 1 and len(held) == 2


class _Fake:
    def __init__(self, name, fitness):
        self.name, self.fitness = name, fitness

    def key(self):
        return (self.name,)


def test_choose_on_holdback_takes_the_best_candidate_on_the_held_dates(monkeypatch):
    rng = np.random.default_rng(0)
    signal = {"good": np.tile(np.arange(10) < 3, (6, 1)), "bad": np.tile(np.arange(10) >= 7, (6, 1)),
              "tiny": np.tile(np.arange(10) < 0, (6, 1))}
    returns = np.tile(np.where(np.arange(10) < 3, 0.02, -0.01), (6, 1)) + rng.normal(0, 0.001, (6, 10))
    market = e9.BookMarket(returns, np.zeros_like(returns), 0.1, 0.4)
    monkeypatch.setattr(e9, "signals", lambda individual, data: {"C": signal[individual.name]})
    population = [_Fake("bad", 0.9), _Fake("bad", 0.9), _Fake("good", 0.5), _Fake("tiny", 0.4),
                  _Fake("good", e9.FLOOR)]
    chosen, score = e9.choose_on_holdback(population, {}, market)
    assert chosen.name == "good" and score > 0
    # only the fittest candidate is looked at when one candidate is allowed
    assert e9.choose_on_holdback(population, {}, market, candidates=1)[0].name == "bad"
    # every candidate at the floor on the held dates (an empty book): the fittest stays
    assert e9.choose_on_holdback([_Fake("tiny", 0.4)], {}, market)[0].name == "tiny"
    assert e9.choose_on_holdback([_Fake("good", e9.FLOOR)], {}, market) is None


def test_run_1a_with_holdback_evolves_on_the_earlier_dates_and_writes_apart(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path)
    kwargs = dict(ctx=ctx, budget={"population": 20, "generations": 2}, test_start=20, selection="holdback")
    a = control.run_1a("P1", 0.05, 300, 12, 0, **kwargs)
    b = control.run_1a("P1", 0.05, 300, 12, 0, write=False, **kwargs)
    assert a["tree"] == b["tree"] and a["gp_test_mean"] == b["gp_test_mean"]
    assert (a["selection"], a["fit_dates"], a["held_dates"]) == ("holdback", 8, 4)
    assert control.path_1a("P1", 0.05, 300, 12, 0, "holdback").exists()
    assert not control.path_1a("P1", 0.05, 300, 12, 0).exists()
    assert "1a-holdback" in control.path_1a("P1", 0.05, 300, 12, 0, "holdback").parts
    # the references see every training date, so they match the first version's
    plain = control.run_1a("P1", 0.05, 300, 12, 0, write=False, **{**kwargs, "selection": "fitness"})
    assert plain["references"] == a["references"] and plain["held_dates"] == 0
    jobs = control.jobs_1a(runs=1, selection="holdback", plants=("P1",))
    assert len(jobs) == 20 and all(j[0] == "P1" and j[5] == "holdback" for j in jobs)
    with pytest.raises(ValueError):
        control.part_1a("other")


# --------------------------------------------------------------- the returns the training fitness is shown
def test_fitness_returns_clip_or_rank_each_date_on_its_own():
    r = np.array([[0.0, 1.0, 2.0, 100.0, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan], [3.0, -50.0, 1.0, 2.0, 0.0]])
    assert e9.fitness_returns(r) is not None and np.array_equal(e9.fitness_returns(r, "raw"), r, equal_nan=True)
    ranks = e9.fitness_returns(r, "ranks")
    assert np.allclose(ranks[0, :4], [-1, -1 / 3, 1 / 3, 1]) and np.isnan(ranks[0, 4]) and np.isnan(ranks[1]).all()
    assert np.allclose(ranks[2], [1, -1, 0, 0.5, -0.5])
    clipped = e9.fitness_returns(r, "clipped")
    assert clipped[0, 3] < 100 and clipped[2, 1] > -50 and clipped[0, 1] == 1.0 and np.isnan(clipped[0, 4])
    # a date's values do not depend on any other date
    assert np.array_equal(e9.fitness_returns(r[:1], "ranks"), ranks[:1], equal_nan=True)
    assert np.array_equal(e9.fitness_returns(r[2:], "clipped"), clipped[2:], equal_nan=True)
    with pytest.raises(ValueError):
        e9.fitness_returns(r, "other")


def test_run_1a_on_ranked_returns_changes_the_fitness_only(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path)
    kwargs = dict(ctx=ctx, budget={"population": 20, "generations": 2}, test_start=20)
    plain = control.run_1a("P1", 0.05, 300, 12, 0, write=False, **kwargs)
    for version in ("ranks", "clipped"):
        a = control.run_1a("P1", 0.05, 300, 12, 0, selection=version, **kwargs)
        assert a["evidence"] == version and a["held_dates"] == 0 and a["fit_dates"] == 12
        assert a["oracle_test_mean"] == plain["oracle_test_mean"] and a["references"] == plain["references"]
        assert control.path_1a("P1", 0.05, 300, 12, 0, version).exists()
    assert plain["evidence"] == "raw"


# ------------------------------------------------------------------------ the scoring GP's two versions (S1b)
def test_the_score_versions_run_end_to_end_and_repeat_exactly(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path)
    kwargs = dict(ctx=ctx, budget={"population": 12, "generations": 2}, test_start=20)
    plain = control.run_1a("P1", 0.05, 300, 12, 0, write=False, **kwargs)
    for version, fitness in (("score-book", "book"), ("score-rank", "rank")):
        a = control.run_1a("P1", 0.05, 300, 12, 0, selection=version, **kwargs)
        b = control.run_1a("P1", 0.05, 300, 12, 0, write=False, selection=version, **kwargs)
        for key in ("model", "tree", "fitness", "inputs_used", "detected", "uses_planted", "gp_test_mean",
                    "oracle_test_mean", "captured", "references", "lambda"):
            assert a[key] == b[key], key
        assert (a["selection"], a["evidence"], a["held_dates"], a["fit_dates"]) == (version, "ranks", 0, 12)
        assert set(a["tree"]) == {"fusion", "branches", "fitness"} and "fusion:" in a["model"]
        assert np.isfinite(a["gp_test_mean"]) and isinstance(a["detected"], bool)
        # the references and the oracle do not depend on the GP's version
        assert a["references"] == plain["references"] and a["oracle_test_mean"] == plain["oracle_test_mean"]
        assert control.path_1a("P1", 0.05, 300, 12, 0, version).exists()
        assert f"1a-{version}" in control.path_1a("P1", 0.05, 300, 12, 0, version).parts
        assert json.loads(control.path_1a("P1", 0.05, 300, 12, 0, version).read_text())["selection"] == version
    assert not control.path_1a("P1", 0.05, 300, 12, 0).exists()
    # the score model's result is the top fifth of its score, evaluated like the references'
    from regime_fusion.dev import score
    model = score.ScoreModel.from_record(json.loads(control.path_1a("P1", 0.05, 300, 12, 0, "score-book").read_text())["tree"])
    data = control.planted_data(ctx, "P1", 0.05, 300, 0)
    test = np.arange(20, len(ctx.dates))
    later = e9.BookMarket(data["returns"][test], np.zeros((len(test), data["returns"].shape[1])), e9.LOW, e9.HIGH)
    tradable = data["tradable"][test]
    expected = later.gross(control.top_book(model.score({n: v[test] for n, v in data["inputs"].items()}, tradable),
                                            tradable)).mean()
    assert expected == pytest.approx(
        json.loads(control.path_1a("P1", 0.05, 300, 12, 0, "score-book").read_text())["gp_test_mean"])


def test_the_gps_seed_is_the_same_call_for_every_version(ctx, monkeypatch):
    calls = []
    real = control.make_rng

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(control, "make_rng", spy)
    kwargs = dict(ctx=ctx, budget={"population": 8, "generations": 1}, test_start=20, write=False)
    seen = {}
    for version in ("fitness", "ranks", "score-book", "score-rank"):
        calls.clear()
        control.run_1a("P3", 0.05, 300, 12, 0, selection=version, **kwargs)
        seen[version] = [c for c in calls if c[3] == "1a-gp"]
    assert len(seen["fitness"]) == 1 and seen["fitness"][0][4:] == (300, 12, 0)
    assert all(v == seen["fitness"] for v in seen.values())
    assert "score-book" in control.SELECTIONS and "score-rank" in control.SELECTIONS
    assert control.part_1a("score-rank") == "1a-score-rank"


# -------------------------------------------------------------------------------------------------- P4
def test_p1_to_p3_data_are_unchanged_by_the_fourth_plant(ctx):
    import hashlib
    # checksums of the data (returns, g, every input) computed before P4 was added
    expected = {("P1", 0.05, 700, 0): "61ba635e2b32b7b3", ("P2", 0.03, 300, 1): "6f268b155869499e",
                ("P3", 0.10, 700, 2): "c80ede26f57eaea6"}
    for (plant, ic, stocks, run), digest in expected.items():
        d = control.planted_data(ctx, plant, ic, stocks, run)
        h = hashlib.sha256()
        h.update(np.nan_to_num(d["returns"], nan=-9).tobytes())
        h.update(np.nan_to_num(d["g"], nan=-9).tobytes())
        for k in sorted(d["inputs"]):
            h.update(np.nan_to_num(d["inputs"][k], nan=-9).tobytes())
        assert h.hexdigest()[:16] == digest, plant


def test_p4_follows_the_state_and_fa_alone_shows_no_overall_correlation():
    big = synthetic_context(stocks=200, dates=80, seed=3)
    data = control.planted_data(big, "P4", 0.10, 700, 0)
    state = data["inputs"]["rg:planted_state"][:, 0]
    u1 = data["inputs"]["fa:planted"]
    assert (state > 0).any() and (state < 0).any()
    for d in range(len(state)):
        assert np.array_equal(data["g"][d], state[d] * u1[d], equal_nan=True)
    assert control.mean_spearman(data["g"], data["returns"]) == pytest.approx(0.10, abs=control.IC_TOLERANCE_1A)
    assert data["lambda"] > 0
    # signed by the state the per-date correlation of fa:planted is the planted strength; unsigned it cancels
    per_date = control._spearman_rows(u1, data["returns"])
    assert np.nanmean(per_date * state) > 0.07
    assert abs(np.nanmean(per_date)) < 0.05
    # same shuffle and inputs as the other plants for the same sample
    p1 = control.planted_data(big, "P1", 0.0, 700, 0)
    zero = control.planted_data(big, "P4", 0.0, 700, 0)
    assert np.array_equal(p1["returns"], zero["returns"], equal_nan=True)
    assert np.array_equal(p1["inputs"]["fa:planted"], u1, equal_nan=True)


def test_p4_detection_rule_and_oracle(ctx):
    assert control.detected("P4", ["fa:planted", "rg:planted_state"])
    assert control.detected("P4", ["fa:planted", "rg:planted_state", "ta:x"])
    assert not control.detected("P4", ["fa:planted"])
    assert not control.detected("P4", ["rg:planted_state", "ta:planted"])
    assert "P4" in control.PLANTS and "P4" not in control.DEFAULT_PLANTS
    data = control.planted_data(ctx, "P4", 0.10, 700, 0)
    test = np.arange(20, len(ctx.dates))
    market = e9.BookMarket(data["returns"][test], np.zeros((len(test), data["returns"].shape[1])))
    book = control.oracle_signal("P4", data["g"][test], data["tradable"][test])
    assert market.gross(book).mean() > 0.01


def test_p4_runs_for_the_references_without_a_detected_flag(ctx):
    a = control.run_1a("P4", 0.05, 300, 12, 0, write=False, ctx=ctx, budget={"population": 8, "generations": 1},
                       test_start=20)
    assert all(a["references"][k]["detected"] is None for k in control.REFERENCES)
    assert a["plant"] == "P4" and a["detected"] in (True, False)


# ------------------------------------------------------------------------------------------- long samples
def test_a_long_sample_resamples_the_dates_and_keeps_the_rest_of_the_machinery(ctx):
    d = control.planted_data(ctx, "P3", 0.05, 700, 0, dates=150)
    assert d["returns"].shape == (150, ctx.hedged.shape[1]) and d["g"].shape == d["returns"].shape
    assert all(v.shape == d["returns"].shape for v in d["inputs"].values())
    pos = d["positions"]
    assert len(pos) == 150 and pos.min() >= 0 and pos.max() < len(ctx.dates)
    assert np.array_equal(d["tradable"], np.isfinite(ctx.hedged)[pos])
    assert control.mean_spearman(d["g"], d["returns"]) == pytest.approx(0.05, abs=control.IC_TOLERANCE_1A)
    # a decoy input is the real input at those dates
    assert np.array_equal(d["inputs"]["fa:earnings_price"], ctx.inputs["fa:earnings_price"][pos], equal_nan=True)
    # blocks: mostly consecutive positions (mean block 6), with some jumps
    steps = np.diff(pos) % len(ctx.dates)
    assert 0.6 < (steps == 1).mean() < 0.95
    # determinism
    again = control.planted_data(ctx, "P3", 0.05, 700, 0, dates=150)
    assert np.array_equal(again["returns"], d["returns"], equal_nan=True) and np.array_equal(again["positions"], pos)
    # the default is the real dates
    short = control.planted_data(ctx, "P3", 0.05, 700, 0)
    assert np.array_equal(short["positions"], np.arange(len(ctx.dates))) and short["returns"].shape[0] == len(ctx.dates)


def test_the_bootstrap_positions_depend_on_dates_and_run_only(ctx):
    base = control.planted_data(ctx, "P3", 0.03, 700, 1, dates=150)["positions"]
    for plant, ic, stocks in (("P4", 0.10, 700), ("P1", 0.0, 700), ("P2", 0.05, 300)):
        assert np.array_equal(control.planted_data(ctx, plant, ic, stocks, 1, dates=150)["positions"], base)
    assert not np.array_equal(control.planted_data(ctx, "P3", 0.03, 700, 2, dates=150)["positions"], base)
    assert not np.array_equal(control.planted_data(ctx, "P3", 0.03, 700, 1, dates=315)["positions"][:150], base)
    # and the helper alone: any length, wrapping round
    got = control.block_positions(10, 100, 6.0, np.random.default_rng(0))
    assert len(got) == 100 and got.min() >= 0 and got.max() < 10


def test_the_planted_characteristic_is_slow_along_the_new_timeline(ctx):
    u1 = control.planted_data(ctx, "P1", 0.0, 700, 0, dates=315)["inputs"]["fa:planted"]
    both = np.isfinite(u1[1:]) & np.isfinite(u1[:-1])
    assert np.corrcoef(u1[1:][both], u1[:-1][both])[0, 1] == pytest.approx(0.9, abs=0.03)


def test_long_split_paths_and_train_check(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path)
    assert control.long_train(150) == 105 and control.long_train(315) == 220
    target = control.path_1a("P4", 0.05, 700, 220, 3, "score-book", 315)
    assert target == (tmp_path / "results" / "dev-control" / "1a-score-book" / "P4" / "ic005" / "n700" / "d315" / "t220"
                      / "run03.json")
    assert control.path_1a("P4", 0.05, 700, 57, 3) == (tmp_path / "results" / "dev-control" / "1a" / "P4" / "ic005"
                                                       / "n700" / "t57" / "run03.json")
    kwargs = dict(ctx=ctx, budget={"population": 8, "generations": 1})
    with pytest.raises(ValueError):
        control.run_1a("P3", 0.05, 700, 57, 0, write=False, dates=40, **kwargs)
    a = control.run_1a("P3", 0.05, 300, control.long_train(40), 0, dates=40, **kwargs)
    b = control.run_1a("P3", 0.05, 300, control.long_train(40), 0, write=False, dates=40, **kwargs)
    assert control.long_train(40) == 28
    assert a["dates"] == 40 and a["train"] == 28 and len(a["test_dates"]) == 12
    assert a["tree"] == b["tree"] and a["gp_test_mean"] == b["gp_test_mean"] and a["references"] == b["references"]
    assert control.path_1a("P3", 0.05, 300, 28, 0, "fitness", 40).exists()
    assert not control.path_1a("P3", 0.05, 300, 28, 0).exists()
    assert all(d.count("-") == 2 for d in a["test_dates"])
    # the positions, so the test dates, do not depend on the GP's version
    for version in ("score-rank", "ranks"):
        c = control.run_1a("P3", 0.05, 300, 28, 0, write=False, dates=40, selection=version, **kwargs)
        assert c["test_dates"] == a["test_dates"] and c["references"] == a["references"]


# ------------------------------------------------------------------------------------------------ the jobs
def test_the_job_lists_for_p4_ics_and_long_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path)
    assert len(control.CELLS_1A) == 60 and len(control.CELLS_P4) == 20 and len(set(control.CELLS_P4)) == 20
    assert {c[0] for c in control.CELLS_P4} == {"P4"} and sum(1 for c in control.CELLS_P4 if c[3] == 36) == 5
    assert len(control.jobs_1a(runs=1)) == 60
    assert len(control.jobs_1a(runs=1, plants=("P4",))) == 20
    assert len(control.jobs_1a(runs=1, plants=control.PLANTS)) == 80
    assert len(control.jobs_1a(runs=2, plants=("P1", "P4"))) == 2 * 40
    ics = control.jobs_1a(runs=1, selection="score-book", plants=("P1",), ics=[0.0, 0.02, 0.03])
    assert len(ics) == 3 * 4 and {j[1] for j in ics} == {0.0, 0.02, 0.03} and all(j[5] == "score-book" and len(j) == 6 for j in ics)
    long = control.jobs_1a(runs=10, long=True)
    assert len(control.CELLS_LONG) == 12 and len(long) == 120
    assert {j[0] for j in long} == {"P3", "P4"} and {j[6] for j in long} == {150, 315} and all(len(j) == 7 for j in long)
    assert {(j[6], j[3]) for j in long} == {(150, 105), (315, 220)} and all(j[2] == 700 for j in long)
    assert len(control.jobs_1a(runs=3, long=True)) == 36
    assert len(control.jobs_1a(runs=10, long=True, ics=[0.05])) == 40
    assert len(control.jobs_1a(runs=10, long=True, plants=("P4",), selection="score-rank")) == 60
    # finished records are skipped in each form
    for p in (control.path_1a("P4", 0.03, 700, 57, 0), control.path_1a("P3", 0.05, 700, 220, 1, "fitness", 315)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    assert ("P4", 0.03, 700, 57, 0) not in control.jobs_1a(runs=1, plants=("P4",)) and \
        len(control.jobs_1a(runs=1, plants=("P4",))) == 19
    skipped = control.jobs_1a(runs=10, long=True)
    assert ("P3", 0.05, 700, 220, 1, "fitness", 315) not in skipped and len(skipped) == 119


def test_run_job_1a_reads_all_three_forms(monkeypatch):
    seen = []

    def fake(*args, **kwargs):
        seen.append((args, kwargs))
        return {"fitness": 0.1, "detected": False, "seconds": 1.0}

    monkeypatch.setattr(control, "run_1a", fake)
    control.run_job_1a(("P1", 0.05, 700, 57, 2))
    control.run_job_1a(("P1", 0.05, 700, 57, 2, "ranks"))
    control.run_job_1a(("P4", 0.05, 700, 220, 2, "score-rank", 315))
    assert seen[0] == (("P1", 0.05, 700, 57, 2), {"selection": "fitness", "dates": None})
    assert seen[1][1] == {"selection": "ranks", "dates": None}
    assert seen[2] == (("P4", 0.05, 700, 220, 2), {"selection": "score-rank", "dates": 315})


def test_the_report_with_a_p4_record_and_long_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(control.small, "SMALL_DIR", tmp_path / "small")
    monkeypatch.setenv("RFGP_DATA_DIR", str(tmp_path / "data"))
    for k in range(4):
        write(control.path_1a("P1", 0.05, 700, 57, k), fake_1a("P1", 0.05, 700, 57, k, k < 3, 0.02))
        write(control.path_1a("P1", 0.0, 700, 57, k), fake_1a("P1", 0.0, 700, 57, k, False, 0.001 * (-1) ** k))
    write(control.path_1a("P4", 0.05, 700, 57, 0), fake_1a("P4", 0.05, 700, 57, 0, False))
    for k in range(3):
        for plant in ("P3", "P4"):
            record = fake_1a(plant, 0.05, 700, 220, k, k == 0, 0.04)
            record["dates"] = 315
            write(control.path_1a(plant, 0.05, 700, 220, k, "fitness", 315), record)
    out = control.report()
    text = capsys.readouterr().out
    assert set(out) == {"1a", "1a_long"}
    table, longs = out["1a"], out["1a_long"]
    assert (table["plant"] == "P4").sum() == 1 and "dates" not in table.columns
    assert len(table) == 3 and table["runs"].sum() == 9           # the long records stay out of the 81-date table
    assert list(longs.columns) == ["plant", "ic", "dates", "runs", "gp_detection", "gp_captured_median",
                                   "best_single_captured_median", "ridge_captured_median", "lightgbm_captured_median"]
    assert longs["runs"].tolist() == [3, 3] and longs["gp_detection"].iloc[0] == pytest.approx(1 / 3)
    assert longs["gp_captured_median"].iloc[0] == pytest.approx(0.8)
    assert "Long samples" in text and "over 4 runs" in text      # the condition counts the 81-date records only
    assert (tmp_path / "small" / "results" / "control_1a_long.csv").exists()
    assert pd.read_csv(tmp_path / "small" / "results" / "control_1a_long.csv")["dates"].tolist() == [315, 315]


def test_shards_split_the_runs_without_overlap_and_do_not_move():
    jobs = control.jobs_1a(runs=3, selection="score-book", plants=("P1",))
    parts = [control.shard_of(j, 7) for j in jobs]
    assert set(parts) <= set(range(1, 8)) and len(set(parts)) == 7
    # a run's part depends on the run alone, not on which other runs are still to do
    assert [control.shard_of(j, 7) for j in jobs[5:]] == parts[5:]
    assert control.shard_of(list(jobs[0]), 7) == parts[0]
