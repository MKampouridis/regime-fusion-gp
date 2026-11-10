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
