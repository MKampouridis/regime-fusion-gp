import numpy as np

from regime_fusion.seeds import derive_seed, make_rng, run_seeds


def test_seeds_are_deterministic():
    assert derive_seed(1, "AAPL", 2019, 0) == derive_seed(1, "AAPL", 2019, 0)
    assert run_seeds(7, "MSFT", 2020, 30) == run_seeds(7, "MSFT", 2020, 30)


def test_seeds_differ_by_key_and_base():
    seeds = run_seeds(7, "MSFT", 2020, 30)
    assert len(set(seeds)) == 30
    assert seeds != run_seeds(7, "MSFT", 2021, 30)
    assert seeds != run_seeds(7, "AAPL", 2020, 30)
    assert seeds != run_seeds(8, "MSFT", 2020, 30)


def test_more_runs_extend_rather_than_reshuffle():
    # Raising the run count from 30 to 50 keeps the first 30 seeds.
    assert run_seeds(7, "MSFT", 2020, 50)[:30] == run_seeds(7, "MSFT", 2020, 30)


def test_generators_reproduce_draws():
    a = make_rng(3, "XOM", 2022, 5).standard_normal(5)
    b = make_rng(3, "XOM", 2022, 5).standard_normal(5)
    np.testing.assert_array_equal(a, b)
