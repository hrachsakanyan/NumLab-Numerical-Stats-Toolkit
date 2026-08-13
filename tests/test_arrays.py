"""Array constructors: shapes, dtypes and reproducibility."""

import numpy as np
import pytest

import toolkit as tk


def test_arange_with_single_argument_starts_at_zero():
    np.testing.assert_allclose(tk.arange(5), [0, 1, 2, 3, 4])


def test_arange_with_step():
    np.testing.assert_allclose(tk.arange(0, 10, 2), [0, 2, 4, 6, 8])


def test_linspace_includes_both_endpoints():
    values = tk.linspace(0, 1, 5)
    assert values[0] == 0.0 and values[-1] == 1.0
    assert values.size == 5


def test_linspace_rejects_negative_count():
    with pytest.raises(ValueError):
        tk.linspace(0, 1, -1)


def test_constructors_produce_expected_shapes():
    assert tk.zeros((2, 3)).shape == (2, 3)
    assert tk.ones((4,)).shape == (4,)
    assert np.all(tk.full((2, 2), 7.0) == 7.0)
    np.testing.assert_allclose(tk.identity(3), np.eye(3))


def test_same_seed_gives_identical_draws():
    np.testing.assert_array_equal(
        tk.random_normal(10, seed=123),
        tk.random_normal(10, seed=123),
    )


def test_different_seeds_give_different_draws():
    assert not np.array_equal(tk.random_normal(10, seed=1), tk.random_normal(10, seed=2))


def test_passing_a_generator_advances_the_stream():
    generator = tk.rng_from(0)
    first = tk.random_normal(5, seed=generator)
    second = tk.random_normal(5, seed=generator)
    assert not np.array_equal(first, second)


def test_random_uniform_stays_in_bounds():
    values = tk.random_uniform(1_000, low=-2.0, high=3.0, seed=0)
    assert values.min() >= -2.0 and values.max() < 3.0


def test_random_uniform_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        tk.random_uniform(10, low=1.0, high=0.0)


def test_random_normal_rejects_negative_scale():
    with pytest.raises(ValueError):
        tk.random_normal(10, scale=-1.0)


def test_random_integers_stay_in_bounds():
    values = tk.random_integers(500, 1, 7, seed=0)
    assert values.min() >= 1 and values.max() <= 6


@pytest.mark.parametrize("name", tk.DISTRIBUTIONS)
def test_every_distribution_returns_the_requested_shape(name):
    assert tk.sample_distribution(name, (3, 4), seed=0).shape == (3, 4)


def test_normal_distribution_recovers_its_parameters():
    sample = tk.sample_distribution("normal", 200_000, seed=0, loc=5.0, scale=2.0)
    assert sample.mean() == pytest.approx(5.0, abs=0.05)
    assert sample.std() == pytest.approx(2.0, abs=0.05)


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError, match="unknown distribution"):
        tk.sample_distribution("cauchy", 5)
