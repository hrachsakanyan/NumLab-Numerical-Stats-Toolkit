"""Monte Carlo pi, random walks, bootstrap intervals, timing and file I/O."""

import numpy as np
import pytest

import toolkit as tk


def test_monte_carlo_pi_converges():
    result = tk.monte_carlo_pi(200_000, seed=0)
    assert result.estimate == pytest.approx(np.pi, abs=0.01)
    assert result.samples == 200_000
    assert 0 < result.inside < result.samples
    assert result.absolute_error == pytest.approx(abs(result.estimate - np.pi))
    assert result.relative_error < 0.01


def test_monte_carlo_pi_is_reproducible():
    assert tk.monte_carlo_pi(5_000, seed=42) == tk.monte_carlo_pi(5_000, seed=42)


def test_more_samples_shrink_the_error_on_average():
    coarse = np.mean([tk.monte_carlo_pi(500, seed=s).absolute_error for s in range(20)])
    fine = np.mean([tk.monte_carlo_pi(50_000, seed=s).absolute_error for s in range(20)])
    assert fine < coarse


def test_monte_carlo_pi_rejects_zero_samples():
    with pytest.raises(ValueError):
        tk.monte_carlo_pi(0)


def test_random_walk_shape_and_origin():
    walks = tk.random_walk(steps=100, walks=7, seed=1)
    assert walks.shape == (7, 101)
    assert np.all(walks[:, 0] == 0.0)


def test_random_walk_steps_are_unit_sized():
    walks = tk.random_walk(steps=50, walks=3, seed=2)
    assert set(np.unique(np.diff(walks, axis=1))) == {-1.0, 1.0}


def test_random_walk_spread_matches_theory():
    # After n steps the standard deviation of the position is sqrt(n).
    walks = tk.random_walk(steps=400, walks=4_000, seed=3)
    assert walks[:, -1].std() == pytest.approx(np.sqrt(400), rel=0.1)


@pytest.mark.parametrize("steps,walks", [(0, 1), (1, 0)])
def test_random_walk_rejects_empty_configurations(steps, walks):
    with pytest.raises(ValueError):
        tk.random_walk(steps=steps, walks=walks)


def test_bootstrap_ci_brackets_the_true_mean():
    data = tk.random_normal(500, loc=4.0, scale=1.0, seed=4)
    low, high = tk.bootstrap_ci(data, resamples=2_000, seed=4)
    assert low < 4.0 < high
    assert low < data.mean() < high


def test_wider_confidence_gives_a_wider_interval():
    data = tk.random_normal(300, seed=5)
    narrow = tk.bootstrap_ci(data, resamples=1_000, confidence=0.80, seed=5)
    wide = tk.bootstrap_ci(data, resamples=1_000, confidence=0.99, seed=5)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_bootstrap_ci_accepts_other_statistics():
    data = tk.random_normal(300, loc=2.0, seed=6)
    low, high = tk.bootstrap_ci(data, statistic=np.median, resamples=1_000, seed=6)
    assert low < np.median(data) < high


@pytest.mark.parametrize("kwargs", [{"confidence": 0.0}, {"confidence": 1.0}])
def test_bootstrap_ci_rejects_invalid_confidence(kwargs):
    with pytest.raises(ValueError):
        tk.bootstrap_ci([1.0, 2.0, 3.0], **kwargs)


def test_bootstrap_ci_rejects_empty_data():
    with pytest.raises(ValueError):
        tk.bootstrap_ci([])


def test_time_callable_reports_a_positive_duration():
    result = tk.time_callable(sum, range(10_000), label="sum", repeats=2)
    assert result.seconds > 0
    assert result.label == "sum" and result.repeats == 2


def test_time_callable_rejects_zero_repeats():
    with pytest.raises(ValueError):
        tk.time_callable(sum, [1], repeats=0)


def test_numpy_beats_the_python_loop():
    report = tk.compare_loop_vs_numpy(200_000, seed=0)
    assert report["speedup"] > 1.0
    assert report["numpy"].seconds < report["loop"].seconds


def test_npy_round_trip(tmp_path):
    data = tk.random_normal((10, 4), seed=7)
    path = tk.save_array(tmp_path / "sample", data)
    assert path.suffix == ".npy" and path.exists()
    np.testing.assert_array_equal(tk.load_array(path), data)


def test_npz_round_trip_preserves_every_key(tmp_path):
    features = tk.random_normal((5, 2), seed=8)
    labels = tk.random_integers(5, 0, 2, seed=8)
    path = tk.save_arrays(tmp_path / "bundle", features=features, labels=labels)
    loaded = tk.load_arrays(path)
    assert sorted(loaded) == ["features", "labels"]
    np.testing.assert_array_equal(loaded["features"], features)
    np.testing.assert_array_equal(loaded["labels"], labels)


def test_save_creates_missing_directories(tmp_path):
    path = tk.save_array(tmp_path / "nested" / "deeper" / "sample", np.arange(3))
    assert path.exists()
