"""Element-wise transforms, broadcasting and outlier handling."""

import numpy as np
import pytest

import toolkit as tk


def test_zscore_has_zero_mean_and_unit_std():
    result = tk.zscore(tk.random_normal(1_000, loc=10.0, scale=3.0, seed=1))
    assert result.mean() == pytest.approx(0.0, abs=1e-12)
    assert result.std() == pytest.approx(1.0)


def test_zscore_of_a_constant_array_is_zero_not_nan():
    result = tk.zscore(np.full(5, 3.0))
    assert np.all(result == 0.0)
    assert not np.isnan(result).any()


def test_zscore_along_an_axis_normalises_each_row():
    result = tk.zscore(tk.random_normal((4, 50), seed=2), axis=1)
    np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.std(axis=1), 1.0)


def test_minmax_scale_hits_the_range_endpoints():
    result = tk.minmax_scale(tk.random_normal(100, seed=3), (-1.0, 1.0))
    assert result.min() == pytest.approx(-1.0)
    assert result.max() == pytest.approx(1.0)


def test_minmax_scale_of_a_constant_array_returns_the_low_end():
    np.testing.assert_allclose(tk.minmax_scale(np.full(4, 2.0), (0.0, 1.0)), 0.0)


def test_minmax_scale_rejects_inverted_range():
    with pytest.raises(ValueError):
        tk.minmax_scale([1.0, 2.0], (1.0, 0.0))


def test_moving_average_smooths_a_known_sequence():
    np.testing.assert_allclose(tk.moving_average([1, 2, 3, 4, 5], 3), [2.0, 3.0, 4.0])


def test_moving_average_output_length():
    values = tk.random_normal(20, seed=4)
    assert tk.moving_average(values, 5).size == 20 - 5 + 1


@pytest.mark.parametrize("window", [0, 21])
def test_moving_average_rejects_bad_windows(window):
    with pytest.raises(ValueError):
        tk.moving_average(tk.random_normal(20, seed=4), window)


def test_softmax_rows_sum_to_one():
    result = tk.softmax(tk.random_normal((3, 5), seed=5), axis=-1)
    np.testing.assert_allclose(result.sum(axis=-1), 1.0)
    assert np.all(result > 0)


def test_softmax_is_stable_for_large_inputs():
    result = tk.softmax([1000.0, 1001.0, 1002.0])
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(1.0)


def test_pairwise_distances_against_a_hand_computed_matrix():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    expected = np.array([[0.0, 5.0, 1.0], [5.0, 0.0, np.hypot(3.0, 3.0)], [1.0, np.hypot(3.0, 3.0), 0.0]])
    np.testing.assert_allclose(tk.pairwise_distances(points), expected)


def test_pairwise_distances_is_symmetric_with_zero_diagonal():
    result = tk.pairwise_distances(tk.random_normal((6, 3), seed=6))
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-12)


def test_pairwise_distances_between_two_sets():
    a, b = tk.random_normal((4, 2), seed=7), tk.random_normal((5, 2), seed=8)
    assert tk.pairwise_distances(a, b).shape == (4, 5)


def test_pairwise_distances_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        tk.pairwise_distances(np.zeros((2, 3)), np.zeros((2, 4)))


def test_outliers_are_detected_and_dropped():
    data = np.append(tk.random_normal(1_000, seed=9), [40.0, -40.0])
    mask = tk.outlier_mask(data, threshold=3.0)
    assert mask[-1] and mask[-2]
    cleaned = tk.drop_outliers(data, threshold=3.0)
    assert cleaned.size == data.size - np.count_nonzero(mask)
    assert cleaned.max() < 40.0


def test_outlier_mask_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        tk.outlier_mask([1.0, 2.0], threshold=0.0)
