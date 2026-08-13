"""The manual statistics must agree with NumPy — that is the whole point."""

import numpy as np
import pytest

import toolkit as tk

SAMPLE = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


@pytest.fixture
def data() -> np.ndarray:
    return tk.random_normal(500, loc=3.0, scale=2.0, seed=7)


def test_mean_matches_numpy(data):
    assert tk.mean_manual(data) == pytest.approx(float(tk.mean(data)))


def test_known_textbook_values():
    # Classic example: mean 5, population std 2.
    assert tk.mean_manual(SAMPLE) == pytest.approx(5.0)
    assert tk.std_manual(SAMPLE) == pytest.approx(2.0)
    assert tk.median_manual(SAMPLE) == pytest.approx(4.5)


@pytest.mark.parametrize("ddof", [0, 1])
def test_variance_and_std_match_numpy(data, ddof):
    assert tk.variance_manual(data, ddof) == pytest.approx(float(tk.variance(data, ddof)))
    assert tk.std_manual(data, ddof) == pytest.approx(float(tk.std(data, ddof)))


@pytest.mark.parametrize("size", [1, 2, 3, 10, 11])
def test_median_matches_numpy_for_odd_and_even_sizes(size):
    values = tk.random_normal(size, seed=size)
    assert tk.median_manual(values) == pytest.approx(float(tk.median(values)))


@pytest.mark.parametrize("q", [0, 1, 25, 50, 75, 99, 100])
def test_percentile_matches_numpy(data, q):
    assert tk.percentile_manual(data, q) == pytest.approx(float(tk.percentile(data, q)))


def test_empty_input_is_rejected():
    for func in (tk.mean_manual, tk.median_manual, tk.variance_manual):
        with pytest.raises(ValueError):
            func([])


def test_variance_rejects_impossible_ddof():
    with pytest.raises(ValueError):
        tk.variance_manual([1.0], ddof=1)


def test_percentile_rejects_out_of_range_q():
    with pytest.raises(ValueError):
        tk.percentile_manual(SAMPLE, 101)


def test_describe_fields_agree_with_manual(data):
    summary = tk.describe(data, ddof=1)
    assert summary.count == data.size
    assert summary.mean == pytest.approx(tk.mean_manual(data))
    assert summary.median == pytest.approx(tk.median_manual(data))
    assert summary.std == pytest.approx(tk.std_manual(data, ddof=1))
    assert summary.minimum == pytest.approx(float(np.min(data)))
    assert summary.maximum == pytest.approx(float(np.max(data)))
    assert summary.iqr == pytest.approx(summary.q3 - summary.q1)
    assert summary.range == pytest.approx(summary.maximum - summary.minimum)
    assert set(summary.as_dict()) >= {"count", "mean", "std", "q1", "q3"}


def test_describe_rejects_empty():
    with pytest.raises(ValueError):
        tk.describe([])
