"""Linear algebra: manual implementations, identities and error handling."""

import numpy as np
import pytest

import toolkit as tk

SINGULAR = np.array([[1.0, 2.0], [2.0, 4.0]])


def test_dot_matches_the_manual_loop():
    a, b = tk.random_normal(50, seed=1), tk.random_normal(50, seed=2)
    assert tk.dot(a, b) == pytest.approx(tk.dot_manual(a, b))


def test_dot_of_known_vectors():
    assert tk.dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)


def test_dot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        tk.dot([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        tk.dot_manual([1, 2], [1, 2, 3])


def test_matmul_matches_the_triple_loop():
    a, b = tk.random_normal((6, 4), seed=3), tk.random_normal((4, 5), seed=4)
    np.testing.assert_allclose(tk.matmul(a, b), tk.matmul_manual(a, b))


def test_matmul_of_known_matrices():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(tk.matmul(a, b), [[2.0, 1.0], [4.0, 3.0]])


def test_matmul_rejects_unaligned_shapes():
    with pytest.raises(ValueError, match="not aligned"):
        tk.matmul(np.zeros((2, 3)), np.zeros((4, 2)))


def test_matrix_helpers_reject_non_2d_input():
    with pytest.raises(ValueError, match="2-dimensional"):
        tk.matmul(np.zeros(3), np.zeros((3, 3)))


def test_square_only_helpers_reject_rectangles():
    for func in (tk.determinant, tk.inverse, tk.trace, tk.is_orthogonal):
        with pytest.raises(ValueError, match="square"):
            func(np.zeros((2, 3)))


def test_transpose_and_identity_round_trip():
    a = tk.random_normal((3, 5), seed=5)
    np.testing.assert_allclose(tk.transpose(tk.transpose(a)), a)


def test_matrix_power_equals_repeated_multiplication():
    a = tk.random_normal((3, 3), seed=6)
    np.testing.assert_allclose(tk.matrix_power(a, 3), a @ a @ a)


def test_trace_and_determinant_of_a_known_matrix():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert tk.trace(a) == pytest.approx(5.0)
    assert tk.determinant(a) == pytest.approx(-2.0)


def test_inverse_times_original_is_the_identity():
    a = tk.random_normal((4, 4), seed=7)
    np.testing.assert_allclose(tk.matmul(a, tk.inverse(a)), np.eye(4), atol=1e-10)


def test_inverse_of_a_singular_matrix_is_rejected():
    with pytest.raises(ValueError, match="singular"):
        tk.inverse(SINGULAR)


def test_solve_recovers_a_planted_solution():
    a = tk.random_normal((5, 5), seed=8)
    expected = np.arange(1.0, 6.0)
    solution = tk.solve(a, a @ expected)
    np.testing.assert_allclose(solution, expected, atol=1e-8)


def test_solve_reports_a_singular_system():
    with pytest.raises(ValueError, match="singular"):
        tk.solve(SINGULAR, [1.0, 2.0])


def test_solve_rejects_incompatible_rhs():
    with pytest.raises(ValueError, match="incompatible"):
        tk.solve(np.eye(3), [1.0, 2.0])


def test_least_squares_fits_a_straight_line():
    x = np.linspace(0, 10, 50)
    design = np.column_stack([x, np.ones_like(x)])
    y = 2.5 * x + 1.0
    slope, intercept = tk.least_squares(design, y)
    assert slope == pytest.approx(2.5)
    assert intercept == pytest.approx(1.0)


def test_eigen_decomposition_satisfies_its_definition():
    a = np.array([[2.0, 0.0], [0.0, 3.0]])
    values, vectors = tk.eigen(a)
    np.testing.assert_allclose(sorted(values.real), [2.0, 3.0])
    for i, value in enumerate(values):
        np.testing.assert_allclose(a @ vectors[:, i], value * vectors[:, i], atol=1e-10)


def test_norm_and_normalize():
    assert tk.norm([3.0, 4.0]) == pytest.approx(5.0)
    assert tk.norm([3.0, 4.0], order=1) == pytest.approx(7.0)
    assert tk.norm(tk.normalize_vector([3.0, 4.0])) == pytest.approx(1.0)


def test_normalizing_the_zero_vector_is_rejected():
    with pytest.raises(ValueError, match="zero vector"):
        tk.normalize_vector([0.0, 0.0])


def test_angle_between_known_vectors():
    assert tk.angle_between([1.0, 0.0], [0.0, 1.0], degrees=True) == pytest.approx(90.0)
    assert tk.angle_between([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-8)
    assert tk.angle_between([1.0, 0.0], [-1.0, 0.0], degrees=True) == pytest.approx(180.0)


def test_projection_lands_on_the_target_direction():
    projection = tk.project_onto([3.0, 3.0], [1.0, 0.0])
    np.testing.assert_allclose(projection, [3.0, 0.0], atol=1e-12)


def test_is_orthogonal_recognises_a_rotation_matrix():
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert tk.is_orthogonal(rotation)
    assert not tk.is_orthogonal(np.array([[1.0, 2.0], [3.0, 4.0]]))
