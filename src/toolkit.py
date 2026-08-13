"""NumLab toolkit — a small numerical & statistics library built on NumPy.

The module is organised in six sections:

1. Array creation      — reproducible generators for the rest of the toolkit.
2. Descriptive stats   — pure-Python ("manual") vs. NumPy implementations.
3. Vectorized ops      — element-wise / broadcasting transformations.
4. Linear algebra      — dot products, matrix multiplication, solving systems.
5. Simulations         — Monte Carlo pi, random walks, bootstrap CIs.
6. Utilities           — timing comparisons and .npy/.npz persistence.

Every function that draws random numbers accepts a ``seed`` (or an already
built ``numpy.random.Generator``) so results are reproducible.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    # array creation
    "rng_from",
    "arange",
    "linspace",
    "zeros",
    "ones",
    "full",
    "identity",
    "random_uniform",
    "random_normal",
    "random_integers",
    "sample_distribution",
    # descriptive statistics
    "Summary",
    "mean_manual",
    "variance_manual",
    "std_manual",
    "median_manual",
    "percentile_manual",
    "mean",
    "variance",
    "std",
    "median",
    "percentile",
    "describe",
    # vectorized operations
    "zscore",
    "minmax_scale",
    "moving_average",
    "softmax",
    "pairwise_distances",
    "outlier_mask",
    "drop_outliers",
    # linear algebra
    "dot",
    "dot_manual",
    "matmul",
    "matmul_manual",
    "transpose",
    "matrix_power",
    "trace",
    "determinant",
    "inverse",
    "solve",
    "least_squares",
    "eigen",
    "norm",
    "normalize_vector",
    "angle_between",
    "project_onto",
    "is_orthogonal",
    # simulations
    "PiEstimate",
    "monte_carlo_pi",
    "random_walk",
    "bootstrap_ci",
    # utilities
    "TimingResult",
    "time_callable",
    "compare_loop_vs_numpy",
    "save_array",
    "load_array",
    "save_arrays",
    "load_arrays",
]

# A 1-D sequence of numbers accepted by the "manual" statistics helpers.
Numeric1D = Sequence[float] | np.ndarray

DISTRIBUTIONS: tuple[str, ...] = (
    "normal",
    "uniform",
    "exponential",
    "poisson",
    "binomial",
    "lognormal",
)


# --------------------------------------------------------------------------- #
# 1. Array creation
# --------------------------------------------------------------------------- #
def rng_from(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a ``Generator``; pass an int for reproducible draws.

    Accepting either a seed or an existing generator lets callers chain several
    random helpers without accidentally reusing the same stream.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def arange(start: float, stop: float | None = None, step: float = 1.0) -> np.ndarray:
    """Evenly spaced values over ``[start, stop)`` with a fixed *step*."""
    if stop is None:
        start, stop = 0.0, start
    return np.arange(start, stop, step, dtype=float)


def linspace(start: float, stop: float, num: int = 50) -> np.ndarray:
    """``num`` evenly spaced values over the closed interval ``[start, stop]``."""
    if num < 0:
        raise ValueError("num must be non-negative")
    return np.linspace(start, stop, num)


def zeros(shape: int | tuple[int, ...]) -> np.ndarray:
    """Array of zeros with the given shape."""
    return np.zeros(shape, dtype=float)


def ones(shape: int | tuple[int, ...]) -> np.ndarray:
    """Array of ones with the given shape."""
    return np.ones(shape, dtype=float)


def full(shape: int | tuple[int, ...], value: float) -> np.ndarray:
    """Array of the given shape filled with ``value``."""
    return np.full(shape, value, dtype=float)


def identity(n: int) -> np.ndarray:
    """``n x n`` identity matrix."""
    return np.eye(n)


def random_uniform(
    shape: int | tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform samples on ``[low, high)``."""
    if high <= low:
        raise ValueError("high must be greater than low")
    return rng_from(seed).uniform(low, high, size=shape)


def random_normal(
    shape: int | tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Gaussian samples with mean ``loc`` and standard deviation ``scale``."""
    if scale < 0:
        raise ValueError("scale must be non-negative")
    return rng_from(seed).normal(loc, scale, size=shape)


def random_integers(
    shape: int | tuple[int, ...],
    low: int,
    high: int,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Integers drawn uniformly from ``[low, high)``."""
    return rng_from(seed).integers(low, high, size=shape)


def sample_distribution(
    name: str,
    size: int | tuple[int, ...],
    seed: int | np.random.Generator | None = None,
    **params: float,
) -> np.ndarray:
    """Draw from one of :data:`DISTRIBUTIONS` by name.

    Examples
    --------
    >>> sample_distribution("poisson", 5, seed=0, lam=3.0).shape
    (5,)
    """
    generator = rng_from(seed)
    match name:
        case "normal":
            return generator.normal(params.get("loc", 0.0), params.get("scale", 1.0), size)
        case "uniform":
            return generator.uniform(params.get("low", 0.0), params.get("high", 1.0), size)
        case "exponential":
            return generator.exponential(params.get("scale", 1.0), size)
        case "poisson":
            return generator.poisson(params.get("lam", 1.0), size).astype(float)
        case "binomial":
            n = int(params.get("n", 10))
            return generator.binomial(n, params.get("p", 0.5), size).astype(float)
        case "lognormal":
            return generator.lognormal(params.get("mean", 0.0), params.get("sigma", 1.0), size)
        case _:
            raise ValueError(f"unknown distribution {name!r}; choose from {DISTRIBUTIONS}")


# --------------------------------------------------------------------------- #
# 2. Descriptive statistics
# --------------------------------------------------------------------------- #
def _as_list(data: Numeric1D) -> list[float]:
    """Flatten *data* into a plain Python list, raising on empty input."""
    values = [float(x) for x in np.ravel(np.asarray(data, dtype=float))]
    if not values:
        raise ValueError("cannot compute a statistic of an empty sequence")
    return values


# --- manual (pure Python, no NumPy) --------------------------------------- #
def mean_manual(data: Numeric1D) -> float:
    """Arithmetic mean computed with an explicit loop."""
    values = _as_list(data)
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def variance_manual(data: Numeric1D, ddof: int = 0) -> float:
    """Variance computed with an explicit loop (``ddof=1`` for the sample form)."""
    values = _as_list(data)
    n = len(values)
    if n - ddof <= 0:
        raise ValueError("not enough data points for the requested ddof")
    mu = mean_manual(values)
    total = 0.0
    for value in values:
        total += (value - mu) ** 2
    return total / (n - ddof)


def std_manual(data: Numeric1D, ddof: int = 0) -> float:
    """Standard deviation computed from :func:`variance_manual`."""
    return variance_manual(data, ddof) ** 0.5


def median_manual(data: Numeric1D) -> float:
    """Median computed by sorting a copy of the data."""
    values = sorted(_as_list(data))
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def percentile_manual(data: Numeric1D, q: float) -> float:
    """Linear-interpolated percentile, matching NumPy's default method."""
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    values = sorted(_as_list(data))
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


# --- NumPy versions -------------------------------------------------------- #
def mean(data: Numeric1D, axis: int | None = None) -> np.ndarray | float:
    """Vectorized mean (``np.mean``)."""
    return np.mean(np.asarray(data, dtype=float), axis=axis)


def variance(data: Numeric1D, ddof: int = 0, axis: int | None = None) -> np.ndarray | float:
    """Vectorized variance (``np.var``)."""
    return np.var(np.asarray(data, dtype=float), ddof=ddof, axis=axis)


def std(data: Numeric1D, ddof: int = 0, axis: int | None = None) -> np.ndarray | float:
    """Vectorized standard deviation (``np.std``)."""
    return np.std(np.asarray(data, dtype=float), ddof=ddof, axis=axis)


def median(data: Numeric1D, axis: int | None = None) -> np.ndarray | float:
    """Vectorized median (``np.median``)."""
    return np.median(np.asarray(data, dtype=float), axis=axis)


def percentile(data: Numeric1D, q: float, axis: int | None = None) -> np.ndarray | float:
    """Vectorized percentile (``np.percentile``)."""
    return np.percentile(np.asarray(data, dtype=float), q, axis=axis)


@dataclass(frozen=True)
class Summary:
    """Descriptive statistics for a 1-D sample."""

    count: int
    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    maximum: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        """Interquartile range (``q3 - q1``)."""
        return self.q3 - self.q1

    @property
    def range(self) -> float:
        """Spread between the largest and smallest observation."""
        return self.maximum - self.minimum

    def as_dict(self) -> dict[str, float]:
        """Field values as a plain dictionary (handy for tables / JSON)."""
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        rows = [
            ("mean", self.mean),
            ("median", self.median),
            ("std", self.std),
            ("variance", self.variance),
            ("min", self.minimum),
            ("q1", self.q1),
            ("q3", self.q3),
            ("max", self.maximum),
            ("iqr", self.iqr),
            ("range", self.range),
        ]
        body = "\n".join(f"{label:<9}{value:>14.4f}" for label, value in rows)
        return f"{'count':<9}{self.count:>14d}\n{body}"


def describe(data: Numeric1D, ddof: int = 0) -> Summary:
    """Compute a :class:`Summary` of *data* in a single pass of NumPy calls."""
    array = np.ravel(np.asarray(data, dtype=float))
    if array.size == 0:
        raise ValueError("cannot describe an empty array")
    q1, q3 = np.percentile(array, [25, 75])
    return Summary(
        count=int(array.size),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=float(np.std(array, ddof=ddof)),
        variance=float(np.var(array, ddof=ddof)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        q1=float(q1),
        q3=float(q3),
    )


# --------------------------------------------------------------------------- #
# 3. Vectorized operations
# --------------------------------------------------------------------------- #
def zscore(data: Numeric1D, ddof: int = 0, axis: int | None = None) -> np.ndarray:
    """Standardise to zero mean and unit variance.

    A constant array has zero spread, so it is returned as all zeros instead of
    producing ``nan`` through a division by zero.
    """
    array = np.asarray(data, dtype=float)
    spread = np.std(array, ddof=ddof, axis=axis, keepdims=axis is not None)
    centred = array - np.mean(array, axis=axis, keepdims=axis is not None)
    return np.divide(centred, spread, out=np.zeros_like(centred), where=spread != 0)


def minmax_scale(
    data: Numeric1D,
    feature_range: tuple[float, float] = (0.0, 1.0),
    axis: int | None = None,
) -> np.ndarray:
    """Rescale values linearly into ``feature_range``."""
    low, high = feature_range
    if high <= low:
        raise ValueError("feature_range must be increasing")
    array = np.asarray(data, dtype=float)
    keepdims = axis is not None
    smallest = np.min(array, axis=axis, keepdims=keepdims)
    largest = np.max(array, axis=axis, keepdims=keepdims)
    span = largest - smallest
    unit = np.divide(array - smallest, span, out=np.zeros_like(array), where=span != 0)
    return unit * (high - low) + low


def moving_average(data: Numeric1D, window: int) -> np.ndarray:
    """Simple moving average over a 1-D array (``mode='valid'``)."""
    array = np.ravel(np.asarray(data, dtype=float))
    if window < 1:
        raise ValueError("window must be at least 1")
    if window > array.size:
        raise ValueError("window cannot be larger than the array")
    kernel = np.ones(window) / window
    return np.convolve(array, kernel, mode="valid")


def softmax(data: Numeric1D, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (the max is subtracted before exponentiating)."""
    array = np.asarray(data, dtype=float)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def pairwise_distances(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Euclidean distances between every row of *a* and every row of *b*.

    Pure broadcasting: ``(n, 1, d)`` against ``(1, m, d)`` gives the full
    ``(n, m)`` matrix without a single Python-level loop.
    """
    first = np.atleast_2d(np.asarray(a, dtype=float))
    second = first if b is None else np.atleast_2d(np.asarray(b, dtype=float))
    if first.shape[1] != second.shape[1]:
        raise ValueError("both inputs must have the same number of columns")
    differences = first[:, None, :] - second[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=-1))


def outlier_mask(data: Numeric1D, threshold: float = 3.0) -> np.ndarray:
    """Boolean mask marking values whose |z-score| exceeds *threshold*."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return np.abs(zscore(data)) > threshold


def drop_outliers(data: Numeric1D, threshold: float = 3.0) -> np.ndarray:
    """Return *data* with the outliers found by :func:`outlier_mask` removed."""
    array = np.ravel(np.asarray(data, dtype=float))
    return array[~outlier_mask(array, threshold)]


# --------------------------------------------------------------------------- #
# 4. Linear algebra
# --------------------------------------------------------------------------- #
def _as_matrix(data: np.ndarray, name: str = "matrix") -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got {array.ndim}-D")
    return array


def _as_square(data: np.ndarray, name: str = "matrix") -> np.ndarray:
    array = _as_matrix(data, name)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {array.shape}")
    return array


def dot(a: Numeric1D, b: Numeric1D) -> float:
    """Dot product of two 1-D vectors."""
    first = np.ravel(np.asarray(a, dtype=float))
    second = np.ravel(np.asarray(b, dtype=float))
    if first.shape != second.shape:
        raise ValueError("vectors must have the same length")
    return float(np.dot(first, second))


def dot_manual(a: Numeric1D, b: Numeric1D) -> float:
    """Dot product written as an explicit loop, for comparison with :func:`dot`."""
    first, second = _as_list(a), _as_list(b)
    if len(first) != len(second):
        raise ValueError("vectors must have the same length")
    total = 0.0
    for x, y in zip(first, second):
        total += x * y
    return total


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product ``a @ b``."""
    first, second = _as_matrix(a, "a"), _as_matrix(b, "b")
    if first.shape[1] != second.shape[0]:
        raise ValueError(f"shapes {first.shape} and {second.shape} are not aligned")
    return first @ second


def matmul_manual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Triple-loop matrix product — the baseline :func:`matmul` is measured against."""
    first, second = _as_matrix(a, "a"), _as_matrix(b, "b")
    if first.shape[1] != second.shape[0]:
        raise ValueError(f"shapes {first.shape} and {second.shape} are not aligned")
    n, k = first.shape
    m = second.shape[1]
    result = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            total = 0.0
            for p in range(k):
                total += first[i, p] * second[p, j]
            result[i, j] = total
    return result


def transpose(a: np.ndarray) -> np.ndarray:
    """Transpose of a matrix (a view, not a copy)."""
    return _as_matrix(a).T


def matrix_power(a: np.ndarray, n: int) -> np.ndarray:
    """Raise a square matrix to the integer power *n*."""
    return np.linalg.matrix_power(_as_square(a), n)


def trace(a: np.ndarray) -> float:
    """Sum of the diagonal entries."""
    return float(np.trace(_as_square(a)))


def determinant(a: np.ndarray) -> float:
    """Determinant of a square matrix."""
    return float(np.linalg.det(_as_square(a)))


def inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of a square matrix, with a clear error when it is singular."""
    array = _as_square(a)
    try:
        return np.linalg.inv(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError("matrix is singular and cannot be inverted") from exc


def solve(a: np.ndarray, b: Numeric1D) -> np.ndarray:
    """Solve ``A x = b`` directly instead of forming ``inv(A) @ b``."""
    array = _as_square(a, "A")
    rhs = np.asarray(b, dtype=float)
    if rhs.shape[0] != array.shape[0]:
        raise ValueError("A and b have incompatible shapes")
    try:
        return np.linalg.solve(array, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("system has no unique solution (A is singular)") from exc


def least_squares(a: np.ndarray, b: Numeric1D) -> np.ndarray:
    """Least-squares solution of an over-determined system ``A x ~= b``."""
    array = _as_matrix(a, "A")
    rhs = np.asarray(b, dtype=float)
    if rhs.shape[0] != array.shape[0]:
        raise ValueError("A and b have incompatible shapes")
    solution, *_ = np.linalg.lstsq(array, rhs, rcond=None)
    return solution


def eigen(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and right eigenvectors (columns) of a square matrix."""
    values, vectors = np.linalg.eig(_as_square(a))
    return values, vectors


def norm(v: Numeric1D, order: float | None = None) -> float:
    """Vector/matrix norm; defaults to the Euclidean (L2) norm."""
    return float(np.linalg.norm(np.asarray(v, dtype=float), ord=order))


def normalize_vector(v: Numeric1D) -> np.ndarray:
    """Scale a vector to unit length."""
    array = np.ravel(np.asarray(v, dtype=float))
    length = np.linalg.norm(array)
    if length == 0:
        raise ValueError("cannot normalize the zero vector")
    return array / length


def angle_between(a: Numeric1D, b: Numeric1D, degrees: bool = False) -> float:
    """Angle between two vectors, clipped to keep ``arccos`` in range."""
    cosine = dot(normalize_vector(a), normalize_vector(b))
    radians = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.degrees(radians) if degrees else radians


def project_onto(v: Numeric1D, onto: Numeric1D) -> np.ndarray:
    """Orthogonal projection of *v* onto the direction of *onto*."""
    direction = normalize_vector(onto)
    return dot(v, direction) * direction


def is_orthogonal(a: np.ndarray, tolerance: float = 1e-8) -> bool:
    """True when ``A.T @ A`` equals the identity within *tolerance*."""
    array = _as_square(a)
    return bool(np.allclose(array.T @ array, np.eye(array.shape[0]), atol=tolerance))


# --------------------------------------------------------------------------- #
# 5. Simulations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PiEstimate:
    """Result of a Monte Carlo estimate of pi."""

    samples: int
    inside: int
    estimate: float

    @property
    def absolute_error(self) -> float:
        """Distance from ``math.pi``."""
        return abs(self.estimate - np.pi)

    @property
    def relative_error(self) -> float:
        """Absolute error as a fraction of ``math.pi``."""
        return self.absolute_error / np.pi

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"samples={self.samples:,}  inside={self.inside:,}  "
            f"pi~={self.estimate:.6f}  abs_err={self.absolute_error:.6f}"
        )


def monte_carlo_pi(
    samples: int = 1_000_000,
    seed: int | np.random.Generator | None = None,
) -> PiEstimate:
    """Estimate pi by sampling the unit square.

    The fraction of points landing inside the quarter circle approaches
    ``pi / 4``, so the estimate is ``4 * inside / samples``. Both coordinate
    arrays are drawn at once and tested with a single vectorized comparison.
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    generator = rng_from(seed)
    points = generator.random((samples, 2))
    inside = int(np.count_nonzero(np.sum(points**2, axis=1) <= 1.0))
    return PiEstimate(samples=samples, inside=inside, estimate=4.0 * inside / samples)


def random_walk(
    steps: int = 1_000,
    walks: int = 1,
    step_size: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate ``walks`` independent +/-1 random walks of ``steps`` steps.

    Returns an array of shape ``(walks, steps + 1)`` whose first column is the
    common origin at 0; the cumulative sum does all the work.
    """
    if steps < 1 or walks < 1:
        raise ValueError("steps and walks must be at least 1")
    generator = rng_from(seed)
    increments = generator.choice([-step_size, step_size], size=(walks, steps))
    positions = np.cumsum(increments, axis=1)
    return np.hstack([np.zeros((walks, 1)), positions])


def bootstrap_ci(
    data: Numeric1D,
    statistic: Callable[..., Any] = np.mean,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for *statistic*.

    All resamples are drawn as one ``(resamples, n)`` index matrix, so the
    statistic is applied along an axis rather than inside a Python loop —
    which means *statistic* must accept an ``axis`` keyword (``np.mean``,
    ``np.median``, ``np.std``, ...).
    """
    array = np.ravel(np.asarray(data, dtype=float))
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    generator = rng_from(seed)
    indices = generator.integers(0, array.size, size=(resamples, array.size))
    estimates = statistic(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(estimates, [tail, 100.0 - tail])
    return float(low), float(high)


# --------------------------------------------------------------------------- #
# 6. Utilities: timing and persistence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimingResult:
    """Best-of-N wall-clock timing for a single callable."""

    label: str
    seconds: float
    repeats: int

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.label:<28}{self.seconds * 1e3:>10.3f} ms"


def time_callable(
    func: Callable[..., object],
    *args: object,
    label: str | None = None,
    repeats: int = 3,
    **kwargs: object,
) -> TimingResult:
    """Time ``func(*args, **kwargs)`` and keep the fastest of *repeats* runs.

    The minimum is reported rather than the average: it is the run least
    disturbed by other activity on the machine.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return TimingResult(label=label or getattr(func, "__name__", "callable"), seconds=best, repeats=repeats)


def compare_loop_vs_numpy(
    size: int = 1_000_000,
    seed: int | np.random.Generator | None = None,
) -> dict[str, TimingResult | float]:
    """Time a Python loop against the equivalent vectorized NumPy call.

    Both compute the sum of squares of the same array, so the speed-up is a
    fair apples-to-apples number rather than a difference in work done.
    """
    array = random_normal(size, seed=seed)
    values = array.tolist()  # convert once, outside the timed region

    def python_loop() -> float:
        total = 0.0
        for value in values:
            total += value * value
        return total

    def numpy_vectorized() -> float:
        return float(np.sum(array**2))

    loop = time_callable(python_loop, label="python loop")
    vectorized = time_callable(numpy_vectorized, label="numpy vectorized")
    return {
        "size": size,
        "loop": loop,
        "numpy": vectorized,
        "speedup": loop.seconds / vectorized.seconds if vectorized.seconds else float("inf"),
    }


def save_array(path: str | Path, array: np.ndarray) -> Path:
    """Save a single array to ``.npy`` and return the written path."""
    destination = Path(path).with_suffix(".npy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(array))
    return destination


def load_array(path: str | Path) -> np.ndarray:
    """Load a single array written by :func:`save_array`."""
    return np.load(Path(path).with_suffix(".npy"))


def save_arrays(path: str | Path, **arrays: np.ndarray) -> Path:
    """Save several named arrays into one compressed ``.npz`` archive."""
    destination = Path(path).with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load every array from an ``.npz`` archive into a dictionary."""
    with np.load(Path(path).with_suffix(".npz")) as archive:
        return {name: archive[name] for name in archive.files}
