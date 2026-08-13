"""Command-line demo for the NumLab toolkit.

Usage
-----
    python src/main.py demo             # run every section end to end
    python src/main.py stats  --size 1000
    python src/main.py matrix --size 4
    python src/main.py pi     --samples 2000000
    python src/main.py bench  --size 2000000
    python src/main.py io     --path data/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import toolkit as tk  # noqa: E402  (path must be set before the import)

DEFAULT_SEED = 42


def _heading(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _row(label: str, manual: float, vectorized: float) -> str:
    return f"{label:<12}{manual:>16.8f}{vectorized:>16.8f}{abs(manual - vectorized):>14.2e}"


def show_arrays(seed: int) -> None:
    """Section 1 — a tour of the array constructors."""
    _heading("1. Array creation")
    print("arange(0, 10, 2)      ->", tk.arange(0, 10, 2))
    print("linspace(0, 1, 5)     ->", tk.linspace(0, 1, 5))
    print("ones((2, 3))          ->", tk.ones((2, 3)).tolist())
    print("identity(3)           ->", tk.identity(3).tolist())
    print("random_normal(5)      ->", np.round(tk.random_normal(5, seed=seed), 4))
    print("random_integers(5)    ->", tk.random_integers(5, 1, 7, seed=seed))
    for name in ("normal", "exponential", "poisson"):
        sample = tk.sample_distribution(name, 6, seed=seed)
        print(f"{name:<22}-> {np.round(sample, 3)}")


def show_stats(size: int, seed: int) -> None:
    """Section 2 — manual statistics checked against the NumPy versions."""
    _heading("2. Descriptive statistics — manual vs NumPy")
    data = tk.random_normal(size, loc=10.0, scale=2.5, seed=seed)

    print(f"{'statistic':<12}{'manual':>16}{'numpy':>16}{'|diff|':>14}")
    print("-" * 58)
    print(_row("mean", tk.mean_manual(data), float(tk.mean(data))))
    print(_row("median", tk.median_manual(data), float(tk.median(data))))
    print(_row("variance", tk.variance_manual(data, ddof=1), float(tk.variance(data, ddof=1))))
    print(_row("std", tk.std_manual(data, ddof=1), float(tk.std(data, ddof=1))))
    print(_row("p25", tk.percentile_manual(data, 25), float(tk.percentile(data, 25))))
    print(_row("p90", tk.percentile_manual(data, 90), float(tk.percentile(data, 90))))

    print(f"\ndescribe(sample of {size}):")
    print(tk.describe(data, ddof=1))

    low, high = tk.bootstrap_ci(data, resamples=2_000, seed=seed)
    print(f"\n95% bootstrap CI for the mean: [{low:.4f}, {high:.4f}]")


def show_vectorized(seed: int) -> None:
    """Section 3 — element-wise transforms and broadcasting."""
    _heading("3. Vectorized operations & broadcasting")
    data = tk.random_normal(8, loc=5.0, scale=2.0, seed=seed)
    print("data           ->", np.round(data, 3))
    print("zscore         ->", np.round(tk.zscore(data), 3))
    print("minmax_scale   ->", np.round(tk.minmax_scale(data), 3))
    print("moving_avg(3)  ->", np.round(tk.moving_average(data, 3), 3))
    print("softmax        ->", np.round(tk.softmax(data), 3), " sum =", round(float(tk.softmax(data).sum()), 6))

    points = tk.random_normal((4, 2), seed=seed)
    print("\npairwise_distances of 4 points in 2-D (broadcasting, no loops):")
    print(np.round(tk.pairwise_distances(points), 3))

    contaminated = np.append(tk.random_normal(200, seed=seed), [50.0, -60.0])
    cleaned = tk.drop_outliers(contaminated, threshold=3.0)
    print(f"\noutliers: {contaminated.size} values in -> {cleaned.size} kept "
          f"({contaminated.size - cleaned.size} dropped at |z| > 3)")


def show_linear_algebra(size: int, seed: int) -> None:
    """Section 4 — matrix products, determinants and linear systems."""
    _heading("4. Linear algebra")
    generator = tk.rng_from(seed)
    a = np.round(tk.random_normal((size, size), seed=generator), 2)
    b = np.round(tk.random_normal((size, size), seed=generator), 2)

    print("A =\n", a)
    print("\nA @ B (first row)     ->", np.round(tk.matmul(a, b)[0], 3))
    print("manual matmul matches ->", np.allclose(tk.matmul(a, b), tk.matmul_manual(a, b)))

    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, -5.0, 6.0])
    print(f"\ndot(u, v)             -> {tk.dot(u, v):.4f} (manual: {tk.dot_manual(u, v):.4f})")
    print(f"norm(u)               -> {tk.norm(u):.4f}")
    print(f"angle(u, v)           -> {tk.angle_between(u, v, degrees=True):.2f} degrees")
    print("projection of u on v  ->", np.round(tk.project_onto(u, v), 4))

    print(f"\ntrace(A)              -> {tk.trace(a):.4f}")
    print(f"det(A)                -> {tk.determinant(a):.4f}")
    print("A @ inv(A) == I       ->", np.allclose(tk.matmul(a, tk.inverse(a)), tk.identity(size)))

    rhs = np.ones(size)
    solution = tk.solve(a, rhs)
    print("\nsolve(A, [1, ...])    ->", np.round(solution, 4))
    print("residual |Ax - b|     -> {:.2e}".format(tk.norm(a @ solution - rhs)))

    values, _ = tk.eigen(a)
    print("eigenvalues           ->", np.round(values, 3))


def show_simulation(samples: int, seed: int) -> None:
    """Section 5 — Monte Carlo pi and a random walk."""
    _heading("5. Simulation — Monte Carlo estimate of pi")
    for n in (1_000, 100_000, samples):
        result = tk.monte_carlo_pi(n, seed=seed)
        print(f"  {result}  rel_err={result.relative_error:.2%}")

    walks = tk.random_walk(steps=1_000, walks=500, seed=seed)
    finals = walks[:, -1]
    print("\nRandom walk: 500 walks x 1000 steps")
    print(f"  mean final position     {finals.mean():>8.3f}  (theory 0)")
    print(f"  std of final position   {finals.std():>8.3f}  (theory sqrt(1000) = {np.sqrt(1000):.3f})")
    print(f"  max |displacement|      {np.abs(walks).max():>8.3f}")


def show_benchmark(size: int, seed: int) -> None:
    """Section 6 — Python loops against vectorized NumPy."""
    _heading("6. Performance — Python loop vs NumPy")
    compare = tk.compare_loop_vs_numpy(size, seed=seed)
    print(f"sum of squares over {compare['size']:,} elements")
    print(" ", compare["loop"])
    print(" ", compare["numpy"])
    print(f"  speed-up: {compare['speedup']:.1f}x")

    n = 120
    generator = tk.rng_from(seed)
    a = tk.random_normal((n, n), seed=generator)
    b = tk.random_normal((n, n), seed=generator)
    naive = tk.time_callable(tk.matmul_manual, a, b, label="matmul (triple loop)", repeats=1)
    fast = tk.time_callable(tk.matmul, a, b, label="matmul (BLAS)", repeats=3)
    print(f"\n{n}x{n} matrix multiplication")
    print(" ", naive)
    print(" ", fast)
    print(f"  speed-up: {naive.seconds / fast.seconds:.1f}x")


def show_io(path: str, seed: int) -> None:
    """Section 7 — round-tripping arrays through .npy / .npz."""
    _heading("7. Save / load arrays")
    data = tk.random_normal((100, 3), seed=seed)
    single = tk.save_array(path, data)
    print(f"saved  {single}  ({single.stat().st_size:,} bytes)")
    print("round-trip identical ->", np.array_equal(tk.load_array(single), data))

    archive = tk.save_arrays(f"{path}_bundle", features=data, labels=tk.random_integers(100, 0, 2, seed=seed))
    loaded = tk.load_arrays(archive)
    print(f"saved  {archive}  keys={sorted(loaded)}")
    print("features shape       ->", loaded["features"].shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numlab",
        description="NumLab — a small numerical & statistics toolkit built on NumPy.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("arrays", help="array creation showcase")

    stats = subparsers.add_parser("stats", help="manual vs NumPy descriptive statistics")
    stats.add_argument("--size", type=int, default=1_000, help="sample size (default: %(default)s)")

    subparsers.add_parser("vector", help="vectorized operations and broadcasting")

    matrix = subparsers.add_parser("matrix", help="linear algebra showcase")
    matrix.add_argument("--size", type=int, default=3, help="matrix dimension (default: %(default)s)")

    pi = subparsers.add_parser("pi", help="Monte Carlo estimate of pi + random walk")
    pi.add_argument("--samples", type=int, default=1_000_000, help="samples (default: %(default)s)")

    bench = subparsers.add_parser("bench", help="Python loop vs NumPy timing")
    bench.add_argument("--size", type=int, default=1_000_000, help="array size (default: %(default)s)")

    io_parser = subparsers.add_parser("io", help="save/load .npy and .npz")
    io_parser.add_argument("--path", default="data/demo", help="output path stem (default: %(default)s)")

    demo = subparsers.add_parser("demo", help="run every section (default)")
    demo.add_argument("--samples", type=int, default=200_000, help="Monte Carlo samples (default: %(default)s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "demo"
    seed = args.seed

    match command:
        case "arrays":
            show_arrays(seed)
        case "stats":
            show_stats(args.size, seed)
        case "vector":
            show_vectorized(seed)
        case "matrix":
            show_linear_algebra(args.size, seed)
        case "pi":
            show_simulation(args.samples, seed)
        case "bench":
            show_benchmark(args.size, seed)
        case "io":
            show_io(args.path, seed)
        case "demo":
            samples = getattr(args, "samples", 200_000)
            show_arrays(seed)
            show_stats(1_000, seed)
            show_vectorized(seed)
            show_linear_algebra(3, seed)
            show_simulation(samples, seed)
            show_benchmark(500_000, seed)
            show_io("data/demo", seed)
            print("\nDone. Run `python src/main.py --help` for individual sections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
