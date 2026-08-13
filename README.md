# NumLab — Numerical & Stats Toolkit

A small numerical and statistics toolkit built on **NumPy**: array creation, descriptive
statistics implemented twice (a pure-Python loop *and* the vectorized NumPy call),
broadcasting-based transforms, linear algebra, and a Monte Carlo simulation of π.

Every "manual" implementation is unit-tested against its NumPy counterpart, so the
library doubles as a demonstration that the vectorized version computes the same
thing — thousands of times faster.

```
python loop        92.185 ms
numpy vectorized    4.577 ms     ->  20x

matmul (triple loop)  2471.443 ms
matmul (BLAS)            0.452 ms  ->  5463x
```

---

## What it demonstrates

| Concept | Where to look |
| --- | --- |
| Array thinking — no Python loops over data | `zscore`, `moving_average`, `softmax` |
| Broadcasting | `pairwise_distances` — an `(n, m)` distance matrix from `(n,1,d)` vs `(1,m,d)` |
| Manual vs vectorized correctness | `mean_manual` / `mean`, `matmul_manual` / `matmul`, plus `tests/` |
| Linear algebra in code | `solve`, `inverse`, `eigen`, `least_squares`, `project_onto` |
| Reproducible randomness | every random helper takes a `seed` and uses `np.random.default_rng` |
| Monte Carlo simulation | `monte_carlo_pi`, `random_walk`, `bootstrap_ci` |
| Vectorization pay-off | `compare_loop_vs_numpy`, `time_callable` |

---

## Features

**Array creation** — `arange`, `linspace`, `zeros`, `ones`, `full`, `identity`,
`random_uniform`, `random_normal`, `random_integers`, and `sample_distribution` for
six distributions (normal, uniform, exponential, poisson, binomial, lognormal).

**Descriptive statistics** — `mean`, `median`, `variance`, `std`, `percentile`, each
with a `*_manual` pure-Python twin, plus `describe()` returning a `Summary` dataclass
(count, mean, median, std, variance, min/max, quartiles, IQR, range).

**Vectorized operations** — `zscore`, `minmax_scale`, `moving_average`, `softmax`
(numerically stable), `pairwise_distances`, `outlier_mask` / `drop_outliers`.

**Linear algebra** — `dot`, `matmul` (both with manual loop versions), `transpose`,
`matrix_power`, `trace`, `determinant`, `inverse`, `solve`, `least_squares`, `eigen`,
`norm`, `normalize_vector`, `angle_between`, `project_onto`, `is_orthogonal`.

**Simulations** — `monte_carlo_pi` (returns a `PiEstimate` with absolute and relative
error), `random_walk` (many independent walks via one `cumsum`), `bootstrap_ci`
(percentile confidence intervals, all resamples drawn as one index matrix).

**Utilities** — `time_callable` / `compare_loop_vs_numpy` for benchmarking, and
`save_array` / `load_array` / `save_arrays` / `load_arrays` for `.npy` and `.npz`.

---

## Installation

```bash
git clone https://github.com/<your-username>/numlab.git
cd numlab
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Only `numpy` is required to use the toolkit; `pytest`, `jupyter` and `matplotlib`
are for the tests and the notebook.

## Usage

### Command line

```bash
python src/main.py demo                  # run every section end to end
python src/main.py arrays                # array-creation showcase
python src/main.py stats  --size 1000    # manual vs NumPy statistics table
python src/main.py vector                # vectorized ops & broadcasting
python src/main.py matrix --size 4       # linear algebra
python src/main.py pi     --samples 2000000
python src/main.py bench  --size 2000000 # loop vs NumPy timings
python src/main.py io     --path data/demo
python src/main.py --seed 7 stats        # any section, different seed
```

### As a library

```python
import sys; sys.path.insert(0, "src")
import toolkit as tk

data = tk.random_normal(1_000, loc=10, scale=2.5, seed=42)

print(tk.describe(data))                    # full summary
print(tk.mean_manual(data), tk.mean(data))  # identical to ~1e-15

tk.zscore(data)                             # standardise
tk.moving_average(data, window=20)          # smooth
tk.drop_outliers(data, threshold=3.0)       # clean

A = tk.random_normal((3, 3), seed=1)
x = tk.solve(A, [1.0, 1.0, 1.0])            # A x = b, no explicit inverse
tk.norm(A @ x - 1.0)                        # ~1e-15 residual

print(tk.monte_carlo_pi(1_000_000, seed=0)) # pi ~= 3.1416
print(tk.bootstrap_ci(data, seed=0))        # 95% CI for the mean
```

## Notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is the visual walkthrough: distribution
histograms, a convergence plot of the Monte Carlo estimate against `1/sqrt(n)`, random
walk trajectories with the `±sqrt(n)` envelope, a broadcasting heatmap, and the
loop-vs-NumPy timing chart.

```bash
jupyter notebook notebooks/demo.ipynb
```

## Tests

```bash
pytest tests -q          # 103 tests
```

The suite checks three things: that each manual implementation agrees with NumPy, that
statistical results match theory (a 400-step random walk has spread `sqrt(400)`, a
95% bootstrap CI brackets the true mean, `softmax` sums to 1), and that invalid input
raises a clear `ValueError` rather than producing `nan`.

## Performance notes

Measured on 500,000 elements / 120×120 matrices, best of three runs:

| Operation | Python loop | NumPy | Speed-up |
| --- | ---: | ---: | ---: |
| Sum of squares (500k) | 92.2 ms | 4.6 ms | **20×** |
| Matrix multiply (120×120) | 2471 ms | 0.45 ms | **5463×** |

Two different effects are on display. The 20× comes from removing the interpreter from
the inner loop — the same arithmetic, executed in compiled code over a contiguous
buffer instead of over boxed Python floats. The 5463× is larger because `@` does not
just avoid the interpreter, it dispatches to BLAS, which blocks the computation for
cache reuse and uses SIMD instructions. The lesson is the one that carries over to
machine learning: *express the operation on whole arrays and let the library pick the
implementation.*

Two smaller habits the code follows for the same reason:

- **`solve(A, b)` rather than `inverse(A) @ b`** — an LU factorisation is faster than
  forming an explicit inverse and numerically better behaved.
- **Draw all randomness at once** — `monte_carlo_pi` samples an `(n, 2)` array and
  tests it with one comparison; `bootstrap_ci` builds a whole `(resamples, n)` index
  matrix instead of looping over resamples.

## Project structure

```
numlab/
├── src/
│   ├── main.py            # CLI demo (argparse subcommands)
│   └── toolkit.py         # the library
├── notebooks/
│   └── demo.ipynb         # visual walkthrough
├── tests/
│   ├── conftest.py        # puts src/ on sys.path
│   ├── test_arrays.py
│   ├── test_stats.py
│   ├── test_vectorized.py
│   ├── test_linalg.py
│   └── test_simulation.py
├── README.md
├── requirements.txt
└── .gitignore
```

## License

MIT
