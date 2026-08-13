# 🔢 NumLab — Numerical & Stats Toolkit

> **A hands-on NumPy toolkit for numerical computing, statistics, linear algebra, vectorization, and simulation.**

NumLab is a small numerical and statistics toolkit built with **NumPy**.

It demonstrates how common mathematical operations can be implemented in two ways:

* 🐍 **Manual** — pure Python loops
* ⚡ **Vectorized** — NumPy operations

Every manual implementation is **unit-tested against its NumPy counterpart**, making the project both a reusable toolkit and a practical demonstration of why vectorization matters.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Tests](https://img.shields.io/badge/tests-103-success?style=for-the-badge\&logo=pytest\&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)

</p>

---

## ⚡ Performance at a Glance

The same mathematical operation can have dramatically different performance depending on how it is expressed.

```text
python loop        92.185 ms
numpy vectorized    4.577 ms     → 20× faster

matmul (triple loop)  2471.443 ms
matmul (BLAS)            0.452 ms → 5463× faster
```

> 💡 **Core lesson:** express operations on whole arrays and let optimized numerical libraries handle the computation.

---

## 🧠 What This Project Demonstrates

| Concept                             | Where to look                                                |
| ----------------------------------- | ------------------------------------------------------------ |
| ⚡ Array thinking                    | `zscore`, `moving_average`, `softmax`                        |
| 📡 Broadcasting                     | `pairwise_distances`                                         |
| 🔬 Manual vs vectorized correctness | `mean_manual` / `mean`, `matmul_manual` / `matmul`           |
| 📐 Linear algebra                   | `solve`, `inverse`, `eigen`, `least_squares`, `project_onto` |
| 🎲 Reproducible randomness          | `seed` + `np.random.default_rng`                             |
| 🎯 Monte Carlo methods              | `monte_carlo_pi`, `random_walk`, `bootstrap_ci`              |
| ⏱️ Benchmarking                     | `compare_loop_vs_numpy`, `time_callable`                     |

---

## ✨ Features

### 📦 Array Creation

Create arrays using NumPy-based helpers:

* `arange`
* `linspace`
* `zeros`
* `ones`
* `full`
* `identity`
* `random_uniform`
* `random_normal`
* `random_integers`
* `sample_distribution`

`sample_distribution` supports:

* Normal
* Uniform
* Exponential
* Poisson
* Binomial
* Lognormal

---

### 📊 Descriptive Statistics

Standard statistical operations are implemented both manually and with NumPy:

* `mean`
* `median`
* `variance`
* `std`
* `percentile`

Each has a corresponding pure-Python implementation:

```text
mean_manual
median_manual
variance_manual
std_manual
percentile_manual
```

The `describe()` function returns a structured `Summary` dataclass containing:

* Count
* Mean
* Median
* Standard deviation
* Variance
* Minimum / maximum
* Quartiles
* IQR
* Range

---

### ⚡ Vectorized Operations

NumPy-based transformations designed to avoid Python loops:

```text
zscore
minmax_scale
moving_average
softmax
pairwise_distances
outlier_mask
drop_outliers
```

The `pairwise_distances` implementation is a practical example of **broadcasting**, producing an `(n, m)` distance matrix from arrays shaped `(n, 1, d)` and `(1, m, d)`.

---

### 📐 Linear Algebra

NumLab includes common linear algebra operations:

```text
dot
matmul
transpose
matrix_power
trace
determinant
inverse
solve
least_squares
eigen
norm
normalize_vector
angle_between
project_onto
is_orthogonal
```

`matmul` is available in both forms:

```text
matmul_manual
matmul
```

This makes the performance difference between a Python triple loop and optimized numerical routines directly measurable.

---

### 🎲 Simulations

NumLab also demonstrates vectorized simulation techniques:

#### π estimation

`monte_carlo_pi` estimates π using random points and returns a `PiEstimate` containing:

* Estimated π
* Absolute error
* Relative error

#### Random walk

`random_walk` generates many independent walks using a single vectorized `cumsum`.

#### Bootstrap confidence intervals

`bootstrap_ci` calculates percentile confidence intervals while generating all resample indices as one matrix rather than looping over individual samples.

---

### ⏱️ Benchmarking

Two utilities make performance comparisons easy:

```text
time_callable
compare_loop_vs_numpy
```

They allow direct comparison between manual Python implementations and NumPy's optimized implementations.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/numlab.git
cd numlab
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

Only **NumPy** is required to use the toolkit.

Additional packages such as `pytest`, `jupyter`, and `matplotlib` are used for testing and visualization.

---

## 🖥️ Usage

### Command Line

Run the complete demonstration:

```bash
python src/main.py demo
```

Run individual sections:

```bash
python src/main.py arrays
python src/main.py stats --size 1000
python src/main.py vector
python src/main.py matrix --size 4
python src/main.py pi --samples 2000000
python src/main.py bench --size 2000000
python src/main.py io --path data/demo
```

Use a custom random seed:

```bash
python src/main.py --seed 7 stats
```

---

## 🐍 As a Python Library

```python
import sys
sys.path.insert(0, "src")

import toolkit as tk

# Generate data
data = tk.random_normal(
    1_000,
    loc=10,
    scale=2.5,
    seed=42
)

# Descriptive statistics
print(tk.describe(data))

# Manual vs NumPy implementation
print(tk.mean_manual(data))
print(tk.mean(data))

# Vectorized transformations
tk.zscore(data)
tk.moving_average(data, window=20)
tk.drop_outliers(data, threshold=3.0)

# Linear algebra
A = tk.random_normal((3, 3), seed=1)
x = tk.solve(A, [1.0, 1.0, 1.0])

# Check residual
print(tk.norm(A @ x - 1.0))

# Monte Carlo π
print(tk.monte_carlo_pi(1_000_000, seed=0))

# Bootstrap confidence interval
print(tk.bootstrap_ci(data, seed=0))
```

---

## 📓 Notebook

The project includes a visual walkthrough:

**[`notebooks/demo.ipynb`](notebooks/demo.ipynb)**

The notebook demonstrates:

* 📈 Distribution histograms
* 🎯 Monte Carlo π convergence
* 📉 The `1 / sqrt(n)` convergence relationship
* 🚶 Random walk trajectories
* 📐 The `±sqrt(n)` envelope
* 🔥 Broadcasting heatmaps
* ⚡ Loop vs NumPy performance

Run it with:

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 🧪 Tests

Run the complete test suite:

```bash
pytest tests -q
```

Current test suite:

```text
103 tests
```

The tests verify three major areas:

### 1. Manual vs NumPy correctness

Manual implementations are compared against NumPy equivalents.

### 2. Statistical correctness

Examples include:

* A 400-step random walk has spread approximately `sqrt(400)`
* A 95% bootstrap confidence interval brackets the true mean
* `softmax` outputs sum to `1`

### 3. Input validation

Invalid input should raise a clear `ValueError` instead of silently producing invalid values such as `nan`.

---

## ⚡ Performance Notes

Measurements on **500,000 elements** and **120 × 120 matrices**, using the best of three runs:

| Operation                       | Python Loop |   NumPy |  Speed-up |
| ------------------------------- | ----------: | ------: | --------: |
| Sum of squares — 500k           |     92.2 ms |  4.6 ms |   **20×** |
| Matrix multiplication — 120×120 |     2471 ms | 0.45 ms | **5463×** |

### Why is NumPy faster?

The **20× improvement** comes mainly from removing the Python interpreter from the inner loop.

Instead of repeatedly operating on Python objects:

```text
Python → Python → Python → Python → ...
```

NumPy performs the operation in optimized compiled code over contiguous numerical arrays.

The matrix multiplication result is even more dramatic.

`@` can dispatch the computation to **BLAS**, which can take advantage of:

* Cache-aware blocking
* SIMD instructions
* Highly optimized compiled routines
* Hardware-level parallelism

This is why the matrix multiplication benchmark reaches a difference of roughly **5463×** in this particular environment.

> ⚠️ Benchmark numbers depend heavily on hardware, NumPy version, BLAS backend, array sizes, and system load. They should be treated as illustrative rather than universal.

---

## 🧮 Two Important Numerical Computing Habits

### `solve(A, b)` instead of `inverse(A) @ b`

When solving:

```text
Ax = b
```

prefer:

```python
x = solve(A, b)
```

rather than:

```python
x = inverse(A) @ b
```

Computing the inverse explicitly is generally unnecessary. A factorization-based solver is typically faster and numerically preferable.

---

### 🎲 Vectorize Randomness Too

NumLab avoids repeatedly generating individual random values inside Python loops.

For example, `monte_carlo_pi` generates an entire `(n, 2)` array of random points and performs the calculation with a single vectorized operation.

Similarly, `bootstrap_ci` generates the complete resampling index matrix instead of iterating over every bootstrap sample.

The general principle is:

> **Move work out of Python loops and into optimized array operations whenever possible.**

---

## 📁 Project Structure

```text
numlab/
│
├── src/
│   ├── main.py
│   │   └── CLI demo using argparse
│   │
│   └── toolkit.py
│       └── Numerical & statistics library
│
├── notebooks/
│   └── demo.ipynb
│       └── Visual walkthrough
│
├── tests/
│   ├── conftest.py
│   ├── test_arrays.py
│   ├── test_stats.py
│   ├── test_vectorized.py
│   ├── test_linalg.py
│   └── test_simulation.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🎯 Why NumLab?

NumLab is intentionally more than a collection of NumPy wrappers.

The project is designed to make several important ideas in **scientific computing and machine learning** concrete:

```text
Mathematics
    ↓
Python implementation
    ↓
NumPy vectorization
    ↓
Correctness testing
    ↓
Performance benchmarking
    ↓
Numerical computing mindset
```

It connects mathematical concepts such as:

* Statistics
* Probability
* Linear algebra
* Random processes
* Numerical methods

with practical programming concepts such as:

* Vectorization
* Broadcasting
* Array operations
* Numerical stability
* Testing
* Benchmarking
* Reproducibility

---

## 🛠️ Tech Stack

| Technology         | Purpose                    |
| ------------------ | -------------------------- |
| 🐍 Python          | Core language              |
| 🔢 NumPy           | Numerical computing        |
| 🧪 Pytest          | Testing                    |
| 📓 Jupyter         | Interactive demonstrations |
| 📊 Matplotlib      | Visualization              |
| 📦 `.npy` / `.npz` | Array persistence          |

---

## 📌 Learning Goals

This project was built to strengthen practical understanding of:

* NumPy arrays
* Vectorization
* Broadcasting
* Descriptive statistics
* Probability simulations
* Linear algebra
* Numerical stability
* Performance optimization
* Unit testing
* Reproducible experiments

These are foundational concepts for **data science, scientific computing, and machine learning**.

---

## 📄 License

This project is licensed under the **MIT License**.

---

<p align="center">

**NumLab** — Learn the mathematics.
**Implement it manually.**
**Vectorize it.**
**Measure the difference.**

</p>
