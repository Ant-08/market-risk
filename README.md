# Market Risk Analysis with Streamlit

Multi-page Streamlit application exploring the statistical behaviour of financial
returns. Real daily prices are downloaded from Yahoo Finance and the observed
log-returns of two assets are compared with synthetic Gaussian returns having the
**same estimated mean vector and covariance matrix**.

The Gaussian model is used as a *reference*, not as an assumption: the point of the
application is to show what it reproduces (mean, variances, correlation) and what
it misses (fat tails, joint extremes, volatility clustering). The project does not
forecast prices and does not implement any trading strategy.

## Installation

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
The Python version (3.12) is pinned in `.python-version`; uv downloads it if needed.

```bash
git clone https://github.com/Ant-08/market-risk.git
cd market-risk
uv sync
```

`uv sync` creates `.venv/` (not committed) with the exact package versions recorded
in `uv.lock`, so every member of the group gets the same environment. No package
has to be installed by hand.

## Running the application

```bash
uv run streamlit run app.py
```

The application opens at <http://localhost:8501>. Choose two assets (or type any
Yahoo Finance ticker) and a period in the sidebar; the selection is kept when
switching pages. An internet connection is needed to download the prices.

| Page | Content |
|---|---|
| **Market Data** | Prices (per asset or rebased to 100), daily log-returns `r_t = log(P_t / P_{t-1})`, mean, standard deviation, skewness, kurtosis, bottom 5 and top 5 returns |
| **Risk Explorer** | Scatter plot of the two return series, mean vector, covariance and correlation matrices, covariance ellipse (1 to 3 standard deviations), outliers by Mahalanobis distance |
| **Gaussian Simulator** | `N(μ, Σ)` sample of the same size as the real data: scatter plots with ellipses, histograms, returns over time, frequency of `|r − μ| > 2σ` days, tail indicators and a written interpretation. Optional Student-t model with the same μ and Σ |

## Tests and code quality

```bash
uv run pytest          # 39 unit tests, no network access needed
uv run ruff check .    # lint (pyflakes, pycodestyle, isort, bugbear, docstrings)
uv run ruff format .   # formatting
```

The same checks run on GitHub Actions for every pull request (`.github/workflows/ci.yml`).

## Code organization

```
app.py                       Home page of the Streamlit app
pages/                       Streamlit pages: interface code only
    1_Market_Data.py         Part 1
    2_Risk_Explorer.py       Part 2
    3_Gaussian_Simulator.py  Part 3
src/marketrisk/              Reusable Python package (installed in the environment by uv)
    data.py                  Yahoo Finance download, missing values, validation, log-returns
    statistics.py            Descriptive statistics, mean vector, covariance/correlation,
                             Mahalanobis distance, outliers, extreme-observation counts
    simulation.py            GaussianSimulator class (+ StudentSimulator extension)
    plots.py                 Plotly figures, including the reusable covariance ellipse
    ui.py                    Shared sidebar (asset/period selection) and cached data loading
tests/                       pytest unit tests on small artificial datasets
pyproject.toml               Project metadata, dependencies, tool configuration
uv.lock                      Exact locked dependency versions
```

Design choices:

- **Separation of concerns.** Pages only arrange widgets and figures; every
  computation lives in `marketrisk` and is unit-tested without Streamlit or network.
- **Validation.** Functions check their inputs (DataFrame type, dates index, missing
  or non-positive prices, symmetric positive semi-definite covariance, sample size)
  and raise `TypeError`/`ValueError` with explicit messages; dates dropped while
  cleaning prices trigger a warning.
- **Covariance ellipse.** Built from the eigendecomposition `Σ = V Λ Vᵀ`: points
  `μ + k V Λ^{1/2} (cos t, sin t)ᵀ`, i.e. all points at Mahalanobis distance `k`.
  The same function is used on the real and simulated data.
- **GaussianSimulator.** Stores `μ`, `Σ` and a NumPy random generator (seeded by
  `random_state` for reproducibility) and samples `μ + Z A^T` with `A Aᵀ = Σ`.
  `StudentSimulator` subclasses it and only changes how the standardised draws are
  generated, keeping the same mean and covariance.

## Git workflow

Each part was developed on its own branch and merged into `main` through a pull request:
`feature/market-data` → `feature/risk-explorer` → `feature/gaussian-simulator` →
`feature/quality-docs`.

```bash
git switch -c feature/my-change
# ... commit ...
git push -u origin feature/my-change
gh pr create --base main
```
