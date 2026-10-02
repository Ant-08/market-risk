# Market Risk Analysis with Streamlit

Multi-page Streamlit application exploring the statistical behaviour of financial
returns (Yahoo Finance data) and comparing them with a Gaussian model having the
same estimated mean and covariance.

## Installation

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`
and uv downloads it automatically if needed.

```bash
git clone <repo-url>
cd market-risk
uv sync
```

`uv sync` recreates the exact environment recorded in `uv.lock` (in `.venv/`, not committed).

## Run the application

```bash
uv run streamlit run app.py
```

## Run the tests

```bash
uv run pytest
```

## Code organization

```
app.py                      # Streamlit home page
pages/                      # Streamlit pages (interface code only)
    1_Market_Data.py        # Part 1: prices, log-returns, descriptive statistics
    2_Risk_Explorer.py      # Part 2: covariance, correlation, covariance ellipse
    3_Gaussian_Simulator.py # Part 3: real returns vs Gaussian simulation
src/marketrisk/             # Reusable Python package
    data.py                 # data download and return computation
    statistics.py           # statistical computations
    simulation.py           # GaussianSimulator class
    plots.py                # reusable Plotly figures
tests/                      # pytest unit tests
pyproject.toml / uv.lock    # dependencies and exact locked versions
```

## Git workflow

Each feature is developed on its own branch (e.g. `feature/market-data`) and merged
into `main` through a Pull Request.
