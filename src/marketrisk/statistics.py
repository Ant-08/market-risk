"""Descriptive statistics of returns (Part 1)."""

from __future__ import annotations

import pandas as pd


def validate_returns(returns: pd.DataFrame, min_obs: int = 2) -> None:
    """Check that ``returns`` is a numeric DataFrame without missing values.

    Parameters
    ----------
    returns : pd.DataFrame
        Return table, one column per asset.
    min_obs : int, default 2
        Minimum number of observations required.

    Raises
    ------
    TypeError
        If ``returns`` is not a DataFrame.
    ValueError
        If it has too few rows, non-numeric columns or missing values.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pandas DataFrame, got {type(returns).__name__}.")
    if returns.shape[1] == 0 or len(returns) < min_obs:
        raise ValueError(f"returns must contain at least {min_obs} observations.")
    non_numeric = [c for c in returns.columns if not pd.api.types.is_numeric_dtype(returns[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric return columns: {non_numeric}.")
    if returns.isna().any().any():
        raise ValueError("returns contains missing values.")


def descriptive_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics of daily returns for each asset.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns, one column per asset.

    Returns
    -------
    pd.DataFrame
        One row per asset with the number of observations, mean, standard
        deviation (``ddof=1``), minimum, maximum, skewness and excess kurtosis.
    """
    validate_returns(returns)
    return pd.DataFrame(
        {
            "observations": returns.count(),
            "mean": returns.mean(),
            "std": returns.std(),
            "min": returns.min(),
            "max": returns.max(),
            "skewness": returns.skew(),
            "excess kurtosis": returns.kurt(),
        }
    )


def extreme_returns(returns: pd.Series, k: int = 5) -> tuple[pd.Series, pd.Series]:
    """Return the ``k`` worst and ``k`` best returns of one asset.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of a single asset, indexed by date.
    k : int, default 5
        Number of observations in each tail.

    Returns
    -------
    bottom, top : tuple of pd.Series
        The ``k`` lowest returns (ascending) and the ``k`` highest returns
        (descending), with their dates.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if k < 1:
        raise ValueError("k must be a positive integer.")
    clean = returns.dropna()
    return clean.nsmallest(k), clean.nlargest(k)
