"""Statistics of returns: descriptive statistics, covariance, outliers (Parts 1-3)."""

from __future__ import annotations

import math

import numpy as np
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


def validate_covariance(covariance, dim: int | None = None, tol: float = 1e-10) -> np.ndarray:
    """Check that ``covariance`` is a valid (symmetric, positive semi-definite) matrix.

    Parameters
    ----------
    covariance : array-like of shape (d, d)
        Candidate covariance matrix.
    dim : int, optional
        Expected dimension ``d``.
    tol : float, default 1e-10
        Numerical tolerance for symmetry and non-negative eigenvalues.

    Returns
    -------
    np.ndarray
        The covariance matrix as a float array.

    Raises
    ------
    ValueError
        If the matrix is not square, has the wrong dimension, contains
        non-finite values, is not symmetric or has negative eigenvalues.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be a square matrix, got shape {cov.shape}.")
    if dim is not None and cov.shape[0] != dim:
        raise ValueError(f"covariance must have shape ({dim}, {dim}), got {cov.shape}.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values.")
    scale = max(1.0, float(np.abs(cov).max()))
    if not np.allclose(cov, cov.T, atol=tol * scale):
        raise ValueError("covariance must be symmetric.")
    if np.linalg.eigvalsh(cov).min() < -tol * scale:
        raise ValueError("covariance must be positive semi-definite.")
    return cov


def mean_vector(returns: pd.DataFrame) -> pd.Series:
    """Estimate the mean vector ``mu`` of the returns (one entry per asset)."""
    validate_returns(returns)
    return returns.mean()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate the (unbiased, ``ddof=1``) covariance matrix ``Sigma`` of the returns."""
    validate_returns(returns)
    return returns.cov()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate the Pearson correlation matrix of the returns."""
    validate_returns(returns)
    return returns.corr()


def mahalanobis_distances(data, mean, covariance) -> np.ndarray:
    """Mahalanobis distance of each observation to ``mean``.

    ``d_i = sqrt((x_i - mu)^T Sigma^{-1} (x_i - mu))``. A point lies on the
    covariance ellipse of size ``k`` standard deviations exactly when
    ``d_i = k``.

    Parameters
    ----------
    data : array-like of shape (n, d)
        Observations (a DataFrame of returns or a NumPy array).
    mean : array-like of shape (d,)
        Centre of the distribution.
    covariance : array-like of shape (d, d)
        Invertible covariance matrix.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    x = np.asarray(data, dtype=float)
    mu = np.asarray(mean, dtype=float)
    if x.ndim != 2 or x.shape[1] != mu.shape[0]:
        raise ValueError(f"data must have shape (n, {mu.shape[0]}), got {x.shape}.")
    cov = validate_covariance(covariance, dim=mu.shape[0])
    centered = x - mu
    try:
        solved = np.linalg.solve(cov, centered.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance is singular; Mahalanobis distance is undefined.") from exc
    return np.sqrt(np.einsum("ij,ij->i", centered, solved))


def gaussian_outside_probability(n_std: float, dim: int = 2) -> float:
    """Probability that a Gaussian vector falls outside its ``n_std`` ellipse.

    For a ``d``-dimensional Gaussian, the squared Mahalanobis distance follows
    a chi-square distribution with ``d`` degrees of freedom. In dimension 2
    this gives the closed form ``P(D > k) = exp(-k^2 / 2)``; in dimension 1,
    ``P(|Z| > k) = erfc(k / sqrt(2))``.
    """
    if n_std <= 0:
        raise ValueError("n_std must be positive.")
    if dim == 1:
        return math.erfc(n_std / math.sqrt(2))
    if dim == 2:
        return math.exp(-(n_std**2) / 2)
    raise NotImplementedError("Only dimensions 1 and 2 are supported.")


def detect_outliers(returns: pd.DataFrame, n_std: float = 2.0) -> pd.DataFrame:
    """Flag the joint observations lying outside the ``n_std`` covariance ellipse.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, one column per asset.
    n_std : float, default 2.0
        Size of the ellipse, in standard deviations (Mahalanobis distance).

    Returns
    -------
    pd.DataFrame
        ``returns`` with two extra columns: ``mahalanobis`` (distance to the
        mean) and ``outlier`` (True when the distance exceeds ``n_std``).
    """
    distances = mahalanobis_distances(returns, mean_vector(returns), covariance_matrix(returns))
    return returns.assign(mahalanobis=distances, outlier=distances > n_std)
