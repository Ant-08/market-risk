"""Simulation of synthetic returns with the same mean and covariance as real data (Part 3).

:class:`GaussianSimulator` draws ``X ~ N(mu, Sigma)``. :class:`StudentSimulator`
(optional extension) draws a multivariate Student-t with the *same* mean and
covariance but heavier tails, to show what the Gaussian model misses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from marketrisk.statistics import covariance_matrix, mean_vector, validate_covariance


class GaussianSimulator:
    """Multivariate Gaussian model ``N(mu, Sigma)``.

    Parameters
    ----------
    mean : array-like of shape (d,)
        Mean vector ``mu``.
    covariance : array-like of shape (d, d)
        Symmetric positive semi-definite covariance matrix ``Sigma``.
    random_state : int, np.random.Generator or None, default None
        Seed (or generator) for reproducible samples.
    columns : sequence of str, optional
        Names of the ``d`` variables, used by :meth:`sample_frame`.

    Examples
    --------
    >>> sim = GaussianSimulator([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], random_state=0)
    >>> sim.sample(1000).shape
    (1000, 2)
    """

    def __init__(self, mean, covariance, random_state=None, columns: Sequence[str] | None = None):
        mean = np.asarray(mean, dtype=float)
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError(f"mean must be a non-empty 1-D vector, got shape {mean.shape}.")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean contains non-finite values.")
        self.mean = mean
        self.covariance = validate_covariance(covariance, dim=mean.size)
        if columns is not None and len(columns) != mean.size:
            raise ValueError(f"columns must contain {mean.size} names, got {len(columns)}.")
        self.columns = list(columns) if columns is not None else [f"X{i + 1}" for i in range(mean.size)]
        self.rng = np.random.default_rng(random_state)
        # Factor A with A A^T = Sigma, via the eigendecomposition (works even if Sigma is singular).
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        self._factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))

    @classmethod
    def from_returns(cls, returns: pd.DataFrame, random_state=None, **kwargs) -> GaussianSimulator:
        """Build a simulator whose parameters are estimated from a returns DataFrame."""
        return cls(
            mean_vector(returns),
            covariance_matrix(returns),
            random_state=random_state,
            columns=list(returns.columns),
            **kwargs,
        )

    @property
    def dim(self) -> int:
        """Number of variables ``d``."""
        return self.mean.size

    def _standard_draws(self, n: int) -> np.ndarray:
        """Draw ``n`` centred vectors with identity covariance, shape ``(n, d)``."""
        return self.rng.standard_normal((n, self.dim))

    def sample(self, n: int) -> np.ndarray:
        """Draw ``n`` independent observations.

        Parameters
        ----------
        n : int
            Number of observations (positive integer).

        Returns
        -------
        np.ndarray of shape (n, d)
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        return self.mean + self._standard_draws(int(n)) @ self._factor.T

    def sample_frame(self, n: int, index: pd.Index | None = None) -> pd.DataFrame:
        """Draw ``n`` observations as a DataFrame with the simulator's column names.

        If ``index`` is given (e.g. the dates of the real returns), its length
        must be ``n``.
        """
        if index is not None and len(index) != n:
            raise ValueError(f"index has length {len(index)}, expected {n}.")
        return pd.DataFrame(self.sample(n), columns=self.columns, index=index)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim}, columns={self.columns})"


class StudentSimulator(GaussianSimulator):
    """Multivariate Student-t model with given mean, covariance and degrees of freedom.

    ``X = mu + Z / sqrt(W / nu)`` with ``Z ~ N(0, S)`` and ``W ~ chi2(nu)``.
    The scale matrix ``S = Sigma (nu - 2) / nu`` is chosen so that ``Cov(X) =
    Sigma``: the model has exactly the same first two moments as the Gaussian
    one, only the tails differ (excess kurtosis ``6 / (nu - 4)`` for ``nu > 4``).

    Parameters
    ----------
    mean, covariance, random_state, columns
        See :class:`GaussianSimulator`.
    df : float, default 5.0
        Degrees of freedom ``nu``; must be greater than 2 for the covariance to exist.
    """

    def __init__(self, mean, covariance, random_state=None, columns=None, df: float = 5.0):
        if not df > 2:
            raise ValueError("df must be greater than 2 so that the covariance is finite.")
        super().__init__(mean, covariance, random_state=random_state, columns=columns)
        self.df = float(df)

    def _standard_draws(self, n: int) -> np.ndarray:
        z = self.rng.standard_normal((n, self.dim))
        w = self.rng.chisquare(self.df, size=(n, 1))
        return z * np.sqrt((self.df - 2) / w)  # unit covariance, Student-t tails

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim}, df={self.df:g}, columns={self.columns})"


def student_df_from_kurtosis(excess_kurtosis: float, lower: float = 4.5, upper: float = 30.0) -> float:
    """Degrees of freedom of a Student-t matching a given excess kurtosis.

    Inverts ``k = 6 / (nu - 4)``, i.e. ``nu = 4 + 6 / k``, and clips the
    result to ``[lower, upper]`` (no or negative excess kurtosis gives ``upper``).
    """
    if not np.isfinite(excess_kurtosis) or excess_kurtosis <= 0:
        return upper
    return float(np.clip(4 + 6 / excess_kurtosis, lower, upper))
