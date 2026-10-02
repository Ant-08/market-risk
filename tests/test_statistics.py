"""Unit tests for marketrisk.statistics and the covariance ellipse."""

import numpy as np
import pandas as pd
import pytest

from marketrisk.plots import covariance_ellipse, ellipse_parameters
from marketrisk.statistics import (
    correlation_matrix,
    covariance_matrix,
    descriptive_statistics,
    detect_outliers,
    extreme_returns,
    gaussian_outside_probability,
    mahalanobis_distances,
    mean_vector,
    validate_covariance,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    """Small artificial returns with simple, hand-computable moments."""
    index = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0, -0.01], "B": [0.02, -0.04, 0.06, 0.0, -0.02]}, index=index)


@pytest.fixture
def market_like() -> pd.DataFrame:
    """Larger correlated sample used for the covariance properties."""
    rng = np.random.default_rng(0)
    values = rng.multivariate_normal([0.0005, 0.0003], [[4e-4, 1.5e-4], [1.5e-4, 1e-4]], size=1000)
    return pd.DataFrame(values, columns=["X", "Y"], index=pd.date_range("2020-01-01", periods=1000, freq="B"))


def test_covariance_matrix_is_symmetric(market_like):
    cov = covariance_matrix(market_like)
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)
    assert list(cov.index) == list(cov.columns) == ["X", "Y"]


def test_covariance_matches_numpy(market_like):
    np.testing.assert_allclose(covariance_matrix(market_like), np.cov(market_like.to_numpy(), rowvar=False))


def test_moments_on_artificial_dataset(returns):
    # B = 2 * A exactly: perfect correlation and Var(B) = 4 Var(A).
    np.testing.assert_allclose(mean_vector(returns), [0.002, 0.004])
    cov = covariance_matrix(returns)
    assert cov.loc["B", "B"] == pytest.approx(4 * cov.loc["A", "A"])
    assert cov.loc["A", "B"] == pytest.approx(2 * cov.loc["A", "A"])
    np.testing.assert_allclose(correlation_matrix(returns), np.ones((2, 2)))


def test_descriptive_statistics(returns):
    stats = descriptive_statistics(returns)
    assert stats.loc["A", "observations"] == 5
    assert stats.loc["A", "mean"] == pytest.approx(0.002)
    assert stats.loc["A", "std"] == pytest.approx(np.std(returns["A"], ddof=1))
    assert stats.loc["B", "max"] == pytest.approx(0.06)


def test_extreme_returns(returns):
    bottom, top = extreme_returns(returns["A"], k=2)
    assert list(bottom) == [-0.02, -0.01]
    assert list(top) == [0.03, 0.01]
    assert bottom.index[0] == returns.index[1]


def test_invalid_returns_raise(returns):
    with pytest.raises(TypeError):
        covariance_matrix(returns.to_numpy())
    with pytest.raises(ValueError, match="missing"):
        covariance_matrix(returns.assign(A=[0.01, np.nan, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        mean_vector(returns.iloc[:1])


@pytest.mark.parametrize(
    "matrix",
    [[[1.0, 0.5], [0.4, 1.0]], [[1.0, 2.0], [2.0, 1.0]], [[1.0, 0.0, 0.0]], [[np.nan, 0.0], [0.0, 1.0]]],
    ids=["asymmetric", "not-psd", "not-square", "nan"],
)
def test_validate_covariance_rejects_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        validate_covariance(matrix)


def test_mahalanobis_identity_is_euclidean():
    points = np.array([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_allclose(mahalanobis_distances(points, [0, 0], np.eye(2)), [5.0, 0.0, 1.0])


def test_ellipse_points_are_at_requested_distance(market_like):
    mu, cov = mean_vector(market_like), covariance_matrix(market_like)
    for n_std in (1.0, 2.0, 3.0):
        points = covariance_ellipse(mu, cov, n_std=n_std)
        assert points.shape == (200, 2)
        np.testing.assert_allclose(mahalanobis_distances(points, mu, cov), n_std, rtol=1e-8)


def test_ellipse_parameters_diagonal_covariance():
    major, minor, angle = ellipse_parameters([[4.0, 0.0], [0.0, 1.0]], n_std=2)
    assert (major, minor, angle) == pytest.approx((4.0, 2.0, 0.0))


def test_outlier_detection_matches_gaussian_rate(market_like):
    flagged = detect_outliers(market_like, n_std=2.0)
    assert {"mahalanobis", "outlier"} <= set(flagged.columns)
    # Gaussian sample: about exp(-2) = 13.5% of points outside the 2-sd ellipse.
    assert flagged["outlier"].mean() == pytest.approx(gaussian_outside_probability(2.0), abs=0.03)


def test_gaussian_outside_probability_known_values():
    assert gaussian_outside_probability(2.0, dim=1) == pytest.approx(0.0455, abs=1e-4)
    assert gaussian_outside_probability(1.0, dim=2) == pytest.approx(np.exp(-0.5))
