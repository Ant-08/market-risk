"""Unit tests for marketrisk.simulation and the comparison of extreme observations."""

import numpy as np
import pandas as pd
import pytest

from marketrisk.simulation import GaussianSimulator, StudentSimulator, student_df_from_kurtosis
from marketrisk.statistics import compare_descriptive, compare_extremes, extreme_counts

MEAN = np.array([0.001, -0.0005])
COV = np.array([[4e-4, 1.2e-4], [1.2e-4, 1e-4]])


def test_sample_has_expected_shape():
    sim = GaussianSimulator(MEAN, COV, random_state=0)
    assert sim.sample(250).shape == (250, 2)
    assert sim.dim == 2


def test_same_seed_gives_same_sample():
    first = GaussianSimulator(MEAN, COV, random_state=123).sample(100)
    second = GaussianSimulator(MEAN, COV, random_state=123).sample(100)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, GaussianSimulator(MEAN, COV, random_state=124).sample(100))


@pytest.mark.parametrize("cls", [GaussianSimulator, StudentSimulator])
def test_large_sample_recovers_mean_and_covariance(cls):
    sample = cls(MEAN, COV, random_state=0).sample(200_000)
    np.testing.assert_allclose(sample.mean(axis=0), MEAN, atol=2e-4)
    np.testing.assert_allclose(np.cov(sample, rowvar=False), COV, rtol=0.05, atol=5e-6)


def test_student_has_fatter_tails_than_gaussian():
    gaussian = GaussianSimulator(MEAN, COV, random_state=0).sample_frame(100_000)
    student = StudentSimulator(MEAN, COV, random_state=0, df=5).sample_frame(100_000)
    assert (student.kurt() > 2).all()
    assert (gaussian.kurt().abs() < 0.1).all()
    beyond = extreme_counts(student, 4.0)["count"] > extreme_counts(gaussian, 4.0)["count"]
    assert beyond.all()


def test_from_returns_and_sample_frame():
    index = pd.date_range("2024-01-01", periods=50, freq="B")
    returns = pd.DataFrame(
        np.random.default_rng(1).normal(size=(50, 2)) * 0.01, columns=["AAPL", "MSFT"], index=index
    )
    sim = GaussianSimulator.from_returns(returns, random_state=0)
    np.testing.assert_allclose(sim.mean, returns.mean())
    np.testing.assert_allclose(sim.covariance, returns.cov())
    frame = sim.sample_frame(len(returns), index=returns.index)
    assert list(frame.columns) == ["AAPL", "MSFT"]
    assert frame.index.equals(returns.index)


def test_singular_covariance_is_supported():
    sample = GaussianSimulator([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]], random_state=0).sample(1000)
    np.testing.assert_allclose(sample[:, 0], sample[:, 1])


@pytest.mark.parametrize(
    "mean, cov",
    [
        ([0.0, 0.0], [[1.0, 0.0], [0.5, 1.0]]),
        ([0.0, 0.0], [[1.0, 0.0, 0.0]] * 3),
        ([[0.0]], [[1.0]]),
        ([np.nan, 0.0], np.eye(2)),
    ],
    ids=["asymmetric", "wrong-dimension", "2d-mean", "nan-mean"],
)
def test_invalid_parameters_raise(mean, cov):
    with pytest.raises(ValueError):
        GaussianSimulator(mean, cov)


@pytest.mark.parametrize("n", [0, -5, 2.5, True])
def test_invalid_sample_size_raises(n):
    with pytest.raises(ValueError):
        GaussianSimulator(MEAN, COV).sample(n)


def test_student_requires_df_above_two():
    with pytest.raises(ValueError):
        StudentSimulator(MEAN, COV, df=2)


def test_student_df_from_kurtosis():
    assert student_df_from_kurtosis(6.0) == pytest.approx(5.0)
    assert student_df_from_kurtosis(-1.0) == 30.0
    assert student_df_from_kurtosis(100.0) == 4.5


def test_extreme_counts_on_artificial_dataset():
    # mean 0, sample std = sqrt(26/19) ~ 1.17: only the two +-3 values are beyond 2 std.
    values = np.array([3.0, -3.0] + [1.0, -1.0] * 4 + [0.0] * 10)
    counts = extreme_counts(pd.DataFrame({"A": values}), n_sigma=2.0)
    assert counts.loc["A", "count"] == 2
    assert counts.loc["A", "proportion"] == pytest.approx(0.1)


def test_compare_extremes_and_descriptive():
    sim = GaussianSimulator(MEAN, COV, random_state=0, columns=["A", "B"])
    datasets = {"Real": sim.sample_frame(500), "Gaussian": sim.sample_frame(500)}
    table = compare_extremes(datasets)
    assert list(table.columns) == [
        "Real count",
        "Real %",
        "Gaussian count",
        "Gaussian %",
        "Gaussian theory %",
    ]
    assert table["Gaussian theory %"].iloc[0] == pytest.approx(0.0455, abs=1e-4)
    assert list(compare_descriptive(datasets, "A").index) == ["Real", "Gaussian"]
    with pytest.raises(ValueError):
        compare_extremes({"Real": datasets["Real"], "Other": datasets["Real"].rename(columns={"A": "C"})})
