"""Tests for marketrisk.data (no network access: artificial prices only)."""

import numpy as np
import pandas as pd
import pytest

from marketrisk.data import clean_prices, compute_log_returns, validate_prices


@pytest.fixture
def prices() -> pd.DataFrame:
    """Small artificial price table with known returns."""
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    return pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0], "B": [50.0, 25.0, 50.0, 100.0]}, index=index)


def test_returns_have_one_fewer_observation(prices):
    returns = compute_log_returns(prices)
    assert len(returns) == len(prices) - 1
    assert list(returns.columns) == list(prices.columns)
    assert returns.index[0] == prices.index[1]


def test_log_returns_values(prices):
    returns = compute_log_returns(prices)
    np.testing.assert_allclose(returns["A"], [np.log(1.1), np.log(0.9), 0.0])
    np.testing.assert_allclose(returns["B"], [-np.log(2), np.log(2), np.log(2)])


def test_clean_prices_fills_short_gaps_and_drops_leading_nans():
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    raw = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0], "B": [np.nan, 1.0, 1.0, 1.0, 1.0]}, index=index)
    with pytest.warns(UserWarning):
        cleaned = clean_prices(raw)
    assert not cleaned.isna().any().any()
    assert cleaned.index[0] == index[1]
    assert cleaned.loc[index[2], "A"] == 2.0


def test_validate_prices_rejects_bad_inputs(prices):
    with pytest.raises(TypeError):
        validate_prices(prices.to_numpy())
    with pytest.raises(TypeError):
        validate_prices(prices.reset_index(drop=True))
    with pytest.raises(ValueError):
        validate_prices(prices.iloc[:1])
    with pytest.raises(ValueError):
        validate_prices(prices.assign(A=-prices["A"]))


def test_compute_log_returns_rejects_missing_values(prices):
    prices.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        compute_log_returns(prices)
