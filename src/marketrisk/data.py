"""Download prices from Yahoo Finance and compute log-returns (Part 1).

Prices and returns are kept in two separate DataFrames: one column per
ticker, indexed by date.
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Predefined assets offered in the application (label -> Yahoo ticker).
ASSETS: dict[str, str] = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Amazon": "AMZN",
    "S&P 500": "^GSPC",
}


def validate_prices(prices: pd.DataFrame) -> None:
    """Check that ``prices`` has the structure expected by the package.

    Parameters
    ----------
    prices : pd.DataFrame
        Price table with one column per asset and a ``DatetimeIndex``.

    Raises
    ------
    TypeError
        If ``prices`` is not a DataFrame or its index is not a DatetimeIndex.
    ValueError
        If the table is empty, has fewer than two rows, contains
        non-numeric columns, or contains non-positive prices.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(f"prices must be a pandas DataFrame, got {type(prices).__name__}.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by dates (pd.DatetimeIndex).")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("prices is empty.")
    if len(prices) < 2:
        raise ValueError("At least two price observations are needed to compute a return.")
    non_numeric = [c for c in prices.columns if not pd.api.types.is_numeric_dtype(prices[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric price columns: {non_numeric}.")
    if (prices.dropna() <= 0).any().any():
        raise ValueError("Prices must be strictly positive to compute log-returns.")


def clean_prices(prices: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Handle missing values in a price table.

    Short gaps (at most ``max_gap`` consecutive missing days, e.g. a holiday on
    one exchange only) are forward-filled; remaining rows with missing values
    (typically before an asset starts trading) are dropped so that all assets
    share the same dates.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price table.
    max_gap : int, default 5
        Maximum number of consecutive missing values filled forward.

    Returns
    -------
    pd.DataFrame
        Price table without missing values, sorted by date.
    """
    cleaned = prices.sort_index().ffill(limit=max_gap).dropna(how="any")
    n_dropped = len(prices) - len(cleaned)
    if n_dropped > 0:
        warnings.warn(f"{n_dropped} dates with missing prices were dropped.", stacklevel=2)
    return cleaned


def download_prices(
    tickers: Sequence[str],
    start: dt.date | str,
    end: dt.date | str,
) -> pd.DataFrame:
    """Download adjusted daily closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers : sequence of str
        Yahoo Finance tickers, e.g. ``["AAPL", "^GSPC"]``.
    start, end : datetime.date or str
        Period of interest (``end`` is exclusive, as in yfinance).

    Returns
    -------
    pd.DataFrame
        Cleaned adjusted close prices, one column per ticker, in the order of
        ``tickers``.

    Raises
    ------
    ValueError
        If no ticker is given, the period is invalid, or Yahoo Finance returns
        no data for one of the tickers.
    """
    import yfinance as yf  # imported here so the rest of the package works offline

    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    if not tickers:
        raise ValueError("At least one ticker is required.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("The start date must be before the end date.")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned by Yahoo Finance for {tickers}.")

    close = raw["Close"]
    if isinstance(close, pd.Series):  # single ticker with flat columns
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise ValueError(f"No data returned by Yahoo Finance for {missing}.")

    prices = close[tickers].astype(float)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices.index.name = "Date"
    prices.columns.name = None
    prices = clean_prices(prices)
    validate_prices(prices)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns ``r_t = log(P_t / P_{t-1})``.

    Parameters
    ----------
    prices : pd.DataFrame
        Price table without missing values (see :func:`clean_prices`).

    Returns
    -------
    pd.DataFrame
        Log-returns with the same columns and one fewer row than ``prices``
        (the first date has no previous price).

    Raises
    ------
    ValueError
        If ``prices`` is invalid or still contains missing values.
    """
    validate_prices(prices)
    if prices.isna().any().any():
        raise ValueError("prices contains missing values; call clean_prices first.")
    returns = np.log(prices / prices.shift(1)).iloc[1:]
    return returns
