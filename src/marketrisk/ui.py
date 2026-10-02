"""Streamlit helpers shared by all pages.

This module only contains interface code: the sidebar used to choose the
assets and the period, and a cached wrapper around :mod:`marketrisk.data`.
The selection is stored in ``st.session_state`` so that it is kept when the
user switches pages.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from marketrisk.data import ASSETS, compute_log_returns, download_prices


@dataclass(frozen=True)
class MarketData:
    """Prices and log-returns of the selected assets over the selected period."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    start: dt.date
    end: dt.date

    @property
    def tickers(self) -> list[str]:
        """Tickers of the selected assets, in selection order."""
        return list(self.prices.columns)


@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance...")
def load_prices(tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Cached call to :func:`marketrisk.data.download_prices` (``end`` inclusive)."""
    return download_prices(list(tickers), start, end + dt.timedelta(days=1))


def _widget_key(key: str) -> str:
    return f"_widget_{key}"


def _restore(key: str, default) -> None:
    """Copy the persistent value of ``key`` into its widget before rendering."""
    if key not in st.session_state:
        st.session_state[key] = default
    st.session_state[_widget_key(key)] = st.session_state[key]


def _store(key: str) -> None:
    """Callback copying a widget value into its persistent key."""
    st.session_state[key] = st.session_state[_widget_key(key)]


def _label(ticker: str) -> str:
    names = {v: k for k, v in ASSETS.items()}
    return f"{names[ticker]} ({ticker})" if ticker in names else ticker


def sidebar_selection() -> tuple[list[str], dt.date, dt.date]:
    """Render the sidebar used to pick two assets and a period.

    Returns
    -------
    tickers, start, end : tuple
        The two selected tickers and the selected period (both dates included).
    """
    today = dt.date.today()
    with st.sidebar:
        st.header("Selection")

        _restore("custom_ticker", "")
        st.text_input(
            "Other ticker (optional)", key=_widget_key("custom_ticker"),
            on_change=_store, args=("custom_ticker",),
            help="Any Yahoo Finance symbol, e.g. GOOGL, TSLA, ^FCHI, BTC-USD.",
        )
        custom = st.session_state["custom_ticker"].strip().upper()
        options = list(ASSETS.values()) + ([custom] if custom and custom not in ASSETS.values() else [])

        for key, default in (("asset_1", "AAPL"), ("asset_2", "^GSPC")):
            if st.session_state.get(key) not in options:
                st.session_state.pop(key, None)
            _restore(key, default)
            st.selectbox(
                f"Asset {key[-1]}", options, format_func=_label,
                key=_widget_key(key), on_change=_store, args=(key,),
            )

        _restore("start", today.replace(year=today.year - 5))
        st.date_input("Start date", key=_widget_key("start"), max_value=today,
                      on_change=_store, args=("start",))
        _restore("end", today)
        st.date_input("End date", key=_widget_key("end"), max_value=today,
                      on_change=_store, args=("end",))

    return [st.session_state["asset_1"], st.session_state["asset_2"]], \
        st.session_state["start"], st.session_state["end"]


def get_market_data() -> MarketData:
    """Render the sidebar, load the data and stop the page on invalid input.

    Returns
    -------
    MarketData
        Prices and returns of the two selected assets.
    """
    tickers, start, end = sidebar_selection()
    if tickers[0] == tickers[1]:
        st.error("Please select two different assets.")
        st.stop()
    if start >= end:
        st.error("The start date must be before the end date.")
        st.stop()
    try:
        prices = load_prices(tuple(tickers), start, end)
        returns = compute_log_returns(prices)
    except ValueError as exc:
        st.error(f"Could not load the data: {exc}")
        st.stop()
    if len(returns) < 30:
        st.warning(f"Only {len(returns)} returns over this period: statistics will be unreliable.")
    return MarketData(prices=prices, returns=returns, start=start, end=end)
