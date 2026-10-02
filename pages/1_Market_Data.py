"""Part 1 - Market data exploration: prices, log-returns and descriptive statistics."""

import streamlit as st

from marketrisk.plots import price_figure
from marketrisk.statistics import descriptive_statistics, extreme_returns
from marketrisk.ui import get_market_data

st.set_page_config(page_title="Market Data", layout="wide")
st.title("Market Data")

data = get_market_data()
prices, returns = data.prices, data.returns

st.caption(
    f"{len(prices)} trading days from {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y} "
    "- adjusted close prices from Yahoo Finance."
)

st.subheader("Historical prices")
normalize = st.toggle("Rebase both series to 100", value=False)
st.plotly_chart(price_figure(prices, normalize=normalize))

st.subheader("Daily log-returns")
st.latex(r"r_t = \log\left(\frac{P_t}{P_{t-1}}\right)")
st.dataframe(returns.head(10).set_axis(returns.index[:10].strftime("%Y-%m-%d")).style.format("{:.4%}"))

st.subheader("Descriptive statistics")
stats = descriptive_statistics(returns)
st.dataframe(
    stats.style.format(
        {
            "observations": "{:.0f}",
            "mean": "{:.4%}",
            "std": "{:.4%}",
            "min": "{:.2%}",
            "max": "{:.2%}",
            "skewness": "{:.2f}",
            "excess kurtosis": "{:.2f}",
        }
    )
)

st.subheader("Bottom 5 and top 5 daily returns")
for col, ticker in zip(st.columns(len(returns.columns)), returns.columns, strict=True):
    bottom, top = extreme_returns(returns[ticker], k=5)
    bottom.index, top.index = bottom.index.strftime("%Y-%m-%d"), top.index.strftime("%Y-%m-%d")
    with col:
        st.markdown(f"**{ticker}**")
        left, right = st.columns(2)
        left.markdown("Bottom 5")
        left.dataframe(bottom.rename("return").to_frame().style.format("{:.2%}"))
        right.markdown("Top 5")
        right.dataframe(top.rename("return").to_frame().style.format("{:.2%}"))
