"""Home page of the Market Risk Streamlit application.

Run with: uv run streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Market Risk Analysis", layout="wide")

st.title("Market Risk Analysis")
st.markdown(
    """
This application explores the statistical behaviour of daily financial returns
downloaded from Yahoo Finance, and compares them with a **Gaussian model having
the same mean vector and covariance matrix**.

The Gaussian model is a *reference*: the goal is to see what it reproduces and
what it misses, not to assume that returns are Gaussian, nor to forecast prices.
"""
)

c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown("#### 1. Market Data")
    st.markdown("Historical prices, daily log-returns and descriptive statistics.")
    st.page_link("pages/1_Market_Data.py", label="Open", icon=":material/show_chart:")
with c2, st.container(border=True):
    st.markdown("#### 2. Risk Explorer")
    st.markdown("Covariance, correlation, covariance ellipse and outliers.")
    st.page_link("pages/2_Risk_Explorer.py", label="Open", icon=":material/scatter_plot:")
with c3, st.container(border=True):
    st.markdown("#### 3. Gaussian Simulator")
    st.markdown("Real returns vs simulated N(μ, Σ) returns: where do they differ?")
    st.page_link("pages/3_Gaussian_Simulator.py", label="Open", icon=":material/casino:")

st.info(
    "Choose the two assets and the period in the sidebar of each analysis page; the selection is shared by all pages."
)
