"""Entry point of the Market Risk Streamlit application.

Run with: uv run streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Market Risk Analysis", layout="wide")

st.title("Market Risk Analysis")
st.markdown(
    """
Explore the statistical behaviour of financial returns and compare them
with a Gaussian model having the same mean and covariance.

Use the sidebar to navigate:

1. **Market Data**: prices, log-returns and descriptive statistics.
2. **Risk Explorer**: covariance, correlation and covariance ellipse.
3. **Gaussian Simulator**: real returns vs. simulated Gaussian returns.
"""
)
