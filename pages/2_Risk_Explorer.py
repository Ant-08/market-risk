"""Part 2 - Correlation, covariance and risk explorer."""

import streamlit as st

from marketrisk.plots import ellipse_parameters, scatter_figure
from marketrisk.statistics import (
    correlation_matrix,
    covariance_matrix,
    detect_outliers,
    gaussian_outside_probability,
    mean_vector,
)
from marketrisk.ui import get_market_data

st.set_page_config(page_title="Risk Explorer", layout="wide")
st.title("Risk Explorer")

data = get_market_data()
returns = data.returns
a, b = data.tickers

mu = mean_vector(returns)
sigma = covariance_matrix(returns)
corr = correlation_matrix(returns)

n_std = st.slider("Ellipse size (number of standard deviations)", 1.0, 3.0, 2.0, step=0.5)
flagged = detect_outliers(returns, n_std=n_std)

left, right = st.columns([3, 2])
with left:
    st.subheader("Joint daily returns")
    st.plotly_chart(
        scatter_figure(
            returns, labels=(a, b), mean=mu, covariance=sigma, n_std=n_std, outliers=flagged["outlier"]
        )
    )
with right:
    st.subheader("Mean vector")
    st.latex(
        r"\mu = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix} = \begin{pmatrix}"
        + rf"{mu.iloc[0]:.2e} \\ {mu.iloc[1]:.2e}"
        + r"\end{pmatrix}"
    )
    st.dataframe(mu.rename("mean").to_frame().style.format("{:.4%}"))

    st.subheader("Covariance matrix")
    st.latex(r"\Sigma = \begin{pmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{12} & \sigma_2^2 \end{pmatrix}")
    st.dataframe(sigma.style.format("{:.3e}"))

    st.subheader("Correlation matrix")
    st.dataframe(corr.style.format("{:.3f}"))

with st.expander("How the ellipse is built"):
    major, minor, angle = ellipse_parameters(sigma, n_std)
    st.markdown(
        f"""
The ellipse is the set of points at Mahalanobis distance **{n_std:g}** from the mean:
$(x-\\mu)^T \\Sigma^{{-1}} (x-\\mu) = {n_std:g}^2$.

- Its axes are the eigenvectors of $\\Sigma$; the major axis makes an angle of **{angle:.1f}°** with the {a} axis.
- Its half-axis lengths are ${n_std:g}\\sqrt{{\\lambda_i}}$:
  **{major:.2%}** (major) and **{minor:.2%}** (minor).
"""
    )

st.subheader("Possible outliers")
n_out = int(flagged["outlier"].sum())
expected = gaussian_outside_probability(n_std, dim=2)
c1, c2, c3 = st.columns(3)
c1.metric("Days outside the ellipse", f"{n_out} / {len(returns)}")
c2.metric("Observed proportion", f"{n_out / len(returns):.2%}")
c3.metric("Expected if Gaussian", f"{expected:.2%}")
st.caption(
    "An observation is flagged when its Mahalanobis distance to the mean exceeds the ellipse size. "
    f"For a bivariate Gaussian, the probability of lying outside the {n_std:g}-sd ellipse is "
    f"exp(-{n_std:g}²/2) = {expected:.2%}."
)
outliers = flagged[flagged["outlier"]].drop(columns="outlier").sort_values("mahalanobis", ascending=False)
outliers.index = outliers.index.strftime("%Y-%m-%d")
st.dataframe(
    outliers.style.format({a: "{:.2%}", b: "{:.2%}", "mahalanobis": "{:.2f}"}),
    height=min(400, 38 + 35 * len(outliers)),
)
