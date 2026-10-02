"""Part 3 - Gaussian simulation: real market returns vs a Gaussian model."""

import streamlit as st

from marketrisk.plots import (
    DATASET_COLORS,
    axis_ranges,
    histogram_figure,
    returns_time_figure,
    scatter_figure,
)
from marketrisk.simulation import GaussianSimulator, StudentSimulator, student_df_from_kurtosis
from marketrisk.statistics import (
    compare_descriptive,
    compare_extremes,
    correlation_matrix,
    covariance_matrix,
    descriptive_statistics,
    detect_outliers,
    gaussian_outside_probability,
    mean_vector,
    tail_summary,
)
from marketrisk.ui import get_market_data

st.set_page_config(page_title="Gaussian Simulator", layout="wide")
st.title("Real Market vs Gaussian Model")

data = get_market_data()
real = data.returns
a, b = data.tickers

st.markdown(
    "The Gaussian model below has **the same mean vector and covariance matrix** as the real "
    f"returns of {a} and {b}, and the **same number of observations** ({len(real)}). "
    "It is a reference model, not an assumption: the goal is to see what it fails to reproduce."
)

# --- Controls ---------------------------------------------------------------
real_stats = descriptive_statistics(real)
c1, c2, c3 = st.columns(3)
seed = c1.number_input("Random seed", min_value=0, value=42, step=1)
n_std = c2.slider("Ellipse size (standard deviations)", 1.0, 3.0, 2.0, step=0.5)
with c3:
    use_student = st.toggle("Also compare with a Student-t model (extension)")
    default_df = student_df_from_kurtosis(real_stats["excess kurtosis"].mean())
    df = st.slider(
        "Student degrees of freedom",
        2.5,
        30.0,
        round(default_df * 2) / 2,
        step=0.5,
        disabled=not use_student,
        help="Default chosen so that the Student kurtosis 6/(ν-4) matches the average real kurtosis.",
    )

# --- Simulation ---------------------------------------------------------------
gaussian = GaussianSimulator.from_returns(real, random_state=int(seed))
datasets = {"Real": real, "Gaussian": gaussian.sample_frame(len(real), index=real.index)}
if use_student:
    student = StudentSimulator.from_returns(real, random_state=int(seed), df=df)
    datasets["Student-t"] = student.sample_frame(len(real), index=real.index)

with st.expander("Estimated parameters μ and Σ (used by every model)"):
    p1, p2 = st.columns(2)
    p1.dataframe(mean_vector(real).rename("μ").to_frame().style.format("{:.4%}"))
    p2.dataframe(covariance_matrix(real).style.format("{:.3e}"))

# --- Scatter plots with ellipses ---------------------------------------------
st.subheader("Scatter plots and covariance ellipses")
st.caption(
    f"Same axes for every panel. The orange ellipse is the {n_std:g}-sd ellipse of the "
    "real data's μ and Σ; red diamonds lie outside it."
)
x_range, y_range = axis_ranges(*datasets.values())
for col, (i, (name, sample)) in zip(st.columns(len(datasets)), enumerate(datasets.items()), strict=True):
    flagged = detect_outliers(sample, n_std=n_std)
    fig = scatter_figure(
        sample,
        labels=(a, b),
        mean=gaussian.mean,
        covariance=gaussian.covariance,
        n_std=n_std,
        outliers=flagged["outlier"],
        color=DATASET_COLORS[i],
        name=name,
        title=f"{name}: {flagged['outlier'].mean():.1%} outside",
    )
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    col.plotly_chart(fig, key=f"scatter_{name}")

# --- Histograms ---------------------------------------------------------------
st.subheader("Distribution of each asset's returns")
log_y = st.toggle("Logarithmic density axis (shows the tails)", value=True)
for col, ticker in zip(st.columns(2), (a, b), strict=True):
    col.plotly_chart(
        histogram_figure(
            {name: sample[ticker] for name, sample in datasets.items()},
            mean=real_stats.loc[ticker, "mean"],
            std=real_stats.loc[ticker, "std"],
            title=ticker,
            log_y=log_y,
        ),
        key=f"hist_{ticker}",
    )

# --- Extreme observations ------------------------------------------------------
st.subheader("Extreme observations: |r − μ| > 2σ")
extremes = compare_extremes(datasets, n_sigma=2.0)
st.dataframe(
    extremes.style.format(
        {c: "{:.2%}" for c in extremes.columns if c.endswith("%")}
        | {c: "{:.0f}" for c in extremes.columns if c.endswith("count")}
    )
)
expected3 = gaussian_outside_probability(3.0, dim=1) * len(real)
tails = tail_summary(datasets)
st.markdown("**Further in the tails**")
st.dataframe(
    tails.style.format("{:.0f}").format("{:.1f}", subset=(tails.index.str.contains("move"), slice(None)))
)
st.caption(f"Gaussian theory: {expected3:.1f} days beyond 3σ per asset out of {len(real)}.")

st.subheader("Returns over time")
st.plotly_chart(
    returns_time_figure(
        {name: sample[a] for name, sample in datasets.items()}, title=f"{a} daily log-returns"
    ),
    key="time",
)

st.subheader("Summary statistics: real vs simulated")
for ticker in (a, b):
    st.markdown(f"**{ticker}**")
    table = compare_descriptive(datasets, ticker)
    st.dataframe(
        table.style.format(
            {
                "mean": "{:.4%}",
                "std": "{:.3%}",
                "min": "{:.2%}",
                "max": "{:.2%}",
                "skewness": "{:.2f}",
                "excess kurtosis": "{:.2f}",
            }
        )
    )

# --- Interpretation ------------------------------------------------------------
st.subheader("Interpretation")
kurt = real_stats["excess kurtosis"]
real_pct = extremes["Real %"]
gauss_pct = extremes["Gaussian %"]
skew = real_stats["skewness"]
joint_real, joint_gauss = tails.loc["days with all assets beyond 2σ", ["Real", "Gaussian"]]
joint_sentence = (
    f"Extreme days are also more often **shared by both assets**: {joint_real:.0f} days with both beyond 2σ "
    f"in the real data, against {joint_gauss:.0f} in the simulation."
    if joint_real > joint_gauss
    else f"Days on which both assets are beyond 2σ are not more frequent in the real data ({joint_real:.0f}) "
    f"than in the simulation ({joint_gauss:.0f}) for this pair."
)
joint_bullet = (
    "- the **dependence in the tails**: joint extreme days are more frequent than the correlation alone implies;\n"
    if joint_real > joint_gauss
    else ""
)
student_note = (
    f"With the Student-t extension (ν = {df:g}), same μ and Σ but heavier tails, the days beyond 3σ become "
    f"{tails.loc[f'{a} days beyond 3σ', 'Student-t']:.0f} ({a}) and {tails.loc[f'{b} days beyond 3σ', 'Student-t']:.0f} ({b}), "
    f"against {tails.loc[f'{a} days beyond 3σ', 'Real']:.0f} and {tails.loc[f'{b} days beyond 3σ', 'Real']:.0f} in the real data."
    if use_student
    else "Switch on the Student-t extension above to compare with a model having the same μ and Σ but heavier tails."
)
st.markdown(
    f"""
**Do the simulated data visually resemble the real returns?** Broadly, yes: by construction the
Gaussian cloud has the same centre, the same spread and the same orientation as the real one
(correlation {correlation_matrix(real).iloc[0, 1]:.2f}), so the two ellipses coincide and the bulk of
both clouds looks alike. The differences are in the shape: the real cloud is **more concentrated near
the centre** and has **isolated points far from it**, while the Gaussian cloud is a smooth elliptical
cloud without such points. On the histograms (log scale), real returns have a sharper peak and much
fatter tails than the Gaussian density.

**Are extreme observations equally frequent?** At the 2σ threshold the frequencies are similar:
{real_pct[a]:.1%} ({a}) and {real_pct[b]:.1%} ({b}) of real days, against {gauss_pct[a]:.1%} and
{gauss_pct[b]:.1%} in the simulation (theory 4.55%). The difference is further in the tails: beyond 3σ,
the real data have **{tails.loc[f"{a} days beyond 3σ", "Real"]:.0f} and {tails.loc[f"{b} days beyond 3σ", "Real"]:.0f} days**
against about **{expected3:.1f}** expected under the Gaussian model, and the
largest real moves reach **{tails.loc[f"{a} largest move (σ)", "Real"]:.1f}σ and {tails.loc[f"{b} largest move (σ)", "Real"]:.1f}σ**, against
{tails.loc[f"{a} largest move (σ)", "Gaussian"]:.1f}σ and {tails.loc[f"{b} largest move (σ)", "Gaussian"]:.1f}σ in the simulation. {joint_sentence}

**Does the Gaussian model reproduce all aspects of the real data?** No. It reproduces the mean, the
variances and the linear correlation, which are its only parameters, but not:
- the **fat tails**: excess kurtosis {kurt[a]:.1f} ({a}) and {kurt[b]:.1f} ({b}), against 0 for a Gaussian;
{joint_bullet}- the **volatility clustering** shown in the time series above: calm and turbulent periods alternate in
  real returns, whereas simulated days are independent and identically distributed;
- the **asymmetry**, which is small here (skewness {skew[a]:.2f} and {skew[b]:.2f}, against 0 for a Gaussian).

A risk measure based on the Gaussian model (e.g. a 3σ loss threshold) therefore **underestimates the
probability of extreme losses**. {student_note}
"""
)
