"""Reusable Plotly figures (Parts 1-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketrisk.statistics import validate_covariance

#: Colour of each role in the figures (asset 1, asset 2, simulated data, ...).
COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626"]
#: One colour per dataset in comparisons (real, Gaussian, Student-t).
DATASET_COLORS = ["#2563eb", "#16a34a", "#9333ea"]
#: Colours readable on both light and dark Streamlit themes.
ELLIPSE_COLOR = "#f59e0b"
OUTLIER_COLOR = "#dc2626"
DENSITY_COLOR = "#94a3b8"


def price_figure(prices: pd.DataFrame, normalize: bool = False) -> go.Figure:
    """Plot the historical prices of each asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Price table, one column per asset.
    normalize : bool, default False
        If True, all series start at 100 on a single chart so that assets with
        very different price levels can be compared. Otherwise each asset is
        drawn on its own panel with its own scale.

    Returns
    -------
    go.Figure
    """
    if normalize:
        rebased = 100 * prices / prices.iloc[0]
        fig = go.Figure()
        for i, col in enumerate(rebased.columns):
            fig.add_scatter(
                x=rebased.index,
                y=rebased[col],
                name=col,
                mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            )
        fig.update_layout(yaxis_title="Value (base 100)")
    else:
        fig = make_subplots(rows=1, cols=prices.shape[1], subplot_titles=list(prices.columns))
        for i, col in enumerate(prices.columns):
            fig.add_scatter(
                x=prices.index,
                y=prices[col],
                name=col,
                mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                row=1,
                col=i + 1,
            )
            fig.update_yaxes(title_text="Adjusted close", row=1, col=i + 1)
    fig.update_layout(hovermode="x unified", height=420, margin=dict(t=50, b=30))
    return fig


def covariance_ellipse(mean, covariance, n_std: float = 2.0, n_points: int = 200) -> np.ndarray:
    """Points of the covariance ellipse of a 2-dimensional distribution.

    The ellipse is the set of points at Mahalanobis distance ``n_std`` from
    ``mean``. With the eigendecomposition ``Sigma = V diag(lambda) V^T``, its
    axes are the eigenvectors ``V`` and its half-axis lengths are
    ``n_std * sqrt(lambda)``, so the points are
    ``mean + n_std * V diag(sqrt(lambda)) (cos t, sin t)^T``.

    Parameters
    ----------
    mean : array-like of shape (2,)
        Centre of the ellipse.
    covariance : array-like of shape (2, 2)
        Symmetric positive semi-definite covariance matrix.
    n_std : float, default 2.0
        Size of the ellipse, in standard deviations.
    n_points : int, default 200
        Number of points used to draw the (closed) ellipse.

    Returns
    -------
    np.ndarray of shape (n_points, 2)
        Coordinates of the ellipse; the first and last points coincide.
    """
    mu = np.asarray(mean, dtype=float)
    if mu.shape != (2,):
        raise ValueError(f"mean must have shape (2,), got {mu.shape}.")
    cov = validate_covariance(covariance, dim=2)
    if n_std <= 0:
        raise ValueError("n_std must be positive.")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # remove tiny negative rounding errors
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.vstack([np.cos(t), np.sin(t)])
    transform = n_std * eigenvectors @ np.diag(np.sqrt(eigenvalues))
    return mu + (transform @ circle).T


def ellipse_parameters(covariance, n_std: float = 2.0) -> tuple[float, float, float]:
    """Geometry of the ``n_std`` covariance ellipse of a 2x2 covariance matrix.

    Returns
    -------
    major, minor, angle : tuple of float
        Half-lengths of the major and minor axes (``n_std * sqrt(lambda)``) and
        angle in degrees between the major axis and the horizontal axis,
        in ``(-90, 90]``.
    """
    cov = validate_covariance(covariance, dim=2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending eigenvalues
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    vx, vy = eigenvectors[:, -1]
    angle = float(np.degrees(np.arctan2(vy, vx)))
    if angle <= -90:
        angle += 180
    elif angle > 90:
        angle -= 180
    return float(n_std * np.sqrt(eigenvalues[-1])), float(n_std * np.sqrt(eigenvalues[0])), angle


def add_ellipse(
    fig: go.Figure,
    mean,
    covariance,
    n_std: float,
    name: str | None = None,
    color: str = ELLIPSE_COLOR,
    dash: str = "solid",
    **trace_kwargs,
) -> go.Figure:
    """Add a covariance ellipse (and its centre) to an existing figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to update in place.
    mean, covariance, n_std
        See :func:`covariance_ellipse`.
    name : str, optional
        Legend label (default: ``"{n_std} sd ellipse"``).
    color, dash : str
        Line style.
    **trace_kwargs
        Extra arguments for ``fig.add_scatter`` (e.g. ``row`` and ``col``).

    Returns
    -------
    go.Figure
        The same figure, for chaining.
    """
    mean = np.asarray(mean, dtype=float)
    points = covariance_ellipse(mean, covariance, n_std)
    fig.add_scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode="lines",
        name=name or f"{n_std:g} sd ellipse",
        line=dict(color=color, width=2, dash=dash),
        hoverinfo="skip",
        **trace_kwargs,
    )
    fig.add_scatter(
        x=[mean[0]],
        y=[mean[1]],
        mode="markers",
        name="mean",
        showlegend=False,
        marker=dict(color=color, symbol="x", size=10),
        **trace_kwargs,
    )
    return fig


def scatter_figure(
    data,
    labels: tuple[str, str],
    mean=None,
    covariance=None,
    n_std: float = 2.0,
    outliers=None,
    color: str = COLORS[0],
    name: str = "daily returns",
    title: str | None = None,
) -> go.Figure:
    """Scatter plot of two return series, with an optional covariance ellipse.

    Parameters
    ----------
    data : array-like of shape (n, 2)
        Joint returns (DataFrame or NumPy array). If a DataFrame, its index is
        shown on hover.
    labels : tuple of str
        Axis titles (names of the two assets).
    mean, covariance : array-like, optional
        If both are given, the ``n_std`` covariance ellipse is drawn.
    n_std : float, default 2.0
        Size of the ellipse.
    outliers : array-like of bool, optional
        Mask of observations to highlight.
    color, name, title : str
        Style of the main markers and figure title.

    Returns
    -------
    go.Figure
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"data must have shape (n, 2), got {values.shape}.")
    hover = [f"{d:%Y-%m-%d}" for d in data.index] if isinstance(data, pd.DataFrame) else None
    mask = np.zeros(len(values), dtype=bool) if outliers is None else np.asarray(outliers, dtype=bool)

    fig = go.Figure()
    fig.add_scatter(
        x=values[~mask, 0],
        y=values[~mask, 1],
        mode="markers",
        name=name,
        marker=dict(color=color, size=5, opacity=0.5),
        text=None if hover is None else np.asarray(hover)[~mask],
    )
    if mask.any():
        fig.add_scatter(
            x=values[mask, 0],
            y=values[mask, 1],
            mode="markers",
            name="outside ellipse",
            marker=dict(color=OUTLIER_COLOR, size=7, symbol="diamond"),
            text=None if hover is None else np.asarray(hover)[mask],
        )
    if mean is not None and covariance is not None:
        add_ellipse(fig, mean, covariance, n_std)
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(t=50 if title else 20, b=30),
        xaxis=dict(title=labels[0], tickformat=".1%", zeroline=True),
        yaxis=dict(title=labels[1], tickformat=".1%", zeroline=True),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def histogram_figure(
    samples: dict[str, pd.Series],
    mean: float | None = None,
    std: float | None = None,
    title: str | None = None,
    log_y: bool = False,
    bins: int = 80,
) -> go.Figure:
    """Overlaid density histograms of one asset's returns in several datasets.

    Parameters
    ----------
    samples : dict of str to pd.Series
        For example ``{"Real": real[a], "Gaussian": simulated[a]}``.
    mean, std : float, optional
        If both are given, the density of ``N(mean, std^2)`` is drawn on top.
    title : str, optional
        Figure title.
    log_y : bool, default False
        Use a logarithmic density axis, which makes the tails visible.
    bins : int, default 80
        Number of bins, shared by all datasets.

    Returns
    -------
    go.Figure
    """
    values = np.concatenate([np.asarray(s, dtype=float) for s in samples.values()])
    edges = np.linspace(values.min(), values.max(), bins + 1)
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    fig = go.Figure()
    positive = []
    for i, (name, series) in enumerate(samples.items()):
        density, _ = np.histogram(np.asarray(series, dtype=float), bins=edges, density=True)
        positive.append(density[density > 0])
        fig.add_bar(
            x=centers,
            y=np.where(density > 0, density, np.nan),  # empty bins are not drawn (log scale)
            width=width,
            name=name,
            marker_color=DATASET_COLORS[i % len(DATASET_COLORS)],
            opacity=0.5,
        )
    densities = np.concatenate(positive)
    if mean is not None and std is not None:
        x = np.linspace(edges[0], edges[-1], 400)
        normal = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        fig.add_scatter(
            x=x, y=normal, mode="lines", name="N(μ, σ²) density", line=dict(color=DENSITY_COLOR, width=2)
        )
    fig.update_layout(
        barmode="overlay",
        bargap=0,
        title=title,
        height=380,
        margin=dict(t=50 if title else 20, b=30),
        xaxis=dict(title="daily log-return", tickformat=".0%"),
        yaxis_title="density",
        legend=dict(orientation="h", y=-0.2),
    )
    if log_y:
        # Range from the smallest non-empty bin to the highest bar, so the Gaussian curve
        # does not stretch the axis down to its (astronomically small) tail values.
        low, high = np.log10(densities.min()) - 0.3, np.log10(densities.max()) + 0.3
        fig.update_yaxes(type="log", range=[low, high], dtick=1, exponentformat="power")
    return fig


def axis_ranges(*datasets, margin: float = 0.05) -> tuple[list[float], list[float]]:
    """Common x and y ranges covering several 2-column datasets.

    Used to draw real and simulated scatter plots on identical axes.
    """
    stacked = np.vstack([np.asarray(d, dtype=float) for d in datasets])
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    pad = margin * (high - low)
    return [low[0] - pad[0], high[0] + pad[0]], [low[1] - pad[1], high[1] + pad[1]]


def returns_time_figure(samples: dict[str, pd.Series], title: str | None = None) -> go.Figure:
    """Daily returns over time, one panel per dataset, on a shared y axis.

    Comparing real and simulated series this way shows volatility clustering:
    real returns alternate calm and turbulent periods, i.i.d. simulated ones do not.
    """
    fig = make_subplots(
        rows=len(samples),
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=list(samples),
        vertical_spacing=0.08,
    )
    for i, (name, series) in enumerate(samples.items()):
        fig.add_scatter(
            x=series.index,
            y=series.to_numpy(),
            mode="lines",
            name=name,
            line=dict(color=DATASET_COLORS[i % len(DATASET_COLORS)], width=0.8),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(tickformat=".0%", row=i + 1, col=1)
    fig.update_layout(
        title=title,
        height=200 + 180 * len(samples),
        showlegend=False,
        margin=dict(t=60 if title else 40, b=30),
    )
    return fig
