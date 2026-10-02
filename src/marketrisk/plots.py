"""Reusable Plotly figures (Parts 1-3)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#: Colour of each role in the figures (asset 1, asset 2, simulated data, ...).
COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626"]


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
                x=rebased.index, y=rebased[col], name=col, mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            )
        fig.update_layout(yaxis_title="Value (base 100)")
    else:
        fig = make_subplots(rows=1, cols=prices.shape[1], subplot_titles=list(prices.columns))
        for i, col in enumerate(prices.columns):
            fig.add_scatter(
                x=prices.index, y=prices[col], name=col, mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=1.5), row=1, col=i + 1,
            )
            fig.update_yaxes(title_text="Adjusted close", row=1, col=i + 1)
    fig.update_layout(hovermode="x unified", height=420, margin=dict(t=50, b=30))
    return fig
