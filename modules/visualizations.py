"""Shared Plotly visualisation helpers."""

from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import BRAND_COLORS


_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15)), **_LAYOUT_DEFAULTS)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


# ── Trend line ──────────────────────────────────────────────────────────────

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    title: str = "",
    color: Optional[str] = None,
    hue: Optional[str] = None,
) -> go.Figure:
    """Generic line chart, optionally coloured by a dimension."""
    if isinstance(y, str):
        y = [y]

    if hue and hue in df.columns:
        fig = px.line(df, x=x, y=y[0], color=hue, title=title)
    elif len(y) == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y[0]],
            mode="lines",
            line=dict(color=color or BRAND_COLORS["secondary"], width=2),
            name=y[0],
        ))
    else:
        fig = go.Figure()
        palette = [
            BRAND_COLORS["secondary"], BRAND_COLORS["accent"],
            BRAND_COLORS["success"], BRAND_COLORS["danger"],
        ]
        for i, col in enumerate(y):
            fig.add_trace(go.Scatter(
                x=df[x], y=df[col],
                mode="lines",
                line=dict(width=2, color=palette[i % len(palette)]),
                name=col,
            ))

    return _apply_defaults(fig, title)


def area_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y],
        mode="lines",
        fill="tozeroy",
        line=dict(color=color or BRAND_COLORS["secondary"], width=2),
        fillcolor=f"rgba(46,134,193,0.2)",
        name=y,
    ))
    return _apply_defaults(fig, title)


# ── Bar ─────────────────────────────────────────────────────────────────────

def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    orientation: str = "v",
    color_col: Optional[str] = None,
) -> go.Figure:
    if color_col:
        fig = px.bar(df, x=x, y=y, color=color_col, title=title,
                     orientation=orientation)
    else:
        fig = go.Figure(go.Bar(
            x=df[x] if orientation == "v" else df[y],
            y=df[y] if orientation == "v" else df[x],
            marker_color=color or BRAND_COLORS["secondary"],
            name=y,
            orientation=orientation,
        ))
    return _apply_defaults(fig, title)


# ── Heatmap ─────────────────────────────────────────────────────────────────

def occupancy_heatmap(
    df: pd.DataFrame,
    x_col: str = "day_name",
    y_col: str = "month_name",
    z_col: str = "occupancy_pct",
    title: str = "Occupancy Heatmap",
) -> go.Figure:
    if df.empty or z_col not in df.columns:
        return go.Figure()

    pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc="mean")

    # Enforce calendar ordering
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [c for c in day_order if c in pivot.columns]
    rows = [r for r in month_order if r in pivot.index]
    if cols:
        pivot = pivot[cols]
    if rows:
        pivot = pivot.reindex(rows)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="Blues",
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Occ %"),
    ))
    return _apply_defaults(fig, title)


def revenue_heatmap(df: pd.DataFrame, title: str = "Revenue Heatmap") -> go.Figure:
    return occupancy_heatmap(df, z_col="revenue", title=title)


# ── Waterfall ───────────────────────────────────────────────────────────────

def waterfall_chart(
    labels: list[str],
    values: list[float],
    title: str = "Waterfall",
) -> go.Figure:
    measure = ["relative"] * len(values)
    measure[-1] = "total"
    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measure,
        connector=dict(line=dict(color="rgba(255,255,255,0.2)")),
        increasing=dict(marker=dict(color=BRAND_COLORS["success"])),
        decreasing=dict(marker=dict(color=BRAND_COLORS["danger"])),
        totals=dict(marker=dict(color=BRAND_COLORS["accent"])),
    ))
    return _apply_defaults(fig, title)


# ── Scatter ─────────────────────────────────────────────────────────────────

def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
) -> go.Figure:
    kwargs = dict(x=x, y=y, title=title)
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
    if size_col and size_col in df.columns:
        kwargs["size"] = size_col
    fig = px.scatter(df, **kwargs)
    return _apply_defaults(fig, title)


# ── Forecast ────────────────────────────────────────────────────────────────

def forecast_chart(
    actuals: pd.DataFrame,
    forecast_df: pd.DataFrame,
    metric: str,
    title: str = "Forecast",
) -> go.Figure:
    fig = go.Figure()

    if not actuals.empty and metric in actuals.columns:
        fig.add_trace(go.Scatter(
            x=actuals["date"], y=actuals[metric],
            mode="lines",
            name="Actual",
            line=dict(color=BRAND_COLORS["secondary"], width=2),
        ))

    if forecast_df is not None and not forecast_df.empty:
        fcast_only = forecast_df[forecast_df.get("is_forecast", True)] if "is_forecast" in forecast_df.columns else forecast_df

        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast"],
            mode="lines",
            name="Forecast",
            line=dict(color=BRAND_COLORS["accent"], width=2, dash="dash"),
        ))

        if "upper_80" in forecast_df.columns and "lower_80" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
                y=list(forecast_df["upper_80"]) + list(forecast_df["lower_80"][::-1]),
                fill="toself",
                fillcolor="rgba(243,156,18,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="80% CI",
                showlegend=True,
            ))

    return _apply_defaults(fig, title)


# ── KPI gauge ───────────────────────────────────────────────────────────────

def gauge_chart(value: float, title: str, suffix: str = "", max_val: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=suffix, font=dict(size=28)),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="white"),
            bar=dict(color=BRAND_COLORS["secondary"]),
            bgcolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[0, max_val * 0.5], color="rgba(231,76,60,0.2)"),
                dict(range=[max_val * 0.5, max_val * 0.75], color="rgba(243,156,18,0.2)"),
                dict(range=[max_val * 0.75, max_val], color="rgba(39,174,96,0.2)"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=value,
            ),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E0E0"), height=220)
    return fig


# ── Booking window histogram ─────────────────────────────────────────────────

def booking_window_hist(df: pd.DataFrame, title: str = "Booking Window Distribution") -> go.Figure:
    if df.empty or "lead_days" not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df, x="lead_days", nbins=50,
        color_discrete_sequence=[BRAND_COLORS["secondary"]],
        title=title, labels={"lead_days": "Lead Days"},
    )
    return _apply_defaults(fig, title)


# ── Pace curve ──────────────────────────────────────────────────────────────

def pace_curve(pace_df: pd.DataFrame, metric: str = "rooms_sold", title: str = "Pace Curve") -> go.Figure:
    if pace_df.empty:
        return go.Figure()

    fig = go.Figure()
    col_map = {
        "current": dict(color=BRAND_COLORS["secondary"], name="Current"),
        "benchmark": dict(color=BRAND_COLORS["accent"], name="Benchmark", dash="dash"),
    }
    for col, style in col_map.items():
        if col in pace_df.columns:
            fig.add_trace(go.Scatter(
                x=pace_df["date"] if "date" in pace_df.columns else pace_df.index,
                y=pace_df[col],
                mode="lines",
                name=style["name"],
                line=dict(color=style["color"], width=2, dash=style.get("dash", "solid")),
            ))

    return _apply_defaults(fig, title)
