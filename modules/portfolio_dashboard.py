"""Portfolio-level dashboard: aggregate view across all hotels."""

import logging

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from modules import kpi_engine, benchmarking, visualizations, forecasting, anomaly_detection
from config.settings import BRAND_COLORS

logger = logging.getLogger(__name__)


def _metric_card(label: str, value: str, col_obj) -> None:
    col_obj.markdown(
        f"""<div style='background:linear-gradient(135deg,#1A2744,#0D1B2A);
            border-radius:10px;padding:18px 22px;text-align:center;
            border:1px solid rgba(255,255,255,0.08);'>
          <p style='margin:0;font-size:11px;color:#94A3B8;letter-spacing:1px;text-transform:uppercase'>{label}</p>
          <p style='margin:8px 0 0;font-size:26px;font-weight:700;color:#F1F5F9'>{value}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def _portfolio_kpi_row(kpis: dict) -> None:
    cols = st.columns(6)
    items = [
        ("Hotels", f"{int(kpis.get('hotel_count', 0))}"),
        ("Occupancy", f"{kpis.get('occupancy_pct', 0):.1f}%"),
        ("ADR", f"${kpis.get('adr', 0):,.2f}"),
        ("RevPAR", f"${kpis.get('revpar', 0):,.2f}"),
        ("Revenue", f"${kpis.get('revenue', 0):,.0f}"),
        ("Rooms Sold", f"{int(kpis.get('rooms_sold', 0)):,}"),
    ]
    for col, (label, val) in zip(cols, items):
        _metric_card(label, val, col)


# ── Tab: Portfolio Overview ───────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame) -> None:
    kpis = benchmarking.portfolio_summary(df)
    _portfolio_kpi_row(kpis)

    st.markdown("---")

    # Portfolio daily trends
    daily = kpi_engine.aggregate_daily(df, hotel_col="hotel_name")
    daily_total = (
        df.groupby("date")[["rooms_sold", "rooms_available", "revenue"]]
        .sum(min_count=1).reset_index()
    )
    if "rooms_sold" in daily_total.columns and "rooms_available" in daily_total.columns:
        daily_total["occupancy_pct"] = (
            daily_total["rooms_sold"] / daily_total["rooms_available"].replace(0, np.nan) * 100
        )
    if "revenue" in daily_total.columns and "rooms_available" in daily_total.columns:
        daily_total["revpar"] = (
            daily_total["revenue"] / daily_total["rooms_available"].replace(0, np.nan)
        )

    c1, c2 = st.columns(2)
    with c1:
        if "occupancy_pct" in daily_total.columns:
            fig = visualizations.area_chart(daily_total, "date", "occupancy_pct",
                                            title="Portfolio Occupancy %")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "revenue" in daily_total.columns:
            fig = visualizations.area_chart(daily_total, "date", "revenue",
                                            title="Portfolio Revenue",
                                            color=BRAND_COLORS["success"])
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "adr" in daily.columns and "hotel_name" in daily.columns:
            fig = visualizations.line_chart(daily, "date", "adr",
                                            hue="hotel_name", title="ADR by Hotel")
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "revpar" in daily_total.columns:
            fig = visualizations.line_chart(daily_total, "date", "revpar",
                                            title="Portfolio RevPAR")
            st.plotly_chart(fig, use_container_width=True)


# ── Tab: Rankings ─────────────────────────────────────────────────────────────

def tab_rankings(df: pd.DataFrame) -> None:
    st.subheader("Hotel Rankings")
    rank_metric = st.selectbox(
        "Rank by",
        [m for m in ["revpar", "revenue", "occupancy_pct", "adr", "rooms_sold"] if m in df.columns],
        key="portfolio_rank_metric",
    )
    ranked = benchmarking.rank_hotels(df, metric=rank_metric)
    if ranked.empty:
        st.info("No hotel data to rank.")
        return

    display_cols = [c for c in ["rank", "hotel_name", "revenue", "rooms_sold", "occupancy_pct", "adr", "revpar"] if c in ranked.columns]

    # Top performers
    st.subheader("Top Performers")
    top = ranked.head(5)
    st.dataframe(top[display_cols].round(2), use_container_width=True, hide_index=True)

    # Bottom performers
    st.subheader("Bottom Performers")
    bottom = ranked.tail(5).sort_values("rank", ascending=False)
    st.dataframe(bottom[display_cols].round(2), use_container_width=True, hide_index=True)

    # Bar chart
    if rank_metric in ranked.columns:
        fig = visualizations.bar_chart(
            ranked, "hotel_name", rank_metric,
            title=f"Hotels Ranked by {rank_metric.replace('_', ' ').title()}",
            orientation="h",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Contribution
    st.subheader("Portfolio Contribution")
    contrib = benchmarking.portfolio_contribution(df)
    if not contrib.empty:
        share_cols = [c for c in contrib.columns if "share_pct" in c]
        if share_cols and "hotel_name" in contrib.columns:
            fig = px.pie(
                contrib, names="hotel_name", values=share_cols[0],
                title="Revenue Share by Hotel",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E0E0"))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(contrib.round(2), use_container_width=True, hide_index=True)


# ── Tab: Heatmaps ─────────────────────────────────────────────────────────────

def tab_heatmaps(df: pd.DataFrame) -> None:
    st.subheader("Portfolio Heatmaps")
    if "day_name" not in df.columns or "month_name" not in df.columns:
        st.info("Time-dimension columns (day_name, month_name) are required.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = visualizations.occupancy_heatmap(df, title="Portfolio Occupancy Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "adr" in df.columns:
            fig = visualizations.occupancy_heatmap(df, z_col="adr", title="Portfolio ADR Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    if "revpar" in df.columns:
        fig = visualizations.occupancy_heatmap(df, z_col="revpar", title="Portfolio RevPAR Heatmap")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab: Portfolio Forecast ────────────────────────────────────────────────────

def tab_forecast(df: pd.DataFrame, method: str, horizon: int) -> None:
    st.subheader("Portfolio Demand Forecast")
    metrics = [m for m in ["revenue", "revpar", "occupancy_pct", "adr"] if m in df.columns]
    if not metrics:
        st.error("No forecastable metrics found.")
        return

    selected = st.selectbox("Metric", metrics, key="portfolio_forecast_metric")

    # Aggregate portfolio into single daily series
    daily_total = df.groupby("date")[selected].mean().reset_index()

    with st.spinner(f"Running {method} portfolio forecast…"):
        fcast = forecasting.run_forecast(daily_total, method=method, metric=selected, horizon=horizon)

    if fcast is None or fcast.empty:
        st.error("Portfolio forecast failed — not enough data.")
        return

    fig = visualizations.forecast_chart(daily_total, fcast, selected,
                                        title=f"Portfolio {selected.upper()} Forecast ({method})")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Forecast Table"):
        disp = fcast[fcast.get("is_forecast", True)] if "is_forecast" in fcast.columns else fcast
        st.dataframe(disp[["date", "forecast", "lower_80", "upper_80"]].round(2),
                     use_container_width=True, hide_index=True)


# ── Tab: Portfolio Anomalies ───────────────────────────────────────────────────

def tab_anomalies(df: pd.DataFrame) -> None:
    st.subheader("Portfolio Anomaly Detection")
    flagged = anomaly_detection.detect_anomalies(df)
    summary = anomaly_detection.anomaly_summary(flagged)

    n = int(flagged["anomaly_flag"].sum()) if "anomaly_flag" in flagged.columns else 0
    st.metric("Total Anomalies", n)

    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.success("No portfolio-level anomalies detected.")


# ── Tab: Variance ─────────────────────────────────────────────────────────────

def tab_variance(df: pd.DataFrame) -> None:
    st.subheader("Budget vs Actual Variance")
    pairs = [("revenue", "budget_revenue"), ("rooms_sold", "budget_rooms")]
    rendered = False
    for actual_col, budget_col in pairs:
        if actual_col in df.columns and budget_col in df.columns:
            var = benchmarking.variance_analysis(df, compare_col=budget_col, actual_col=actual_col)
            if not var.empty:
                st.markdown(f"**{actual_col.title()} vs {budget_col.replace('_', ' ').title()}**")
                st.dataframe(var.round(2), use_container_width=True, hide_index=True)
                rendered = True
    if not rendered:
        st.info("Budget columns (budget_revenue, budget_rooms) not found in data.")


# ── Main entry ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame, forecast_method: str = "Prophet", forecast_horizon: int = 90) -> None:
    """Render the complete portfolio dashboard."""
    if df.empty:
        st.warning("No portfolio data available.")
        return

    n_hotels = df["hotel_name"].nunique() if "hotel_name" in df.columns else 1
    date_min = df["date"].min().date() if "date" in df.columns else "N/A"
    date_max = df["date"].max().date() if "date" in df.columns else "N/A"

    st.markdown(f"## 📊 Portfolio Dashboard — {n_hotels} Hotels")
    st.caption(f"Date range: {date_min} → {date_max} | {len(df):,} total records")

    tabs = st.tabs([
        "📊 Overview", "🏆 Rankings", "🗺️ Heatmaps",
        "🔮 Forecast", "⚠️ Anomalies", "📉 Variance",
    ])

    with tabs[0]:
        tab_overview(df)
    with tabs[1]:
        tab_rankings(df)
    with tabs[2]:
        tab_heatmaps(df)
    with tabs[3]:
        tab_forecast(df, forecast_method, forecast_horizon)
    with tabs[4]:
        tab_anomalies(df)
    with tabs[5]:
        tab_variance(df)
