"""Hotel-level dashboard: renders all tabs for a single selected hotel."""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st

from modules import kpi_engine, pickup_pace, forecasting, anomaly_detection, visualizations
from config.settings import BRAND_COLORS, COMPRESSION_THRESHOLD, KPI_FORMAT

logger = logging.getLogger(__name__)


# ── KPI Card helper ──────────────────────────────────────────────────────────

def _kpi_card(label: str, value: float | str, fmt: str = "{:,.2f}", delta: Optional[float] = None) -> str:
    if isinstance(value, float) and not np.isnan(value):
        formatted = fmt.format(value)
    elif isinstance(value, str):
        formatted = value
    else:
        formatted = "N/A"

    delta_html = ""
    if delta is not None and not np.isnan(delta):
        color = "#27AE60" if delta >= 0 else "#E74C3C"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f"<span style='color:{color};font-size:13px'>{arrow} {abs(delta):,.1f}</span>"

    return f"""
    <div style='background:linear-gradient(135deg,#1A2744,#1E3A5F);
                border-radius:12px;padding:20px 24px;
                border:1px solid rgba(255,255,255,0.08);
                box-shadow:0 4px 15px rgba(0,0,0,0.3);'>
      <p style='margin:0;font-size:12px;color:#94A3B8;letter-spacing:1px;text-transform:uppercase'>{label}</p>
      <p style='margin:6px 0 4px;font-size:28px;font-weight:700;color:#F1F5F9'>{formatted}</p>
      {delta_html}
    </div>"""


def render_kpi_cards(kpis: dict, cols: int = 5) -> None:
    """Render a row of KPI metric cards."""
    card_defs = [
        ("Occupancy", "occupancy_pct", "{:.1f}%"),
        ("ADR", "adr", "${:,.2f}"),
        ("RevPAR", "revpar", "${:,.2f}"),
        ("Revenue", "revenue", "${:,.0f}"),
        ("Rooms Sold", "rooms_sold", "{:,.0f}"),
    ]
    columns = st.columns(len(card_defs))
    for col, (label, key, fmt) in zip(columns, card_defs):
        val = kpis.get(key, float("nan"))
        with col:
            st.markdown(_kpi_card(label, val, fmt), unsafe_allow_html=True)


# ── Tab: Overview ────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame, hotel: str) -> None:
    kpis = kpi_engine.compute_hotel_kpis(df)
    render_kpi_cards(kpis)

    st.markdown("---")
    daily = kpi_engine.aggregate_daily(df)
    if daily.empty:
        st.info("No daily data available.")
        return

    c1, c2 = st.columns(2)
    with c1:
        if "occupancy_pct" in daily.columns:
            fig = visualizations.area_chart(
                daily, "date", "occupancy_pct",
                title="Daily Occupancy %",
                color=BRAND_COLORS["secondary"],
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "revpar" in daily.columns:
            fig = visualizations.area_chart(
                daily, "date", "revpar",
                title="Daily RevPAR",
                color=BRAND_COLORS["accent"],
            )
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "adr" in daily.columns:
            fig = visualizations.line_chart(daily, "date", "adr", title="Daily ADR")
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "revenue" in daily.columns:
            fig = visualizations.area_chart(
                daily, "date", "revenue", title="Daily Revenue",
                color=BRAND_COLORS["success"],
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Tab: Performance ────────────────────────────────────────────────────────

def tab_performance(df: pd.DataFrame) -> None:
    st.subheader("Monthly Performance")
    monthly = kpi_engine.aggregate_period(df, "M", group_hotel=False)
    if not monthly.empty and "period" in monthly.columns:
        c1, c2 = st.columns(2)
        with c1:
            if "occupancy_pct" in monthly.columns:
                fig = visualizations.bar_chart(monthly, "period", "occupancy_pct",
                                               title="Monthly Occupancy %",
                                               color=BRAND_COLORS["secondary"])
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "revenue" in monthly.columns:
                fig = visualizations.bar_chart(monthly, "period", "revenue",
                                               title="Monthly Revenue",
                                               color=BRAND_COLORS["success"])
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday Performance")
    weekday = kpi_engine.weekday_performance(df)
    if not weekday.empty:
        c3, c4 = st.columns(2)
        with c3:
            if "occupancy_pct" in weekday.columns:
                fig = visualizations.bar_chart(weekday, "day_name", "occupancy_pct",
                                               title="Occupancy by Day of Week",
                                               color=BRAND_COLORS["primary"])
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            if "adr" in weekday.columns:
                fig = visualizations.bar_chart(weekday, "day_name", "adr",
                                               title="ADR by Day of Week",
                                               color=BRAND_COLORS["accent"])
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonality — Monthly Average")
    seasonal = kpi_engine.monthly_performance(df)
    if not seasonal.empty:
        c5, c6 = st.columns(2)
        with c5:
            if "revpar" in seasonal.columns:
                fig = visualizations.bar_chart(seasonal, "month_name", "revpar",
                                               title="RevPAR by Month",
                                               color=BRAND_COLORS["secondary"])
                st.plotly_chart(fig, use_container_width=True)
        with c6:
            if "adr" in seasonal.columns:
                fig = visualizations.line_chart(seasonal, "month_name", "adr",
                                                title="ADR Seasonality")
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmaps")
    if "day_name" in df.columns and "month_name" in df.columns:
        c7, c8 = st.columns(2)
        with c7:
            fig = visualizations.occupancy_heatmap(df, title="Occupancy % Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        with c8:
            if "revpar" in df.columns:
                fig = visualizations.occupancy_heatmap(df, z_col="revpar",
                                                        title="RevPAR Heatmap")
                st.plotly_chart(fig, use_container_width=True)

    # Compression nights
    compression = kpi_engine.compression_nights(df, COMPRESSION_THRESHOLD)
    st.subheader(f"Compression Nights (Occ ≥ {COMPRESSION_THRESHOLD}%)")
    if not compression.empty:
        st.metric("Compression Nights", len(compression))
        display_cols = [c for c in ["date", "occupancy_pct", "adr", "revpar", "revenue"] if c in compression.columns]
        st.dataframe(compression[display_cols].sort_values("date"), use_container_width=True, hide_index=True)
    else:
        st.info("No compression nights found in the selected date range.")

    # Rolling averages
    st.subheader("7-Day Rolling Average")
    daily = kpi_engine.aggregate_daily(df)
    if not daily.empty:
        daily = kpi_engine.rolling_kpis(daily, 7)
        for col in ["occupancy_pct_rolling7", "revpar_rolling7"]:
            if col in daily.columns:
                fig = visualizations.line_chart(daily, "date", col,
                                                title=col.replace("_", " ").title())
                st.plotly_chart(fig, use_container_width=True)


# ── Tab: Pickup ──────────────────────────────────────────────────────────────

def tab_pickup(df: pd.DataFrame) -> None:
    st.subheader("Pickup Analysis")

    has_snapshots = "snapshot_date" in df.columns and df["snapshot_date"].notna().any()
    available_metrics = [m for m in ["rooms_sold", "revenue", "adr"] if m in df.columns]
    if not available_metrics:
        st.info("No pickup metrics found in data.")
        return

    metric = st.selectbox("Pickup Metric", available_metrics, key="hotel_pickup_metric")

    if has_snapshots:
        n_snaps = df["snapshot_date"].nunique()
        st.caption(
            f"📅 {n_snaps} daily snapshots detected — showing true pickup "
            f"(on-books change between snapshot dates)."
        )

        pickup_df = pickup_pace.compute_pickup(df, metrics=available_metrics)
        if pickup_df.empty:
            st.info("Not enough snapshots to compute pickup.")
            return

        # Aggregate pickup across all arrival dates per snapshot date
        snap_col = "snapshot_date"
        for w in [1, 7, 14, 30]:
            col = f"{metric}_pickup_{w}d"
            if col not in pickup_df.columns:
                continue
            agg = pickup_df.groupby(snap_col)[col].sum().reset_index()
            agg.columns = [snap_col, col]
            fig = visualizations.area_chart(
                agg, snap_col, col,
                title=f"{w}-Day Pickup: {metric.replace('_',' ').title()} (by Snapshot Date)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Pickup curve for a selected arrival date
        st.subheader("Booking Build-Up Curve")
        future_dates = sorted(df["date"].dropna().unique())
        if future_dates:
            sel_date = st.selectbox(
                "Select Arrival Date",
                future_dates,
                format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
                key="hotel_pickup_arrival",
            )
            hotel = df["hotel_name"].iloc[0] if "hotel_name" in df.columns else None
            if hotel:
                curve = pickup_pace.pickup_curve(df, hotel, pd.Timestamp(sel_date), metric)
                if not curve.empty:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=curve["days_to_arrival"][::-1],
                        y=curve[metric],
                        mode="lines+markers",
                        line=dict(color="#2E86C1", width=2),
                        name=metric,
                    ))
                    fig.update_layout(
                        title=f"Booking Build-Up: {metric} for {pd.Timestamp(sel_date).strftime('%d %b %Y')}",
                        xaxis_title="Days Before Arrival",
                        xaxis_autorange="reversed",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        # Single snapshot fallback
        st.caption("Single snapshot detected — showing day-over-day differences.")
        pickup_df = pickup_pace.compute_pickup(df, metrics=available_metrics)
        if pickup_df.empty:
            st.info("Pickup analysis requires at least 2 data points.")
            return
        for w in [1, 7, 14, 30]:
            col = f"{metric}_pickup_{w}d"
            if col in pickup_df.columns:
                fig = visualizations.area_chart(
                    pickup_df, "date", col,
                    title=f"{w}-Day Pickup: {metric.replace('_', ' ').title()}",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Waterfall summary
    pickup_df2 = pickup_pace.compute_pickup(df, metrics=available_metrics)
    cols = [f"{metric}_pickup_{w}d" for w in [1, 7, 14, 30] if f"{metric}_pickup_{w}d" in pickup_df2.columns]
    if cols:
        vals = [float(pickup_df2[c].sum()) for c in cols]
        labels = ["1-Day", "7-Day", "14-Day", "30-Day"][:len(vals)]
        fig = visualizations.waterfall_chart(labels, vals, title=f"Pickup Waterfall: {metric}")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab: Pace ─────────────────────────────────────────────────────────────────

def tab_pace(df: pd.DataFrame) -> None:
    st.subheader("Pace Analysis")

    has_snapshots = "snapshot_date" in df.columns and df["snapshot_date"].notna().any()
    if has_snapshots:
        snaps = sorted(df["snapshot_date"].dropna().unique())
        latest = pd.Timestamp(snaps[-1])
        st.caption(
            f"📅 {len(snaps)} snapshots available. "
            f"Current position: **{latest.strftime('%d %b %Y')}**"
        )

    benchmark = st.selectbox(
        "Benchmark",
        ["last_year", "budget", "forecast", "portfolio_avg"],
        key="hotel_pace_benchmark",
    )
    metrics = [m for m in ["rooms_sold", "revenue", "adr", "occupancy_pct"] if m in df.columns]
    selected_metric = st.selectbox("Pace Metric", metrics, key="hotel_pace_metric")

    pace_df = pickup_pace.compute_pace(df, benchmark=benchmark, metrics=metrics)
    if pace_df.empty:
        st.info(f"Pace vs '{benchmark}' not available — missing required reference columns.")
        return

    pace_for_metric = pace_df[pace_df["metric"] == selected_metric] if "metric" in pace_df.columns else pace_df

    if not pace_for_metric.empty:
        fig = visualizations.pace_curve(pace_for_metric, title=f"Pace: {selected_metric} vs {benchmark}")
        st.plotly_chart(fig, use_container_width=True)

        if "pace_var" in pace_for_metric.columns:
            fig2 = visualizations.bar_chart(
                pace_for_metric,
                x="date" if "date" in pace_for_metric.columns else "month_day",
                y="pace_var",
                title="Pace Variance",
                color=BRAND_COLORS["warning"],
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Booking window
    bw = pickup_pace.booking_window_distribution(df)
    if bw is not None and not bw.empty:
        st.subheader("Booking Window Distribution")
        fig = visualizations.booking_window_hist(bw)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Booking window analysis requires 'booking_date' column.")


# ── Tab: Forecasting ──────────────────────────────────────────────────────────

def tab_forecasting(df: pd.DataFrame, method: str, horizon: int) -> None:
    st.subheader("Demand Forecasting")
    metrics = [m for m in ["revpar", "revenue", "occupancy_pct", "adr", "rooms_sold"] if m in df.columns]
    if not metrics:
        st.error("No forecastable metrics found.")
        return

    selected = st.selectbox("Forecast Metric", metrics, key="hotel_forecast_metric")

    with st.spinner(f"Running {method} forecast for {selected}…"):
        fcast = forecasting.run_forecast(df, method=method, metric=selected, horizon=horizon)

    if fcast is None or fcast.empty:
        st.error("Forecast failed — not enough data or missing dependency.")
        return

    daily = kpi_engine.aggregate_daily(df)
    fig = visualizations.forecast_chart(daily, fcast, selected, title=f"{selected.upper()} Forecast ({method})")
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy
    acc = forecasting.compute_forecast_accuracy(daily, fcast, selected)
    if acc:
        st.subheader("Forecast Accuracy")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", f"{acc.get('MAPE_%', 'N/A')}%")
        c2.metric("MAE", f"{acc.get('MAE', 'N/A'):,.2f}")
        c3.metric("RMSE", f"{acc.get('RMSE', 'N/A'):,.2f}")

    # Table
    with st.expander("Forecast Table"):
        disp = fcast[fcast.get("is_forecast", True)].copy() if "is_forecast" in fcast.columns else fcast.copy()
        st.dataframe(disp[["date", "forecast", "lower_80", "upper_80"]].round(2),
                     use_container_width=True, hide_index=True)


# ── Tab: Anomalies ────────────────────────────────────────────────────────────

def tab_anomalies(df: pd.DataFrame) -> None:
    st.subheader("Anomaly Detection")
    flagged = anomaly_detection.detect_anomalies(df)
    summary = anomaly_detection.anomaly_summary(flagged)

    n_anom = int(flagged["anomaly_flag"].sum()) if "anomaly_flag" in flagged.columns else 0
    st.metric("Anomalies Detected", n_anom)

    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if "date" in flagged.columns and "revenue" in flagged.columns:
            daily = kpi_engine.aggregate_daily(df)
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["revenue"],
                mode="lines", name="Revenue",
                line=dict(color=BRAND_COLORS["secondary"]),
            ))
            anom_pts = flagged[flagged["anomaly_flag"]] if "anomaly_flag" in flagged.columns else pd.DataFrame()
            if not anom_pts.empty and "revenue" in anom_pts.columns:
                fig.add_trace(go.Scatter(
                    x=anom_pts["date"], y=anom_pts["revenue"],
                    mode="markers", name="Anomaly",
                    marker=dict(color=BRAND_COLORS["danger"], size=10, symbol="x"),
                ))
            fig.update_layout(title="Revenue with Anomalies", **{
                "template": "plotly_dark",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            })
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No anomalies detected in the selected date range.")


# ── Main entry point ──────────────────────────────────────────────────────────

def render(
    df: pd.DataFrame,
    hotel: str,
    forecast_method: str = "Prophet",
    forecast_horizon: int = 90,
) -> None:
    """Render the full hotel-level dashboard with tabs."""
    hotel_df = df[df["hotel_name"] == hotel].copy() if "hotel_name" in df.columns else df.copy()

    if hotel_df.empty:
        st.warning(f"No data found for hotel: {hotel}")
        return

    st.markdown(f"## 🏨 {hotel}")
    st.caption(f"{len(hotel_df):,} records | {hotel_df['date'].min().date()} → {hotel_df['date'].max().date()}" if "date" in hotel_df.columns else "")

    tabs = st.tabs(["📊 Overview", "📈 Performance", "🔄 Pickup", "⏱ Pace",
                    "🔮 Forecasting", "⚠️ Anomalies"])

    with tabs[0]:
        tab_overview(hotel_df, hotel)
    with tabs[1]:
        tab_performance(hotel_df)
    with tabs[2]:
        tab_pickup(hotel_df)
    with tabs[3]:
        tab_pace(hotel_df)
    with tabs[4]:
        tab_forecasting(hotel_df, forecast_method, forecast_horizon)
    with tabs[5]:
        tab_anomalies(hotel_df)
