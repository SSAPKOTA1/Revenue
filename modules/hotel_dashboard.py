"""Hotel-level dashboard: renders all tabs for a single selected hotel."""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st

from modules import kpi_engine, pickup_pace, forecasting, anomaly_detection, visualizations
from config.settings import BRAND_COLORS, COMPRESSION_THRESHOLD, KPI_FORMAT

logger = logging.getLogger(__name__)


# ── Revenue Management aggregation helper ────────────────────────────────────

def _rm_agg(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    """
    Aggregate raw room/revenue data using correct RM formulas.

    ADR       = Σ Revenue / Σ Rooms Sold          (never average ADR directly)
    Occupancy = Σ Rooms Sold / Σ Rooms Available  (never average occ% directly)
    RevPAR    = Σ Revenue / Σ Rooms Available
    Revenue   = Σ Revenue
    Rooms Sold= Σ Rooms Sold

    Returns a DataFrame with columns [group_col, metric] where metric
    is calculated with the correct formula.
    """
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    # Always sum the additive base columns
    sum_cols = [c for c in ["rooms_sold", "rooms_available", "revenue"] if c in df.columns]
    if not sum_cols:
        return pd.DataFrame()

    base = df.groupby(group_col)[sum_cols].sum(min_count=1).reset_index()

    # Derive requested metric from the summed bases
    if metric == "occupancy_pct":
        if "rooms_sold" in base.columns and "rooms_available" in base.columns:
            base[metric] = (
                base["rooms_sold"] / base["rooms_available"].replace(0, np.nan) * 100
            ).clip(0, 100)
        elif "occupancy_pct" in df.columns:
            # Fall back: weighted mean using rooms_available as weight
            wm = (
                df.groupby(group_col)
                .apply(lambda g: np.average(
                    g["occupancy_pct"].fillna(0),
                    weights=g.get("rooms_available", pd.Series(np.ones(len(g)))).fillna(1)
                ))
                .reset_index(name=metric)
            )
            base = base.merge(wm, on=group_col, how="left")

    elif metric == "adr":
        if "revenue" in base.columns and "rooms_sold" in base.columns:
            base[metric] = base["revenue"] / base["rooms_sold"].replace(0, np.nan)
        elif "adr" in df.columns:
            # Weighted mean using rooms_sold as weight
            wm = (
                df.groupby(group_col)
                .apply(lambda g: np.average(
                    g["adr"].fillna(0),
                    weights=g.get("rooms_sold", pd.Series(np.ones(len(g)))).fillna(1)
                ))
                .reset_index(name=metric)
            )
            base = base.merge(wm, on=group_col, how="left")

    elif metric == "revpar":
        if "revenue" in base.columns and "rooms_available" in base.columns:
            base[metric] = base["revenue"] / base["rooms_available"].replace(0, np.nan)
        elif "revpar" in df.columns:
            wm = (
                df.groupby(group_col)
                .apply(lambda g: np.average(
                    g["revpar"].fillna(0),
                    weights=g.get("rooms_available", pd.Series(np.ones(len(g)))).fillna(1)
                ))
                .reset_index(name=metric)
            )
            base = base.merge(wm, on=group_col, how="left")

    elif metric == "revenue":
        pass  # already summed above

    elif metric == "rooms_sold":
        pass  # already summed above

    else:
        # Generic numeric: sum if additive, mean otherwise
        if metric in df.columns:
            base[metric] = df.groupby(group_col)[metric].sum(min_count=1).values

    return base[[group_col, metric]].copy()


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


# ── Snapshot diagnostics ─────────────────────────────────────────────────────

def _snapshot_status(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Returns (has_snapshots, sorted_snapshot_list).
    A DataFrame has meaningful snapshots only if there are ≥ 2 distinct dates.
    """
    if "snapshot_date" not in df.columns:
        return False, []
    snaps = sorted(pd.to_datetime(df["snapshot_date"], errors="coerce").dropna().unique())
    return len(snaps) >= 2, snaps


def _render_snapshot_info(df: pd.DataFrame) -> tuple[bool, list]:
    """Show a compact snapshot status badge and return (has_snapshots, snaps)."""
    has_snaps, snaps = _snapshot_status(df)

    if has_snaps:
        first = pd.Timestamp(snaps[0]).strftime("%d %b %Y")
        last  = pd.Timestamp(snaps[-1]).strftime("%d %b %Y")
        st.success(
            f"✅ **{len(snaps)} snapshots detected** — from {first} to {last}. "
            f"Each snapshot = one day-folder export of the 365-day occupancy picture."
        )
    else:
        n = len(snaps)
        if n == 0:
            st.error(
                "⚠️ **No snapshot dates found.**  \n"
                "The app reads the snapshot date from your folder structure "
                "(`year / month / day / hotel.xlsx`).  \n"
                "Check the **Data Quality** panel → Ingestion Report to see what "
                "snapshot_date was detected for each file.  \n"
                "If the folder names don't contain a recognisable date the app "
                "falls back to the file's last-modified date."
            )
        else:
            st.warning(
                f"⚠️ **Only 1 snapshot detected** ({pd.Timestamp(snaps[0]).strftime('%d %b %Y')}).  \n"
                "Pickup and Pace need ≥ 2 snapshots to show change over time.  \n"
                "Make sure you have data from at least two different day-folders."
            )
    return has_snaps, snaps


# ── Tab: Pickup ──────────────────────────────────────────────────────────────

def tab_pickup(df: pd.DataFrame) -> None:
    """
    Pickup = latest snapshot on-books  MINUS  a chosen past snapshot on-books,
    for each arrival date. Grouped by Day / Week / Month / Year.
    """
    import plotly.graph_objects as go

    st.subheader("Pickup Analysis")
    st.caption(
        "Pickup measures how many rooms/revenue/ADR were added (or lost) "
        "between the current booking position and a past snapshot."
    )

    has_snapshots, all_snaps_raw = _render_snapshot_info(df)

    available_metrics = [m for m in ["rooms_sold", "revenue", "adr", "occupancy_pct"] if m in df.columns]
    if not available_metrics:
        st.info("No pickup metrics found in data.")
        return

    # ── Controls row ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        metric = st.selectbox("Metric", available_metrics, key="pu_metric")
    with c2:
        group_by = st.selectbox(
            "Group Arrival Dates By",
            ["Daily", "Weekly", "Monthly", "Yearly"],
            key="pu_groupby",
        )
    with c3:
        pass  # comparison selector placed below

    if not has_snapshots:
        st.info(
            "No snapshot dates found in this data. "
            "Pickup requires files organised in day-folders so each folder date "
            "becomes a booking snapshot. Showing day-over-day totals instead."
        )
        daily = df.groupby("date")[metric].sum(min_count=1).reset_index().sort_values("date")
        daily["pickup"] = daily[metric].diff()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"], y=daily["pickup"],
                             marker_color=BRAND_COLORS["secondary"], name="Day-over-Day"))
        fig.update_layout(title=f"Day-over-Day {metric}", template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        return

    # ── Snapshot selector ───────────────────────────────────────────────────
    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["snapshot_date", "date"])

    all_snaps = [pd.Timestamp(s) for s in all_snaps_raw]
    latest_snap = all_snaps[-1]

    # Preset options mapped to number of days back
    preset_options = {
        "1 day ago": 1,
        "3 days ago": 3,
        "7 days ago (1 week)": 7,
        "14 days ago (2 weeks)": 14,
        "30 days ago (1 month)": 30,
        "60 days ago": 60,
        "Custom date": -1,
    }
    with c3:
        preset = st.selectbox("Compare latest vs.", list(preset_options.keys()), key="pu_preset")

    if preset_options[preset] == -1:
        compare_date = st.date_input(
            "Pick comparison snapshot date",
            value=pd.Timestamp(all_snaps[0]).date(),
            min_value=pd.Timestamp(all_snaps[0]).date(),
            max_value=pd.Timestamp(all_snaps[-2]).date(),
            key="pu_custom_date",
        )
        target_snap_ts = pd.Timestamp(compare_date)
    else:
        target_snap_ts = latest_snap - pd.Timedelta(days=preset_options[preset])

    # Find closest available snapshot to the target
    snap_index = pd.DatetimeIndex(all_snaps)
    diffs = abs(snap_index - target_snap_ts)
    comparison_snap = pd.Timestamp(all_snaps[diffs.argmin()])

    st.markdown(
        f"**Latest snapshot:** `{latest_snap.strftime('%d %b %Y')}`   →   "
        f"**Comparison snapshot:** `{comparison_snap.strftime('%d %b %Y')}`   "
        f"*(Δ {(latest_snap - comparison_snap).days} days)*"
    )

    # ── Extract the two views ───────────────────────────────────────────────
    latest_view = df2[df2["snapshot_date"] == latest_snap].copy()
    compare_view = df2[df2["snapshot_date"] == comparison_snap].copy()

    if latest_view.empty or compare_view.empty:
        st.warning("One of the snapshots has no data for this hotel.")
        return

    # ── Grouping period column ──────────────────────────────────────────────
    freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME", "Yearly": "YE"}
    freq = freq_map[group_by]

    def _add_period(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.copy()
        if group_by == "Daily":
            f["period"] = f["date"].dt.normalize()
        elif group_by == "Weekly":
            f["period"] = f["date"].dt.to_period("W").apply(lambda p: p.start_time)
        elif group_by == "Monthly":
            f["period"] = f["date"].dt.to_period("M").apply(lambda p: p.start_time)
        else:
            f["period"] = f["date"].dt.to_period("Y").apply(lambda p: p.start_time)
        return f

    # Use proper RM formulas — ADR = Σrev/Σrooms, Occ% = Σsold/Σavail
    latest_agg = _rm_agg(_add_period(latest_view), "period", metric).set_index("period")[metric].rename("latest")
    compare_agg = _rm_agg(_add_period(compare_view), "period", metric).set_index("period")[metric].rename("compare")

    combined = pd.concat([latest_agg, compare_agg], axis=1).dropna(how="all").reset_index()
    combined["pickup"] = combined["latest"].fillna(0) - combined["compare"].fillna(0)

    if combined.empty:
        st.info("No overlapping arrival dates between the two snapshots.")
        return

    # ── Main pickup bar chart ───────────────────────────────────────────────
    metric_label = metric.replace("_", " ").title()
    period_labels = combined["period"].dt.strftime(
        "%d %b %Y" if group_by == "Daily" else
        "W %W %Y" if group_by == "Weekly" else
        "%b %Y" if group_by == "Monthly" else "%Y"
    )

    colors = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"] for v in combined["pickup"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=period_labels,
        y=combined["pickup"],
        marker_color=colors,
        name=f"Pickup ({metric_label})",
        text=[f"{v:+,.1f}" for v in combined["pickup"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{group_by} Pickup — {metric_label}  |  "
              f"{comparison_snap.strftime('%d %b %Y')} → {latest_snap.strftime('%d %b %Y')}",
        xaxis_title=f"Arrival Date ({group_by})",
        yaxis_title=f"Change in {metric_label}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Side-by-side comparison ─────────────────────────────────────────────
    st.subheader("Side-by-Side: Latest vs Comparison Snapshot")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=period_labels, y=combined["compare"],
        name=f"As of {comparison_snap.strftime('%d %b %Y')}",
        marker_color=BRAND_COLORS["neutral"],
        opacity=0.8,
    ))
    fig2.add_trace(go.Bar(
        x=period_labels, y=combined["latest"],
        name=f"As of {latest_snap.strftime('%d %b %Y')}",
        marker_color=BRAND_COLORS["secondary"],
        opacity=0.9,
    ))
    fig2.update_layout(
        barmode="group",
        title=f"{metric_label} On-Books: {group_by} View",
        xaxis_title=f"Arrival Date ({group_by})",
        yaxis_title=metric_label,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Summary KPIs — use correct RM formula for the whole period ──────────
    # For additive metrics (rooms, revenue): sum all periods
    # For derived metrics (ADR, Occ%): re-derive from the totals, not sum of derived values
    is_derived = metric in ("adr", "occupancy_pct", "revpar")

    if is_derived:
        total_latest = float(_rm_agg(latest_view, pd.Series(["all"] * len(latest_view), index=latest_view.index)
                                     .rename("_all").reset_index(drop=True).pipe(
                                         lambda _: latest_view.assign(_all="all")), "_all", metric)[metric].iloc[0]) \
            if False else float(_rm_agg(latest_view.assign(_all="all"), "_all", metric)[metric].iloc[0])
        total_compare = float(_rm_agg(compare_view.assign(_all="all"), "_all", metric)[metric].iloc[0])
    else:
        total_latest = combined["latest"].sum()
        total_compare = combined["compare"].sum()

    total_pickup = total_latest - total_compare
    pct_change = (total_pickup / abs(total_compare) * 100) if total_compare else 0

    k1, k2, k3, k4 = st.columns(4)
    fmt = "{:.1f}%" if metric == "occupancy_pct" else "${:,.2f}" if metric in ("adr", "revpar") else "{:,.1f}"
    k1.metric(f"{metric_label} (Latest)", fmt.format(total_latest))
    k2.metric(f"{metric_label} (Comparison)", fmt.format(total_compare))
    k3.metric("Net Pickup", f"{total_pickup:+,.1f}")
    k4.metric("% Change", f"{pct_change:+.1f}%")


# ── Tab: Pace ─────────────────────────────────────────────────────────────────

def tab_pace(df: pd.DataFrame) -> None:
    """
    Pace = for a selected arrival date (or range), how did the on-books
    occupancy / rooms_sold / revenue change over time across snapshot dates?

    X-axis = snapshot date (time the report was exported)
    Y-axis = on-books value for the selected arrival dates at that snapshot
    """
    import plotly.graph_objects as go

    st.subheader("Pace Analysis")
    st.caption(
        "Select an arrival date or date range below. "
        "The chart shows how the on-books position for those dates "
        "evolved over time as bookings accumulated."
    )

    has_snapshots, _ = _render_snapshot_info(df)
    metrics = [m for m in ["rooms_sold", "occupancy_pct", "revenue", "adr"] if m in df.columns]

    if not metrics:
        st.info("No metrics available for pace analysis.")
        return

    if not has_snapshots:
        return

    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["snapshot_date", "date"])

    all_snaps = sorted(df2["snapshot_date"].dropna().unique())
    all_arrival_dates = sorted(df2["date"].dropna().unique())

    if not all_arrival_dates:
        st.info("No arrival dates found.")
        return

    # ── Controls ────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        metric = st.selectbox("Metric to track", metrics, key="pace_metric")
    with c2:
        selection_mode = st.radio(
            "Date selection",
            ["Single date", "Date range"],
            horizontal=True,
            key="pace_sel_mode",
        )

    min_arrival = pd.Timestamp(all_arrival_dates[0]).date()
    max_arrival = pd.Timestamp(all_arrival_dates[-1]).date()

    if selection_mode == "Single date":
        # Default to a date roughly 30 days out from last snapshot
        default_date = min(
            pd.Timestamp(all_snaps[-1]) + pd.Timedelta(days=30),
            pd.Timestamp(all_arrival_dates[-1]),
        ).date()
        default_date = max(default_date, min_arrival)

        sel_date = st.date_input(
            "Arrival date",
            value=default_date,
            min_value=min_arrival,
            max_value=max_arrival,
            key="pace_single_date",
        )
        mask = df2["date"] == pd.Timestamp(sel_date)
        date_label = pd.Timestamp(sel_date).strftime("%d %b %Y")
    else:
        default_start = min_arrival
        default_end = max_arrival
        date_range = st.date_input(
            "Arrival date range",
            value=(default_start, default_end),
            min_value=min_arrival,
            max_value=max_arrival,
            key="pace_date_range",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        else:
            start_dt = end_dt = pd.Timestamp(date_range[0])
        mask = (df2["date"] >= start_dt) & (df2["date"] <= end_dt)
        date_label = f"{start_dt.strftime('%d %b %Y')} – {end_dt.strftime('%d %b %Y')}"

    selected = df2[mask].copy()

    if selected.empty:
        st.warning("No data for the selected arrival date(s). Try a different date.")
        return

    # ── Aggregate by snapshot date using correct RM formulas ───────────────
    # ADR = Σrev/Σrooms_sold,  Occ% = Σrooms_sold/Σrooms_avail,  RevPAR = Σrev/Σrooms_avail
    pace_series = (
        _rm_agg(selected, "snapshot_date", metric)
        .sort_values("snapshot_date")
        .reset_index(drop=True)
    )

    if pace_series.empty or len(pace_series) < 2:
        st.info("Not enough snapshots to draw a pace curve for this date. "
                "Try a broader date range.")
        return

    # ── Pace curve ──────────────────────────────────────────────────────────
    metric_label = metric.replace("_", " ").title()
    latest_val = float(pace_series[metric].iloc[-1])
    first_val = float(pace_series[metric].iloc[0])
    total_change = latest_val - first_val
    pct_change = (total_change / abs(first_val) * 100) if first_val else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pace_series["snapshot_date"],
        y=pace_series[metric],
        mode="lines+markers",
        line=dict(color=BRAND_COLORS["secondary"], width=2.5),
        marker=dict(size=7, color=BRAND_COLORS["accent"]),
        name=metric_label,
        hovertemplate="%{x|%d %b %Y}<br>" + metric_label + ": %{y:,.1f}<extra></extra>",
    ))
    # Shade the area under the curve
    fig.add_trace(go.Scatter(
        x=pace_series["snapshot_date"],
        y=pace_series[metric],
        fill="tozeroy",
        fillcolor="rgba(46,134,193,0.12)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"Pace of {metric_label} for Arrival Date(s): {date_label}",
        xaxis_title="Snapshot Date (as-of date of the report)",
        yaxis_title=metric_label,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI row ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Earliest on-books ({pd.Timestamp(pace_series['snapshot_date'].iloc[0]).strftime('%d %b')})",
              f"{first_val:,.1f}")
    k2.metric(f"Latest on-books ({pd.Timestamp(pace_series['snapshot_date'].iloc[-1]).strftime('%d %b')})",
              f"{latest_val:,.1f}")
    k3.metric("Total Change", f"{total_change:+,.1f}")
    k4.metric("% Change", f"{pct_change:+.1f}%")

    # ── Period-over-period pace comparison ──────────────────────────────────
    st.subheader("Weekly Pace Change")
    st.caption("Change in on-books from one snapshot to the one 7 days earlier.")
    # diff on the correctly-derived metric series (already one row per snapshot)
    pace_series = pace_series.copy()
    pace_series["weekly_change"] = pace_series[metric].diff(
        min(7, max(1, len(pace_series) // 4))
    ).fillna(pace_series[metric].diff(1))
    fig2 = go.Figure()
    colors = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"]
              for v in pace_series["weekly_change"]]
    fig2.add_trace(go.Bar(
        x=pace_series["snapshot_date"],
        y=pace_series["weekly_change"],
        marker_color=colors,
        name="7-Day Change",
        hovertemplate="%{x|%d %b %Y}<br>Change: %{y:+,.1f}<extra></extra>",
    ))
    fig2.update_layout(
        title=f"Week-over-Week Change in {metric_label} (for {date_label})",
        xaxis_title="Snapshot Date",
        yaxis_title=f"Change in {metric_label}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Multi-metric comparison (if date range selected) ────────────────────
    if selection_mode == "Date range" and len(metrics) > 1:
        st.subheader("Multi-Metric Pace Overview")
        available = [m for m in ["rooms_sold", "occupancy_pct", "revenue"] if m in df2.columns]
        if len(available) >= 2:
            fig3 = go.Figure()
            color_cycle = [BRAND_COLORS["secondary"], BRAND_COLORS["accent"],
                           BRAND_COLORS["success"], BRAND_COLORS["warning"]]
            for i, m in enumerate(available):
                s = _rm_agg(selected, "snapshot_date", m).sort_values("snapshot_date")
                if s.empty:
                    continue
                # Normalise to index (100 = first snapshot) for comparison
                base = s[m].iloc[0]
                if base and base != 0:
                    s["indexed"] = s[m] / base * 100
                    fig3.add_trace(go.Scatter(
                        x=s["snapshot_date"], y=s["indexed"],
                        mode="lines+markers",
                        name=m.replace("_", " ").title(),
                        line=dict(color=color_cycle[i % len(color_cycle)], width=2),
                    ))
            fig3.add_hline(y=100, line_dash="dash", line_color="white", opacity=0.3,
                           annotation_text="Baseline (first snapshot)")
            fig3.update_layout(
                title="Pace Index (100 = first snapshot) — All Metrics",
                xaxis_title="Snapshot Date",
                yaxis_title="Index (100 = first snapshot)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Raw data table ──────────────────────────────────────────────────────
    with st.expander("Show raw pace data"):
        display = pace_series.copy()
        display["snapshot_date"] = display["snapshot_date"].dt.strftime("%d %b %Y")
        st.dataframe(display.rename(columns={metric: metric_label}),
                     use_container_width=True, hide_index=True)


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
