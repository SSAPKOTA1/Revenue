"""
Hotel-level dashboard — 6 tabs:
  1. Current Position  — what's on books now for future arrival dates
  2. Pickup            — change between latest snapshot and a past snapshot
  3. Pace              — how did bookings for a selected arrival date build up
  4. Performance       — historical actuals (past arrival dates)
  5. Forecast          — demand forecast
  6. Anomalies         — outlier detection
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from modules import kpi_engine, forecasting, anomaly_detection, exports
from config.settings import BRAND_COLORS, COMPRESSION_THRESHOLD

logger = logging.getLogger(__name__)

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E0E0E0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val: float, metric: str) -> str:
    if pd.isna(val):
        return "N/A"
    if metric == "occupancy_pct":
        return f"{val:.1f}%"
    if metric in ("adr", "revpar", "revenue"):
        return f"€{val:,.2f}" if val < 1_000_000 else f"€{val/1_000:,.0f}k"
    return f"{val:,.1f}"


def _kpi_card(label: str, value: str, delta: Optional[str] = None, delta_pos: bool = True) -> str:
    delta_html = ""
    if delta:
        color = "#22c55e" if delta_pos else "#ef4444"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f"<span style='color:{color};font-size:13px'>{arrow} {delta}</span>"
    return f"""
    <div style='background:linear-gradient(135deg,#1a2744,#1e3a5f);border-radius:10px;
                padding:18px 22px;border:1px solid rgba(255,255,255,0.08);
                box-shadow:0 4px 15px rgba(0,0,0,0.3);'>
      <p style='margin:0;font-size:11px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>{label}</p>
      <p style='margin:6px 0 2px;font-size:26px;font-weight:700;color:#f1f5f9'>{value}</p>
      {delta_html}
    </div>"""


def _kpi_row(df: pd.DataFrame) -> None:
    """KPI cards: Combined (large) + Closed / Open split (small, below)."""
    closed_df, open_df = kpi_engine.split_closed_open(df)
    c = kpi_engine._kpis_from_df(closed_df)
    o = kpi_engine._kpis_from_df(open_df)
    t = kpi_engine._kpis_from_df(df)       # combined — re-derived from full sums
    today = pd.Timestamp.today()

    st.markdown(
        f"<p style='font-size:0.78rem;color:#64748b;margin-bottom:6px;'>"
        f"✅ <b>Closed</b> = up to {(today - pd.Timedelta(days=1)).strftime('%d %b %Y')} &nbsp;·&nbsp; "
        f"📋 <b>Open</b> = from {today.strftime('%d %b %Y')} on-books &nbsp;·&nbsp; "
        f"⬜ <b>Combined</b> = full period</p>",
        unsafe_allow_html=True,
    )

    items = [
        ("Occupancy %", "occupancy_pct", "{:.1f}%"),
        ("ADR",         "adr",           "€{:,.2f}"),
        ("RevPAR",      "revpar",        "€{:,.2f}"),
        ("Revenue",     "revenue",       "€{:,.0f}"),
        ("Rooms Sold",  "rooms_sold",    "{:,.0f}"),
    ]
    cols = st.columns(len(items))
    for col, (lbl, key, fmt) in zip(cols, items):
        cv = c.get(key, 0)
        ov = o.get(key, 0)
        tv = t.get(key, 0)
        col.markdown(
            f"""<div style='background:linear-gradient(135deg,#1a2744,#0d1b2a);
                border-radius:10px;padding:14px 16px;text-align:center;
                border:1px solid rgba(255,255,255,0.08);'>
              <p style='margin:0 0 4px;font-size:11px;color:#94a3b8;letter-spacing:1px;
                        text-transform:uppercase'>{lbl}</p>
              <p style='margin:0 0 8px;font-size:22px;font-weight:800;color:#f1f5f9'>
                {fmt.format(tv)}</p>
              <div style='display:flex;justify-content:space-around;
                          border-top:1px solid rgba(255,255,255,0.07);padding-top:7px;'>
                <div>
                  <p style='margin:0;font-size:9px;color:#64748b'>✅ Closed</p>
                  <p style='margin:2px 0 0;font-size:13px;font-weight:700;color:#34d399'>
                    {fmt.format(cv)}</p>
                </div>
                <div style='border-left:1px solid rgba(255,255,255,0.08)'></div>
                <div>
                  <p style='margin:0;font-size:9px;color:#64748b'>📋 Open</p>
                  <p style='margin:2px 0 0;font-size:13px;font-weight:700;color:#60a5fa'>
                    {fmt.format(ov)}</p>
                </div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def _bar(x, y_pos, y_neg=None, title="", x_title="", y_title="", labels_pos=None, labels_neg=None):
    fig = go.Figure()
    if y_neg is None:
        colors = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"] for v in y_pos]
        fig.add_trace(go.Bar(x=x, y=y_pos, marker_color=colors,
                             text=[f"{v:+,.1f}" for v in y_pos], textposition="outside"))
    else:
        fig.add_trace(go.Bar(x=x, y=y_neg, name="Comparison",
                             marker_color=BRAND_COLORS["neutral"], opacity=0.8))
        fig.add_trace(go.Bar(x=x, y=y_pos, name="Latest",
                             marker_color=BRAND_COLORS["secondary"], opacity=0.9))
        fig.update_layout(barmode="group")
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, **_DARK)
    return fig


def _line(x, y, title="", x_title="", y_title="", color=None, fill=False, name=""):
    fig = go.Figure()
    c = color or BRAND_COLORS["secondary"]
    kwargs = dict(x=x, y=y, mode="lines+markers",
                  line=dict(color=c, width=2.5),
                  marker=dict(size=6, color=BRAND_COLORS["accent"]),
                  name=name or y_title,
                  hovertemplate="%{x|%d %b %Y}<br>%{y:,.1f}<extra></extra>")
    fig.add_trace(go.Scatter(**kwargs))
    if fill:
        fig.add_trace(go.Scatter(x=x, y=y, fill="tozeroy",
                                 fillcolor="rgba(46,134,193,0.12)",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      hovermode="x unified", **_DARK)
    return fig


# ── Snapshot helper ───────────────────────────────────────────────────────────

def _snap_banner(df: pd.DataFrame):
    snaps = kpi_engine.get_snapshots(df)
    if len(snaps) >= 2:
        st.success(
            f"✅ **{len(snaps)} snapshots** — "
            f"{pd.Timestamp(snaps[0]).strftime('%d %b %Y')} → "
            f"{pd.Timestamp(snaps[-1]).strftime('%d %b %Y')}"
        )
    elif len(snaps) == 1:
        st.warning("⚠️ Only 1 snapshot found. Pickup and Pace need ≥ 2 snapshots.")
    else:
        st.error("⚠️ No snapshot dates detected. Check folder structure: year/month/day/Hotel.xlsx")
    return snaps


# ── Tab 1: Current Position ───────────────────────────────────────────────────

def tab_current_position(df: pd.DataFrame, hotel: str) -> None:
    """What's on books right now (latest snapshot) for future arrival dates."""

    snaps = kpi_engine.get_snapshots(df)
    latest_snap = snaps[-1] if snaps else None

    # Current view = latest snapshot
    if latest_snap:
        current = kpi_engine.latest_snapshot_view(df)
        st.caption(f"📅 Showing on-books as of **{latest_snap.strftime('%d %b %Y')}** (latest snapshot)")
    else:
        current = df.copy()
        st.caption("📅 No snapshots — showing all data combined")

    _kpi_row(current)
    st.markdown("---")

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        group_by = st.selectbox(
            "Group arrival dates by",
            ["Monthly", "Weekly", "Quarterly", "Yearly", "Daily"],
            key="cp_groupby",
        )
    with c2:
        available = [m for m in ["rooms_sold", "revenue", "occupancy_pct", "adr", "revpar"]
                     if m in current.columns]
        metric = st.selectbox("Metric", available, key="cp_metric")

    if current.empty:
        st.info("No data in the current snapshot.")
        return

    # Add period and aggregate
    agg = kpi_engine.rm_aggregate(
        kpi_engine.add_period_col(current, "date", group_by),
        ["period", "period_label"],
    ).sort_values("period")

    if agg.empty or metric not in agg.columns:
        st.info("Not enough data to aggregate.")
        return

    metric_label = metric.replace("_", " ").title()

    # Chart: current on-books by arrival period
    fig = go.Figure(go.Bar(
        x=agg["period_label"],
        y=agg[metric],
        marker_color=BRAND_COLORS["secondary"],
        text=[_fmt(v, metric) for v in agg[metric]],
        textposition="outside",
        name=metric_label,
    ))
    fig.update_layout(
        title=f"On-Books {metric_label} by Arrival {group_by} — as of {latest_snap.strftime('%d %b %Y') if latest_snap else 'Latest'}",
        xaxis_title=f"Arrival Date ({group_by})",
        yaxis_title=metric_label,
        **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table summary
    with st.expander("📋 Detailed breakdown"):
        display_cols = ["period_label"] + [c for c in
                        ["rooms_sold", "rooms_available", "occupancy_pct", "adr", "revpar", "revenue"]
                        if c in agg.columns]
        show = agg[display_cols].copy()
        show.columns = [c.replace("_", " ").title() for c in show.columns]
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)

    # Snapshot count info
    if len(snaps) > 0:
        st.markdown("---")
        st.markdown(f"**Available Snapshots:** {len(snaps)} total")
        snap_df = pd.DataFrame({
            "Snapshot Date": [pd.Timestamp(s).strftime("%d %b %Y") for s in snaps],
        })
        with st.expander("View all snapshot dates"):
            st.dataframe(snap_df, use_container_width=True, hide_index=True)


# ── Tab 2: Pickup ─────────────────────────────────────────────────────────────

def tab_pickup(df: pd.DataFrame) -> None:
    """
    Pickup = (on-books at latest snapshot) − (on-books at chosen past snapshot)
    for the same future arrival dates.
    """
    snaps = _snap_banner(df)
    if len(snaps) < 2:
        return

    latest_snap = snaps[-1]
    earliest_snap = snaps[0]

    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    df2["date"]          = pd.to_datetime(df2["date"],          errors="coerce")
    df2 = df2.dropna(subset=["snapshot_date", "date"])

    # ── Section 1: Snapshot date picker ──────────────────────────────────────
    st.markdown("#### 📅 Choose comparison snapshot date")
    st.caption(
        "The app compares what is on-books **today** (latest snapshot) "
        "versus what was on-books on the date you pick below."
    )

    col_date, col_quick = st.columns([2, 3])

    with col_date:
        comp_date_input = st.date_input(
            "Comparison snapshot date",
            value=max(
                pd.Timestamp(latest_snap - pd.Timedelta(days=7)).date(),
                pd.Timestamp(earliest_snap).date(),
            ),
            min_value=pd.Timestamp(earliest_snap).date(),
            max_value=pd.Timestamp(snaps[-2]).date(),
            key="pu_comp_date",
        )
        target = pd.Timestamp(comp_date_input)

    with col_quick:
        st.markdown("**Quick presets** (click to jump):")
        preset_cols = st.columns(5)
        presets = [("−7d", 7), ("−14d", 14), ("−30d", 30), ("−60d", 60), ("−90d", 90)]
        for i, (label, days) in enumerate(presets):
            t = latest_snap - pd.Timedelta(days=days)
            if t >= earliest_snap:
                if preset_cols[i].button(label, key=f"pu_preset_{days}"):
                    target = t

    # Snap to nearest available snapshot
    diffs = [abs((s - target).days) for s in snaps[:-1]]
    comp_snap = snaps[diffs.index(min(diffs))]
    delta_days = (latest_snap - comp_snap).days

    st.markdown(
        f"<div style='background:#162032;border-radius:8px;padding:10px 16px;"
        f"margin:8px 0;font-size:0.9rem;'>"
        f"🗓️ <b>Comparing</b>: "
        f"<span style='color:#60a5fa'>{comp_snap.strftime('%d %b %Y')}</span> "
        f"→ "
        f"<span style='color:#34d399'>{latest_snap.strftime('%d %b %Y')}</span> "
        f"&nbsp;·&nbsp; <b>Δ {delta_days} days</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Section 2: Arrival date filter ───────────────────────────────────────
    st.markdown("#### 🛎️ Filter arrival dates (optional)")
    all_arrival_dates = sorted(df2["date"].dropna().unique())
    arr_min = pd.Timestamp(all_arrival_dates[0]).date()
    arr_max = pd.Timestamp(all_arrival_dates[-1]).date()

    arr_filter = st.date_input(
        "Show pickup for arrivals between",
        value=(arr_min, arr_max),
        min_value=arr_min,
        max_value=arr_max,
        key="pu_arr_filter",
    )
    if isinstance(arr_filter, (list, tuple)) and len(arr_filter) == 2:
        arr_from, arr_to = pd.Timestamp(arr_filter[0]), pd.Timestamp(arr_filter[1])
    else:
        arr_from = arr_to = pd.Timestamp(arr_filter[0])

    # ── Section 3: Metric & grouping ─────────────────────────────────────────
    available_metrics = [m for m in ["rooms_sold", "revenue", "occupancy_pct", "adr"]
                         if m in df2.columns]
    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox("Metric", available_metrics, key="pu_metric")
    with c2:
        group_by = st.selectbox(
            "Group arrival dates by",
            ["Monthly", "Weekly", "Daily", "Yearly"],
            key="pu_groupby",
        )

    # ── Extract the two snapshot views ───────────────────────────────────────
    latest_view  = df2[df2["snapshot_date"] == latest_snap].copy()
    compare_view = df2[df2["snapshot_date"] == comp_snap].copy()

    # Apply arrival date filter
    latest_view  = latest_view[(latest_view["date"]  >= arr_from) & (latest_view["date"]  <= arr_to)]
    compare_view = compare_view[(compare_view["date"] >= arr_from) & (compare_view["date"] <= arr_to)]

    if latest_view.empty or compare_view.empty:
        st.warning("No data for the selected arrival date range in one of the snapshots.")
        return

    def _agg_view(view):
        v = kpi_engine.add_period_col(view, "date", group_by)
        return kpi_engine.rm_aggregate(v, ["period", "period_label"]).set_index("period")

    agg_l = _agg_view(latest_view)
    agg_c = _agg_view(compare_view)

    all_periods = agg_l.index.union(agg_c.index)
    agg_l = agg_l.reindex(all_periods)
    agg_c = agg_c.reindex(all_periods)

    metric_label = metric.replace("_", " ").title()
    labels  = agg_l["period_label"].fillna(agg_c["period_label"]).values
    vals_l  = agg_l[metric].fillna(0).values if metric in agg_l.columns else np.zeros(len(all_periods))
    vals_c  = agg_c[metric].fillna(0).values if metric in agg_c.columns else np.zeros(len(all_periods))
    pickup  = vals_l - vals_c

    # ── KPI summary row ───────────────────────────────────────────────────────
    rs_l  = float(latest_view["rooms_sold"].sum())  if "rooms_sold"  in latest_view.columns  else np.nan
    rs_c  = float(compare_view["rooms_sold"].sum()) if "rooms_sold"  in compare_view.columns else np.nan
    rev_l = float(latest_view["revenue"].sum())     if "revenue"     in latest_view.columns  else np.nan
    rev_c = float(compare_view["revenue"].sum())    if "revenue"     in compare_view.columns else np.nan
    ra_l  = float(latest_view["rooms_available"].sum()) if "rooms_available" in latest_view.columns else np.nan

    occ_l = (rs_l / ra_l * 100) if (ra_l and ra_l > 0) else np.nan
    ra_c  = float(compare_view["rooms_available"].sum()) if "rooms_available" in compare_view.columns else np.nan
    occ_c = (rs_c / ra_c * 100) if (ra_c and ra_c > 0) else np.nan

    def _fmt(v, is_pct=False, is_money=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if is_pct:   return f"{v:.1f}%"
        if is_money: return f"€{v:,.0f}"
        return f"{v:,.0f}"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Rooms (Latest)",      _fmt(rs_l),  delta=_fmt(rs_l  - rs_c)  if not np.isnan(rs_c)  else None)
    k2.metric("Rooms (Comparison)",  _fmt(rs_c))
    k3.metric("Revenue (Latest)",    _fmt(rev_l, is_money=True), delta=_fmt(rev_l - rev_c, is_money=True) if not np.isnan(rev_c) else None)
    k4.metric("Revenue (Comparison)",_fmt(rev_c, is_money=True))
    k5.metric("Occ % (Latest)",      _fmt(occ_l, is_pct=True))
    k6.metric("Occ % (Comparison)",  _fmt(occ_c, is_pct=True))

    st.markdown("---")

    # ── Chart 1: Pickup delta bars ────────────────────────────────────────────
    colors = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"] for v in pickup]
    fig1 = go.Figure(go.Bar(
        x=labels, y=pickup,
        marker_color=colors,
        text=[f"{v:+,.1f}" for v in pickup],
        textposition="outside",
        name="Pickup",
        hovertemplate="%{x}<br>Pickup: %{y:+,.1f}<extra></extra>",
    ))
    fig1.update_layout(
        title=f"{group_by} Pickup — {metric_label}  |  "
              f"{comp_snap.strftime('%d %b %Y')} → {latest_snap.strftime('%d %b %Y')}",
        xaxis_title=f"Arrival Date ({group_by})",
        yaxis_title=f"Change in {metric_label}",
        showlegend=False,
        **_DARK,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Side-by-side comparison ─────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=labels, y=vals_c,
        name=f"As of {comp_snap.strftime('%d %b %Y')}",
        marker_color=BRAND_COLORS["neutral"], opacity=0.75,
        hovertemplate="%{x}<br>%{y:,.1f}<extra></extra>",
    ))
    fig2.add_trace(go.Bar(
        x=labels, y=vals_l,
        name=f"As of {latest_snap.strftime('%d %b %Y')}",
        marker_color=BRAND_COLORS["secondary"], opacity=0.9,
        hovertemplate="%{x}<br>%{y:,.1f}<extra></extra>",
    ))
    fig2.update_layout(
        barmode="group",
        title=f"On-Books Comparison — {metric_label} by Arrival {group_by}",
        xaxis_title=f"Arrival Date ({group_by})",
        yaxis_title=metric_label,
        **_DARK,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Pickup detail table ───────────────────────────────────────────────────
    with st.expander("📋 View pickup detail table"):
        pickup_table = pd.DataFrame({
            "Arrival Period": labels,
            f"Comparison {comp_snap.strftime('%d %b %Y')}": vals_c.round(2),
            f"Latest {latest_snap.strftime('%d %b %Y')}":   vals_l.round(2),
            "Pickup (Δ)": pickup.round(2),
        })
        st.dataframe(pickup_table, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Export to Excel",
            data=exports.export_pickup(pickup_table),
            file_name="pickup_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ── Tab 3: Pace ───────────────────────────────────────────────────────────────

def tab_pace(df: pd.DataFrame) -> None:
    """
    Pace: for a selected arrival date (or range), show how the on-books
    position evolved across ALL snapshot dates over time.

    X-axis = snapshot_date (the date the file was exported)
    Y-axis = metric on-books for selected arrival dates at that snapshot
    """
    st.caption(
        "Select arrival dates below. The chart shows how bookings for those "
        "dates built up over time across all snapshot exports."
    )

    snaps = _snap_banner(df)
    if len(snaps) < 2:
        return

    available_metrics = [m for m in ["rooms_sold", "occupancy_pct", "revenue", "adr"]
                         if m in df.columns]

    c1, c2 = st.columns([3, 2])
    with c1:
        metric = st.selectbox("Metric to track", available_metrics, key="pace_metric")
    with c2:
        mode = st.radio("Date selection", ["Single date", "Date range"],
                        horizontal=True, key="pace_mode")

    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["snapshot_date", "date"])

    all_dates = sorted(df2["date"].dropna().unique())
    min_d = pd.Timestamp(all_dates[0]).date()
    max_d = pd.Timestamp(all_dates[-1]).date()

    # Default: first future date relative to latest snapshot
    latest_snap = snaps[-1]
    future_dates = [d for d in all_dates if pd.Timestamp(d) > latest_snap]
    default_date = (pd.Timestamp(future_dates[0]) if future_dates
                    else pd.Timestamp(all_dates[len(all_dates)//2])).date()

    if mode == "Single date":
        sel_date = st.date_input(
            "Arrival date", value=default_date,
            min_value=min_d, max_value=max_d, key="pace_single",
        )
        mask = df2["date"] == pd.Timestamp(sel_date)
        date_label = pd.Timestamp(sel_date).strftime("%d %b %Y")
    else:
        rng = st.date_input(
            "Arrival date range",
            value=(default_date, min(max_d, (pd.Timestamp(default_date) + pd.Timedelta(days=30)).date())),
            min_value=min_d, max_value=max_d, key="pace_range",
        )
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            s_dt, e_dt = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
        else:
            s_dt = e_dt = pd.Timestamp(rng[0])
        mask = (df2["date"] >= s_dt) & (df2["date"] <= e_dt)
        date_label = f"{s_dt.strftime('%d %b %Y')} – {e_dt.strftime('%d %b %Y')}"

    selected = df2[mask].copy()
    if selected.empty:
        st.warning("No data for the selected arrival date(s). Try a different date.")
        return

    # Aggregate by snapshot using correct RM formulas
    pace_agg = kpi_engine.rm_aggregate(selected, ["snapshot_date"]).sort_values("snapshot_date")

    if len(pace_agg) < 2 or metric not in pace_agg.columns:
        st.info("Not enough snapshots for the selected date(s). Try a broader date range.")
        return

    metric_label = metric.replace("_", " ").title()
    x = pace_agg["snapshot_date"]
    y = pace_agg[metric]

    # ── Pace curve ───────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=BRAND_COLORS["secondary"], width=2.5),
        marker=dict(size=7, color=BRAND_COLORS["accent"]),
        name=metric_label,
        hovertemplate="%{x|%d %b %Y}<br>" + metric_label + ": %{y:,.2f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(46,134,193,0.10)",
    ))

    # Annotate days-before-arrival milestones (90d, 60d, 30d, 14d, 7d)
    if mode == "Single date":
        arrival_ts = pd.Timestamp(sel_date)
        for days_before, label in [(90, "90d"), (60, "60d"), (30, "30d"), (14, "14d"), (7, "7d")]:
            milestone_date = arrival_ts - pd.Timedelta(days=days_before)
            if x.min() <= milestone_date <= x.max():
                fig.add_vline(
                    x=milestone_date.timestamp() * 1000,
                    line_dash="dot", line_color="rgba(255,255,255,0.25)",
                    annotation_text=label,
                    annotation_position="top",
                    annotation_font_color="#94a3b8",
                )

    fig.update_layout(
        title=f"Pace of {metric_label} for Arrival: {date_label}",
        xaxis_title="Snapshot Date (when file was exported)",
        yaxis_title=metric_label,
        hovermode="x unified",
        **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI summary ──────────────────────────────────────────────────────────
    first_val  = float(y.iloc[0])
    latest_val = float(y.iloc[-1])
    net_change = latest_val - first_val
    pct_change = (net_change / abs(first_val) * 100) if first_val else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"First on-books ({pd.Timestamp(x.iloc[0]).strftime('%d %b')})",
              _fmt(first_val, metric))
    k2.metric(f"Latest on-books ({pd.Timestamp(x.iloc[-1]).strftime('%d %b')})",
              _fmt(latest_val, metric))
    k3.metric("Total Change",   f"{net_change:+,.1f}")
    k4.metric("% Change",       f"{pct_change:+.1f}%" if not pd.isna(pct_change) else "N/A")

    # ── Week-over-week change ────────────────────────────────────────────────
    st.subheader("Weekly Pickup Velocity")
    st.caption("How much the on-books changed from snapshot to snapshot (7-day step).")
    step = min(7, max(1, len(pace_agg) // 5))
    pace_agg = pace_agg.copy()
    pace_agg["weekly_chg"] = y.diff(step).values
    pace_agg = pace_agg.dropna(subset=["weekly_chg"])

    if not pace_agg.empty:
        wcolors = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"]
                   for v in pace_agg["weekly_chg"]]
        fig2 = go.Figure(go.Bar(
            x=pace_agg["snapshot_date"],
            y=pace_agg["weekly_chg"],
            marker_color=wcolors,
            hovertemplate="%{x|%d %b %Y}<br>Change: %{y:+,.1f}<extra></extra>",
        ))
        fig2.update_layout(
            title=f"{step}-Step Pickup Velocity — {metric_label} for {date_label}",
            xaxis_title="Snapshot Date",
            yaxis_title=f"Change in {metric_label}",
            showlegend=False,
            **_DARK,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Multi-metric pace index (date range mode) ────────────────────────────
    if mode == "Date range":
        st.subheader("Multi-Metric Pace Index")
        st.caption("All metrics normalised to 100 at the first snapshot to compare momentum.")
        multi_metrics = [m for m in ["rooms_sold", "occupancy_pct", "revenue"] if m in df2.columns]
        if len(multi_metrics) >= 2:
            fig3 = go.Figure()
            palette = [BRAND_COLORS["secondary"], BRAND_COLORS["accent"],
                       BRAND_COLORS["success"], BRAND_COLORS["warning"]]
            for i, m in enumerate(multi_metrics):
                s = kpi_engine.rm_aggregate(selected, ["snapshot_date"]).sort_values("snapshot_date")
                if m not in s.columns:
                    continue
                base = s[m].iloc[0]
                if base and base != 0:
                    s["idx"] = s[m] / base * 100
                    fig3.add_trace(go.Scatter(
                        x=s["snapshot_date"], y=s["idx"],
                        mode="lines+markers",
                        name=m.replace("_", " ").title(),
                        line=dict(color=palette[i % len(palette)], width=2),
                    ))
            fig3.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                           annotation_text="Baseline")
            fig3.update_layout(
                title="Pace Index — 100 = First Snapshot",
                xaxis_title="Snapshot Date",
                yaxis_title="Index (100 = baseline)",
                **_DARK,
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Raw data ─────────────────────────────────────────────────────────────
    with st.expander("📋 Raw pace data"):
        show = pace_agg[["snapshot_date", metric]].copy()
        show["snapshot_date"] = show["snapshot_date"].dt.strftime("%d %b %Y")
        show.columns = ["Snapshot Date", metric_label]
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)


# ── Tab 4: Performance ────────────────────────────────────────────────────────

def tab_performance(df: pd.DataFrame) -> None:
    """Historical performance — aggregated from all snapshot data."""

    st.caption(
        "Performance uses the latest snapshot to show actuals for each arrival date. "
        "Aggregations use correct RM formulas (ADR = Σrev/Σrooms, Occ = Σsold/Σavail)."
    )

    # Use latest snapshot for performance
    snaps = kpi_engine.get_snapshots(df)
    current = kpi_engine.latest_snapshot_view(df) if snaps else df.copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce")

    _kpi_row(current)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(2), st.columns(2), st.columns(2), st.columns(2)

    # Monthly breakdown
    st.subheader("Monthly KPIs")
    monthly = kpi_engine.aggregate_period(current, "Monthly")
    if not monthly.empty:
        col1, col2 = st.columns(2)
        with col1:
            if "occupancy_pct" in monthly.columns:
                fig = go.Figure(go.Bar(
                    x=monthly["period_label"], y=monthly["occupancy_pct"],
                    marker_color=BRAND_COLORS["secondary"],
                    text=[f"{v:.1f}%" for v in monthly["occupancy_pct"]],
                    textposition="outside",
                ))
                fig.update_layout(title="Monthly Occupancy %", **_DARK)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "revenue" in monthly.columns:
                fig = go.Figure(go.Bar(
                    x=monthly["period_label"], y=monthly["revenue"],
                    marker_color=BRAND_COLORS["success"],
                    text=[f"€{v/1000:,.0f}k" for v in monthly["revenue"]],
                    textposition="outside",
                ))
                fig.update_layout(title="Monthly Revenue", **_DARK)
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "adr" in monthly.columns:
                fig = go.Figure(go.Bar(
                    x=monthly["period_label"], y=monthly["adr"],
                    marker_color=BRAND_COLORS["accent"],
                    text=[f"€{v:,.0f}" for v in monthly["adr"]],
                    textposition="outside",
                ))
                fig.update_layout(title="Monthly ADR (Σrev / Σrooms_sold)", **_DARK)
                st.plotly_chart(fig, use_container_width=True)
        with col4:
            if "revpar" in monthly.columns:
                fig = go.Figure(go.Bar(
                    x=monthly["period_label"], y=monthly["revpar"],
                    marker_color=BRAND_COLORS["warning"],
                    text=[f"€{v:,.0f}" for v in monthly["revpar"]],
                    textposition="outside",
                ))
                fig.update_layout(title="Monthly RevPAR (Σrev / Σrooms_avail)", **_DARK)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Monthly KPI Table"):
            show_cols = [c for c in ["period_label","rooms_sold","rooms_available",
                                      "occupancy_pct","adr","revpar","revenue"] if c in monthly.columns]
            show = monthly[show_cols].copy()
            show.columns = [c.replace("_"," ").title() for c in show.columns]
            st.dataframe(show.round(2), use_container_width=True, hide_index=True)

    # Weekday analysis
    st.subheader("Weekday Analysis")
    if "day_of_week" in current.columns:
        weekday = kpi_engine.weekday_performance(current)
        if not weekday.empty:
            col5, col6 = st.columns(2)
            with col5:
                if "occupancy_pct" in weekday.columns:
                    fig = go.Figure(go.Bar(
                        x=weekday["day_name"], y=weekday["occupancy_pct"],
                        marker_color=BRAND_COLORS["primary"],
                        text=[f"{v:.1f}%" for v in weekday["occupancy_pct"]],
                        textposition="outside",
                    ))
                    fig.update_layout(title="Occupancy by Weekday", **_DARK)
                    st.plotly_chart(fig, use_container_width=True)
            with col6:
                if "adr" in weekday.columns:
                    fig = go.Figure(go.Bar(
                        x=weekday["day_name"], y=weekday["adr"],
                        marker_color=BRAND_COLORS["accent"],
                        text=[f"€{v:,.0f}" for v in weekday["adr"]],
                        textposition="outside",
                    ))
                    fig.update_layout(title="ADR by Weekday", **_DARK)
                    st.plotly_chart(fig, use_container_width=True)

    # Compression nights
    st.subheader(f"High-Demand Dates (Occupancy ≥ {COMPRESSION_THRESHOLD}%)")
    comp = kpi_engine.compression_nights(current, COMPRESSION_THRESHOLD)
    if not comp.empty:
        st.metric("High-Demand Dates", len(comp))
        show_cols = [c for c in ["date","occupancy_pct","adr","revpar","revenue"] if c in comp.columns]
        st.dataframe(comp[show_cols].sort_values("date").round(2),
                     use_container_width=True, hide_index=True)
    else:
        st.info(f"No dates with occupancy ≥ {COMPRESSION_THRESHOLD}%.")


# ── Tab 5: Forecast ───────────────────────────────────────────────────────────

def tab_forecast(df: pd.DataFrame, method: str, horizon: int) -> None:
    st.subheader("Demand Forecast")

    metrics = [m for m in ["rooms_sold", "revenue", "occupancy_pct", "adr"] if m in df.columns]
    if not metrics:
        st.error("No forecastable metrics found.")
        return

    metric = st.selectbox("Metric", metrics, key="fc_metric")

    # Use the latest snapshot for forecasting
    snaps = kpi_engine.get_snapshots(df)
    source = kpi_engine.latest_snapshot_view(df) if snaps else df.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")

    # Daily series for the chosen metric (correct RM formula)
    daily = kpi_engine.rm_aggregate(source, ["date"]).sort_values("date")
    if daily.empty or metric not in daily.columns:
        st.info("Not enough data for forecast.")
        return

    with st.spinner(f"Running {method} forecast…"):
        fcast = forecasting.run_forecast(daily, method=method, metric=metric, horizon=horizon)

    if fcast is None or fcast.empty:
        st.error("Forecast failed — not enough history or missing dependency.")
        return

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily[metric],
        mode="lines", name="Actual/On-Books",
        line=dict(color=BRAND_COLORS["secondary"], width=2),
    ))
    fig_fcast = fcast[fcast["is_forecast"]] if "is_forecast" in fcast.columns else fcast
    fig.add_trace(go.Scatter(
        x=fig_fcast["date"], y=fig_fcast["forecast"],
        mode="lines", name=f"{method} Forecast",
        line=dict(color=BRAND_COLORS["accent"], width=2, dash="dash"),
    ))
    if "upper_80" in fcast.columns and "lower_80" in fcast.columns:
        fig.add_trace(go.Scatter(
            x=list(fig_fcast["date"]) + list(fig_fcast["date"][::-1]),
            y=list(fig_fcast["upper_80"]) + list(fig_fcast["lower_80"][::-1]),
            fill="toself", fillcolor="rgba(243,156,18,0.12)",
            line=dict(width=0), name="80% CI",
        ))
    fig.update_layout(
        title=f"{metric.replace('_',' ').title()} — {method} Forecast ({horizon} days)",
        xaxis_title="Date", yaxis_title=metric.replace("_", " ").title(),
        hovermode="x unified", **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy
    acc = forecasting.compute_forecast_accuracy(daily, fcast, metric)
    if acc:
        st.subheader("Forecast Accuracy (on historical data)")
        a1, a2, a3 = st.columns(3)
        a1.metric("MAPE", f"{acc.get('MAPE_%', 'N/A')}%")
        a2.metric("MAE",  f"{acc.get('MAE', 'N/A'):,.2f}")
        a3.metric("RMSE", f"{acc.get('RMSE', 'N/A'):,.2f}")

    with st.expander("📋 Forecast Table"):
        show = (fig_fcast[["date","forecast","lower_80","upper_80"]].round(2)
                if "lower_80" in fig_fcast.columns
                else fig_fcast[["date","forecast"]].round(2))
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Export Forecast",
        data=exports.export_forecast(fcast, acc),
        file_name="forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Tab 6: Anomalies ──────────────────────────────────────────────────────────

def tab_anomalies(df: pd.DataFrame) -> None:
    st.subheader("Anomaly Detection")

    source = kpi_engine.latest_snapshot_view(df) if kpi_engine.get_snapshots(df) else df.copy()
    daily = kpi_engine.rm_aggregate(source, ["date"]).sort_values("date") if not source.empty else pd.DataFrame()

    if daily.empty:
        st.info("No data available for anomaly detection.")
        return

    flagged = anomaly_detection.detect_anomalies(daily)
    summary = anomaly_detection.anomaly_summary(flagged)
    n = int(flagged["anomaly_flag"].sum()) if "anomaly_flag" in flagged.columns else 0
    st.metric("Anomalies Detected", n)

    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # Revenue with anomaly flags
    if "date" in flagged.columns and "revenue" in flagged.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flagged["date"], y=flagged["revenue"],
            mode="lines", name="Revenue",
            line=dict(color=BRAND_COLORS["secondary"]),
        ))
        anom_pts = flagged[flagged["anomaly_flag"]] if "anomaly_flag" in flagged.columns else pd.DataFrame()
        if not anom_pts.empty:
            fig.add_trace(go.Scatter(
                x=anom_pts["date"], y=anom_pts["revenue"],
                mode="markers", name="Anomaly",
                marker=dict(color=BRAND_COLORS["danger"], size=10, symbol="x"),
            ))
        fig.update_layout(title="Revenue with Anomaly Flags", **_DARK)
        st.plotly_chart(fig, use_container_width=True)

    if summary.empty:
        st.success("✅ No anomalies detected in the current data.")


# ── Main entry ────────────────────────────────────────────────────────────────

def render(
    df: pd.DataFrame,
    hotel: str,
    fc_method: str = "Prophet",
    fc_horizon: int = 90,
    file_reports: Optional[list] = None,
) -> None:
    hotel_df = df[df["hotel_name"] == hotel].copy() if "hotel_name" in df.columns else df.copy()
    hotel_df["date"] = pd.to_datetime(hotel_df["date"], errors="coerce")
    hotel_df["snapshot_date"] = pd.to_datetime(hotel_df.get("snapshot_date", pd.NaT), errors="coerce")

    if hotel_df.empty:
        st.warning(f"No data for hotel: {hotel}")
        return

    # Header
    snaps = kpi_engine.get_snapshots(hotel_df)
    latest_snap = snaps[-1] if snaps else None
    n_snaps = len(snaps)
    date_min = hotel_df["date"].min()
    date_max = hotel_df["date"].max()

    st.markdown(f"## 🏨 {hotel}")
    cols = st.columns(4)
    cols[0].metric("Records", f"{len(hotel_df):,}")
    cols[1].metric("Snapshots", f"{n_snaps}")
    cols[2].metric("Arrival dates", f"{date_min.strftime('%d %b %Y') if pd.notna(date_min) else 'N/A'}")
    cols[3].metric("→", f"{date_max.strftime('%d %b %Y') if pd.notna(date_max) else 'N/A'}")

    if latest_snap:
        st.caption(f"📅 Latest snapshot: **{latest_snap.strftime('%d %b %Y')}**  ·  "
                   f"On-books for {(date_max - latest_snap).days if pd.notna(date_max) else '?'} days ahead")

    st.markdown("---")

    tabs = st.tabs([
        "📍 Current Position",
        "📈 Pickup",
        "⏱️ Pace",
        "📊 Performance",
        "🔮 Forecast",
        "⚠️ Anomalies",
    ])

    with tabs[0]:
        tab_current_position(hotel_df, hotel)
    with tabs[1]:
        tab_pickup(hotel_df)
    with tabs[2]:
        tab_pace(hotel_df)
    with tabs[3]:
        tab_performance(hotel_df)
    with tabs[4]:
        tab_forecast(hotel_df, fc_method, fc_horizon)
    with tabs[5]:
        tab_anomalies(hotel_df)
