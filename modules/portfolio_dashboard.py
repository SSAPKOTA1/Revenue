"""Portfolio dashboard — aggregate view across all hotels."""

import logging

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from modules import kpi_engine, forecasting, anomaly_detection, exports
from config.settings import BRAND_COLORS

logger = logging.getLogger(__name__)

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E0E0E0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _card(label: str, value: str, col) -> None:
    col.markdown(
        f"""<div style='background:linear-gradient(135deg,#1a2744,#0d1b2a);
            border-radius:10px;padding:18px 22px;text-align:center;
            border:1px solid rgba(255,255,255,0.08);'>
          <p style='margin:0;font-size:11px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>{label}</p>
          <p style='margin:6px 0 0;font-size:24px;font-weight:700;color:#f1f5f9'>{value}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def _safe(v, default=0.0):
    """Return v if finite, else default."""
    try:
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    except Exception:
        return default


def _kpi_row(df: pd.DataFrame) -> None:
    # Compute totals directly from the DataFrame — never rely on pre-aggregated KPI values
    rs  = float(df["rooms_sold"].sum())      if "rooms_sold"      in df.columns else 0.0
    ra  = float(df["rooms_available"].sum()) if "rooms_available" in df.columns else 0.0
    rev = float(df["revenue"].sum())         if "revenue"         in df.columns else 0.0
    occ = (rs / ra * 100) if ra > 0 else 0.0
    adr = (rev / rs)      if rs > 0 else 0.0
    rp  = (rev / ra)      if ra > 0 else 0.0
    n_hotels = int(df["hotel_name"].nunique()) if "hotel_name" in df.columns else 0

    cols = st.columns(6)
    _card("Hotels",     str(n_hotels),            cols[0])
    _card("Occupancy",  f"{occ:.1f}%",            cols[1])
    _card("ADR",        f"€{adr:,.2f}",           cols[2])
    _card("RevPAR",     f"€{rp:,.2f}",            cols[3])
    _card("Revenue",    f"€{rev:,.0f}",           cols[4])
    _card("Rooms Sold", f"{int(rs):,}",           cols[5])


# ── Tab: Overview ─────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame) -> None:
    snaps = kpi_engine.get_snapshots(df)
    current = kpi_engine.latest_snapshot_view(df) if snaps else df.copy()
    latest_label = f" (as of {pd.Timestamp(snaps[-1]).strftime('%d %b %Y')})" if snaps else ""

    st.caption(f"Portfolio KPIs{latest_label}")
    _kpi_row(current)
    st.markdown("---")

    c1, _ = st.columns(2)
    with c1:
        group_by = st.selectbox("Group by", ["Monthly","Weekly","Quarterly","Yearly"], key="po_groupby")

    agg = kpi_engine.rm_aggregate(
        kpi_engine.add_period_col(current, "date", group_by),
        ["period", "period_label"],
    ).sort_values("period")

    if agg.empty:
        st.info("No data to display.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if "occupancy_pct" in agg.columns:
            fig = go.Figure(go.Bar(
                x=agg["period_label"], y=agg["occupancy_pct"],
                marker_color=BRAND_COLORS["secondary"],
                text=[f"{v:.1f}%" for v in agg["occupancy_pct"]],
                textposition="outside",
            ))
            fig.update_layout(title=f"Portfolio Occupancy % by {group_by}", **_DARK)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "revenue" in agg.columns:
            fig = go.Figure(go.Bar(
                x=agg["period_label"], y=agg["revenue"],
                marker_color=BRAND_COLORS["success"],
                text=[f"€{v/1000:,.0f}k" for v in agg["revenue"]],
                textposition="outside",
            ))
            fig.update_layout(title=f"Portfolio Revenue by {group_by}", **_DARK)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "adr" in agg.columns:
            fig = go.Figure(go.Scatter(
                x=agg["period_label"], y=agg["adr"],
                mode="lines+markers",
                line=dict(color=BRAND_COLORS["accent"], width=2),
                marker=dict(size=7),
            ))
            fig.update_layout(title=f"Portfolio ADR by {group_by}", **_DARK)
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if "revpar" in agg.columns:
            fig = go.Figure(go.Scatter(
                x=agg["period_label"], y=agg["revpar"],
                mode="lines+markers",
                line=dict(color=BRAND_COLORS["warning"], width=2),
                marker=dict(size=7),
            ))
            fig.update_layout(title=f"Portfolio RevPAR by {group_by}", **_DARK)
            st.plotly_chart(fig, use_container_width=True)

    # ── Monthly KPI table (always Monthly, regardless of chart grouping) ──────
    st.markdown("---")
    st.markdown("#### 📋 Monthly KPI Summary — All Hotels")

    monthly = kpi_engine.rm_aggregate(
        kpi_engine.add_period_col(current, "date", "Monthly"),
        ["period", "period_label"],
    ).sort_values("period")

    if not monthly.empty:
        # Build display table with formatted columns
        display_cols = {
            "period_label":   "Month",
            "rooms_sold":     "Rooms Sold",
            "rooms_available":"Rooms Available",
            "occupancy_pct":  "Occupancy %",
            "adr":            "ADR (€)",
            "revpar":         "RevPAR (€)",
            "revenue":        "Revenue (€)",
        }
        existing = {k: v for k, v in display_cols.items() if k in monthly.columns}
        tbl = monthly[list(existing.keys())].copy()
        tbl = tbl.rename(columns=existing)

        # Format numeric columns
        def _fmt_col(series, col_name):
            if "%" in col_name:
                return series.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
            if "€" in col_name:
                return series.map(lambda x: f"€{x:,.2f}" if pd.notna(x) else "—")
            return series.map(lambda x: f"{int(x):,}" if pd.notna(x) else "—")

        for col in tbl.columns:
            if col != "Month":
                tbl[col] = _fmt_col(tbl[col], col)

        # Totals / averages footer
        footer = {"Month": "TOTAL / AVG"}
        for raw_col, display_col in existing.items():
            if raw_col == "period_label":
                continue
            col_data = monthly[raw_col].dropna()
            if raw_col in ("rooms_sold", "rooms_available", "revenue"):
                footer[display_col] = f"{'€' if raw_col == 'revenue' else ''}{col_data.sum():,.0f}"
            elif raw_col == "occupancy_pct":
                # Correct: re-derive from totals
                tot_rs = monthly["rooms_sold"].sum()   if "rooms_sold"      in monthly.columns else 0
                tot_ra = monthly["rooms_available"].sum() if "rooms_available" in monthly.columns else 0
                footer[display_col] = f"{(tot_rs / tot_ra * 100):.1f}%" if tot_ra > 0 else "—"
            elif raw_col == "adr":
                tot_rev = monthly["revenue"].sum()     if "revenue"    in monthly.columns else 0
                tot_rs  = monthly["rooms_sold"].sum()  if "rooms_sold" in monthly.columns else 0
                footer[display_col] = f"€{(tot_rev / tot_rs):.2f}" if tot_rs > 0 else "—"
            elif raw_col == "revpar":
                tot_rev = monthly["revenue"].sum()     if "revenue"         in monthly.columns else 0
                tot_ra  = monthly["rooms_available"].sum() if "rooms_available" in monthly.columns else 0
                footer[display_col] = f"€{(tot_rev / tot_ra):.2f}" if tot_ra > 0 else "—"

        footer_df = pd.DataFrame([footer])
        full_table = pd.concat([tbl, footer_df], ignore_index=True)

        st.dataframe(
            full_table,
            use_container_width=True,
            hide_index=True,
            column_config={"Month": st.column_config.TextColumn("Month", width="medium")},
        )

        # Export
        export_tbl = monthly[list(existing.keys())].rename(columns=existing).copy()
        buf = __import__("io").BytesIO()
        export_tbl.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.download_button(
            "⬇️ Export Monthly KPI Table to Excel",
            data=buf.getvalue(),
            file_name="portfolio_monthly_kpis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ── Tab: Rankings ─────────────────────────────────────────────────────────────

def tab_rankings(df: pd.DataFrame) -> None:
    st.subheader("Hotel Rankings")
    if "hotel_name" not in df.columns:
        st.info("No hotel dimension available.")
        return

    current = kpi_engine.latest_snapshot_view(df) if kpi_engine.get_snapshots(df) else df.copy()

    rank_metric = st.selectbox(
        "Rank by",
        [m for m in ["revpar","revenue","occupancy_pct","adr","rooms_sold"] if m in current.columns],
        key="rank_metric",
    )

    ranked = kpi_engine.rm_aggregate(current, ["hotel_name"]).sort_values(rank_metric, ascending=False)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))

    fig = go.Figure(go.Bar(
        x=ranked["hotel_name"], y=ranked[rank_metric],
        marker_color=BRAND_COLORS["secondary"],
        text=[f"{v:,.1f}" for v in ranked[rank_metric]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Hotels Ranked by {rank_metric.replace('_',' ').title()}",
        xaxis_title="Hotel", yaxis_title=rank_metric.replace("_"," ").title(),
        **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [c for c in ["Rank","hotel_name","rooms_sold","rooms_available",
                                 "occupancy_pct","adr","revpar","revenue"] if c in ranked.columns]
    show = ranked[display_cols].copy()
    show.columns = [c.replace("_"," ").title() for c in show.columns]
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)

    if "revenue" in ranked.columns:
        st.subheader("Revenue Share")
        fig2 = px.pie(ranked, names="hotel_name", values="revenue",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E0E0"))
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab: Pickup ───────────────────────────────────────────────────────────────

def tab_pickup(df: pd.DataFrame) -> None:
    st.subheader("Portfolio Pickup")
    st.caption("Change in on-books across all hotels between latest and comparison snapshot.")

    snaps = kpi_engine.get_snapshots(df)
    if len(snaps) < 2:
        st.warning("Need ≥ 2 snapshots for pickup analysis.")
        return

    latest_snap = snaps[-1]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox(
            "Metric",
            [m for m in ["rooms_sold","revenue","occupancy_pct","adr"] if m in df.columns],
            key="pp_metric",
        )
    with c2:
        group_by = st.selectbox("Group by", ["Monthly","Weekly","Daily"], key="pp_groupby")
    with c3:
        days_back = st.selectbox("Compare vs.", [7, 14, 30, 60, 90], key="pp_days", index=1)

    target = latest_snap - pd.Timedelta(days=days_back)
    diffs  = [abs((s - target).days) for s in snaps[:-1]]
    comp_snap = snaps[diffs.index(min(diffs))]

    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")

    latest_view  = df2[df2["snapshot_date"] == latest_snap]
    compare_view = df2[df2["snapshot_date"] == comp_snap]

    def _agg(view):
        return kpi_engine.rm_aggregate(
            kpi_engine.add_period_col(view, "date", group_by),
            ["period", "period_label"],
        ).set_index("period")

    agg_l = _agg(latest_view)
    agg_c = _agg(compare_view)
    all_p = agg_l.index.union(agg_c.index)
    agg_l = agg_l.reindex(all_p)
    agg_c = agg_c.reindex(all_p)

    metric_label = metric.replace("_", " ").title()
    labels = agg_l["period_label"].fillna(agg_c["period_label"]).values
    vals_l = agg_l[metric].fillna(0).values if metric in agg_l.columns else np.zeros(len(all_p))
    vals_c = agg_c[metric].fillna(0).values if metric in agg_c.columns else np.zeros(len(all_p))
    pickup = vals_l - vals_c

    colors = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"] for v in pickup]
    fig = go.Figure(go.Bar(
        x=labels, y=pickup, marker_color=colors,
        text=[f"{v:+,.1f}" for v in pickup], textposition="outside",
    ))
    fig.update_layout(
        title=f"Portfolio {group_by} Pickup — {metric_label} | "
              f"{comp_snap.strftime('%d %b %Y')} → {latest_snap.strftime('%d %b %Y')}",
        xaxis_title=f"Arrival ({group_by})", yaxis_title=f"Change in {metric_label}",
        showlegend=False, **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    if "hotel_name" in df.columns:
        st.subheader("Pickup by Hotel")
        hotel_l = kpi_engine.rm_aggregate(latest_view,  ["hotel_name"])
        hotel_c = kpi_engine.rm_aggregate(compare_view, ["hotel_name"])
        hotel_m = hotel_l.merge(hotel_c, on="hotel_name", suffixes=("_l","_c"))
        if f"{metric}_l" in hotel_m.columns and f"{metric}_c" in hotel_m.columns:
            hotel_m["pickup"] = hotel_m[f"{metric}_l"] - hotel_m[f"{metric}_c"]
            hotel_m = hotel_m.sort_values("pickup", ascending=True)
            col2 = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"]
                    for v in hotel_m["pickup"]]
            fig2 = go.Figure(go.Bar(
                x=hotel_m["pickup"], y=hotel_m["hotel_name"],
                orientation="h", marker_color=col2,
                text=[f"{v:+,.1f}" for v in hotel_m["pickup"]], textposition="outside",
            ))
            fig2.update_layout(
                title=f"Pickup by Hotel — {metric_label}",
                xaxis_title=f"Change in {metric_label}", yaxis_title="Hotel",
                showlegend=False, **_DARK,
            )
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab: Pace ─────────────────────────────────────────────────────────────────

def tab_pace(df: pd.DataFrame) -> None:
    st.subheader("Portfolio Pace")
    st.caption("How did on-books for selected arrival dates evolve across all snapshot dates?")

    snaps = kpi_engine.get_snapshots(df)
    if len(snaps) < 2:
        st.warning("Need ≥ 2 snapshots for pace analysis.")
        return

    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["snapshot_date", "date"])

    all_dates = sorted(df2["date"].dropna().unique())
    min_d = pd.Timestamp(all_dates[0]).date()
    max_d = pd.Timestamp(all_dates[-1]).date()

    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox(
            "Metric",
            [m for m in ["rooms_sold","revenue","occupancy_pct"] if m in df2.columns],
            key="pp_pace_metric",
        )
    with c2:
        rng = st.date_input(
            "Arrival date range",
            value=(min_d, min(max_d, (pd.Timestamp(min_d) + pd.Timedelta(days=30)).date())),
            min_value=min_d, max_value=max_d, key="pp_pace_range",
        )

    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        s_dt, e_dt = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    else:
        s_dt = e_dt = pd.Timestamp(rng[0])

    selected = df2[(df2["date"] >= s_dt) & (df2["date"] <= e_dt)]
    if selected.empty:
        st.warning("No data for the selected date range.")
        return

    date_label  = f"{s_dt.strftime('%d %b %Y')} – {e_dt.strftime('%d %b %Y')}"
    metric_label = metric.replace("_", " ").title()

    if "hotel_name" in selected.columns:
        pace_all = kpi_engine.rm_aggregate(selected, ["snapshot_date", "hotel_name"]).sort_values("snapshot_date")
        fig = go.Figure()
        palette = px.colors.qualitative.Bold
        for i, (hotel, hdf) in enumerate(pace_all.groupby("hotel_name")):
            if metric not in hdf.columns:
                continue
            fig.add_trace(go.Scatter(
                x=hdf["snapshot_date"], y=hdf[metric],
                mode="lines+markers", name=hotel,
                line=dict(width=2, color=palette[i % len(palette)]),
            ))
    else:
        pace_all = kpi_engine.rm_aggregate(selected, ["snapshot_date"]).sort_values("snapshot_date")
        fig = go.Figure(go.Scatter(
            x=pace_all["snapshot_date"], y=pace_all[metric],
            mode="lines+markers", line=dict(color=BRAND_COLORS["secondary"], width=2),
        ))

    fig.update_layout(
        title=f"Portfolio Pace — {metric_label} for Arrivals: {date_label}",
        xaxis_title="Snapshot Date", yaxis_title=metric_label,
        hovermode="x unified", **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab: Forecast ─────────────────────────────────────────────────────────────

def tab_forecast(df: pd.DataFrame, method: str, horizon: int) -> None:
    st.subheader("Portfolio Demand Forecast")
    metrics = [m for m in ["revenue","rooms_sold","occupancy_pct","adr"] if m in df.columns]
    if not metrics:
        st.error("No forecastable metrics found.")
        return

    metric = st.selectbox("Metric", metrics, key="pf_metric")
    current = kpi_engine.latest_snapshot_view(df) if kpi_engine.get_snapshots(df) else df.copy()
    daily = kpi_engine.rm_aggregate(current, ["date"]).sort_values("date")

    if daily.empty or metric not in daily.columns:
        st.info("Not enough data.")
        return

    with st.spinner(f"Running {method} portfolio forecast…"):
        fcast = forecasting.run_forecast(daily, method=method, metric=metric, horizon=horizon)

    if fcast is None or fcast.empty:
        st.error("Forecast failed.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily[metric],
                             mode="lines", name="On-Books",
                             line=dict(color=BRAND_COLORS["secondary"], width=2)))
    fig_f = fcast[fcast["is_forecast"]] if "is_forecast" in fcast.columns else fcast
    fig.add_trace(go.Scatter(x=fig_f["date"], y=fig_f["forecast"],
                             mode="lines", name=f"{method}",
                             line=dict(color=BRAND_COLORS["accent"], width=2, dash="dash")))
    if "upper_80" in fig_f.columns:
        fig.add_trace(go.Scatter(
            x=list(fig_f["date"]) + list(fig_f["date"][::-1]),
            y=list(fig_f["upper_80"]) + list(fig_f["lower_80"][::-1]),
            fill="toself", fillcolor="rgba(243,156,18,0.12)",
            line=dict(width=0), name="80% CI",
        ))
    fig.update_layout(
        title=f"Portfolio {metric.replace('_',' ').title()} — {method} Forecast",
        xaxis_title="Date", yaxis_title=metric.replace("_", " ").title(),
        hovermode="x unified", **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab: Anomalies ────────────────────────────────────────────────────────────

def tab_anomalies(df: pd.DataFrame) -> None:
    st.subheader("Portfolio Anomaly Detection")
    current = kpi_engine.latest_snapshot_view(df) if kpi_engine.get_snapshots(df) else df.copy()
    daily = kpi_engine.rm_aggregate(current, ["date"]).sort_values("date")

    flagged = anomaly_detection.detect_anomalies(daily)
    summary = anomaly_detection.anomaly_summary(flagged)
    n = int(flagged["anomaly_flag"].sum()) if "anomaly_flag" in flagged.columns else 0
    st.metric("Portfolio Anomalies", n)

    if not summary.empty:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.success("No portfolio-level anomalies detected.")


# ── Main entry ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame, fc_method: str = "Prophet", fc_horizon: int = 90) -> None:
    if df.empty:
        st.warning("No portfolio data available.")
        return

    snaps = kpi_engine.get_snapshots(df)
    latest_label = (f" · Latest snapshot: {pd.Timestamp(snaps[-1]).strftime('%d %b %Y')}"
                    if snaps else "")
    n_hotels = df["hotel_name"].nunique() if "hotel_name" in df.columns else "?"
    date_min = df["date"].min().strftime("%d %b %Y") if "date" in df.columns and df["date"].notna().any() else "?"
    date_max = df["date"].max().strftime("%d %b %Y") if "date" in df.columns and df["date"].notna().any() else "?"

    st.markdown(f"## 📊 Portfolio — {n_hotels} Hotels")
    st.caption(f"Arrival dates: {date_min} → {date_max}  ·  {len(df):,} records  ·  "
               f"{len(snaps)} snapshots{latest_label}")

    tabs = st.tabs(["📊 Overview", "🏆 Rankings", "📈 Pickup", "⏱️ Pace", "🔮 Forecast", "⚠️ Anomalies"])
    with tabs[0]: tab_overview(df)
    with tabs[1]: tab_rankings(df)
    with tabs[2]: tab_pickup(df)
    with tabs[3]: tab_pace(df)
    with tabs[4]: tab_forecast(df, fc_method, fc_horizon)
    with tabs[5]: tab_anomalies(df)
