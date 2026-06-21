"""Portfolio dashboard — aggregate view across all hotels."""

import logging

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from modules import kpi_engine, forecasting, anomaly_detection, exports
from config.settings import BRAND_COLORS, HOTEL_COMPANY_MAP

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


# ── Company helpers ───────────────────────────────────────────────────────────

def _assign_company(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'company' column using HOTEL_COMPANY_MAP. Unmatched hotels → 'Other'."""
    if "hotel_name" not in df.columns:
        return df
    df = df.copy()
    # Case-insensitive lookup
    lookup = {k.lower(): v for k, v in HOTEL_COMPANY_MAP.items()}
    df["company"] = df["hotel_name"].str.strip().str.lower().map(lookup).fillna("Other")
    return df


def _monthly_company_table(df: pd.DataFrame, month_label: str) -> pd.DataFrame:
    """
    Build the hierarchical Firma / Hotel monthly KPI table matching the
    Excel pivot style:  hotel rows + company subtotal rows + grand total.

    Columns: Firma, Hotel, Total Rooms, Sales (€), Zimmer, ADR (€), Occ %
    """
    df = _assign_company(df)

    # Per-hotel aggregation
    hotel_agg = kpi_engine.rm_aggregate(df, ["company", "hotel_name"])
    if hotel_agg.empty:
        return pd.DataFrame()

    # Add total rooms (rooms_available mode = capacity per hotel)
    if "rooms_available" in df.columns:
        cap = (
            df.groupby("hotel_name")["rooms_available"]
            .apply(lambda s: int(s.mode().iloc[0]) if not s.dropna().empty else 0)
            .reset_index()
            .rename(columns={"rooms_available": "total_rooms"})
        )
        hotel_agg = hotel_agg.merge(cap, on="hotel_name", how="left")
    else:
        hotel_agg["total_rooms"] = 0

    rows = []
    companies = hotel_agg["company"].unique()

    for company in sorted(companies):
        grp = hotel_agg[hotel_agg["company"] == company].copy()
        first = True
        for _, h in grp.sort_values("hotel_name").iterrows():
            rs  = h.get("rooms_sold", 0)    or 0
            ra  = h.get("rooms_available", 0) or 0
            rev = h.get("revenue", 0)        or 0
            occ = (rs / ra * 100) if ra > 0 else 0.0
            adr = (rev / rs)      if rs > 0 else 0.0
            rows.append({
                "Firma":       company if first else "",
                "Hotel":       h["hotel_name"],
                "Total Rooms": int(h.get("total_rooms", 0) or 0),
                "Sales (€)":   rev,
                "Zimmer":      int(rs),
                "ADR (€)":     adr,
                "Occ %":       occ,
                "_sort":       0,
                "_is_subtotal": False,
                "_company":    company,
            })
            first = False

        # Company subtotal
        tot_rs  = grp["rooms_sold"].sum()      if "rooms_sold"      in grp.columns else 0
        tot_ra  = grp["rooms_available"].sum() if "rooms_available" in grp.columns else 0
        tot_rev = grp["revenue"].sum()          if "revenue"         in grp.columns else 0
        rows.append({
            "Firma":       f"{company} Total",
            "Hotel":       "",
            "Total Rooms": "",
            "Sales (€)":   tot_rev,
            "Zimmer":      int(tot_rs),
            "ADR (€)":     (tot_rev / tot_rs) if tot_rs > 0 else 0.0,
            "Occ %":       (tot_rs / tot_ra * 100) if tot_ra > 0 else 0.0,
            "_sort":       1,
            "_is_subtotal": True,
            "_company":    company,
        })

    # Grand total
    g_rs  = hotel_agg["rooms_sold"].sum()      if "rooms_sold"      in hotel_agg.columns else 0
    g_ra  = hotel_agg["rooms_available"].sum() if "rooms_available" in hotel_agg.columns else 0
    g_rev = hotel_agg["revenue"].sum()          if "revenue"         in hotel_agg.columns else 0
    rows.append({
        "Firma":       "Grand Total",
        "Hotel":       "",
        "Total Rooms": "",
        "Sales (€)":   g_rev,
        "Zimmer":      int(g_rs),
        "ADR (€)":     (g_rev / g_rs) if g_rs > 0 else 0.0,
        "Occ %":       (g_rs / g_ra * 100) if g_ra > 0 else 0.0,
        "_sort":       2,
        "_is_subtotal": True,
        "_company":    "zzz",
    })

    return pd.DataFrame(rows)


def _render_company_table(raw: pd.DataFrame) -> None:
    """Render the hierarchical company/hotel table with styled subtotal rows."""
    if raw.empty:
        st.info("No data available.")
        return

    display = raw.drop(columns=["_sort", "_is_subtotal", "_company"], errors="ignore").copy()

    # Format numeric columns
    for col in ["Sales (€)", "ADR (€)"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"€{x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else x
            )
    if "Occ %" in display.columns:
        display["Occ %"] = display["Occ %"].apply(
            lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and not pd.isna(x) else x
        )
    if "Zimmer" in display.columns:
        display["Zimmer"] = display["Zimmer"].apply(
            lambda x: f"{x:,}" if isinstance(x, int) else x
        )

    # Style subtotal/total rows bold
    is_subtotal = raw["_is_subtotal"].tolist()

    def _highlight(row):
        idx = row.name
        if is_subtotal[idx]:
            return ["font-weight:bold;background:#1e3a5f;color:#f0f8ff"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(_highlight, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── Tab: Company Analysis ─────────────────────────────────────────────────────

def tab_company(df: pd.DataFrame) -> None:
    st.subheader("🏢 Company (Firma) Analysis")

    snaps = kpi_engine.get_snapshots(df)
    current = kpi_engine.latest_snapshot_view(df) if snaps else df.copy()

    # ── Month selector ────────────────────────────────────────────────────────
    current = kpi_engine.add_period_col(current, "date", "Monthly")
    available_months = (
        current[["period", "period_label"]]
        .drop_duplicates()
        .sort_values("period")
    )

    if available_months.empty:
        st.info("No monthly data available.")
        return

    month_options = available_months["period_label"].tolist()
    selected_month_label = st.selectbox("📅 Select Month", month_options,
                                        index=0, key="co_month")
    selected_period = available_months.loc[
        available_months["period_label"] == selected_month_label, "period"
    ].iloc[0]

    month_df = current[current["period"] == selected_period].copy()

    st.markdown(f"#### Firm / Hotel KPIs — **{selected_month_label}**")

    # ── Hierarchical table ────────────────────────────────────────────────────
    raw_table = _monthly_company_table(month_df, selected_month_label)
    _render_company_table(raw_table)

    # Export
    if not raw_table.empty:
        export_df = raw_table.drop(columns=["_sort", "_is_subtotal", "_company"], errors="ignore")
        buf = __import__("io").BytesIO()
        export_df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.download_button(
            f"⬇️ Export {selected_month_label} Table to Excel",
            data=buf.getvalue(),
            file_name=f"company_kpi_{selected_month_label.replace(' ','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")

    # ── Company-level KPI cards ───────────────────────────────────────────────
    st.markdown("#### Company KPI Comparison")
    df_co = _assign_company(month_df)
    company_agg = kpi_engine.rm_aggregate(df_co, ["company"]).sort_values("company")

    if not company_agg.empty:
        companies = company_agg["company"].tolist()
        cols = st.columns(len(companies))
        for i, (_, row) in enumerate(company_agg.iterrows()):
            rs  = row.get("rooms_sold", 0) or 0
            ra  = row.get("rooms_available", 0) or 0
            rev = row.get("revenue", 0) or 0
            occ = (rs / ra * 100) if ra > 0 else 0.0
            adr = (rev / rs)      if rs > 0 else 0.0
            rp  = (rev / ra)      if ra > 0 else 0.0
            with cols[i]:
                st.markdown(
                    f"<div style='background:#1a2744;border-radius:10px;padding:14px;"
                    f"border:1px solid rgba(255,255,255,0.1);'>"
                    f"<p style='font-size:13px;font-weight:700;color:#60a5fa;margin:0 0 10px'>"
                    f"{row['company']}</p>"
                    f"<p style='margin:3px 0;font-size:12px;color:#94a3b8'>Revenue: "
                    f"<b style='color:#f1f5f9'>€{rev:,.0f}</b></p>"
                    f"<p style='margin:3px 0;font-size:12px;color:#94a3b8'>Rooms: "
                    f"<b style='color:#f1f5f9'>{int(rs):,}</b></p>"
                    f"<p style='margin:3px 0;font-size:12px;color:#94a3b8'>Occ: "
                    f"<b style='color:#f1f5f9'>{occ:.1f}%</b></p>"
                    f"<p style='margin:3px 0;font-size:12px;color:#94a3b8'>ADR: "
                    f"<b style='color:#f1f5f9'>€{adr:.2f}</b></p>"
                    f"<p style='margin:3px 0;font-size:12px;color:#94a3b8'>RevPAR: "
                    f"<b style='color:#f1f5f9'>€{rp:.2f}</b></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Revenue bar chart by company ──────────────────────────────────────────
    if not company_agg.empty and "revenue" in company_agg.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Bar(
                x=company_agg["company"], y=company_agg["revenue"],
                marker_color=[BRAND_COLORS["secondary"], BRAND_COLORS["accent"], BRAND_COLORS["success"]],
                text=[f"€{v:,.0f}" for v in company_agg["revenue"]],
                textposition="outside",
            ))
            fig.update_layout(title=f"Revenue by Company — {selected_month_label}",
                              xaxis_title="Company", yaxis_title="Revenue (€)", **_DARK)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if "occupancy_pct" in company_agg.columns:
                occs = [
                    (row["rooms_sold"] / row["rooms_available"] * 100)
                    if row.get("rooms_available", 0) > 0 else 0
                    for _, row in company_agg.iterrows()
                ]
                fig2 = go.Figure(go.Bar(
                    x=company_agg["company"], y=occs,
                    marker_color=[BRAND_COLORS["warning"], BRAND_COLORS["danger"], BRAND_COLORS["primary"]],
                    text=[f"{v:.1f}%" for v in occs],
                    textposition="outside",
                ))
                fig2.update_layout(title=f"Occupancy % by Company — {selected_month_label}",
                                   xaxis_title="Company", yaxis_title="Occupancy %", **_DARK)
                st.plotly_chart(fig2, use_container_width=True)

    # ── Multi-month trend per company ─────────────────────────────────────────
    st.markdown("#### Revenue Trend by Company (all months)")
    df_all = _assign_company(kpi_engine.add_period_col(
        kpi_engine.latest_snapshot_view(df) if snaps else df.copy(),
        "date", "Monthly"
    ))
    trend = kpi_engine.rm_aggregate(df_all, ["company", "period", "period_label"]).sort_values("period")
    if not trend.empty and "revenue" in trend.columns:
        fig3 = go.Figure()
        palette = [BRAND_COLORS["secondary"], BRAND_COLORS["accent"], BRAND_COLORS["success"],
                   BRAND_COLORS["warning"], BRAND_COLORS["danger"]]
        for i, (co, cdf) in enumerate(trend.groupby("company")):
            fig3.add_trace(go.Scatter(
                x=cdf["period_label"], y=cdf["revenue"],
                mode="lines+markers", name=co,
                line=dict(width=2.5, color=palette[i % len(palette)]),
                marker=dict(size=7),
            ))
        fig3.update_layout(
            title="Monthly Revenue Trend by Company",
            xaxis_title="Month", yaxis_title="Revenue (€)",
            hovermode="x unified", **_DARK,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── Tab: Year-over-Year ───────────────────────────────────────────────────────

_MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
_MONTH_NUM   = {m: i+1 for i, m in enumerate(_MONTH_ORDER)}


def tab_yoy(df: pd.DataFrame) -> None:
    """
    Year-over-Year monthly comparison.

    Uses best_view() so past arrival dates show their final on-books
    position (not just what the latest snapshot covers).
    """
    st.subheader("📅 Year-over-Year Monthly Comparison")
    st.caption(
        "Compares the same calendar month across two years. "
        "Past dates use the last available snapshot for that day."
    )

    if "date" not in df.columns:
        st.info("No date column available.")
        return

    # ── Year selector ─────────────────────────────────────────────────────────
    bv = kpi_engine.best_view(df)
    bv["date"] = pd.to_datetime(bv["date"], errors="coerce")
    bv["year"]  = bv["date"].dt.year
    bv["month"] = bv["date"].dt.month
    bv["month_abbr"] = bv["date"].dt.strftime("%b")

    available_years = sorted(bv["year"].dropna().unique().astype(int), reverse=True)
    if len(available_years) < 2:
        st.warning("Need at least 2 years of data for Year-over-Year comparison. "
                   "Load your historical files first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        year_cur = st.selectbox("Current Year",  available_years,
                                index=0, key="yoy_cur")
    with c2:
        prev_options = [y for y in available_years if y != year_cur]
        year_prev = st.selectbox("Comparison Year", prev_options,
                                 index=0, key="yoy_prev")
    with c3:
        metric = st.selectbox(
            "Primary metric",
            [m for m in ["revenue", "rooms_sold", "occupancy_pct", "adr", "revpar"]
             if m in bv.columns],
            key="yoy_metric",
        )

    metric_label = metric.replace("_", " ").title()

    # ── Aggregate by month for each year ──────────────────────────────────────
    cur_df  = bv[bv["year"] == year_cur]
    prev_df = bv[bv["year"] == year_prev]

    def _monthly(frame):
        return kpi_engine.rm_aggregate(frame, ["month", "month_abbr"]).sort_values("month")

    cur_m  = _monthly(cur_df)
    prev_m = _monthly(prev_df)

    if cur_m.empty and prev_m.empty:
        st.warning("No data found for the selected years.")
        return

    # ── KPI summary cards ─────────────────────────────────────────────────────
    def _total(frame, col):
        if frame.empty or col not in frame.columns:
            return 0.0
        if col in ("revenue", "rooms_sold", "rooms_available"):
            return float(frame[col].sum())
        # For rate metrics re-derive from totals
        rs  = frame["rooms_sold"].sum()  if "rooms_sold"      in frame.columns else 0
        ra  = frame["rooms_available"].sum() if "rooms_available" in frame.columns else 0
        rev = frame["revenue"].sum()     if "revenue"         in frame.columns else 0
        if col == "occupancy_pct": return (rs / ra * 100) if ra > 0 else 0.0
        if col == "adr":           return (rev / rs)       if rs > 0 else 0.0
        if col == "revpar":        return (rev / ra)       if ra > 0 else 0.0
        return 0.0

    def _fmt(v, col):
        if col == "occupancy_pct": return f"{v:.1f}%"
        if col in ("adr", "revpar", "revenue"): return f"€{v:,.0f}" if col == "revenue" else f"€{v:,.2f}"
        return f"{v:,.0f}"

    metrics_show = [m for m in ["revenue", "rooms_sold", "occupancy_pct", "adr", "revpar"]
                    if m in bv.columns]
    cols = st.columns(len(metrics_show))
    for col_w, m in zip(cols, metrics_show):
        v_cur  = _total(cur_m,  m)
        v_prev = _total(prev_m, m)
        chg    = ((v_cur - v_prev) / abs(v_prev) * 100) if v_prev else 0.0
        col_w.metric(
            label=m.replace("_", " ").title(),
            value=_fmt(v_cur, m),
            delta=f"{chg:+.1f}%  vs {year_prev}",
            delta_color="normal" if chg >= 0 else "inverse",
        )

    st.markdown("---")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    months = _MONTH_ORDER
    cur_vals  = []
    prev_vals = []
    for mn in months:
        mn_num = _MONTH_NUM[mn]
        cr  = cur_m[cur_m["month"]  == mn_num]
        pr  = prev_m[prev_m["month"] == mn_num]
        cur_vals.append( _total(cr,  metric) if not cr.empty  else 0.0)
        prev_vals.append(_total(pr,  metric) if not pr.empty  else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months, y=prev_vals, name=str(year_prev),
        marker_color=BRAND_COLORS["neutral"], opacity=0.80,
        text=[_fmt(v, metric) for v in prev_vals], textposition="outside",
        hovertemplate=f"%{{x}} {year_prev}<br>{metric_label}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=months, y=cur_vals, name=str(year_cur),
        marker_color=BRAND_COLORS["secondary"],
        text=[_fmt(v, metric) for v in cur_vals], textposition="outside",
        hovertemplate=f"%{{x}} {year_cur}<br>{metric_label}: %{{y:,.2f}}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title=f"{metric_label} — {year_cur} vs {year_prev} (monthly)",
        xaxis_title="Month", yaxis_title=metric_label,
        hovermode="x unified", **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── % Change line ─────────────────────────────────────────────────────────
    pct_changes = []
    for c, p in zip(cur_vals, prev_vals):
        pct_changes.append(((c - p) / abs(p) * 100) if p else 0.0)

    colors_line = [BRAND_COLORS["success"] if v >= 0 else BRAND_COLORS["danger"]
                   for v in pct_changes]
    fig2 = go.Figure()
    fig2.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
    fig2.add_trace(go.Bar(
        x=months, y=pct_changes,
        marker_color=colors_line,
        text=[f"{v:+.1f}%" for v in pct_changes], textposition="outside",
        hovertemplate="%{x}<br>YoY Change: %{y:+.1f}%<extra></extra>",
    ))
    fig2.update_layout(
        title=f"{metric_label} YoY Change % ({year_cur} vs {year_prev})",
        xaxis_title="Month", yaxis_title="Change %",
        showlegend=False, **_DARK,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Full KPI comparison table ─────────────────────────────────────────────
    st.markdown("#### Monthly KPI Table")

    kpi_cols = [m for m in ["revenue", "rooms_sold", "occupancy_pct", "adr", "revpar"]
                if m in bv.columns]

    table_rows = []
    for mn in months:
        mn_num = _MONTH_NUM[mn]
        cr  = cur_m[cur_m["month"]  == mn_num]
        pr  = prev_m[prev_m["month"] == mn_num]
        row = {"Month": mn}
        for m in kpi_cols:
            v_c = _total(cr, m)
            v_p = _total(pr, m)
            chg = ((v_c - v_p) / abs(v_p) * 100) if v_p else 0.0
            lbl = m.replace("_", " ").title()
            row[f"{lbl} {year_cur}"]  = _fmt(v_c, m)
            row[f"{lbl} {year_prev}"] = _fmt(v_p, m)
            row[f"{lbl} Δ%"]          = f"{chg:+.1f}%"
        table_rows.append(row)

    # Totals row
    tot_row = {"Month": "TOTAL / AVG"}
    for m in kpi_cols:
        v_c = _total(cur_m,  m)
        v_p = _total(prev_m, m)
        chg = ((v_c - v_p) / abs(v_p) * 100) if v_p else 0.0
        lbl = m.replace("_", " ").title()
        tot_row[f"{lbl} {year_cur}"]  = _fmt(v_c, m)
        tot_row[f"{lbl} {year_prev}"] = _fmt(v_p, m)
        tot_row[f"{lbl} Δ%"]          = f"{chg:+.1f}%"
    table_rows.append(tot_row)

    tbl_df = pd.DataFrame(table_rows)

    def _style_row(row):
        if row["Month"] == "TOTAL / AVG":
            return ["font-weight:bold;background:#1e3a5f"] * len(row)
        # Colour the Δ% columns
        styles = [""] * len(row)
        for i, col in enumerate(row.index):
            if "Δ%" in col:
                try:
                    v = float(str(row[col]).replace("%","").replace("+",""))
                    styles[i] = "color:#27ae60" if v >= 0 else "color:#e74c3c"
                except Exception:
                    pass
        return styles

    st.dataframe(
        tbl_df.style.apply(_style_row, axis=1),
        use_container_width=True, hide_index=True,
    )

    # Export
    buf = __import__("io").BytesIO()
    tbl_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    st.download_button(
        f"⬇️ Export YoY Table ({year_cur} vs {year_prev})",
        data=buf.getvalue(),
        file_name=f"yoy_{year_cur}_vs_{year_prev}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


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

    tabs = st.tabs(["📊 Overview", "🏢 Company", "📅 YoY", "🏆 Rankings", "📈 Pickup", "⏱️ Pace", "🔮 Forecast", "⚠️ Anomalies"])
    with tabs[0]: tab_overview(df)
    with tabs[1]: tab_company(df)
    with tabs[2]: tab_yoy(df)
    with tabs[3]: tab_rankings(df)
    with tabs[4]: tab_pickup(df)
    with tabs[5]: tab_pace(df)
    with tabs[6]: tab_forecast(df, fc_method, fc_horizon)
    with tabs[7]: tab_anomalies(df)
