"""
Rule-based chatbot for hotel revenue analytics.

No external API or LLM needed — works entirely on the loaded DataFrame.
Each answer returns: text, optional table, optional Plotly figure.
"""

import re
import io
import logging
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from modules import kpi_engine

logger = logging.getLogger(__name__)

# ── Shared chart theme ────────────────────────────────────────────────────────

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E0E0E0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)

_C_BLUE   = "#3b82f6"
_C_GREEN  = "#10b981"
_C_AMBER  = "#f59e0b"
_C_RED    = "#ef4444"
_C_GREY   = "rgba(148,163,184,0.6)"
_PALETTE  = [_C_BLUE, _C_GREEN, _C_AMBER, "#a78bfa", "#f472b6", "#34d399", "#fb923c"]

# ── Month maps ────────────────────────────────────────────────────────────────

_MONTHS = {
    "jan": 1, "january": 1, "januar": 1,
    "feb": 2, "february": 2, "februar": 2,
    "mar": 3, "march": 3, "märz": 3,
    "apr": 4, "april": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "juni": 6,
    "jul": 7, "july": 7, "juli": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10, "oktober": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12, "dezember": 12,
}

_MONTH_NAMES = {v: k.title() for k, v in _MONTHS.items()
                if len(k) == 3 or k in (
                    "january","february","march","april","june","july",
                    "august","september","october","november","december")}

_MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

_METRIC_ALIASES = {
    "revenue": "revenue", "umsatz": "revenue", "sales": "revenue",
    "occupancy": "occupancy_pct", "occ": "occupancy_pct", "belegung": "occupancy_pct",
    "adr": "adr", "average daily rate": "adr", "rate": "adr",
    "revpar": "revpar", "rev par": "revpar",
    "rooms": "rooms_sold", "zimmer": "rooms_sold", "sold": "rooms_sold",
    "available": "rooms_available",
}

_FMT_FN = {
    "revenue":         lambda v: f"€{v:,.0f}",
    "occupancy_pct":   lambda v: f"{v:.1f}%",
    "adr":             lambda v: f"€{v:,.2f}",
    "revpar":          lambda v: f"€{v:,.2f}",
    "rooms_sold":      lambda v: f"{v:,.0f}",
    "rooms_available": lambda v: f"{v:,.0f}",
}

_EXAMPLE_QUESTIONS = [
    "Show revenue by month as a bar chart",
    "Which hotel had the highest occupancy?",
    "Compare all hotels revenue as a bar chart",
    "How did 2025 compare to 2026?",
    "Show occupancy trend over time",
    "Top 5 hotels by ADR",
    "Which months are below last year?",
    "Any anomalies in the data?",
    "Show a pie chart of revenue by hotel",
    "Compare Closed vs Open revenue",
]


# ── Entity extraction ─────────────────────────────────────────────────────────

def _extract_years(text: str) -> list[int]:
    return [int(y) for y in re.findall(r'\b(20\d\d)\b', text)]

def _extract_months(text: str) -> list[int]:
    found = []
    for name, num in _MONTHS.items():
        if re.search(rf'\b{re.escape(name)}\b', text, re.I):
            if num not in found:
                found.append(num)
    return found

def _extract_metric(text: str) -> str:
    t = text.lower()
    for alias, col in _METRIC_ALIASES.items():
        if alias in t:
            return col
    return "revenue"

def _extract_hotels(text: str, available: list[str]) -> list[str]:
    t = text.lower()
    return [h for h in available if h.lower() in t]

def _extract_n(text: str, default: int = 5) -> int:
    m = re.search(r'\b(\d+)\b', text)
    return int(m.group(1)) if m else default

def _wants_pie(text: str) -> bool:
    return any(w in text.lower() for w in ["pie", "share", "proportion", "breakdown", "distribution"])

def _wants_line(text: str) -> bool:
    return any(w in text.lower() for w in ["line", "trend", "over time", "timeline", "evolution"])


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(val, metric: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    fn = _FMT_FN.get(metric)
    return fn(val) if fn else str(val)

def _filter(df: pd.DataFrame, years=None, months=None, hotels=None) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if years:
        out = out[out["date"].dt.year.isin(years)]
    if months:
        out = out[out["date"].dt.month.isin(months)]
    if hotels:
        out = out[out["hotel_name"].isin(hotels)]
    return out


# ── Chart builders ────────────────────────────────────────────────────────────

def _bar_chart(x, y, title, x_title="", y_title="", color=_C_BLUE,
               text_labels=None, horizontal=False) -> go.Figure:
    if horizontal:
        fig = go.Figure(go.Bar(
            x=y, y=x, orientation="h",
            marker_color=color,
            text=text_labels or [str(v) for v in y],
            textposition="outside",
        ))
        fig.update_layout(title=title, xaxis_title=y_title, yaxis_title=x_title,
                          yaxis=dict(autorange="reversed"), **_DARK)
    else:
        fig = go.Figure(go.Bar(
            x=x, y=y, marker_color=color,
            text=text_labels or [str(v) for v in y],
            textposition="outside",
        ))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, **_DARK)
    return fig


def _line_chart(x, y_dict: dict, title, x_title="", y_title="") -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(y_dict.items()):
        fig.add_trace(go.Scatter(
            x=x, y=vals, mode="lines+markers", name=name,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            marker=dict(size=6),
        ))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      hovermode="x unified", **_DARK)
    return fig


def _pie_chart(labels, values, title) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=_PALETTE),
        hole=0.35,
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title=title, **_DARK)
    return fig


def _grouped_bar(months, series: dict, title, metric) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=months, y=vals, name=str(name),
            marker_color=_PALETTE[i % len(_PALETTE)],
            text=[_fmt(v, metric) for v in vals],
            textposition="outside",
        ))
    fig.update_layout(barmode="group", title=title,
                      hovermode="x unified", **_DARK)
    return fig


def _delta_bar(months, deltas, title) -> go.Figure:
    colors = [_C_GREEN if v >= 0 else _C_RED for v in deltas]
    fig = go.Figure(go.Bar(
        x=months, y=deltas,
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in deltas],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
    fig.update_layout(title=title, yaxis_title="Change %",
                      showlegend=False, **_DARK)
    return fig


def _closed_open_bar(metric, closed_val, open_val, label) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Closed (Actuals)", "Open (On-books)"],
        y=[closed_val, open_val],
        marker_color=[_C_GREEN, _C_BLUE],
        text=[_fmt(closed_val, metric), _fmt(open_val, metric)],
        textposition="outside",
    ))
    fig.update_layout(title=f"{label} — Closed vs Open", **_DARK)
    return fig


# ── Intent handlers ───────────────────────────────────────────────────────────

def _handle_total_kpi(q, df, years, months, hotels, metric) -> dict:
    filtered = _filter(df, years, months, hotels)
    if filtered.empty:
        return {"text": "No data found for that filter.", "table": None, "chart": None}

    k   = kpi_engine._kpis_from_df(filtered)
    c_k = kpi_engine._kpis_from_df(kpi_engine.split_closed_open(filtered)[0])
    o_k = kpi_engine._kpis_from_df(kpi_engine.split_closed_open(filtered)[1])
    val = k.get(metric, 0) or 0
    cv  = c_k.get(metric, 0) or 0
    ov  = o_k.get(metric, 0) or 0

    parts = []
    if hotels:  parts.append(f"for {', '.join(hotels)}")
    if months:  parts.append("in " + ", ".join(_MONTH_NAMES.get(m, str(m)) for m in months))
    if years:   parts.append("(" + "/".join(str(y) for y in years) + ")")
    context = " ".join(parts) if parts else "all hotels & dates"

    label = metric.replace("_", " ").title()
    chart = _closed_open_bar(metric, cv, ov, label)

    return {
        "text": (f"**{label}** {context}: **{_fmt(val, metric)}**\n\n"
                 f"  - ✅ Closed: {_fmt(cv, metric)}\n"
                 f"  - 📋 Open:   {_fmt(ov, metric)}"),
        "table": None,
        "chart": chart,
    }


def _handle_compare_hotels(q, df, years, months, hotels, metric) -> dict:
    filtered = _filter(df, years, months, hotels or None)
    if filtered.empty or "hotel_name" not in filtered.columns:
        return {"text": "No hotel data available.", "table": None, "chart": None}

    agg = kpi_engine.rm_aggregate(filtered, ["hotel_name"])
    if agg.empty or metric not in agg.columns:
        return {"text": f"Could not compute {metric} per hotel.", "table": None, "chart": None}

    agg = agg.sort_values(metric, ascending=False).reset_index(drop=True)
    label = metric.replace("_", " ").title()

    hotel_names = agg["hotel_name"].tolist()
    values      = agg[metric].tolist()

    if _wants_pie(q):
        chart = _pie_chart(hotel_names, values, f"{label} by Hotel")
    else:
        chart = _bar_chart(
            hotel_names, values, f"{label} by Hotel",
            x_title="Hotel", y_title=label,
            color=[_PALETTE[i % len(_PALETTE)] for i in range(len(hotel_names))],
            text_labels=[_fmt(v, metric) for v in values],
            horizontal=True,
        )

    tbl = agg[["hotel_name"] + [c for c in ["revenue","rooms_sold","occupancy_pct","adr","revpar"]
                                  if c in agg.columns]].copy()
    tbl.insert(0, "Rank", range(1, len(tbl)+1))
    tbl = tbl.rename(columns={"hotel_name":"Hotel","revenue":"Revenue (€)",
                               "rooms_sold":"Rooms Sold","occupancy_pct":"Occ %",
                               "adr":"ADR (€)","revpar":"RevPAR (€)"})
    for col, fn in [("Revenue (€)", lambda x: f"€{x:,.0f}"), ("Occ %", lambda x: f"{x:.1f}%"),
                    ("ADR (€)", lambda x: f"€{x:,.2f}"), ("RevPAR (€)", lambda x: f"€{x:,.2f}"),
                    ("Rooms Sold", lambda x: f"{int(x):,}")]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(lambda v: fn(v) if pd.notna(v) else "—")

    return {
        "text": f"Hotel ranking by **{label}**: **{tbl.iloc[0]['Hotel']}** leads.",
        "table": tbl,
        "chart": chart,
    }


def _handle_top_n(q, df, years, months, hotels, metric) -> dict:
    n = _extract_n(q, 5)
    r = _handle_compare_hotels(q, df, years, months, hotels, metric)
    if r["table"] is not None:
        r["table"] = r["table"].head(n)
        r["text"]  = f"**Top {n} hotels by {metric.replace('_',' ').title()}:**"
        # Rebuild chart for top N only
        top_hotels = r["table"]["Hotel"].tolist()
        filtered   = _filter(df, years, months, top_hotels)
        agg        = kpi_engine.rm_aggregate(filtered, ["hotel_name"])
        if not agg.empty and metric in agg.columns:
            agg = agg.sort_values(metric, ascending=False)
            r["chart"] = _bar_chart(
                agg["hotel_name"].tolist(), agg[metric].tolist(),
                f"Top {n} — {metric.replace('_',' ').title()}",
                x_title="Hotel", y_title=metric.replace("_"," ").title(),
                color=[_PALETTE[i % len(_PALETTE)] for i in range(len(agg))],
                text_labels=[_fmt(v, metric) for v in agg[metric]],
                horizontal=True,
            )
    return r


def _handle_yoy(q, df, years, months, hotels, metric) -> dict:
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2["year"] = df2["date"].dt.year
    all_years   = sorted(df2["year"].dropna().unique().astype(int), reverse=True)

    if len(all_years) < 2:
        return {"text": "Need at least 2 years of data.", "table": None, "chart": None}

    y2, y1 = (sorted(years)[-1], sorted(years)[-2]) if len(years) >= 2 else (all_years[0], all_years[1])
    label  = metric.replace("_", " ").title()

    cur_vals, prev_vals, month_labels = [], [], []
    for m_num in range(1, 13):
        m_name = _MONTH_NAMES.get(m_num, str(m_num))
        vc = kpi_engine._kpis_from_df(_filter(df2, [y2], [m_num], hotels)).get(metric, 0) or 0
        vp = kpi_engine._kpis_from_df(_filter(df2, [y1], [m_num], hotels)).get(metric, 0) or 0
        cur_vals.append(vc)
        prev_vals.append(vp)
        month_labels.append(m_name)

    v_cur  = kpi_engine._kpis_from_df(_filter(df2, [y2], months, hotels)).get(metric, 0) or 0
    v_prev = kpi_engine._kpis_from_df(_filter(df2, [y1], months, hotels)).get(metric, 0) or 0
    delta  = ((v_cur - v_prev) / abs(v_prev) * 100) if v_prev else 0
    arrow  = "▲" if delta >= 0 else "▼"

    deltas = [((c - p) / abs(p) * 100) if p else 0 for c, p in zip(cur_vals, prev_vals)]

    if _wants_line(q):
        chart = _line_chart(month_labels, {str(y2): cur_vals, str(y1): prev_vals},
                            f"{label} — {y2} vs {y1}", x_title="Month", y_title=label)
    else:
        chart = _grouped_bar(month_labels, {str(y2): cur_vals, str(y1): prev_vals},
                             f"{label} — {y2} vs {y1} (monthly)", metric)

    rows = []
    for m_name, vc, vp, ch in zip(month_labels, cur_vals, prev_vals, deltas):
        rows.append({"Month": m_name, str(y2): _fmt(vc, metric),
                     str(y1): _fmt(vp, metric), "Δ%": f"{ch:+.1f}%"})
    tbl = pd.DataFrame(rows)

    return {
        "text": (f"**{label}: {y2} vs {y1}**\n\n"
                 f"  - {y2}: {_fmt(v_cur, metric)}\n"
                 f"  - {y1}: {_fmt(v_prev, metric)}\n"
                 f"  - Change: {arrow} {abs(delta):.1f}%"),
        "table": tbl,
        "chart": chart,
    }


def _handle_monthly_trend(q, df, years, months, hotels, metric) -> dict:
    filtered = _filter(df, years or None, months, hotels)
    if filtered.empty:
        return {"text": "No data found.", "table": None, "chart": None}

    monthly = kpi_engine.rm_aggregate(
        kpi_engine.add_period_col(filtered, "date", "Monthly"),
        ["period", "period_label"],
    ).sort_values("period")

    if monthly.empty or metric not in monthly.columns:
        return {"text": f"No monthly {metric} data.", "table": None, "chart": None}

    labels = monthly["period_label"].tolist()
    vals   = monthly[metric].tolist()
    label  = metric.replace("_", " ").title()
    best   = monthly.loc[monthly[metric].idxmax(), "period_label"]

    if _wants_pie(q):
        chart = _pie_chart(labels, vals, f"{label} by Month")
    elif _wants_line(q) or metric in ("adr", "revpar", "occupancy_pct"):
        chart = _line_chart(labels, {label: vals}, f"{label} Trend",
                            x_title="Month", y_title=label)
    else:
        chart = _bar_chart(labels, vals, f"{label} by Month",
                           x_title="Month", y_title=label,
                           color=_C_BLUE, text_labels=[_fmt(v, metric) for v in vals])

    tbl = pd.DataFrame({"Month": labels, label: [_fmt(v, metric) for v in vals]})
    return {
        "text": f"**{label} by month** — best: **{best}** ({_fmt(max(vals), metric)})",
        "table": tbl,
        "chart": chart,
    }


def _handle_anomalies(q, df, years, months, hotels, metric) -> dict:
    try:
        from modules import anomaly_detection
        filtered = _filter(df, years, months, hotels)
        daily    = kpi_engine.rm_aggregate(filtered, ["date"]).sort_values("date")
        flagged  = anomaly_detection.detect_anomalies(daily)
        n = int(flagged["anomaly_flag"].sum()) if "anomaly_flag" in flagged.columns else 0

        if n == 0:
            return {"text": "No anomalies detected. ✅", "table": None, "chart": None}

        anom = flagged[flagged["anomaly_flag"] == True].head(20)
        chart = go.Figure()
        if metric in daily.columns:
            chart.add_trace(go.Scatter(
                x=daily["date"], y=daily[metric], mode="lines",
                name=metric.replace("_"," ").title(),
                line=dict(color=_C_BLUE, width=1.5),
            ))
        if metric in anom.columns:
            chart.add_trace(go.Scatter(
                x=anom["date"], y=anom[metric], mode="markers",
                name="Anomaly", marker=dict(color=_C_RED, size=10, symbol="x"),
            ))
        chart.update_layout(title=f"Anomaly Detection — {metric.replace('_',' ').title()}",
                            hovermode="x unified", **_DARK)

        cols = ["date"] + [c for c in ["occupancy_pct","adr","revenue"] if c in anom.columns]
        return {"text": f"Found **{n} anomalous days**:", "table": anom[cols].head(10), "chart": chart}

    except Exception as e:
        return {"text": f"Could not run anomaly detection: {e}", "table": None, "chart": None}


def _handle_below_ly(q, df, years, months, hotels, metric) -> dict:
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2["year"] = df2["date"].dt.year
    all_years   = sorted(df2["year"].dropna().unique().astype(int), reverse=True)
    if len(all_years) < 2:
        return {"text": "Need 2 years of data.", "table": None, "chart": None}
    y2, y1 = all_years[0], all_years[1]

    rows, below_months, cur_vals, prev_vals = [], [], [], []
    for m_num in range(1, 13):
        m_name = _MONTH_NAMES.get(m_num, str(m_num))
        vc = kpi_engine._kpis_from_df(_filter(df2, [y2], [m_num], hotels)).get(metric, 0) or 0
        vp = kpi_engine._kpis_from_df(_filter(df2, [y1], [m_num], hotels)).get(metric, 0) or 0
        if vp > 0 and vc < vp:
            ch = (vc - vp) / vp * 100
            rows.append({"Month": m_name, str(y2): _fmt(vc, metric),
                         str(y1): _fmt(vp, metric), "Δ%": f"{ch:+.1f}%"})
            below_months.append(m_name)
            cur_vals.append(vc)
            prev_vals.append(vp)

    if not rows:
        return {"text": f"No months in {y2} are below {y1} for {metric.replace('_',' ')}. 🎉",
                "table": None, "chart": None}

    chart = _grouped_bar(below_months,
                         {str(y2): cur_vals, str(y1): prev_vals},
                         f"Months below {y1} — {metric.replace('_',' ').title()}", metric)
    return {
        "text": f"**{len(rows)} month(s)** in {y2} are below {y1}:",
        "table": pd.DataFrame(rows),
        "chart": chart,
    }


def _handle_pie_hotel(q, df, years, months, hotels, metric) -> dict:
    filtered = _filter(df, years, months, hotels or None)
    agg = kpi_engine.rm_aggregate(filtered, ["hotel_name"])
    if agg.empty or metric not in agg.columns:
        return {"text": "No data.", "table": None, "chart": None}
    agg = agg.sort_values(metric, ascending=False)
    chart = _pie_chart(agg["hotel_name"].tolist(), agg[metric].tolist(),
                       f"{metric.replace('_',' ').title()} Distribution by Hotel")
    return {
        "text": f"**{metric.replace('_',' ').title()} share by hotel:**",
        "table": None,
        "chart": chart,
    }


# ── Intent classifier ─────────────────────────────────────────────────────────

def _classify(q: str) -> str:
    t = q.lower()
    if any(w in t for w in ["pie", "share", "proportion", "distribution"]) and \
       any(w in t for w in ["hotel", "property"]):
        return "pie_hotel"
    if any(w in t for w in ["anomal", "unusual", "strange", "outlier", "drop", "spike"]):
        return "anomaly"
    if any(w in t for w in ["below", "worse than", "lower than"]) and \
       any(w in t for w in ["last year", "previous year", "ly", "prior year"]):
        return "below_ly"
    if any(w in t for w in ["vs", "compare", "versus", "against"]):
        if re.search(r'\b20\d\d\b.*\b20\d\d\b', t):
            return "yoy"
        return "compare_hotels"
    if any(w in t for w in ["yoy", "year over year", "year-over-year",
                              "last year", "prior year", "previous year"]) and \
       re.search(r'\b20\d\d\b', t):
        return "yoy"
    if any(w in t for w in ["top", "best", "highest", "most", "worst", "lowest", "ranking"]):
        if any(w in t for w in ["hotel", "property"]) or re.search(r'\btop\s+\d+\b', t):
            return "top_n"
        return "compare_hotels"
    if any(w in t for w in ["by month", "monthly", "each month", "per month",
                              "trend", "over time", "line chart", "bar chart"]):
        return "monthly_trend"
    if any(w in t for w in ["compare hotel", "all hotel", "hotel comparison",
                              "each hotel", "per hotel"]):
        return "compare_hotels"
    return "kpi_single"


# ── Main entry point ──────────────────────────────────────────────────────────

def answer(question: str, df: pd.DataFrame) -> dict:
    """
    Answer a natural-language question about the hotel data.

    Returns:
        {"text": str, "table": pd.DataFrame|None, "chart": go.Figure|None}
    """
    if df is None or df.empty:
        return {"text": "No data loaded yet.", "table": None, "chart": None}

    q = question.strip()
    if not q:
        return {"text": "Please ask a question.", "table": None, "chart": None}

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2.get("date"), errors="coerce")
    all_hotels = sorted(df2["hotel_name"].dropna().unique().tolist()) \
        if "hotel_name" in df2.columns else []

    years  = _extract_years(q)
    months = _extract_months(q)
    hotels = _extract_hotels(q, all_hotels)
    metric = _extract_metric(q)

    bv = kpi_engine.best_view(df2) if kpi_engine.get_snapshots(df2) else df2

    intent = _classify(q)
    logger.debug("intent=%s years=%s months=%s hotels=%s metric=%s",
                 intent, years, months, hotels, metric)

    handlers = {
        "pie_hotel":     _handle_pie_hotel,
        "anomaly":       _handle_anomalies,
        "below_ly":      _handle_below_ly,
        "yoy":           _handle_yoy,
        "compare_hotels":_handle_compare_hotels,
        "top_n":         _handle_top_n,
        "monthly_trend": _handle_monthly_trend,
        "kpi_single":    _handle_total_kpi,
    }

    try:
        return handlers[intent](q, bv, years, months, hotels, metric)
    except Exception as e:
        logger.error("chatbot error: %s", e, exc_info=True)
        return {"text": f"Sorry, couldn't process that. Try rephrasing.\n\n*{e}*",
                "table": None, "chart": None}


def fig_to_png_bytes(fig: go.Figure) -> bytes:
    """Export a Plotly figure to PNG bytes for download."""
    try:
        return fig.to_image(format="png", width=1200, height=600, scale=2)
    except Exception:
        # kaleido not installed — fall back to HTML
        return fig.to_html(include_plotlyjs="cdn").encode()


def example_questions() -> list[str]:
    return _EXAMPLE_QUESTIONS
