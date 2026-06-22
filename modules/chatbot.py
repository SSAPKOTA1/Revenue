"""
Rule-based chatbot for hotel revenue analytics.

No external API or LLM needed — works entirely on the loaded DataFrame.
Uses regex + keyword matching to detect intent, then runs pandas queries.
"""

import re
import logging
from typing import Optional

import pandas as pd
import numpy as np

from modules import kpi_engine

logger = logging.getLogger(__name__)

# ── Month name → number ───────────────────────────────────────────────────────

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
                if len(k) == 3 or k in ("january","february","march","april","june","july",
                                         "august","september","october","november","december")}

_METRIC_ALIASES = {
    "revenue": "revenue", "umsatz": "revenue", "sales": "revenue",
    "occupancy": "occupancy_pct", "occ": "occupancy_pct", "belegung": "occupancy_pct",
    "adr": "adr", "average daily rate": "adr", "rate": "adr",
    "revpar": "revpar", "rev par": "revpar",
    "rooms": "rooms_sold", "zimmer": "rooms_sold", "sold": "rooms_sold",
    "available": "rooms_available",
}

_FMT = {
    "revenue":        lambda v: f"€{v:,.0f}",
    "occupancy_pct":  lambda v: f"{v:.1f}%",
    "adr":            lambda v: f"€{v:,.2f}",
    "revpar":         lambda v: f"€{v:,.2f}",
    "rooms_sold":     lambda v: f"{v:,.0f}",
    "rooms_available":lambda v: f"{v:,.0f}",
}

_EXAMPLE_QUESTIONS = [
    "What is the total revenue this year?",
    "Which hotel had the highest occupancy?",
    "Compare Mainz and Koblenz revenue",
    "How did 2025 compare to 2026?",
    "What was the occupancy in January?",
    "Top 5 hotels by ADR",
    "Which months are below last year?",
    "Any anomalies in the data?",
    "What is the average ADR across all hotels?",
    "Show me revenue by month",
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


def _extract_metric(text: str) -> Optional[str]:
    t = text.lower()
    for alias, col in _METRIC_ALIASES.items():
        if alias in t:
            return col
    return "revenue"  # sensible default


def _extract_hotels(text: str, available: list[str]) -> list[str]:
    found = []
    t = text.lower()
    for h in available:
        if h.lower() in t:
            found.append(h)
    return found


def _extract_n(text: str, default: int = 5) -> int:
    m = re.search(r'\b(\d+)\b', text)
    return int(m.group(1)) if m else default


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(val, metric: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    fmt_fn = _FMT.get(metric)
    return fmt_fn(val) if fmt_fn else str(val)


def _kpis_row(df: pd.DataFrame) -> dict:
    return kpi_engine._kpis_from_df(df)


# ── Filter helpers ────────────────────────────────────────────────────────────

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


# ── Intent handlers ───────────────────────────────────────────────────────────

def _handle_total_kpi(q: str, df: pd.DataFrame, years, months, hotels, metric) -> dict:
    """Single KPI question: 'What is the total revenue in March?'"""
    bv = kpi_engine.best_view(df)
    filtered = _filter(bv, years, months, hotels)

    if filtered.empty:
        return {"text": "No data found for that filter.", "table": None}

    k = _kpis_row(filtered)
    val = k.get(metric, 0)

    parts = []
    if hotels:
        parts.append(f"for {', '.join(hotels)}")
    if months:
        parts.append("in " + ", ".join(_MONTH_NAMES.get(m, str(m)) for m in months))
    if years:
        parts.append("(" + "/".join(str(y) for y in years) + ")")
    context = " ".join(parts) if parts else "across all hotels and dates"

    label = metric.replace("_", " ").title()
    return {
        "text": f"**{label}** {context}: **{_fmt(val, metric)}**\n\n"
                f"  - Closed (actuals): {_fmt(kpi_engine._kpis_from_df(kpi_engine.split_closed_open(filtered)[0]).get(metric,0), metric)}\n"
                f"  - Open (on-books):  {_fmt(kpi_engine._kpis_from_df(kpi_engine.split_closed_open(filtered)[1]).get(metric,0), metric)}",
        "table": None,
    }


def _handle_compare_hotels(q: str, df: pd.DataFrame, years, months, hotels, metric) -> dict:
    """Compare hotels or rank them."""
    bv = kpi_engine.best_view(df)
    filtered = _filter(bv, years, months, hotels if hotels else None)

    if filtered.empty or "hotel_name" not in filtered.columns:
        return {"text": "No hotel data available.", "table": None}

    agg = kpi_engine.rm_aggregate(filtered, ["hotel_name"])
    if agg.empty or metric not in agg.columns:
        return {"text": f"Could not compute {metric} per hotel.", "table": None}

    agg = agg.sort_values(metric, ascending=False).reset_index(drop=True)
    agg["Rank"] = agg.index + 1

    label = metric.replace("_", " ").title()
    tbl = agg[["Rank", "hotel_name"] + [c for c in ["revenue","rooms_sold","occupancy_pct","adr","revpar"] if c in agg.columns]].copy()
    tbl = tbl.rename(columns={"hotel_name": "Hotel", "revenue": "Revenue (€)",
                                "rooms_sold": "Rooms Sold", "occupancy_pct": "Occ %",
                                "adr": "ADR (€)", "revpar": "RevPAR (€)"})

    # Format
    for col, fmt_fn in [("Revenue (€)", lambda x: f"€{x:,.0f}"),
                         ("Occ %", lambda x: f"{x:.1f}%"),
                         ("ADR (€)", lambda x: f"€{x:,.2f}"),
                         ("RevPAR (€)", lambda x: f"€{x:,.2f}"),
                         ("Rooms Sold", lambda x: f"{int(x):,}")]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(lambda v: fmt_fn(v) if pd.notna(v) else "—")

    winner = tbl.iloc[0]["Hotel"]
    top_val = _fmt(agg[metric].iloc[0], metric)
    return {
        "text": f"Hotel ranking by **{label}**: **{winner}** leads with {top_val}.",
        "table": tbl,
    }


def _handle_top_n(q: str, df: pd.DataFrame, years, months, hotels, metric) -> dict:
    """Top N hotels."""
    n = _extract_n(q, 5)
    result = _handle_compare_hotels(q, df, years, months, hotels, metric)
    if result["table"] is not None:
        result["table"] = result["table"].head(n)
        result["text"] = f"**Top {n} hotels by {metric.replace('_',' ').title()}:**"
    return result


def _handle_yoy(q: str, df: pd.DataFrame, years, months, hotels, metric) -> dict:
    """Year-over-year comparison."""
    bv = kpi_engine.best_view(df)
    bv["date"] = pd.to_datetime(bv["date"], errors="coerce")
    bv["year"] = bv["date"].dt.year

    all_years = sorted(bv["year"].dropna().unique().astype(int), reverse=True)
    if len(all_years) < 2:
        return {"text": "Need at least 2 years of data for year-over-year comparison.", "table": None}

    if len(years) >= 2:
        y1, y2 = sorted(years)[-2], sorted(years)[-1]
    else:
        y2, y1 = all_years[0], all_years[1]

    def _month_agg(year):
        f = _filter(bv, [year], months, hotels)
        return kpi_engine.rm_aggregate(
            kpi_engine.add_period_col(f, "date", "Monthly"),
            ["month", "month_abbr"] if "month_abbr" in f.columns else ["period", "period_label"],
        ).sort_values("month" if "month" in kpi_engine.rm_aggregate(f, ["month"]) else "period")

    cur = _filter(bv, [y2], months, hotels)
    prev = _filter(bv, [y1], months, hotels)

    k_cur  = kpi_engine._kpis_from_df(cur)
    k_prev = kpi_engine._kpis_from_df(prev)

    v_cur  = k_cur.get(metric, 0) or 0
    v_prev = k_prev.get(metric, 0) or 0
    delta  = ((v_cur - v_prev) / abs(v_prev) * 100) if v_prev else 0

    arrow = "▲" if delta >= 0 else "▼"
    label = metric.replace("_", " ").title()

    # Monthly breakdown table
    bv2 = bv.copy()
    bv2["month_abbr"] = bv2["date"].dt.strftime("%b")
    rows = []
    for m_num in range(1, 13):
        m_name = _MONTH_NAMES.get(m_num, str(m_num))
        vc = kpi_engine._kpis_from_df(_filter(bv, [y2], [m_num], hotels)).get(metric, 0) or 0
        vp = kpi_engine._kpis_from_df(_filter(bv, [y1], [m_num], hotels)).get(metric, 0) or 0
        ch = ((vc - vp) / abs(vp) * 100) if vp else 0
        rows.append({
            "Month": m_name,
            f"{y2}": _fmt(vc, metric),
            f"{y1}": _fmt(vp, metric),
            "Δ%": f"{ch:+.1f}%",
        })

    tbl = pd.DataFrame(rows)
    return {
        "text": (f"**{label}: {y2} vs {y1}**\n\n"
                 f"  - {y2}: {_fmt(v_cur, metric)}\n"
                 f"  - {y1}: {_fmt(v_prev, metric)}\n"
                 f"  - Change: {arrow} {abs(delta):.1f}%"),
        "table": tbl,
    }


def _handle_monthly_trend(q: str, df: pd.DataFrame, years, months, hotels, metric) -> dict:
    """Monthly breakdown / trend."""
    bv = kpi_engine.best_view(df)
    filtered = _filter(bv, years if years else None, months, hotels)

    if filtered.empty:
        return {"text": "No data found.", "table": None}

    monthly = kpi_engine.rm_aggregate(
        kpi_engine.add_period_col(filtered, "date", "Monthly"),
        ["period", "period_label"],
    ).sort_values("period")

    if monthly.empty or metric not in monthly.columns:
        return {"text": f"No monthly {metric} data available.", "table": None}

    best_idx = monthly[metric].idxmax()
    best_label = monthly.loc[best_idx, "period_label"]
    best_val = monthly.loc[best_idx, metric]

    tbl = monthly[["period_label", metric]].copy()
    tbl.columns = ["Month", metric.replace("_", " ").title()]
    tbl[metric.replace("_", " ").title()] = tbl[metric.replace("_", " ").title()].apply(
        lambda v: _fmt(v, metric))

    return {
        "text": f"**{metric.replace('_',' ').title()} by month** — best month: **{best_label}** ({_fmt(best_val, metric)})",
        "table": tbl,
    }


def _handle_anomalies(q: str, df: pd.DataFrame, years, months, hotels, metric) -> dict:
    """Flag anomalies."""
    try:
        from modules import anomaly_detection
        bv = kpi_engine.best_view(df)
        filtered = _filter(bv, years, months, hotels)
        daily = kpi_engine.rm_aggregate(filtered, ["date"]).sort_values("date")
        flagged = anomaly_detection.detect_anomalies(daily)
        n = int(flagged["anomaly_flag"].sum()) if "anomaly_flag" in flagged.columns else 0
        if n == 0:
            return {"text": "No anomalies detected in the selected data.", "table": None}
        anom = flagged[flagged["anomaly_flag"] == True][
            ["date"] + [c for c in ["hotel_name","occupancy_pct","adr","revenue"] if c in flagged.columns]
        ].head(10)
        return {"text": f"Found **{n} anomalous days** in the data. Showing top {min(n,10)}:", "table": anom}
    except Exception as e:
        return {"text": f"Could not run anomaly detection: {e}", "table": None}


def _handle_below_ly(q: str, df: pd.DataFrame, years, months, hotels, metric) -> dict:
    """Which months are below last year?"""
    bv = kpi_engine.best_view(df)
    bv["date"] = pd.to_datetime(bv["date"], errors="coerce")
    bv["year"] = bv["date"].dt.year
    all_years = sorted(bv["year"].dropna().unique().astype(int), reverse=True)
    if len(all_years) < 2:
        return {"text": "Need 2 years of data.", "table": None}
    y2, y1 = all_years[0], all_years[1]

    rows = []
    for m_num in range(1, 13):
        vc = kpi_engine._kpis_from_df(_filter(bv, [y2], [m_num], hotels)).get(metric, 0) or 0
        vp = kpi_engine._kpis_from_df(_filter(bv, [y1], [m_num], hotels)).get(metric, 0) or 0
        if vp > 0 and vc < vp:
            ch = (vc - vp) / vp * 100
            rows.append({
                "Month": _MONTH_NAMES.get(m_num, str(m_num)),
                f"{y2}": _fmt(vc, metric),
                f"{y1}": _fmt(vp, metric),
                "Δ%": f"{ch:+.1f}%",
            })

    if not rows:
        return {"text": f"No months in {y2} are below {y1} for {metric.replace('_',' ')}. 🎉", "table": None}

    tbl = pd.DataFrame(rows)
    return {
        "text": f"**{len(rows)} month(s)** in {y2} are below {y1} for {metric.replace('_',' ').title()}:",
        "table": tbl,
    }


# ── Intent classifier ─────────────────────────────────────────────────────────

def _classify(q: str) -> str:
    t = q.lower()

    if any(w in t for w in ["anomal", "unusual", "strange", "outlier", "weird", "drop", "spike"]):
        return "anomaly"
    if any(w in t for w in ["below", "worse than", "lower than", "under"]) and any(
            w in t for w in ["last year", "previous year", "ly", "prior year"]):
        return "below_ly"
    if any(w in t for w in ["vs", "compare", "versus", "against", "compared"]):
        if re.search(r'\b20\d\d\b.*\b20\d\d\b', t):
            return "yoy"
        return "compare_hotels"
    if any(w in t for w in ["yoy", "year over year", "year-over-year", "last year", "prior year",
                              "previous year", "2024", "2025", "2026"]) and \
       re.search(r'\b20\d\d\b', t):
        return "yoy"
    if any(w in t for w in ["top", "best", "highest", "most", "worst", "lowest", "ranking", "rank"]):
        if any(w in t for w in ["hotel", "property"]) or re.search(r'\btop\s+\d+\b', t):
            return "top_n"
        return "compare_hotels"
    if any(w in t for w in ["by month", "monthly", "each month", "per month", "trend", "over time"]):
        return "monthly_trend"
    if any(w in t for w in ["compare hotel", "vs hotel", "hotel vs", "between hotel"]):
        return "compare_hotels"
    # Default: single KPI lookup
    return "kpi_single"


# ── Main entry point ──────────────────────────────────────────────────────────

def answer(question: str, df: pd.DataFrame) -> dict:
    """
    Answer a natural-language question about the hotel data.

    Returns:
        {"text": str, "table": pd.DataFrame | None}
    """
    if df is None or df.empty:
        return {"text": "No data loaded yet. Please load your hotel data first.", "table": None}

    q = question.strip()
    if not q:
        return {"text": "Please ask a question.", "table": None}

    # Extract entities
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2.get("date"), errors="coerce")
    all_hotels = sorted(df2["hotel_name"].dropna().unique().tolist()) \
        if "hotel_name" in df2.columns else []

    years  = _extract_years(q)
    months = _extract_months(q)
    hotels = _extract_hotels(q, all_hotels)
    metric = _extract_metric(q)

    # Use best_view for correctness
    bv = kpi_engine.best_view(df2) if kpi_engine.get_snapshots(df2) else df2

    intent = _classify(q)
    logger.debug("chatbot intent=%s years=%s months=%s hotels=%s metric=%s",
                 intent, years, months, hotels, metric)

    try:
        if intent == "anomaly":
            return _handle_anomalies(q, bv, years, months, hotels, metric)
        elif intent == "below_ly":
            return _handle_below_ly(q, bv, years, months, hotels, metric)
        elif intent == "yoy":
            return _handle_yoy(q, bv, years, months, hotels, metric)
        elif intent == "compare_hotels":
            return _handle_compare_hotels(q, bv, years, months, hotels, metric)
        elif intent == "top_n":
            return _handle_top_n(q, bv, years, months, hotels, metric)
        elif intent == "monthly_trend":
            return _handle_monthly_trend(q, bv, years, months, hotels, metric)
        else:
            return _handle_total_kpi(q, bv, years, months, hotels, metric)

    except Exception as e:
        logger.error("chatbot error: %s", e, exc_info=True)
        return {
            "text": f"Sorry, I couldn't process that question. Try rephrasing it.\n\n*Error: {e}*",
            "table": None,
        }


def example_questions() -> list[str]:
    return _EXAMPLE_QUESTIONS
