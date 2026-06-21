"""
Hotel Revenue Management Analytics Platform
============================================
Run:  streamlit run app.py
"""

import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    APP_TITLE, APP_ICON, APP_VERSION,
    LOG_FILE,
    FORECAST_HORIZONS, FORECAST_METHODS,
    DEFAULT_DATA_FOLDER,
)
from modules import (
    data_loader, data_validator, data_cleaner,
    kpi_engine, exports,
    hotel_dashboard, portfolio_dashboard,
    cache_manager,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""<style>
/* ── Base ── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main { background: #080d18 !important; }

[data-testid="stSidebar"] {
    background: #0e1422 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebarContent"] { padding: 1.2rem 1rem !important; }

/* ── Typography ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
h1 { color: #f1f5f9 !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
h2 { color: #e2e8f0 !important; font-weight: 700 !important; letter-spacing: -0.01em !important; }
h3 { color: #cbd5e1 !important; font-weight: 600 !important; }
h4 { color: #94a3b8 !important; font-weight: 600 !important; }
p, li, span { color: #94a3b8; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #111827 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 3px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    margin-bottom: 20px !important;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 7px !important;
    color: #6b7280 !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 6px 16px !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(29,78,216,0.35) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"]    { display:none !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #111827 !important;
    border-radius: 10px !important;
    padding: 18px 20px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.4) !important;
}
[data-testid="stMetricLabel"] p {
    color: #6b7280 !important;
    font-size: 0.70rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.80rem !important; font-weight: 600 !important; }

/* ── Buttons ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg,#1e40af,#2563eb) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
    transition: all 0.2s !important;
    color: #fff !important;
}
[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}
[data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #e2e8f0 !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-baseweb="select"] > div {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.84rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
[data-testid="stDataFrame"] th {
    background: #111827 !important;
    color: #6b7280 !important;
    font-size: 0.70rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-width: 1px !important;
    font-size: 0.84rem !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio span,
[data-testid="stSidebar"] p {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #94a3b8 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 700 !important;
    margin: 16px 0 6px !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 1rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080d18; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p { color: #475569 !important; font-size: 0.74rem !important; }

/* ── Plotly chart wrapper ── */
.stPlotlyChart { border-radius: 12px !important; overflow: hidden !important; }
.js-plotly-plot .plotly .main-svg { border-radius: 12px !important; }
</style>""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
def _header() -> None:
    st.markdown(
        f"""<div style='display:flex;align-items:center;justify-content:space-between;
                        padding:18px 4px 16px;border-bottom:1px solid rgba(255,255,255,0.06);
                        margin-bottom:20px;'>
          <div style='display:flex;align-items:center;gap:14px;'>
            <div style='font-size:2rem;line-height:1'>{APP_ICON}</div>
            <div>
              <h1 style='margin:0;font-size:1.45rem;font-weight:800;color:#f1f5f9;
                         letter-spacing:-0.03em;line-height:1.1'>{APP_TITLE}</h1>
              <p style='margin:2px 0 0;font-size:0.73rem;color:#475569;letter-spacing:0.04em;
                        text-transform:uppercase;font-weight:500'>
                Revenue Management Analytics &nbsp;·&nbsp; v{APP_VERSION}</p>
            </div>
          </div>
          <div style='display:flex;gap:8px;align-items:center;'>
            <span style='background:#0d2145;border:1px solid #1d4ed8;border-radius:20px;
                         padding:4px 12px;font-size:0.70rem;color:#60a5fa;font-weight:600;
                         letter-spacing:0.04em;text-transform:uppercase'>LIVE</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_from_sqlite() -> pd.DataFrame:
    """Load full dataset from SQLite — cached by Streamlit for 1 hour."""
    df, _ = cache_manager.load_cache()
    return df if df is not None else pd.DataFrame()


def _ingest(folder: str, force: bool = False) -> tuple[pd.DataFrame, dict, str]:
    """
    Incremental ingest: only parse files that are new or changed.
    Returns (df, summary, source_label).
    """
    if force:
        with st.spinner("Rebuilding database from scratch…"):
            summary = cache_manager.full_rebuild(folder)
        source = "full rebuild"
    else:
        has_new, n_new = cache_manager.has_new_files(folder)
        if not has_new:
            df = _load_from_sqlite()
            return df, {"new": 0, "skipped": 0}, "cache"
        with st.spinner(f"Loading {n_new} new file(s)…"):
            summary = cache_manager.ingest_new_files(folder)
        source = f"+{summary.get('new', 0)} new files"

    _load_from_sqlite.clear()   # bust Streamlit's in-memory cache
    df = _load_from_sqlite()
    return df, summary, source


# ── Sidebar ─────────────────────────────────────────────────────────────────
def _sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        # Logo / brand
        st.markdown(
            f"<div style='text-align:center;padding:8px 0 16px;border-bottom:1px solid "
            f"rgba(255,255,255,0.05);margin-bottom:16px;'>"
            f"<span style='font-size:1.6rem'>{APP_ICON}</span>"
            f"<p style='margin:4px 0 0;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;"
            f"color:#334155;font-weight:700'>Revenue Intelligence</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Data source ────────────────────────────────────────────────────
        st.markdown("<p style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;"
                    "color:#334155;font-weight:700;margin:0 0 8px'>📁 Data Source</p>",
                    unsafe_allow_html=True)
        default_folder = st.session_state.get("last_folder", DEFAULT_DATA_FOLDER or "")
        folder = st.text_input(
            "folder",
            value=default_folder,
            placeholder=r"U:\Revenue\Data",
            label_visibility="collapsed",
            help="Full path: Year / Month / Day / HotelName.xlsx",
        )
        load_btn  = st.button("🔄 Check for New Files",  use_container_width=True, type="primary")
        force_btn = st.button("⚡ Force Full Rebuild",    use_container_width=True,
                              help="Re-parse ALL files from scratch.")

        # ── Cache info ─────────────────────────────────────────────────────
        info = cache_manager.get_cache_info()
        if info:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);"
                f"border-radius:8px;padding:10px 12px;margin:10px 0 4px;'>"
                f"<p style='margin:0;font-size:0.68rem;color:#334155;text-transform:uppercase;"
                f"letter-spacing:0.08em;font-weight:700;margin-bottom:6px'>Cache</p>"
                f"<p style='margin:0;font-size:0.75rem;color:#4b5563;line-height:1.6'>"
                f"⏱ {info.get('saved_at_display','?')}<br>"
                f"📦 {info.get('rows',0):,} rows &nbsp;·&nbsp; {info.get('cache_size_mb',0)} MB<br>"
                f"🏨 {info.get('hotels',0)} hotels &nbsp;·&nbsp; {info.get('snapshots',0)} snapshots"
                f"</p></div>",
                unsafe_allow_html=True,
            )
            if st.button("🗑️ Clear Cache", use_container_width=True):
                cache_manager.clear_cache()
                st.rerun()

        # ── Snapshot range ─────────────────────────────────────────────────
        if not df.empty:
            snaps = kpi_engine.get_snapshots(df)
            if snaps:
                st.markdown(
                    f"<div style='background:rgba(29,78,216,0.08);border:1px solid rgba(29,78,216,0.2);"
                    f"border-radius:8px;padding:8px 12px;margin:6px 0;'>"
                    f"<p style='margin:0;font-size:0.73rem;color:#3b82f6;font-weight:500'>"
                    f"📅 {len(snaps)} snapshots &nbsp;·&nbsp; "
                    f"{snaps[0].strftime('%d %b %Y')} → {snaps[-1].strftime('%d %b %Y')}"
                    f"</p></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

        # ── Mode & hotel ───────────────────────────────────────────────────
        st.markdown("<p style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;"
                    "color:#334155;font-weight:700;margin:0 0 8px'>🎯 Analysis Mode</p>",
                    unsafe_allow_html=True)
        mode = st.radio("mode", ["Hotel Level", "Portfolio Level"], label_visibility="collapsed")

        selected_hotel = None
        if mode == "Hotel Level" and not df.empty and "hotel_name" in df.columns:
            hotels = sorted(df["hotel_name"].dropna().unique())
            st.markdown("<p style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;"
                        "color:#334155;font-weight:700;margin:8px 0 4px'>🏨 Hotel</p>",
                        unsafe_allow_html=True)
            selected_hotel = st.selectbox("hotel", hotels, label_visibility="collapsed")

        st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

        # ── Date range ─────────────────────────────────────────────────────
        st.markdown("<p style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;"
                    "color:#334155;font-weight:700;margin:0 0 8px'>📅 Arrival Date Range</p>",
                    unsafe_allow_html=True)
        date_min = date_max = None
        if not df.empty and "date" in df.columns:
            d_min = df["date"].min().date()
            d_max = df["date"].max().date()
            date_range = st.date_input(
                "date_range", value=(d_min, d_max),
                min_value=d_min, max_value=d_max,
                label_visibility="collapsed",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                date_min, date_max = date_range

        st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

        # ── Forecast ───────────────────────────────────────────────────────
        st.markdown("<p style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;"
                    "color:#334155;font-weight:700;margin:0 0 8px'>🔮 Forecasting</p>",
                    unsafe_allow_html=True)
        fc_method  = st.selectbox("method",  FORECAST_METHODS,    label_visibility="collapsed")
        fc_horizon = st.selectbox("horizon", FORECAST_HORIZONS,   label_visibility="collapsed", index=1)

        st.markdown(
            f"<p style='text-align:center;margin-top:24px;font-size:0.64rem;color:#1e293b;"
            f"letter-spacing:0.06em'>v{APP_VERSION} · Revenue Analytics</p>",
            unsafe_allow_html=True,
        )

    return {
        "folder": folder,
        "load": load_btn,
        "force": force_btn,
        "mode": mode,
        "hotel": selected_hotel,
        "date_min": date_min,
        "date_max": date_max,
        "fc_method": fc_method,
        "fc_horizon": fc_horizon,
    }


# ── Data quality panel ──────────────────────────────────────────────────────
def _quality_panel(df: pd.DataFrame, file_reports: list[dict]) -> None:
    if df.empty:
        return
    validation = data_validator.validate_dataframe(df)
    score = validation["score"]
    label = ("Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Poor")

    with st.expander(f"🔍 Data Quality — Score {score}/100 ({label})", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality Score", f"{score}/100")
        c2.metric("Total Rows", f"{validation['total_rows']:,}")
        c3.metric("Valid Rows", f"{validation['valid_rows']:,}")

        if validation.get("issues"):
            st.dataframe(pd.DataFrame(validation["issues"]), use_container_width=True, hide_index=True)

        if file_reports:
            st.subheader("Ingestion Report")
            rep_df = pd.DataFrame(file_reports)
            cols = [c for c in ["file", "snapshot_date", "hotel", "status", "rows", "error"] if c in rep_df.columns]
            st.dataframe(rep_df[cols], use_container_width=True, hide_index=True)

            if "snapshot_date" in rep_df.columns:
                n_snaps = rep_df["snapshot_date"].dropna().nunique()
                if n_snaps > 0:
                    st.caption(f"📅 {n_snaps} distinct snapshot dates across {len(rep_df)} files.")
                else:
                    st.warning(
                        "⚠️ No snapshot dates parsed from folder names.  \n"
                        "Expected: `Year / Month / Day / HotelName.xlsx`  \n"
                        "e.g. `2026 / Januar / 15 / Aschaffenburg.xlsx`  \n"
                        "Fell back to file modification dates."
                    )

        try:
            val_bytes = exports.export_validation(validation)
            st.download_button(
                "⬇️ Download Validation Report", data=val_bytes,
                file_name="validation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            pass


# ── Exports tab ─────────────────────────────────────────────────────────────
def _exports_tab(df: pd.DataFrame, sel: dict) -> None:
    st.subheader("📥 Export Data")
    if df.empty:
        st.warning("Load data first.")
        return

    hotel_df = df
    if sel["hotel"] and "hotel_name" in df.columns:
        hotel_df = df[df["hotel_name"] == sel["hotel"]]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Master Dataset**")
        st.download_button("⬇️ Excel", data=exports.export_master(df),
                           file_name="master_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_master")
        st.download_button("⬇️ CSV", data=exports.to_csv_bytes(df),
                           file_name="master_data.csv", mime="text/csv", key="dl_csv")
    with c2:
        st.markdown("**Monthly KPIs**")
        monthly = kpi_engine.aggregate_period(df, "Monthly")
        if not monthly.empty:
            st.download_button("⬇️ Monthly KPI Excel", data=exports.export_kpi_summary(monthly),
                               file_name="kpi_monthly.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_kpi")
    with c3:
        st.markdown("**Validation Report**")
        v = data_validator.validate_dataframe(df)
        st.download_button("⬇️ Validation Excel", data=exports.export_validation(v),
                           file_name="validation_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_val")

    st.markdown("---")
    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**Anomaly Report**")
        try:
            from modules.anomaly_detection import detect_anomalies, anomaly_summary
            flagged = detect_anomalies(hotel_df)
            anom_df = anomaly_summary(flagged)
            if not anom_df.empty:
                st.download_button("⬇️ Anomaly Report", data=exports.export_anomalies(anom_df),
                                   file_name="anomaly_report.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="dl_anom")
            else:
                st.info("No anomalies detected.")
        except Exception as e:
            st.warning(f"Anomaly export failed: {e}")

    with c5:
        st.markdown("**Forecast Export**")
        try:
            from modules.forecasting import run_forecast
            fcast_metric = next((m for m in ["revpar", "revenue"] if m in hotel_df.columns), None)
            if fcast_metric:
                with st.spinner("Generating forecast…"):
                    fcast_df = run_forecast(hotel_df, method=sel["fc_method"],
                                            metric=fcast_metric, horizon=sel["fc_horizon"])
                if fcast_df is not None:
                    st.download_button("⬇️ Forecast Excel", data=exports.export_forecast(fcast_df),
                                       file_name="forecast.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       key="dl_fcast")
        except Exception as e:
            st.warning(f"Forecast export failed: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    _header()

    # Session state init
    for key, default in [("df", pd.DataFrame()), ("file_reports", []),
                          ("auto_load_done", False), ("last_folder", "")]:
        if key not in st.session_state:
            st.session_state[key] = default

    df: pd.DataFrame = st.session_state["df"]

    # Sidebar (uses current df for hotel list / date range / snapshot info)
    sel = _sidebar(df)

    # ── Auto-load on first startup ─────────────────────────────────────────
    if not st.session_state["auto_load_done"] and df.empty:
        auto_folder = (DEFAULT_DATA_FOLDER or "").strip()
        if auto_folder and Path(auto_folder).exists():
            try:
                new_df, summary, source = _ingest(auto_folder, force=False)
                if not new_df.empty:
                    st.session_state["df"] = new_df
                    st.session_state["file_reports"] = summary.get("file_reports", [])
                    st.session_state["last_folder"] = auto_folder
                    df = new_df
                    logger.info("Auto-loaded %d rows (%s)", len(new_df), source)
            except Exception as e:
                logger.error("Auto-load failed: %s", e)
        st.session_state["auto_load_done"] = True

    # ── User-triggered load ─────────────────────────────────────────────────
    if sel["load"] or sel["force"]:
        folder = sel["folder"].strip()
        if not folder:
            st.sidebar.warning("Enter a folder path first.")
        elif not Path(folder).exists():
            st.sidebar.error(f"Folder not found: `{folder}`")
        else:
            st.session_state["last_folder"] = folder
            try:
                new_df, summary, source = _ingest(folder, force=sel["force"])
                if new_df.empty:
                    st.sidebar.warning("No hotel files found.")
                else:
                    st.session_state["df"] = new_df
                    st.session_state["file_reports"] = summary.get("file_reports", [])
                    df = new_df
                    n_new  = summary.get("new", 0)
                    n_skip = summary.get("skipped", 0)
                    snaps  = kpi_engine.get_snapshots(new_df)
                    if source == "cache":
                        st.sidebar.info(f"✅ Already up-to-date · {len(new_df):,} rows · {len(snaps)} snapshots")
                    else:
                        st.sidebar.success(
                            f"✅ {n_new} new file(s) loaded · {n_skip} skipped (already cached)  \n"
                            f"{len(new_df):,} total rows · {len(snaps)} snapshots"
                        )
                    st.rerun()
            except Exception as e:
                logger.error("Load error: %s\n%s", e, traceback.format_exc())
                st.sidebar.error(f"Load error: {e}")

    df = st.session_state["df"]
    file_reports = st.session_state["file_reports"]

    # ── Empty state ────────────────────────────────────────────────────────
    if df.empty:
        st.markdown(
            """<div style='display:flex;flex-direction:column;align-items:center;
                           justify-content:center;padding:80px 40px;text-align:center;'>
              <div style='font-size:4rem;margin-bottom:20px'>🏨</div>
              <h2 style='color:#f1f5f9;font-size:1.8rem;font-weight:800;
                         letter-spacing:-0.03em;margin:0 0 12px'>
                Revenue Management Platform</h2>
              <p style='color:#475569;font-size:0.95rem;max-width:480px;line-height:1.7;margin:0 0 32px'>
                Enter the path to your data folder in the sidebar and click
                <strong style='color:#60a5fa'>Check for New Files</strong> to load your hotel data.
              </p>
              <div style='display:flex;gap:16px;flex-wrap:wrap;justify-content:center;'>
                <div style='background:#111827;border:1px solid rgba(255,255,255,0.06);
                            border-radius:10px;padding:16px 24px;min-width:180px;'>
                  <p style='margin:0;font-size:0.65rem;color:#334155;text-transform:uppercase;
                             letter-spacing:0.1em;font-weight:700'>Formats</p>
                  <p style='margin:6px 0 0;color:#60a5fa;font-weight:600;font-size:0.9rem'>
                    .xlsx &nbsp;·&nbsp; .xls &nbsp;·&nbsp; .csv</p>
                </div>
                <div style='background:#111827;border:1px solid rgba(255,255,255,0.06);
                            border-radius:10px;padding:16px 24px;min-width:180px;'>
                  <p style='margin:0;font-size:0.65rem;color:#334155;text-transform:uppercase;
                             letter-spacing:0.1em;font-weight:700'>Folder Structure</p>
                  <p style='margin:6px 0 0;color:#94a3b8;font-size:0.80rem;font-family:monospace'>
                    Year / Month / Day / Hotel.xlsx</p>
                </div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── Date filter ────────────────────────────────────────────────────────
    filtered = df.copy()
    if sel["date_min"] and sel["date_max"] and "date" in filtered.columns:
        filtered = filtered[
            (filtered["date"] >= pd.Timestamp(sel["date_min"])) &
            (filtered["date"] <= pd.Timestamp(sel["date_max"]))
        ]

    # ── Data quality panel ─────────────────────────────────────────────────
    _quality_panel(filtered, file_reports)

    # ── Dashboard routing ──────────────────────────────────────────────────
    try:
        if sel["mode"] == "Hotel Level":
            if not sel["hotel"]:
                st.warning("Select a hotel from the sidebar.")
                return
            hotel_dashboard.render(
                filtered,
                hotel=sel["hotel"],
                fc_method=sel["fc_method"],
                fc_horizon=sel["fc_horizon"],
            )
        else:
            tabs = st.tabs(["📊 Portfolio Dashboard", "📥 Exports"])
            with tabs[0]:
                portfolio_dashboard.render(
                    filtered,
                    fc_method=sel["fc_method"],
                    fc_horizon=sel["fc_horizon"],
                )
            with tabs[1]:
                _exports_tab(filtered, sel)

    except Exception as e:
        logger.error("Render error: %s\n%s", e, traceback.format_exc())
        st.error(
            f"Dashboard render error: `{e}`\n\n"
            "Check the log file for a full traceback."
        )


if __name__ == "__main__":
    main()
