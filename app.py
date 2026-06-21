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

# ── Inline CSS (avoids Windows file encoding issues) ──────────────────────
st.markdown("""<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1f2e; }
h1,h2,h3,h4 { color: #e2e8f0; }
.main-header { padding: 1rem 0 0.5rem; border-bottom: 1px solid #2d3748; margin-bottom: 1rem; }
.main-header h1 { font-size: 1.8rem; color: #60a5fa; margin: 0; }
.main-header p  { color: #94a3b8; margin: 0; font-size: 0.85rem; }
.kpi-card { background: #1e2a3a; border-radius: 8px; padding: 1rem 1.2rem;
            border-left: 3px solid #3b82f6; margin-bottom: 0.5rem; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #60a5fa; }
.kpi-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
</style>""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
def _header() -> None:
    st.markdown(
        f'<div class="main-header"><h1>{APP_ICON} {APP_TITLE}</h1>'
        f'<p>Professional Revenue Management Analytics · v{APP_VERSION}</p></div>',
        unsafe_allow_html=True,
    )


# ── Data loading ────────────────────────────────────────────────────────────
def _do_scan(folder: str) -> tuple[pd.DataFrame, list[dict]]:
    raw_df, file_reports = data_loader.load_all_files(folder)
    if raw_df.empty:
        return raw_df, file_reports
    cleaned = data_cleaner.clean(raw_df)
    return cleaned, file_reports


def _load_from_cache_or_scan(folder: str, force: bool = False) -> tuple[pd.DataFrame, list[dict], str]:
    """Returns (df, file_reports, source) where source is 'cache' or 'scan'."""
    if not force:
        is_valid, reason = cache_manager.cache_is_valid(folder)
        if is_valid:
            cached_df, _ = cache_manager.load_cache()
            if cached_df is not None:
                return cached_df, [], "cache"
    df, reports = _do_scan(folder)
    if not df.empty:
        cache_manager.save_cache(df, folder, reports)
    return df, reports, "scan"


# ── Sidebar ─────────────────────────────────────────────────────────────────
def _sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Controls")

        # ── Data source ────────────────────────────────────────────────────
        st.markdown("### 📁 Data Source")
        default_folder = st.session_state.get("last_folder", DEFAULT_DATA_FOLDER or "")
        folder = st.text_input(
            "Root Data Folder",
            value=default_folder,
            placeholder=r"U:\Your\Data\Folder",
            help=(
                "Full path to your root data folder.\n"
                "Subfolders: Year / Month / Day / HotelName.xlsx\n\n"
                r"Example: U:\FFM_ZENTRALE\Sudip\REVENUE MANAGEMENT\2026\Belegung Data\ALl itsels"
            ),
        )
        load_btn = st.button("🔄 Load / Refresh Data", use_container_width=True, type="primary")
        force_btn = st.button("⚡ Force Re-scan (bypass cache)", use_container_width=True)

        # ── Cache info ─────────────────────────────────────────────────────
        info = cache_manager.get_cache_info()
        if info:
            st.markdown(
                f"<div style='background:#1e2a3a;border-radius:6px;padding:8px 12px;"
                f"font-size:0.78rem;color:#94A3B8;margin-bottom:4px;'>"
                f"💾 Cache: {info.get('saved_at_display','?')} · "
                f"{info.get('rows',0):,} rows · {info.get('hotels',0)} hotels · "
                f"{info.get('cache_size_mb',0)} MB</div>",
                unsafe_allow_html=True,
            )
            if st.button("🗑️ Clear Cache", use_container_width=True):
                cache_manager.clear_cache()
                st.sidebar.success("Cache cleared.")
                st.rerun()

        # ── Snapshot summary ───────────────────────────────────────────────
        if not df.empty:
            snaps = kpi_engine.get_snapshots(df)
            if snaps:
                st.markdown(
                    f"<div style='background:#162032;border-radius:6px;padding:6px 12px;"
                    f"font-size:0.78rem;color:#60a5fa;'>"
                    f"📅 {len(snaps)} snapshots · "
                    f"{snaps[0].strftime('%d %b')} → {snaps[-1].strftime('%d %b %Y')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Mode & hotel ───────────────────────────────────────────────────
        st.markdown("### 🎯 Analysis Mode")
        mode = st.radio("Mode", ["Hotel Level", "Portfolio Level"], label_visibility="collapsed")

        selected_hotel = None
        if mode == "Hotel Level" and not df.empty and "hotel_name" in df.columns:
            hotels = sorted(df["hotel_name"].dropna().unique())
            selected_hotel = st.selectbox("🏨 Select Hotel", hotels)

        st.markdown("---")

        # ── Date range ─────────────────────────────────────────────────────
        st.markdown("### 📅 Date Range (Arrival)")
        date_min = date_max = None
        if not df.empty and "date" in df.columns:
            d_min = df["date"].min().date()
            d_max = df["date"].max().date()
            date_range = st.date_input(
                "Date Range",
                value=(d_min, d_max),
                min_value=d_min,
                max_value=d_max,
                label_visibility="collapsed",
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                date_min, date_max = date_range

        st.markdown("---")

        # ── Forecast ───────────────────────────────────────────────────────
        st.markdown("### 🔮 Forecasting")
        fc_method = st.selectbox("Method", FORECAST_METHODS)
        fc_horizon = st.selectbox("Horizon (days)", FORECAST_HORIZONS, index=1)

        st.markdown("---")
        st.caption(f"v{APP_VERSION} · Hotel Revenue Analytics")

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
            with st.spinner(f"Auto-loading data from:\n{auto_folder}"):
                try:
                    new_df, reports, source = _load_from_cache_or_scan(auto_folder)
                    if not new_df.empty:
                        st.session_state["df"] = new_df
                        st.session_state["file_reports"] = reports
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
            with st.spinner("Loading data…"):
                try:
                    prog = st.progress(0, text="Scanning…")
                    new_df, reports, source = _load_from_cache_or_scan(folder, force=sel["force"])
                    prog.progress(1.0, text="Done.")
                    prog.empty()
                    if new_df.empty:
                        st.sidebar.warning("No hotel files found. Check path and file formats.")
                    else:
                        st.session_state["df"] = new_df
                        st.session_state["file_reports"] = reports
                        df = new_df
                        st.session_state["file_reports"] = reports
                        snaps = kpi_engine.get_snapshots(new_df)
                        st.sidebar.success(
                            f"✅ {len(new_df):,} rows · {len(snaps)} snapshots  "
                            f"({'cache' if source == 'cache' else f'{len(reports)} files'})"
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
            "<div style='text-align:center;padding:80px;'>"
            "<h2 style='color:#60a5fa;'>🏨 Welcome to the Revenue Management Platform</h2>"
            "<p style='color:#94a3b8;font-size:1rem;'>Enter the path to your root data folder in the sidebar "
            "and click <b>Load / Refresh Data</b>.</p>"
            "<p style='color:#64748b;'>Supported formats: <b>.xlsx · .xls · .csv</b></p>"
            "<p style='color:#64748b;font-size:0.85rem;'>Expected folder structure:<br>"
            "<code>Root / 2026 / Januar / 15 / Aschaffenburg.xlsx</code></p>"
            "</div>",
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
