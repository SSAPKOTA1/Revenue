"""
Hotel Revenue Management Analytics Platform
============================================
Main Streamlit application entry point.

Run:  streamlit run app.py
"""

import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# ── Ensure project root on path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    APP_TITLE, APP_ICON, APP_VERSION,
    LOG_FILE, OUTPUT_DIR,
    FORECAST_HORIZONS, FORECAST_METHODS,
)
from modules import (
    data_loader, data_validator, data_cleaner,
    kpi_engine, exports,
    hotel_dashboard, portfolio_dashboard,
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


# ── CSS ────────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_css() -> None:
    css_path = ROOT / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


_load_css()


# ── Cached data loading ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_data(root_folder: str) -> tuple[pd.DataFrame, list[dict]]:
    """Load and clean all data from the selected folder."""
    logger.info("Loading data from: %s", root_folder)
    raw_df, file_reports = data_loader.load_all_files(root_folder)
    if raw_df.empty:
        return raw_df, file_reports
    cleaned = data_cleaner.clean(raw_df)
    return cleaned, file_reports


# ── Header ────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        f"""<div class="main-header">
          <h1>{APP_ICON} {APP_TITLE}</h1>
          <p>Professional Revenue Management Analytics · v{APP_VERSION}</p>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> dict:
    """Render the sidebar and return user selections as a dict."""
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.markdown("---")

        # ── Folder selection ───────────────────────────────────────────────
        st.markdown("### 📁 Data Source")
        st.caption(
            "Enter the **main/root folder** path. "
            "The app will automatically scan all subfolders "
            "(year → month → day → hotel files)."
        )
        folder_input = st.text_input(
            "Root Data Folder",
            value=st.session_state.get("last_folder", ""),
            placeholder="/path/to/your/data/root",
            help=(
                "Absolute path to the top-level folder. "
                "Expected layout:\n"
                "  root/\n"
                "    2024/\n"
                "      January/\n"
                "        01/\n"
                "          HotelName.xlsx\n"
                "Any nesting depth is supported."
            ),
        )
        load_btn = st.button("🔄 Load / Refresh Data", use_container_width=True, type="primary")

        st.markdown("---")

        # ── Analysis mode ──────────────────────────────────────────────────
        st.markdown("### 🎯 Analysis Mode")
        mode = st.radio(
            "Mode", ["Hotel Level", "Portfolio Level"],
            label_visibility="collapsed",
        )

        # ── Hotel selector (only in hotel mode) ───────────────────────────
        selected_hotel = None
        if mode == "Hotel Level" and not df.empty and "hotel_name" in df.columns:
            hotels = sorted(df["hotel_name"].unique())
            selected_hotel = st.selectbox("🏨 Select Hotel", hotels)

        st.markdown("---")

        # ── Date range filter ──────────────────────────────────────────────
        st.markdown("### 📅 Date Range")
        date_min = date_max = None
        if not df.empty and "date" in df.columns:
            d_min = df["date"].min().date()
            d_max = df["date"].max().date()
            date_min, date_max = st.date_input(
                "Date Range",
                value=(d_min, d_max),
                min_value=d_min,
                max_value=d_max,
                label_visibility="collapsed",
            )

        st.markdown("---")

        # ── Forecast settings ──────────────────────────────────────────────
        st.markdown("### 🔮 Forecasting")
        forecast_method = st.selectbox("Method", FORECAST_METHODS)
        forecast_horizon = st.selectbox(
            "Horizon (days)", FORECAST_HORIZONS, index=1,
        )

        st.markdown("---")

        # ── Theme ─────────────────────────────────────────────────────────
        st.markdown("### 🎨 Theme")
        st.selectbox("Color Theme", ["Dark (Default)", "Midnight Blue", "Slate"], key="theme")

        st.markdown("---")
        st.caption(f"v{APP_VERSION} · Built with Streamlit")

    return {
        "folder": folder_input,
        "load": load_btn,
        "mode": mode,
        "hotel": selected_hotel,
        "date_min": date_min,
        "date_max": date_max,
        "forecast_method": forecast_method,
        "forecast_horizon": forecast_horizon,
    }


# ── Data quality panel ─────────────────────────────────────────────────────

def render_data_quality(df: pd.DataFrame, file_reports: list[dict]) -> None:
    """Show the data validation panel in an expander."""
    if df.empty:
        return

    validation = data_validator.validate_dataframe(df)
    score = validation["score"]

    grade_class = (
        "quality-excellent" if score >= 90 else
        "quality-good" if score >= 75 else
        "quality-fair" if score >= 60 else
        "quality-poor"
    )
    grade_label = (
        "Excellent" if score >= 90 else
        "Good" if score >= 75 else
        "Fair" if score >= 60 else
        "Poor"
    )

    with st.expander(f"🔍 Data Quality — Score: {score}/100 ({grade_label})", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Quality Score", f"{score}/100")
        col2.metric("Total Rows", f"{validation['total_rows']:,}")
        col3.metric("Valid Rows", f"{validation['valid_rows']:,}")

        if validation["issues"]:
            issues_df = pd.DataFrame(validation["issues"])
            st.dataframe(issues_df, use_container_width=True, hide_index=True)

        # File ingestion summary
        if file_reports:
            st.subheader("Ingestion Report")
            rep_df = pd.DataFrame(file_reports)
            cols_to_show = [c for c in ["file", "path", "hotel", "status", "rows", "error"] if c in rep_df.columns]
            st.dataframe(rep_df[cols_to_show], use_container_width=True, hide_index=True)

        # Export validation
        val_bytes = exports.export_validation(validation)
        st.download_button(
            "⬇️ Download Validation Report",
            data=val_bytes,
            file_name="validation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ── Exports tab ────────────────────────────────────────────────────────────

def render_exports_tab(df: pd.DataFrame, sel: dict) -> None:
    st.subheader("📥 Export Data")

    if df.empty:
        st.warning("Load data first before exporting.")
        return

    hotel_df = df
    if sel["hotel"] and "hotel_name" in df.columns:
        hotel_df = df[df["hotel_name"] == sel["hotel"]]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Master Dataset**")
        st.download_button(
            "⬇️ Download Excel",
            data=exports.export_master(df),
            file_name="master_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_master_xlsx",
        )
        st.download_button(
            "⬇️ Download CSV",
            data=exports.to_csv_bytes(df),
            file_name="master_data.csv",
            mime="text/csv",
            key="dl_master_csv",
        )

    with col2:
        st.markdown("**KPI Summary**")
        monthly = kpi_engine.aggregate_period(df, "M")
        if not monthly.empty:
            st.download_button(
                "⬇️ Monthly KPI Excel",
                data=exports.export_kpi_summary(monthly),
                file_name="kpi_monthly.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_kpi",
            )

    with col3:
        st.markdown("**Validation Report**")
        validation = data_validator.validate_dataframe(df)
        st.download_button(
            "⬇️ Validation Excel",
            data=exports.export_validation(validation),
            file_name="validation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_val",
        )

    st.markdown("---")

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**Anomaly Report**")
        from modules.anomaly_detection import detect_anomalies, anomaly_summary
        flagged = detect_anomalies(hotel_df)
        anom_df = anomaly_summary(flagged)
        if not anom_df.empty:
            st.download_button(
                "⬇️ Anomaly Report Excel",
                data=exports.export_anomalies(anom_df),
                file_name="anomaly_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_anom",
            )
        else:
            st.info("No anomalies to export.")

    with col5:
        st.markdown("**Forecast Export**")
        from modules.forecasting import run_forecast
        fcast_metric = [m for m in ["revpar", "revenue"] if m in hotel_df.columns]
        if fcast_metric:
            with st.spinner("Generating forecast for export…"):
                fcast_df = run_forecast(
                    hotel_df,
                    method=sel["forecast_method"],
                    metric=fcast_metric[0],
                    horizon=sel["forecast_horizon"],
                )
            if fcast_df is not None:
                st.download_button(
                    "⬇️ Forecast Excel",
                    data=exports.export_forecast(fcast_df),
                    file_name="forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_fcast",
                )
        else:
            st.info("No forecastable metrics found.")


# ── Demo data generator ────────────────────────────────────────────────────

def _generate_demo_data() -> pd.DataFrame:
    """Generate synthetic hotel data for demonstration."""
    rng = np.random.default_rng(42)
    hotels = [
        "Grand Hyatt Downtown", "Marriott Airport", "Hilton Garden Inn",
        "Sheraton Midtown", "Courtyard by Marriott", "Holiday Inn Express",
        "Radisson Blu", "Best Western Plus", "DoubleTree Resort",
        "Wyndham Grand", "Four Points by Sheraton", "Hampton Inn",
        "Comfort Suites", "Embassy Suites",
    ]
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    rows = []
    for hotel in hotels:
        capacity = rng.integers(80, 300)
        base_occ = rng.uniform(0.55, 0.85)
        base_adr = rng.uniform(80, 350)
        for d in dates:
            dow_effect = 1.15 if d.dayofweek >= 4 else 1.0
            seasonality = 1 + 0.2 * np.sin(2 * np.pi * d.dayofyear / 365)
            noise = rng.normal(1.0, 0.05)
            occ = min(base_occ * dow_effect * seasonality * noise, 0.99)
            rooms_sold = int(capacity * occ)
            adr = base_adr * dow_effect * (0.9 + 0.2 * rng.random())
            revenue = rooms_sold * adr
            rows.append({
                "date": d,
                "hotel_name": hotel,
                "rooms_sold": rooms_sold,
                "rooms_available": capacity,
                "revenue": round(revenue, 2),
                "adr": round(adr, 2),
                "occupancy_pct": round(occ * 100, 2),
                "revpar": round(revenue / capacity, 2),
            })
    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    render_header()

    # ── Session state ──────────────────────────────────────────────────────
    if "df" not in st.session_state:
        st.session_state["df"] = pd.DataFrame()
    if "file_reports" not in st.session_state:
        st.session_state["file_reports"] = []
    if "demo_loaded" not in st.session_state:
        st.session_state["demo_loaded"] = False

    df: pd.DataFrame = st.session_state["df"]
    file_reports: list[dict] = st.session_state["file_reports"]

    # ── Sidebar controls ───────────────────────────────────────────────────
    sel = render_sidebar(df)

    # ── Load demo data on first run (only when no folder provided) ────────────
    if not st.session_state["demo_loaded"] and df.empty and not sel["folder"].strip():
        with st.spinner("Loading demo data (14 hotels, 365 days each)…"):
            demo_df = _generate_demo_data()
            demo_df = data_cleaner.clean(demo_df)
            st.session_state["df"] = demo_df
            st.session_state["file_reports"] = [{"file": "demo_data", "status": "ok", "rows": len(demo_df)}]
            st.session_state["demo_loaded"] = True
            df = demo_df
        st.info(
            "📌 **Demo Mode** — Showing synthetic data for 14 hotels. "
            "Enter your root data folder path in the sidebar and click **Load / Refresh Data** to use your own files."
        )

    # ── User-triggered data load ───────────────────────────────────────────
    if sel["load"]:
        folder = sel["folder"].strip()
        if not folder:
            st.sidebar.warning("Please enter a folder path first.")
        elif not Path(folder).exists():
            st.sidebar.error(
                f"Folder not found:\n`{folder}`\n\n"
                "Please enter an absolute path to your data root folder."
            )
        else:
            st.session_state["last_folder"] = folder
            prog = st.progress(0, text="Scanning folder structure…")
            try:
                new_df, new_reports = load_data(folder)
                prog.progress(1.0, text="Processing complete.")
                if new_df.empty:
                    st.sidebar.warning("No readable hotel files found. Check the folder path and file formats (.xlsx/.xls/.csv).")
                else:
                    st.session_state["df"] = new_df
                    st.session_state["file_reports"] = new_reports
                    st.session_state["demo_loaded"] = True  # suppress demo
                    df = new_df
                    file_reports = new_reports
                    st.sidebar.success(f"✅ Loaded {len(new_df):,} rows from {len(new_reports)} files.")
            except Exception as e:
                logger.error("Load failed: %s\n%s", e, traceback.format_exc())
                st.sidebar.error(f"Load error: {e}")
            finally:
                prog.empty()

    df = st.session_state["df"]
    file_reports = st.session_state["file_reports"]

    # ── No data state ──────────────────────────────────────────────────────
    if df.empty:
        st.markdown(
            """<div style='text-align:center;padding:60px;'>
              <h2>🏨 Welcome to the Revenue Management Platform</h2>
              <p style='color:#94A3B8'>Enter the path to your hotel data folder in the sidebar and click <b>Load / Refresh Data</b>.</p>
              <p style='color:#94A3B8'>Supported formats: <b>.xlsx · .xls · .csv</b></p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── Date filter ────────────────────────────────────────────────────────
    filtered_df = df.copy()
    if sel["date_min"] and sel["date_max"] and "date" in filtered_df.columns:
        start = pd.Timestamp(sel["date_min"])
        end = pd.Timestamp(sel["date_max"])
        filtered_df = filtered_df[(filtered_df["date"] >= start) & (filtered_df["date"] <= end)]

    # ── Data quality banner ────────────────────────────────────────────────
    render_data_quality(filtered_df, file_reports)

    # ── Render mode ────────────────────────────────────────────────────────
    try:
        if sel["mode"] == "Hotel Level":
            if sel["hotel"]:
                hotel_dashboard.render(
                    filtered_df,
                    hotel=sel["hotel"],
                    forecast_method=sel["forecast_method"],
                    forecast_horizon=sel["forecast_horizon"],
                )
            else:
                st.warning("Please select a hotel from the sidebar.")
        else:
            # Portfolio tabs including Exports
            ptabs = st.tabs([
                "📊 Portfolio", "📥 Exports"
            ])
            with ptabs[0]:
                portfolio_dashboard.render(
                    filtered_df,
                    forecast_method=sel["forecast_method"],
                    forecast_horizon=sel["forecast_horizon"],
                )
            with ptabs[1]:
                render_exports_tab(filtered_df, sel)

    except Exception as e:
        logger.error("Dashboard render error: %s\n%s", e, traceback.format_exc())
        st.error(
            f"An error occurred while rendering the dashboard.\n\n"
            f"**Details:** `{e}`\n\n"
            "Check the logs for a full traceback."
        )


if __name__ == "__main__":
    main()
