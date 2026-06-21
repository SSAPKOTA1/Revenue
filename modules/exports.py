"""Export engine: generate Excel and CSV downloads for Streamlit."""

import io
import logging
from typing import Optional

import pandas as pd
import xlsxwriter  # noqa: F401 – needed by pandas ExcelWriter

logger = logging.getLogger(__name__)


def _to_excel_bytes(sheets: dict[str, pd.DataFrame], title: str = "Export") -> bytes:
    """
    Write multiple DataFrames to a single Excel workbook (in-memory).

    Parameters
    ----------
    sheets : dict mapping sheet_name → DataFrame
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        # Formats
        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#1B4F72", "font_color": "white",
            "border": 1, "align": "center",
        })
        num_fmt = wb.add_format({"num_format": "#,##0.00"})
        int_fmt = wb.add_format({"num_format": "#,##0"})
        pct_fmt = wb.add_format({"num_format": "0.0%"})
        date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            sheet_name = sheet_name[:31]  # Excel sheet name limit
            df = df.copy()

            # Convert period columns to string
            for col in df.columns:
                if hasattr(df[col], "dt") and hasattr(df[col].dt, "to_timestamp"):
                    try:
                        df[col] = df[col].dt.to_timestamp()
                    except Exception:
                        df[col] = df[col].astype(str)

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            # Header row formatting
            for col_num, col_name in enumerate(df.columns):
                ws.write(0, col_num, col_name, header_fmt)
                # Column widths
                col_width = max(len(str(col_name)) + 4, 12)
                ws.set_column(col_num, col_num, col_width)

        logger.info("Excel export created: %d sheets", len(sheets))
    return buf.getvalue()


def export_master(df: pd.DataFrame) -> bytes:
    """Export the cleaned master dataset to Excel."""
    return _to_excel_bytes({"Master Data": df}, "Master Data Export")


def export_kpi_summary(kpi_df: pd.DataFrame) -> bytes:
    """Export KPI summary table to Excel."""
    return _to_excel_bytes({"KPI Summary": kpi_df}, "KPI Summary")


def export_pickup(pickup_df: pd.DataFrame, summary_df: Optional[pd.DataFrame] = None) -> bytes:
    sheets: dict[str, pd.DataFrame] = {"Pickup Detail": pickup_df}
    if summary_df is not None and not summary_df.empty:
        sheets["Pickup Summary"] = summary_df
    return _to_excel_bytes(sheets, "Pickup Analysis")


def export_pace(pace_df: pd.DataFrame) -> bytes:
    return _to_excel_bytes({"Pace Analysis": pace_df}, "Pace Analysis")


def export_forecast(forecast_df: pd.DataFrame, accuracy: Optional[dict] = None) -> bytes:
    sheets: dict[str, pd.DataFrame] = {"Forecast": forecast_df}
    if accuracy:
        acc_df = pd.DataFrame([accuracy])
        sheets["Accuracy"] = acc_df
    return _to_excel_bytes(sheets, "Forecast")


def export_validation(validation: dict) -> bytes:
    """Export the validation report."""
    issues_df = pd.DataFrame(validation.get("issues", []))
    summary_df = pd.DataFrame([{
        "score": validation.get("score", 0),
        "total_rows": validation.get("total_rows", 0),
        "valid_rows": validation.get("valid_rows", 0),
        "summary": validation.get("summary", ""),
    }])
    sheets = {"Summary": summary_df, "Issues": issues_df}
    return _to_excel_bytes(sheets, "Validation Report")


def export_anomalies(anomaly_df: pd.DataFrame) -> bytes:
    return _to_excel_bytes({"Anomalies": anomaly_df}, "Anomaly Report")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
