"""Data ingestion engine: recursively scan folders, read Excel/CSV files, merge into master DataFrame."""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import COLUMN_MAP, NUMERIC_COLS

logger = logging.getLogger(__name__)


def _normalise_col(col: str) -> str:
    """Lowercase and strip a column name for fuzzy matching."""
    return re.sub(r"\s+", " ", str(col).strip().lower())


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to canonical internal names using COLUMN_MAP."""
    rename = {}
    for raw in df.columns:
        key = _normalise_col(raw)
        if key in COLUMN_MAP:
            canonical = COLUMN_MAP[key]
            if canonical not in rename.values():  # avoid double-mapping
                rename[raw] = canonical
    df = df.rename(columns=rename)
    logger.debug("Column mapping applied: %s", rename)
    return df


def _infer_hotel_name(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Add hotel_name column if missing, inferring from filename."""
    if "hotel_name" not in df.columns:
        # Try to derive from filename: strip extension and common suffixes
        stem = filepath.stem
        stem = re.sub(r"[_\-]+", " ", stem).strip()
        stem = re.sub(r"\s*(forecast|data|report|daily|monthly)\s*", " ", stem, flags=re.I).strip()
        stem = stem.title() if stem else filepath.stem.replace("_", " ").title()
        df["hotel_name"] = stem
        logger.info("Hotel name inferred from filename: '%s'", stem)
    return df


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float, replacing non-parseable values with NaN."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[,$% ]", "", regex=True),
                errors="coerce",
            )
    return df


def _parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Parse date column with multiple format fallbacks."""
    if date_col not in df.columns:
        return df
    raw = df[date_col].astype(str)
    parsed = pd.to_datetime(raw, infer_datetime_format=True, errors="coerce")
    # Second pass: try common explicit formats
    mask = parsed.isna()
    if mask.any():
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d", "%d %b %Y", "%b %d %Y"):
            parsed[mask] = pd.to_datetime(raw[mask], format=fmt, errors="coerce")
            mask = parsed.isna()
            if not mask.any():
                break
    df[date_col] = parsed
    invalid = mask.sum()
    if invalid:
        logger.warning("%d unparseable date values set to NaT in '%s'", invalid, date_col)
    return df


def _read_single_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Read a single Excel or CSV file and return a raw DataFrame.
    Returns None on failure.
    """
    try:
        suffix = filepath.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            # Try each sheet; take the first with recognisable columns
            xl = pd.ExcelFile(filepath, engine="openpyxl" if suffix == ".xlsx" else "xlrd")
            frames = []
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet, dtype=str)
                    df = df.dropna(how="all").dropna(axis=1, how="all")
                    if df.empty:
                        continue
                    df["_source_sheet"] = sheet
                    frames.append(df)
                except Exception as e:
                    logger.warning("Sheet '%s' in '%s' failed: %s", sheet, filepath.name, e)
            if not frames:
                return None
            df = pd.concat(frames, ignore_index=True)
        elif suffix == ".csv":
            df = pd.read_csv(filepath, dtype=str, on_bad_lines="skip")
        else:
            return None

        df["_source_file"] = str(filepath)
        logger.info("Loaded '%s' — %d rows, %d cols", filepath.name, len(df), len(df.columns))
        return df

    except Exception as e:
        logger.error("Failed to read '%s': %s", filepath, e)
        return None


def scan_folder(root: str | Path) -> list[Path]:
    """Recursively find all Excel and CSV files under root."""
    root = Path(root)
    found = list(root.rglob("*.xlsx")) + list(root.rglob("*.xls")) + list(root.rglob("*.csv"))
    logger.info("Found %d data files under '%s'", len(found), root)
    return found


def load_all_files(
    root: str | Path,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Scan root folder, read all files, map columns, coerce types.

    Returns
    -------
    master_df : pd.DataFrame
        Merged, partially cleaned DataFrame with canonical columns.
    file_reports : list[dict]
        Per-file ingestion metadata for the UI.
    """
    files = scan_folder(root)
    frames: list[pd.DataFrame] = []
    file_reports: list[dict] = []

    for i, fp in enumerate(files):
        report: dict = {"file": fp.name, "path": str(fp), "status": "ok", "rows": 0, "error": ""}
        df = _read_single_file(fp)
        if df is None:
            report["status"] = "error"
            report["error"] = "Could not read file"
            file_reports.append(report)
            if progress_callback:
                progress_callback((i + 1) / len(files))
            continue

        df = _map_columns(df)
        df = _infer_hotel_name(df, fp)
        df = _parse_dates(df)
        df = _coerce_numerics(df)
        # Drop completely blank rows after coercion
        df = df.dropna(how="all")

        report["rows"] = len(df)
        report["columns"] = list(df.columns)
        file_reports.append(report)
        frames.append(df)

        if progress_callback:
            progress_callback((i + 1) / len(files))

    if not frames:
        logger.warning("No valid data loaded from '%s'", root)
        return pd.DataFrame(), file_reports

    master = pd.concat(frames, ignore_index=True)
    master = master.drop_duplicates()

    # Derive computed columns if possible
    master = _derive_missing_kpis(master)

    logger.info("Master DataFrame: %d rows, %d cols", len(master), len(master.columns))
    return master, file_reports


def _derive_missing_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute occupancy_pct, ADR, revpar from raw columns when missing."""
    has = lambda c: c in df.columns  # noqa: E731

    # Occupancy %
    if not has("occupancy_pct") and has("rooms_sold") and has("rooms_available"):
        df["occupancy_pct"] = np.where(
            df["rooms_available"] > 0,
            (df["rooms_sold"] / df["rooms_available"]) * 100,
            np.nan,
        )

    # ADR
    if not has("adr") and has("revenue") and has("rooms_sold"):
        df["adr"] = np.where(
            df["rooms_sold"] > 0,
            df["revenue"] / df["rooms_sold"],
            np.nan,
        )

    # RevPAR
    if not has("revpar"):
        if has("revenue") and has("rooms_available"):
            df["revpar"] = np.where(
                df["rooms_available"] > 0,
                df["revenue"] / df["rooms_available"],
                np.nan,
            )
        elif has("adr") and has("occupancy_pct"):
            df["revpar"] = df["adr"] * df["occupancy_pct"] / 100

    return df
