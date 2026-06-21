"""
Disk cache manager — SQLite backend.

Stores the master DataFrame in a local SQLite database so the app loads
instantly on restart.  SQLite is built into Python (no extra install).

Why SQLite over Parquet:
  - Faster partial reads: we can SELECT only the columns/rows we need
  - Faster filtered queries: SQL WHERE pushes filtering to disk
  - Atomic writes: no corrupt half-written files
  - Works without pyarrow

Cache is invalidated when:
  - The source folder changes
  - Any source file is newer than the DB
  - The user clicks "Force Refresh"
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import SQLITE_FILE, CACHE_META_FILE

logger = logging.getLogger(__name__)

_TABLE = "master"

# Columns that must be stored as TEXT in SQLite and parsed back on load
_DATE_COLS    = ["date", "snapshot_date"]
_NUMERIC_COLS = [
    "rooms_sold", "rooms_available", "revenue", "adr",
    "occupancy_pct", "revpar", "fnb_revenue", "room_revenue",
    "year", "month", "quarter", "day_of_week", "week", "day_of_year",
]


# ── Save ───────────────────────────────────────────────────────────────────────

def save_cache(df: pd.DataFrame, source_folder: str, file_reports: list[dict]) -> None:
    """Write master DataFrame to SQLite + metadata JSON."""
    try:
        save_df = df.copy()

        # Convert timestamps to ISO strings for SQLite TEXT storage
        for col in _DATE_COLS:
            if col in save_df.columns:
                save_df[col] = pd.to_datetime(save_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

        with sqlite3.connect(SQLITE_FILE) as con:
            # Replace entire table — fastest for full refresh
            save_df.to_sql(_TABLE, con, if_exists="replace", index=False, chunksize=10_000)
            # Index the two most-queried columns for fast WHERE/JOIN
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_hotel    ON {_TABLE}(hotel_name)")
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_snap     ON {_TABLE}(snapshot_date)")
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_date     ON {_TABLE}(date)")
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_snap_hotel ON {_TABLE}(snapshot_date, hotel_name)")

        meta = {
            "source_folder": source_folder,
            "saved_at": datetime.now().isoformat(),
            "rows": len(df),
            "hotels": int(df["hotel_name"].nunique()) if "hotel_name" in df.columns else 0,
            "snapshots": int(df["snapshot_date"].nunique()) if "snapshot_date" in df.columns else 0,
            "files_loaded": len(file_reports),
        }
        CACHE_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("SQLite cache saved: %d rows → %s", len(df), SQLITE_FILE)

    except Exception as e:
        logger.error("Failed to save SQLite cache: %s", e)


# ── Load ───────────────────────────────────────────────────────────────────────

def load_cache() -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load full master DataFrame from SQLite. Returns (df, meta) or (None, None)."""
    if not SQLITE_FILE.exists() or not CACHE_META_FILE.exists():
        return None, None
    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))

        with sqlite3.connect(SQLITE_FILE) as con:
            df = pd.read_sql(f"SELECT * FROM {_TABLE}", con)

        # Restore datetime columns
        for col in _DATE_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Restore numeric columns (SQLite may store them as TEXT)
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("SQLite cache loaded: %d rows (saved %s)", len(df), meta.get("saved_at", "?"))
        return df, meta

    except Exception as e:
        logger.error("Failed to load SQLite cache: %s", e)
        return None, None


# ── Partial / filtered load ────────────────────────────────────────────────────

def load_for_hotel(hotel_name: str) -> Optional[pd.DataFrame]:
    """Load only rows for one hotel — much faster than loading everything."""
    if not SQLITE_FILE.exists():
        return None
    try:
        with sqlite3.connect(SQLITE_FILE) as con:
            df = pd.read_sql(
                f"SELECT * FROM {_TABLE} WHERE hotel_name = ?",
                con, params=(hotel_name,),
            )
        for col in _DATE_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.error("load_for_hotel failed: %s", e)
        return None


def load_snapshot(snapshot_date: str) -> Optional[pd.DataFrame]:
    """Load only rows for one snapshot date (YYYY-MM-DD string)."""
    if not SQLITE_FILE.exists():
        return None
    try:
        with sqlite3.connect(SQLITE_FILE) as con:
            df = pd.read_sql(
                f"SELECT * FROM {_TABLE} WHERE snapshot_date = ?",
                con, params=(snapshot_date,),
            )
        for col in _DATE_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.error("load_snapshot failed: %s", e)
        return None


def list_hotels() -> list[str]:
    """Return sorted list of hotel names from the DB (no full table scan)."""
    if not SQLITE_FILE.exists():
        return []
    try:
        with sqlite3.connect(SQLITE_FILE) as con:
            rows = con.execute(
                f"SELECT DISTINCT hotel_name FROM {_TABLE} ORDER BY hotel_name"
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def list_snapshots() -> list[str]:
    """Return sorted list of snapshot dates (YYYY-MM-DD strings)."""
    if not SQLITE_FILE.exists():
        return []
    try:
        with sqlite3.connect(SQLITE_FILE) as con:
            rows = con.execute(
                f"SELECT DISTINCT snapshot_date FROM {_TABLE} ORDER BY snapshot_date"
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


# ── Validity check ─────────────────────────────────────────────────────────────

def cache_is_valid(source_folder: str) -> tuple[bool, str]:
    if not SQLITE_FILE.exists() or not CACHE_META_FILE.exists():
        return False, "No cache found"
    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False, "Cache metadata unreadable"

    if meta.get("source_folder") != source_folder:
        return False, "Source folder changed"

    db_mtime = SQLITE_FILE.stat().st_mtime
    folder = Path(source_folder)
    if folder.exists():
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            for fp in folder.rglob(ext):
                if fp.stat().st_mtime > db_mtime:
                    return False, f"New/modified file: {fp.name}"

    return True, "Cache is up-to-date"


# ── Cache info for UI ──────────────────────────────────────────────────────────

def get_cache_info() -> Optional[dict]:
    if not CACHE_META_FILE.exists():
        return None
    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(meta["saved_at"])
        meta["saved_at_display"] = saved_at.strftime("%d %b %Y %H:%M")
        if SQLITE_FILE.exists():
            meta["cache_size_mb"] = round(SQLITE_FILE.stat().st_size / 1_048_576, 2)
        return meta
    except Exception:
        return None


def clear_cache() -> None:
    for f in [SQLITE_FILE, CACHE_META_FILE]:
        if f.exists():
            f.unlink()
    logger.info("SQLite cache cleared")
