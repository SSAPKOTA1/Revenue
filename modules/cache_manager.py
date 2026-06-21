"""
Disk cache manager.

Saves the processed master DataFrame to Parquet so the app loads
instantly on restart — no need to re-scan the folder every time.

Cache is invalidated automatically when:
  - The source folder path changes
  - Any source file is newer than the cache
  - The user clicks "Force Refresh"
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import CACHE_FILE, CACHE_META_FILE

logger = logging.getLogger(__name__)


# ── Save ───────────────────────────────────────────────────────────────────

def save_cache(df: pd.DataFrame, source_folder: str, file_reports: list[dict]) -> None:
    """Save master DataFrame + metadata to disk."""
    try:
        # Parquet can't store mixed-type columns or some timestamp formats
        save_df = df.copy()
        for col in save_df.columns:
            if save_df[col].dtype == object:
                save_df[col] = save_df[col].astype(str)

        save_df.to_parquet(CACHE_FILE, index=False, engine="pyarrow")

        meta = {
            "source_folder": source_folder,
            "saved_at": datetime.now().isoformat(),
            "rows": len(df),
            "hotels": int(df["hotel_name"].nunique()) if "hotel_name" in df.columns else 0,
            "files_loaded": len(file_reports),
            "snapshots": int(df["snapshot_date"].nunique()) if "snapshot_date" in df.columns else 0,
        }
        with open(CACHE_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Cache saved: %d rows → %s", len(df), CACHE_FILE)
    except Exception as e:
        logger.error("Failed to save cache: %s", e)


# ── Load ───────────────────────────────────────────────────────────────────

def load_cache() -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    Load cached DataFrame from disk.
    Returns (df, meta) or (None, None) if no valid cache exists.
    """
    if not CACHE_FILE.exists() or not CACHE_META_FILE.exists():
        return None, None

    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
        df = pd.read_parquet(CACHE_FILE, engine="pyarrow")

        # Restore datetime columns
        for col in ["date", "snapshot_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Restore numeric columns that may have become strings
        numeric_cols = [
            "rooms_sold", "rooms_available", "revenue", "adr",
            "occupancy_pct", "revpar", "fnb_revenue", "room_revenue",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("Cache loaded: %d rows from %s", len(df), meta.get("saved_at", "?"))
        return df, meta

    except Exception as e:
        logger.error("Failed to load cache: %s", e)
        return None, None


# ── Validity check ─────────────────────────────────────────────────────────

def cache_is_valid(source_folder: str) -> tuple[bool, str]:
    """
    Check whether the on-disk cache is still valid.

    Returns (is_valid, reason_string)
    """
    if not CACHE_FILE.exists() or not CACHE_META_FILE.exists():
        return False, "No cache found"

    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False, "Cache metadata unreadable"

    # Different source folder → invalid
    if meta.get("source_folder") != source_folder:
        return False, f"Source folder changed"

    # Check if any source file is newer than the cache
    cache_mtime = CACHE_FILE.stat().st_mtime
    folder = Path(source_folder)
    if folder.exists():
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            for fp in folder.rglob(ext):
                if fp.stat().st_mtime > cache_mtime:
                    return False, f"New/modified file detected: {fp.name}"

    return True, "Cache is up-to-date"


# ── Cache info for UI ──────────────────────────────────────────────────────

def get_cache_info() -> Optional[dict]:
    """Return cache metadata for display in the UI, or None if no cache."""
    if not CACHE_META_FILE.exists():
        return None
    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(meta["saved_at"])
        meta["saved_at_display"] = saved_at.strftime("%d %b %Y %H:%M")
        meta["cache_size_mb"] = round(CACHE_FILE.stat().st_size / 1_048_576, 2)
        return meta
    except Exception:
        return None


def clear_cache() -> None:
    """Delete the cache files."""
    for f in [CACHE_FILE, CACHE_META_FILE]:
        if f.exists():
            f.unlink()
    logger.info("Cache cleared")
