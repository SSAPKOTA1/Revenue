"""
Disk cache manager — SQLite backend with incremental loading.

On first run: scans all files, writes to SQLite.
On subsequent runs: only parses NEW files added since last scan.
Each file is tracked by its path + modification time in a 'files' table.

This means adding today's snapshot folder takes seconds, not minutes.
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

_TABLE       = "master"
_FILES_TABLE = "scanned_files"   # tracks which files have been ingested

_DATE_COLS    = ["date", "snapshot_date"]
_NUMERIC_COLS = [
    "rooms_sold", "rooms_available", "revenue", "adr",
    "occupancy_pct", "revpar", "fnb_revenue", "room_revenue",
    "year", "month", "quarter", "day_of_week", "week", "day_of_year",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    con = sqlite3.connect(SQLITE_FILE)
    con.execute("PRAGMA journal_mode=WAL")   # faster concurrent reads
    con.execute("PRAGMA synchronous=NORMAL") # faster writes, still safe
    return con


def _ensure_schema(con: sqlite3.Connection) -> None:
    """Create tables and indexes if they don't exist yet."""
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_FILES_TABLE} (
            filepath TEXT PRIMARY KEY,
            mtime    REAL NOT NULL,
            rows     INTEGER DEFAULT 0,
            loaded_at TEXT
        )
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            hotel_name    TEXT,
            date          TEXT,
            snapshot_date TEXT,
            rooms_sold    REAL,
            rooms_available REAL,
            revenue       REAL,
            adr           REAL,
            occupancy_pct REAL,
            revpar        REAL,
            room_revenue  REAL,
            fnb_revenue   REAL,
            year          INTEGER,
            month         INTEGER,
            quarter       INTEGER,
            day_of_week   INTEGER,
            week          INTEGER,
            day_of_year   INTEGER,
            month_name    TEXT,
            day_name      TEXT,
            is_weekend    INTEGER,
            event         TEXT,
            _source_file  TEXT
        )
    """)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_hotel      ON {_TABLE}(hotel_name)")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_snap       ON {_TABLE}(snapshot_date)")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_date       ON {_TABLE}(date)")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_snap_hotel ON {_TABLE}(snapshot_date, hotel_name)")
    con.commit()


def _df_to_sql(df: pd.DataFrame, con: sqlite3.Connection) -> None:
    """Append a cleaned DataFrame to the master table."""
    save = df.copy()
    for col in _DATE_COLS:
        if col in save.columns:
            save[col] = pd.to_datetime(save[col], errors="coerce").dt.strftime("%Y-%m-%d")
    # Boolean → int for SQLite
    if "is_weekend" in save.columns:
        save["is_weekend"] = save["is_weekend"].astype(int)
    # Only insert columns that exist in the schema
    existing = [row[1] for row in con.execute(f"PRAGMA table_info({_TABLE})").fetchall()]
    cols = [c for c in save.columns if c in existing]
    save[cols].to_sql(_TABLE, con, if_exists="append", index=False, chunksize=5_000)


def _restore_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _update_meta(con: sqlite3.Connection, source_folder: str) -> None:
    row_count = con.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
    hotels    = con.execute(f"SELECT COUNT(DISTINCT hotel_name) FROM {_TABLE}").fetchone()[0]
    snaps     = con.execute(f"SELECT COUNT(DISTINCT snapshot_date) FROM {_TABLE}").fetchone()[0]
    files_n   = con.execute(f"SELECT COUNT(*) FROM {_FILES_TABLE}").fetchone()[0]
    meta = {
        "source_folder": source_folder,
        "saved_at": datetime.now().isoformat(),
        "rows": row_count,
        "hotels": hotels,
        "snapshots": snaps,
        "files_loaded": files_n,
    }
    CACHE_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── Public: incremental ingest ────────────────────────────────────────────────

def ingest_new_files(
    source_folder: str,
    progress_callback=None,
) -> dict:
    """
    Scan source_folder for Excel/CSV files.  Only parse files that are
    NEW or MODIFIED since they were last ingested.  Append results to
    the SQLite database.

    Returns a summary dict: {new, skipped, errors, total_rows_added}
    """
    from modules import data_loader, data_cleaner   # local import avoids circular

    root = Path(source_folder)
    all_files = data_loader.scan_folder(root)

    if not all_files:
        return {"new": 0, "skipped": 0, "errors": 0, "total_rows_added": 0}

    with _get_conn() as con:
        _ensure_schema(con)

        # Load already-ingested file registry
        known = {
            row[0]: row[1]
            for row in con.execute(
                f"SELECT filepath, mtime FROM {_FILES_TABLE}"
            ).fetchall()
        }

        new_files = []
        for fp in all_files:
            fp_str = str(fp)
            try:
                mtime = fp.stat().st_mtime
            except Exception:
                continue
            if fp_str not in known or known[fp_str] != mtime:
                new_files.append((fp, mtime))

    skipped = len(all_files) - len(new_files)
    logger.info(
        "Incremental scan: %d total files, %d new/changed, %d already cached",
        len(all_files), len(new_files), skipped,
    )

    if not new_files:
        return {"new": 0, "skipped": skipped, "errors": 0, "total_rows_added": 0}

    rows_added = 0
    errors = 0
    file_reports = []

    with _get_conn() as con:
        _ensure_schema(con)

        for i, (fp, mtime) in enumerate(new_files):
            try:
                snap_date = data_loader._extract_snapshot_date(fp, root)
                df = data_loader._read_single_file(fp)

                if df is None or df.empty:
                    errors += 1
                    file_reports.append({"file": fp.name, "status": "empty", "rows": 0})
                    continue

                df["snapshot_date"] = snap_date
                # Use full path as source key so same-named files in different
                # year/month folders don't overwrite each other
                df["_source_file"] = str(fp)
                df = data_cleaner.clean(df)

                # Delete previous rows for this exact file path before re-inserting
                con.execute(
                    f"DELETE FROM {_TABLE} WHERE _source_file = ?", (str(fp),)
                )

                _df_to_sql(df, con)
                rows_added += len(df)

                # Record this file as ingested
                con.execute(
                    f"INSERT OR REPLACE INTO {_FILES_TABLE}(filepath, mtime, rows, loaded_at) "
                    f"VALUES (?, ?, ?, ?)",
                    (str(fp), mtime, len(df), datetime.now().isoformat()),
                )

                file_reports.append({
                    "file": fp.name,
                    "status": "ok",
                    "rows": len(df),
                    "snapshot_date": str(snap_date.date()) if snap_date else "unknown",
                    "hotel": df["hotel_name"].iloc[0] if "hotel_name" in df.columns else "",
                })

            except Exception as e:
                logger.error("Failed to ingest '%s': %s", fp.name, e)
                errors += 1
                file_reports.append({"file": fp.name, "status": "error", "error": str(e)})

            if progress_callback:
                progress_callback((i + 1) / len(new_files))

        con.commit()
        compact_past_years(con)
        _update_meta(con, source_folder)

    logger.info("Incremental ingest done: +%d rows, %d errors", rows_added, errors)
    return {
        "new": len(new_files),
        "skipped": skipped,
        "errors": errors,
        "total_rows_added": rows_added,
        "file_reports": file_reports,
    }


def compact_past_years(con: sqlite3.Connection) -> int:
    """
    For arrival dates in years before the current year, keep only the row
    with the LATEST snapshot_date per (hotel_name, date).

    Past-year dates are 100% closed — their final value is the latest
    snapshot that covered them.  Older snapshots for those same dates
    are redundant and waste space / slow queries.

    Returns the number of rows deleted.
    """
    from datetime import date
    current_year_start = f"{date.today().year}-01-01"

    # Find rows to delete: past-year dates where a newer snapshot exists
    # for the same (hotel_name, date) pair.
    result = con.execute(f"""
        DELETE FROM {_TABLE}
        WHERE date < ?
          AND snapshot_date != (
              SELECT MAX(m2.snapshot_date)
              FROM   {_TABLE} m2
              WHERE  m2.hotel_name = {_TABLE}.hotel_name
                AND  m2.date       = {_TABLE}.date
          )
    """, (current_year_start,))
    deleted = result.rowcount
    con.commit()
    logger.info("compact_past_years: removed %d redundant past-year rows", deleted)
    return deleted


def full_rebuild(source_folder: str, progress_callback=None) -> dict:
    """
    Drop and rebuild the entire database from scratch.
    Use this only when you want a clean slate (Force Re-scan).
    """
    # Wipe existing DB
    if SQLITE_FILE.exists():
        SQLITE_FILE.unlink()
    with _get_conn() as con:
        _ensure_schema(con)
        con.commit()

    return ingest_new_files(source_folder, progress_callback)


# ── Load ───────────────────────────────────────────────────────────────────────

def load_cache() -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load full master DataFrame from SQLite."""
    if not SQLITE_FILE.exists():
        return None, None
    try:
        meta = {}
        if CACHE_META_FILE.exists():
            meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))

        with _get_conn() as con:
            df = pd.read_sql(f"SELECT * FROM {_TABLE}", con)

        df = _restore_df(df)
        logger.info("SQLite cache loaded: %d rows", len(df))
        return df, meta

    except Exception as e:
        logger.error("Failed to load cache: %s", e)
        return None, None


def load_for_hotel(hotel_name: str) -> Optional[pd.DataFrame]:
    if not SQLITE_FILE.exists():
        return None
    try:
        with _get_conn() as con:
            df = pd.read_sql(
                f"SELECT * FROM {_TABLE} WHERE hotel_name = ?",
                con, params=(hotel_name,),
            )
        return _restore_df(df)
    except Exception as e:
        logger.error("load_for_hotel failed: %s", e)
        return None


def list_hotels() -> list[str]:
    if not SQLITE_FILE.exists():
        return []
    try:
        with _get_conn() as con:
            rows = con.execute(
                f"SELECT DISTINCT hotel_name FROM {_TABLE} ORDER BY hotel_name"
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def list_snapshots() -> list[str]:
    if not SQLITE_FILE.exists():
        return []
    try:
        with _get_conn() as con:
            rows = con.execute(
                f"SELECT DISTINCT snapshot_date FROM {_TABLE} ORDER BY snapshot_date"
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


# ── Validity check ─────────────────────────────────────────────────────────────

def has_new_files(source_folder: str) -> tuple[bool, int]:
    """
    Quick check: are there any files in source_folder that are not yet
    in the database?  Returns (has_new, count_of_new_files).
    Does NOT read any Excel files — just compares mtimes.
    """
    if not SQLITE_FILE.exists():
        return True, -1   # no DB at all

    from modules import data_loader
    all_files = data_loader.scan_folder(Path(source_folder))

    try:
        with _get_conn() as con:
            known = {
                row[0]: row[1]
                for row in con.execute(
                    f"SELECT filepath, mtime FROM {_FILES_TABLE}"
                ).fetchall()
            }
    except Exception:
        return True, len(all_files)

    new_count = sum(
        1 for fp in all_files
        if str(fp) not in known or known[str(fp)] != fp.stat().st_mtime
    )
    return new_count > 0, new_count


def cache_is_valid(source_folder: str) -> tuple[bool, str]:
    """Legacy compatibility — used by app.py to decide whether to scan."""
    if not SQLITE_FILE.exists() or not CACHE_META_FILE.exists():
        return False, "No cache found"
    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False, "Metadata unreadable"
    if meta.get("source_folder") != source_folder:
        return False, "Source folder changed"
    has_new, n = has_new_files(source_folder)
    if has_new:
        return False, f"{n} new/modified file(s) found"
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
