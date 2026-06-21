"""
Data ingestion engine.

Folder structure supported:
    <root>/
        <year>/
            <month>/
                <day>/
                    HotelName.xlsx   (or HotelName.xls / .csv)

Two Excel report formats are auto-detected:

FORMAT A — "Belegung" (occupation by room):
    Identified by cell(0,0) containing "Belegung"
    Header row is row 6; data starts row 7.
    Key columns (0-indexed in raw DataFrame):
        2  = Date
        3  = Free rooms  (Frei Zim.)
        13 = OoO rooms
        14 = Rooms sold  (second Gesa Zim.)
        15 = Occupancy % (%)
        16 = ADR         (Logis/Zi.)
        17 = Room revenue(Logis)
        18 = F&B revenue
        19 = Total revenue (Summe)

FORMAT B — "Forecast" (Forecast - Zimmerbezogen):
    Identified by cell(3,0) containing "Forecast"
    Data rows begin where col 0 parses as a date.
    Fixed column positions (confirmed across samples):
        0  = Date
        19 = %ROCC (occupancy %)
        21 = Gesamt Zimmer (rooms_sold)
        22 = Gesamt Personen
        23 = Gesamt A.R.R. (ADR)
        24 = Gesamt Umsatz (revenue)
    rooms_available is derived from rooms_sold / (%ROCC/100).

Hotel name is taken from the filename stem, stripping any leading
hex-UUID prefix (e.g. "6b81a1f2-Aschaffenburg" → "Aschaffenburg").
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Filename helpers ───────────────────────────────────────────────────────

_UUID_PREFIX = re.compile(r'^[0-9a-f]{6,32}[-_]', re.I)

def _hotel_name_from_path(filepath: Path) -> str:
    """
    Derive a clean hotel name from the file path stem.
    Strips leading UUID/hash prefix, then normalises separators.
    """
    stem = filepath.stem
    stem = _UUID_PREFIX.sub('', stem)            # remove hash prefix
    # Replace underscores/dashes ONLY between word chars (not inside words)
    stem = re.sub(r'(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])', ' ', stem)
    stem = re.sub(r'(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])', ' ', stem)
    # Collapse any remaining underscores/dashes and extra whitespace
    stem = re.sub(r'[_\-]+', '', stem)
    stem = re.sub(r'\s+', ' ', stem).strip()
    return stem or filepath.stem


# ── Format detection ───────────────────────────────────────────────────────

def _detect_format(raw: pd.DataFrame) -> str:
    """
    Return 'A' (Belegung), 'B' (Forecast), or 'unknown'.
    """
    def _cell(r, c):
        try:
            v = raw.iat[r, c]
            return '' if pd.isna(v) else str(v).strip()
        except Exception:
            return ''

    if 'belegung' in _cell(0, 0).lower():
        return 'A'
    # Format B: "Forecast" appears in row 3 col 0
    for r in range(min(10, len(raw))):
        if 'forecast' in _cell(r, 0).lower():
            return 'B'
    return 'unknown'


# ── Format A parser ────────────────────────────────────────────────────────

def _parse_format_a(raw: pd.DataFrame, hotel_name: str) -> pd.DataFrame:
    """
    Parse 'Belegung' format (Aschaffenburg, Giessen style).

    Column layout (0-indexed in raw DataFrame):
        0  = Event label (or '\\n')
        1  = Day-of-week abbreviation (DO, FR, …)
        2  = Date
        3  = Frei Zim. (free rooms)
        13 = OoO Zim.
        14 = Gesa Zim. occupied (rooms_sold)
        15 = % (occupancy)
        16 = Logis/Zi. (ADR)
        17 = Logis (room revenue)
        18 = F&B revenue
        19 = Summe (total revenue)
    """
    # Data rows start at index 7 (row 0-6 are header/metadata)
    data = raw.iloc[7:].copy().reset_index(drop=True)

    rows = []
    for _, r in data.iterrows():
        def g(col):
            try:
                v = r.iat[col]
                return np.nan if pd.isna(v) else v
            except Exception:
                return np.nan

        date_raw = g(2)
        date = pd.to_datetime(date_raw, errors='coerce')
        if pd.isna(date):
            continue  # skip summary / blank rows

        event     = str(g(0)).strip() if not pd.isna(g(0)) else ''
        event     = '' if event in ('\n', 'nan', '') else event

        free_rooms  = _to_float(g(3))
        ooo_rooms   = _to_float(g(13))
        rooms_sold  = _to_float(g(14))
        occ_pct     = _to_float(g(15))
        adr         = _to_float(g(16))
        room_rev    = _to_float(g(17))
        fnb_rev     = _to_float(g(18))
        total_rev   = _to_float(g(19))

        # rooms_available = free + occupied (OoO excluded from denominator per the % formula)
        rooms_avail = np.nan
        if not np.isnan(free_rooms) and not np.isnan(rooms_sold):
            rooms_avail = free_rooms + rooms_sold
            # Verify against reported % (sanity check)
            if not np.isnan(occ_pct) and rooms_avail > 0:
                calc_pct = rooms_sold / rooms_avail * 100
                if abs(calc_pct - occ_pct) > 2:
                    # Try including OoO
                    if not np.isnan(ooo_rooms):
                        rooms_avail2 = free_rooms + rooms_sold + ooo_rooms
                        if rooms_avail2 > 0 and abs(rooms_sold / rooms_avail2 * 100 - occ_pct) < 2:
                            rooms_avail = rooms_avail2

        # Re-derive occupancy to ensure consistency
        if not np.isnan(rooms_sold) and not np.isnan(rooms_avail) and rooms_avail > 0:
            occ_pct = rooms_sold / rooms_avail * 100

        # Use total revenue (Summe) as the primary revenue figure
        revenue = total_rev if not np.isnan(total_rev) else room_rev

        # ADR from revenue / rooms_sold if not provided
        if np.isnan(adr) and not np.isnan(revenue) and not np.isnan(rooms_sold) and rooms_sold > 0:
            adr = revenue / rooms_sold

        rows.append({
            'date':           date,
            'hotel_name':     hotel_name,
            'rooms_sold':     rooms_sold,
            'rooms_available': rooms_avail,
            'occupancy_pct':  occ_pct,
            'adr':            adr,
            'revenue':        revenue,
            'room_revenue':   room_rev,
            'fnb_revenue':    fnb_rev,
            'event':          event,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = _add_revpar(out)
    return out


# ── Format B parser ────────────────────────────────────────────────────────

# Fixed column positions confirmed across Badsoden + Kaiser samples
_FMT_B_DATE    = 0
_FMT_B_ROCC    = 19   # %ROCC (occupancy %)
_FMT_B_ROOMS   = 21   # Gesamt Zimmer (rooms sold)
_FMT_B_PERSONS = 22   # Gesamt Personen
_FMT_B_ARR     = 23   # Gesamt A.R.R. (ADR)
_FMT_B_UMSATZ  = 24   # Gesamt Umsatz (revenue)


def _parse_format_b(raw: pd.DataFrame, hotel_name: str) -> pd.DataFrame:
    """
    Parse 'Forecast - Zimmerbezogen' format (Badsoden, Kaiser style).

    Fixed data columns (see module docstring):
        0  = Date
        19 = %ROCC
        21 = rooms_sold (Gesamt Zimmer)
        22 = persons
        23 = ADR (Gesamt A.R.R.)
        24 = revenue (Gesamt Umsatz)

    rooms_available is computed as rooms_sold / (%ROCC/100), rounded.
    """
    rows = []
    for _, r in raw.iterrows():
        def g(col):
            try:
                v = r.iat[col]
                return np.nan if pd.isna(v) else v
            except Exception:
                return np.nan

        date = pd.to_datetime(g(_FMT_B_DATE), errors='coerce')
        if pd.isna(date):
            continue

        rocc       = _to_float(g(_FMT_B_ROCC))    # occupancy %
        rooms_sold = _to_float(g(_FMT_B_ROOMS))
        adr        = _to_float(g(_FMT_B_ARR))
        revenue    = _to_float(g(_FMT_B_UMSATZ))

        # Derive rooms_available from %ROCC
        rooms_avail = np.nan
        if not np.isnan(rocc) and rocc > 0 and not np.isnan(rooms_sold):
            rooms_avail = round(rooms_sold / (rocc / 100))
        elif not np.isnan(rooms_sold) and rooms_sold == 0:
            # Blank weekend row — keep as zero-occupancy day
            rooms_avail = np.nan  # can't determine capacity; will fill later

        # Compute occupancy from solved capacity for consistency
        occ_pct = rocc if not np.isnan(rocc) else np.nan

        # ADR fallback
        if np.isnan(adr) and not np.isnan(revenue) and not np.isnan(rooms_sold) and rooms_sold > 0:
            adr = revenue / rooms_sold

        rows.append({
            'date':           date,
            'hotel_name':     hotel_name,
            'rooms_sold':     rooms_sold if not np.isnan(rooms_sold) else 0,
            'rooms_available': rooms_avail,
            'occupancy_pct':  occ_pct,
            'adr':            adr,
            'revenue':        revenue if not np.isnan(revenue) else 0,
        })

    out = pd.DataFrame(rows)

    # Back-fill rooms_available (for zero-occupancy weekend rows)
    if not out.empty and 'rooms_available' in out.columns:
        most_common = (
            out['rooms_available']
            .dropna()
            .round()
            .mode()
        )
        if not most_common.empty:
            capacity = most_common.iloc[0]
            out['rooms_available'] = out['rooms_available'].fillna(capacity)
            # Re-derive occ_pct for rows where it was missing
            mask = out['occupancy_pct'].isna() & (out['rooms_available'] > 0)
            out.loc[mask, 'occupancy_pct'] = (
                out.loc[mask, 'rooms_sold'] / out.loc[mask, 'rooms_available'] * 100
            )

    if not out.empty:
        out = _add_revpar(out)
    return out


# ── Shared helpers ─────────────────────────────────────────────────────────

def _to_float(v) -> float:
    if pd.isna(v):
        return np.nan
    try:
        return float(str(v).replace(',', '.').replace(' ', ''))
    except (ValueError, TypeError):
        return np.nan


def _add_revpar(df: pd.DataFrame) -> pd.DataFrame:
    if 'revenue' in df.columns and 'rooms_available' in df.columns:
        df['revpar'] = np.where(
            df['rooms_available'] > 0,
            df['revenue'] / df['rooms_available'],
            np.nan,
        )
    return df


# ── Single file reader ──────────────────────────────────────────────────────

def _read_single_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Read one Excel/CSV file, detect its format, and return a
    clean DataFrame with canonical columns.  Returns None on failure.
    """
    hotel_name = _hotel_name_from_path(filepath)
    suffix = filepath.suffix.lower()

    try:
        if suffix in ('.xlsx', '.xls'):
            engine = 'openpyxl' if suffix == '.xlsx' else 'xlrd'
            raw = pd.read_excel(filepath, header=None, engine=engine, dtype=object)
        elif suffix == '.csv':
            raw = pd.read_csv(filepath, header=None, dtype=str, on_bad_lines='skip')
        else:
            return None

        fmt = _detect_format(raw)
        logger.info("'%s' → format=%s, hotel='%s'", filepath.name, fmt, hotel_name)

        if fmt == 'A':
            df = _parse_format_a(raw, hotel_name)
        elif fmt == 'B':
            df = _parse_format_b(raw, hotel_name)
        else:
            # Fallback: treat as generic tabular data with header on first non-blank row
            df = _parse_generic(raw, hotel_name)

        if df is None or df.empty:
            logger.warning("'%s': parser returned empty DataFrame", filepath.name)
            return None

        df['_source_file'] = filepath.name
        logger.info("'%s': %d rows parsed", filepath.name, len(df))
        return df

    except Exception as e:
        logger.error("Failed to read '%s': %s", filepath, e)
        return None


def _parse_generic(raw: pd.DataFrame, hotel_name: str) -> Optional[pd.DataFrame]:
    """
    Last-resort parser: find the first row that looks like a header,
    use it as column names, then map via COLUMN_MAP.
    """
    from config.settings import COLUMN_MAP

    def _norm(s):
        return re.sub(r'\s+', ' ', str(s).strip().lower())

    # Find header row: first row with ≥3 non-null string cells
    header_row = None
    for i, row in raw.iterrows():
        non_null = [v for v in row if not pd.isna(v) and str(v).strip()]
        if len(non_null) >= 3:
            header_row = i
            break
    if header_row is None:
        return None

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [_norm(c) for c in raw.iloc[header_row]]

    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename)

    if 'date' not in df.columns:
        return None

    df['hotel_name'] = hotel_name
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    from config.settings import NUMERIC_COLS
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[,$% ]', '', regex=True),
                errors='coerce',
            )
    return df


# ── Folder scanner ──────────────────────────────────────────────────────────

def scan_folder(root: str | Path) -> list[Path]:
    """
    Recursively find all Excel (.xlsx, .xls) and CSV files under root.
    Skips hidden files and __pycache__ directories.
    """
    root = Path(root)
    found = []
    for glob_pat in ('*.xlsx', '*.xls', '*.csv'):
        for p in root.rglob(glob_pat):
            if any(part.startswith('.') or part == '__pycache__' for part in p.parts):
                continue
            found.append(p)

    # Sort by path so year/month/day order is preserved
    found.sort()
    logger.info("Found %d data files under '%s'", len(found), root)
    return found


# ── Master loader ───────────────────────────────────────────────────────────

def load_all_files(
    root: str | Path,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Scan root folder recursively, parse all hotel files, and merge into
    a single master DataFrame.

    Returns
    -------
    master_df    : merged and lightly cleaned DataFrame
    file_reports : list[dict] with per-file ingestion metadata
    """
    files = scan_folder(root)
    if not files:
        logger.warning("No data files found under '%s'", root)
        return pd.DataFrame(), []

    frames: list[pd.DataFrame] = []
    file_reports: list[dict] = []

    for i, fp in enumerate(files):
        report = {
            'file': fp.name,
            'path': str(fp.relative_to(root) if root in fp.parents or fp.is_relative_to(root) else fp),
            'status': 'ok',
            'rows': 0,
            'hotel': '',
            'format': '',
            'error': '',
        }

        df = _read_single_file(fp)
        if df is None or df.empty:
            report['status'] = 'empty' if df is not None else 'error'
            report['error'] = 'No data parsed' if df is not None else 'Read failure'
        else:
            report['rows'] = len(df)
            report['hotel'] = df['hotel_name'].iloc[0] if 'hotel_name' in df.columns else ''
            report['format'] = df.get('_source_file', pd.Series(['']))[0]
            frames.append(df)

        file_reports.append(report)

        if progress_callback:
            progress_callback((i + 1) / len(files))

    if not frames:
        logger.warning("No valid data loaded from '%s'", root)
        return pd.DataFrame(), file_reports

    master = pd.concat(frames, ignore_index=True)

    # Remove exact duplicates (same date + hotel from identical files)
    before = len(master)
    master = master.drop_duplicates(subset=['date', 'hotel_name'], keep='last')
    if len(master) < before:
        logger.info("Removed %d duplicate rows after merge", before - len(master))

    logger.info(
        "Master DataFrame: %d rows, %d hotels, %d cols",
        len(master),
        master['hotel_name'].nunique() if 'hotel_name' in master.columns else 0,
        len(master.columns),
    )
    return master, file_reports
