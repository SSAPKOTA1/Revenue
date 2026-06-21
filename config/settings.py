"""Global configuration and constants for the Revenue Management Platform."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
CACHE_DIR = BASE_DIR / "cache"

LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"
CACHE_FILE = CACHE_DIR / "master_data.parquet"
CACHE_META_FILE = CACHE_DIR / "cache_meta.json"

# ── App Meta ───────────────────────────────────────────────────────────────
APP_TITLE = "Hotel Revenue Management Analytics Platform"
APP_ICON = "🏨"
APP_VERSION = "1.0.0"

# ── Default Data Folder ────────────────────────────────────────────────────
# Set this to your root data folder. The app will auto-load on startup.
# Use raw string (r"...") or double backslashes for Windows paths.
# Set to "" or None to disable auto-load.
DEFAULT_DATA_FOLDER = r"U:\FFM_ZENTRALE\Sudip\REVENUE MANAGEMENT\2026\Belegung Data\ALl itsels"

# ── Column Mapping ─────────────────────────────────────────────────────────
# Maps raw column names (lowercased, stripped) → canonical internal names
COLUMN_MAP: dict[str, str] = {
    # Date
    "date": "date",
    "arrival date": "date",
    "stay date": "date",
    "business date": "date",
    "report date": "date",
    "staydate": "date",
    "arrivaldate": "date",
    # Hotel
    "hotel": "hotel_name",
    "hotel name": "hotel_name",
    "property": "hotel_name",
    "property name": "hotel_name",
    "hotelname": "hotel_name",
    "hotel id": "hotel_id",
    "property id": "hotel_id",
    # Rooms sold
    "rooms sold": "rooms_sold",
    "roomssold": "rooms_sold",
    "rooms occupied": "rooms_sold",
    "occupied rooms": "rooms_sold",
    "roomsoccupied": "rooms_sold",
    "sold rooms": "rooms_sold",
    "occ rooms": "rooms_sold",
    "room nights": "rooms_sold",
    "room nights sold": "rooms_sold",
    "rns": "rooms_sold",
    "occ": "rooms_sold",
    "occupied": "rooms_sold",
    # Rooms available
    "rooms available": "rooms_available",
    "roomsavailable": "rooms_available",
    "available rooms": "rooms_available",
    "capacity": "rooms_available",
    "total rooms": "rooms_available",
    "supply": "rooms_available",
    "hotel rooms": "rooms_available",
    "totalrooms": "rooms_available",
    # Revenue
    "revenue": "revenue",
    "room revenue": "revenue",
    "roomrevenue": "revenue",
    "rooms revenue": "revenue",
    "net revenue": "revenue",
    "gross revenue": "revenue",
    "total revenue": "revenue",
    "rev": "revenue",
    "actual revenue": "revenue",
    # ADR
    "adr": "adr",
    "average daily rate": "adr",
    "avg daily rate": "adr",
    "average rate": "adr",
    "avg rate": "adr",
    "rate": "adr",
    "daily rate": "adr",
    # Occupancy
    "occupancy": "occupancy_pct",
    "occupancy %": "occupancy_pct",
    "occupancy pct": "occupancy_pct",
    "occ %": "occupancy_pct",
    "occ%": "rooms_sold",  # sometimes labelled differently
    "occupancy rate": "occupancy_pct",
    "occ rate": "occupancy_pct",
    # RevPAR
    "revpar": "revpar",
    "rev par": "revpar",
    "revenue per available room": "revpar",
    # Booking / arrival
    "booking date": "booking_date",
    "bookingdate": "booking_date",
    "reservation date": "booking_date",
    "created date": "booking_date",
    "booked date": "booking_date",
    # LOS
    "los": "los",
    "length of stay": "los",
    "lengthofstay": "los",
    "avg los": "los",
    "average los": "los",
    # Segment / channel / market
    "segment": "segment",
    "market segment": "segment",
    "marketsegment": "segment",
    "channel": "channel",
    "booking channel": "channel",
    "source": "channel",
    "market": "market",
    "market code": "market",
    # Forecast
    "forecast rooms": "forecast_rooms",
    "forecast revenue": "forecast_revenue",
    "forecast adr": "forecast_adr",
    "forecast occupancy": "forecast_occupancy_pct",
    "budgeted rooms": "budget_rooms",
    "budget rooms": "budget_rooms",
    "budget revenue": "budget_revenue",
    "budget adr": "budget_adr",
    # Snapshot / as-of
    "snapshot date": "snapshot_date",
    "as of date": "snapshot_date",
    "report snapshot": "snapshot_date",
    "pickup date": "snapshot_date",
}

# ── Numeric Columns ────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "rooms_sold", "rooms_available", "revenue", "adr",
    "occupancy_pct", "revpar", "los",
    "forecast_rooms", "forecast_revenue", "forecast_adr",
    "budget_rooms", "budget_revenue", "budget_adr",
]

# ── Validation Thresholds ──────────────────────────────────────────────────
VALIDATION_RULES = {
    "occupancy_pct": {"min": 0, "max": 100},
    "adr": {"min": 0, "max": 50_000},
    "revenue": {"min": 0, "max": 100_000_000},
    "rooms_sold": {"min": 0, "max": 10_000},
    "rooms_available": {"min": 1, "max": 10_000},
    "revpar": {"min": 0, "max": 50_000},
    "los": {"min": 0, "max": 365},
}

# ── KPI Display ────────────────────────────────────────────────────────────
KPI_FORMAT = {
    "occupancy_pct": "{:.1f}%",
    "adr": "${:,.2f}",
    "revpar": "${:,.2f}",
    "revenue": "${:,.0f}",
    "rooms_sold": "{:,.0f}",
    "rooms_available": "{:,.0f}",
    "los": "{:.2f}",
}

# ── Chart Colors ───────────────────────────────────────────────────────────
BRAND_COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#F39C12",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "neutral": "#7F8C8D",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1A2E",
}

COLOR_SCALE_OCC = "Blues"
COLOR_SCALE_ADR = "Oranges"
COLOR_SCALE_REV = "Greens"

# ── Forecasting ────────────────────────────────────────────────────────────
FORECAST_HORIZONS = [30, 90, 180, 365]
FORECAST_METHODS = ["Prophet", "Exponential Smoothing", "Moving Average"]

# ── Anomaly Detection ──────────────────────────────────────────────────────
ANOMALY_ZSCORE_THRESHOLD = 2.5
ANOMALY_ROLLING_WINDOW = 14

# ── Compression Threshold ──────────────────────────────────────────────────
COMPRESSION_THRESHOLD = 85.0  # occupancy %
