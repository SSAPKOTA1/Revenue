"""KPI calculation engine — daily, monthly, quarterly, yearly aggregations."""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _safe_wavg(num: pd.Series, denom: pd.Series) -> float:
    """Weighted average: sum(num) / sum(denom), safe for zero denom."""
    d = denom.sum()
    return float(num.sum() / d) if d > 0 else np.nan


def compute_hotel_kpis(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute aggregate KPIs for a single hotel (or any filtered DataFrame).

    Returns dict with occupancy_pct, adr, revpar, revenue, rooms_sold, etc.
    """
    result: dict[str, float] = {}

    if df.empty:
        return result

    if "rooms_sold" in df.columns:
        result["rooms_sold"] = float(df["rooms_sold"].sum())
    if "rooms_available" in df.columns:
        result["rooms_available"] = float(df["rooms_available"].sum())
    if "revenue" in df.columns:
        result["revenue"] = float(df["revenue"].sum())

    # Occupancy — weighted by available rooms
    if "rooms_sold" in df.columns and "rooms_available" in df.columns:
        result["occupancy_pct"] = _safe_wavg(df["rooms_sold"], df["rooms_available"]) * 100
    elif "occupancy_pct" in df.columns:
        result["occupancy_pct"] = float(df["occupancy_pct"].mean())

    # ADR — weighted by rooms sold
    if "revenue" in df.columns and "rooms_sold" in df.columns:
        result["adr"] = _safe_wavg(df["revenue"], df["rooms_sold"])
    elif "adr" in df.columns:
        result["adr"] = float(df["adr"].mean())

    # RevPAR — revenue / total available rooms
    if "revenue" in df.columns and "rooms_available" in df.columns:
        result["revpar"] = _safe_wavg(df["revenue"], df["rooms_available"])
    elif "revpar" in df.columns:
        result["revpar"] = float(df["revpar"].mean())

    if "los" in df.columns:
        result["avg_los"] = float(df["los"].mean())

    if "booking_date" in df.columns and "date" in df.columns:
        lead = (df["date"] - pd.to_datetime(df["booking_date"], errors="coerce")).dt.days
        result["avg_booking_window"] = float(lead.mean()) if lead.notna().any() else np.nan

    return result


def aggregate_period(
    df: pd.DataFrame,
    period: str = "M",
    hotel_col: str = "hotel_name",
    group_hotel: bool = True,
) -> pd.DataFrame:
    """
    Aggregate the master DataFrame by time period.

    Parameters
    ----------
    period : str  pandas period alias — 'D', 'W', 'M', 'Q', 'Y'
    group_hotel : bool  whether to keep hotel dimension
    """
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["_period"] = df["date"].dt.to_period(period)

    group_keys = ["_period"]
    if group_hotel and hotel_col in df.columns:
        group_keys.append(hotel_col)

    agg_dict: dict[str, object] = {}
    for col, func in [
        ("rooms_sold", "sum"),
        ("rooms_available", "sum"),
        ("revenue", "sum"),
        ("los", "mean"),
    ]:
        if col in df.columns:
            agg_dict[col] = func

    if not agg_dict:
        return pd.DataFrame()

    out = df.groupby(group_keys).agg(agg_dict).reset_index()
    out["_period"] = out["_period"].dt.to_timestamp()
    out = out.rename(columns={"_period": "period"})

    # Derived KPIs from aggregated totals
    if "rooms_sold" in out.columns and "rooms_available" in out.columns:
        out["occupancy_pct"] = (out["rooms_sold"] / out["rooms_available"].replace(0, np.nan)) * 100
    if "revenue" in out.columns and "rooms_sold" in out.columns:
        out["adr"] = out["revenue"] / out["rooms_sold"].replace(0, np.nan)
    if "revenue" in out.columns and "rooms_available" in out.columns:
        out["revpar"] = out["revenue"] / out["rooms_available"].replace(0, np.nan)

    return out


def aggregate_daily(df: pd.DataFrame, hotel_col: str = "hotel_name") -> pd.DataFrame:
    """Daily aggregation keeping hotel dimension."""
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    group_keys = ["date"]
    if hotel_col in df.columns:
        group_keys.append(hotel_col)

    numeric_cols = [c for c in ["rooms_sold", "rooms_available", "revenue", "los"] if c in df.columns]
    if not numeric_cols:
        return pd.DataFrame()

    out = df.groupby(group_keys)[numeric_cols].sum(min_count=1).reset_index()

    if "rooms_sold" in out.columns and "rooms_available" in out.columns:
        out["occupancy_pct"] = (out["rooms_sold"] / out["rooms_available"].replace(0, np.nan)) * 100
    if "revenue" in out.columns and "rooms_sold" in out.columns:
        out["adr"] = out["revenue"] / out["rooms_sold"].replace(0, np.nan)
    if "revenue" in out.columns and "rooms_available" in out.columns:
        out["revpar"] = out["revenue"] / out["rooms_available"].replace(0, np.nan)

    return out.sort_values("date").reset_index(drop=True)


def weekday_performance(df: pd.DataFrame) -> pd.DataFrame:
    """KPIs by day-of-week using correct RM formulas."""
    if df.empty or "day_of_week" not in df.columns:
        return pd.DataFrame()

    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    sum_cols = [c for c in ["rooms_sold", "rooms_available", "revenue"] if c in df.columns]
    if not sum_cols:
        return pd.DataFrame()

    # Sum the bases, then derive — gives the true weighted average per weekday
    out = df.groupby("day_of_week")[sum_cols].sum(min_count=1).reset_index()

    if "rooms_sold" in out.columns and "rooms_available" in out.columns:
        out["occupancy_pct"] = (out["rooms_sold"] / out["rooms_available"].replace(0, np.nan) * 100).clip(0, 100)
    if "revenue" in out.columns and "rooms_sold" in out.columns:
        out["adr"] = out["revenue"] / out["rooms_sold"].replace(0, np.nan)
    if "revenue" in out.columns and "rooms_available" in out.columns:
        out["revpar"] = out["revenue"] / out["rooms_available"].replace(0, np.nan)

    out["day_name"] = out["day_of_week"].map(day_map)
    return out.sort_values("day_of_week")


def monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    """KPIs by month-of-year (seasonality) using correct RM formulas."""
    if df.empty or "month" not in df.columns:
        return pd.DataFrame()

    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    sum_cols = [c for c in ["rooms_sold", "rooms_available", "revenue"] if c in df.columns]
    if not sum_cols:
        return pd.DataFrame()

    out = df.groupby("month")[sum_cols].sum(min_count=1).reset_index()

    if "rooms_sold" in out.columns and "rooms_available" in out.columns:
        out["occupancy_pct"] = (out["rooms_sold"] / out["rooms_available"].replace(0, np.nan) * 100).clip(0, 100)
    if "revenue" in out.columns and "rooms_sold" in out.columns:
        out["adr"] = out["revenue"] / out["rooms_sold"].replace(0, np.nan)
    if "revenue" in out.columns and "rooms_available" in out.columns:
        out["revpar"] = out["revenue"] / out["rooms_available"].replace(0, np.nan)

    out["month_name"] = out["month"].map(month_map)
    return out.sort_values("month")


def compression_nights(df: pd.DataFrame, threshold: float = 85.0) -> pd.DataFrame:
    """Return rows where occupancy exceeds the compression threshold."""
    if df.empty or "occupancy_pct" not in df.columns:
        return pd.DataFrame()
    return df[df["occupancy_pct"] >= threshold].copy()


def rolling_kpis(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Add rolling average columns for key KPIs."""
    if df.empty or "date" not in df.columns:
        return df
    df = df.sort_values("date").copy()
    for col in ["occupancy_pct", "adr", "revpar", "revenue"]:
        if col in df.columns:
            df[f"{col}_rolling{window}"] = (
                df[col].rolling(window, min_periods=1).mean()
            )
    return df


def forecast_accuracy(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute MAPE and MAE between actuals and forecast columns if both present.
    Returns None if insufficient data.
    """
    pairs = [
        ("rooms_sold", "forecast_rooms"),
        ("revenue", "forecast_revenue"),
        ("adr", "forecast_adr"),
        ("occupancy_pct", "forecast_occupancy_pct"),
    ]
    rows = []
    for actual_col, fcast_col in pairs:
        if actual_col in df.columns and fcast_col in df.columns:
            mask = df[actual_col].notna() & df[fcast_col].notna() & (df[actual_col] != 0)
            if mask.sum() < 5:
                continue
            actual = df.loc[mask, actual_col]
            forecast = df.loc[mask, fcast_col]
            mape = float((((actual - forecast).abs() / actual.abs()) * 100).mean())
            mae = float((actual - forecast).abs().mean())
            rows.append({"metric": actual_col, "MAPE_%": round(mape, 2), "MAE": round(mae, 2)})

    return pd.DataFrame(rows) if rows else None
