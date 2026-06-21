"""
KPI Engine — all aggregations use correct Revenue Management formulas.

Core rule:
  ADR       = Σ Revenue / Σ Rooms Sold          (never average ADR directly)
  Occupancy = Σ Rooms Sold / Σ Rooms Available  (never average occ% directly)
  RevPAR    = Σ Revenue / Σ Rooms Available
  Revenue   = Σ Revenue
  Rooms Sold= Σ Rooms Sold
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Base (additive) columns that are always summed
_BASE_COLS = ["rooms_sold", "rooms_available", "revenue", "room_revenue", "fnb_revenue"]


def rm_aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    extra_sum: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate a DataFrame using correct RM formulas.

    Always sums the base columns first, then derives:
      occupancy_pct = Σrooms_sold / Σrooms_available × 100
      adr           = Σrevenue    / Σrooms_sold
      revpar        = Σrevenue    / Σrooms_available

    Parameters
    ----------
    group_cols : columns to group by (e.g. ["period"], ["hotel_name", "period"])
    extra_sum  : any additional columns to sum (e.g. ["los"])
    """
    if df.empty:
        return pd.DataFrame()

    g_cols = [c for c in group_cols if c in df.columns]
    if not g_cols:
        return pd.DataFrame()

    sum_cols = [c for c in _BASE_COLS if c in df.columns]
    if extra_sum:
        sum_cols += [c for c in extra_sum if c in df.columns and c not in sum_cols]

    if not sum_cols:
        return pd.DataFrame()

    out = df.groupby(g_cols, dropna=False)[sum_cols].sum(min_count=1).reset_index()

    denom_avail = out["rooms_available"].replace(0, np.nan) if "rooms_available" in out else None
    denom_sold  = out["rooms_sold"].replace(0, np.nan)      if "rooms_sold"      in out else None
    denom_rev   = out["revenue"].replace(0, np.nan)         if "revenue"         in out else None

    if denom_sold is not None and denom_avail is not None:
        out["occupancy_pct"] = (out["rooms_sold"] / denom_avail * 100).clip(0, 100)

    if denom_rev is not None and denom_sold is not None:
        out["adr"] = (out["revenue"] / denom_sold).clip(0)

    if denom_rev is not None and denom_avail is not None:
        out["revpar"] = (out["revenue"] / denom_avail).clip(0)

    return out


def hotel_kpis(df: pd.DataFrame) -> dict:
    """Single-row KPI dict for KPI cards (works on any filtered slice)."""
    if df.empty:
        return {}

    agg = rm_aggregate(df, ["_all"] if False else []).copy()

    # Simpler: compute directly
    rs  = df["rooms_sold"].sum()      if "rooms_sold"      in df.columns else np.nan
    ra  = df["rooms_available"].sum() if "rooms_available" in df.columns else np.nan
    rev = df["revenue"].sum()         if "revenue"         in df.columns else np.nan

    occ = (rs / ra * 100) if (ra and ra > 0) else np.nan
    adr = (rev / rs)      if (rs and rs > 0) else np.nan
    rp  = (rev / ra)      if (ra and ra > 0) else np.nan

    return {
        "rooms_sold":     float(rs)  if not np.isnan(rs)  else np.nan,
        "rooms_available": float(ra) if not np.isnan(ra)  else np.nan,
        "revenue":        float(rev) if not np.isnan(rev) else np.nan,
        "occupancy_pct":  float(occ) if not np.isnan(occ) else np.nan,
        "adr":            float(adr) if not np.isnan(adr) else np.nan,
        "revpar":         float(rp)  if not np.isnan(rp)  else np.nan,
        "hotel_count":    int(df["hotel_name"].nunique()) if "hotel_name" in df.columns else 1,
    }


def add_period_col(df: pd.DataFrame, date_col: str, group_by: str) -> pd.DataFrame:
    """Add a 'period' column for grouping by Day/Week/Month/Quarter/Year."""
    df = df.copy()
    s = pd.to_datetime(df[date_col], errors="coerce")
    if group_by == "Daily":
        df["period"] = s.dt.normalize()
        df["period_label"] = s.dt.strftime("%d %b %Y")
    elif group_by == "Weekly":
        df["period"] = s.dt.to_period("W").apply(lambda p: p.start_time)
        df["period_label"] = s.dt.to_period("W").astype(str)
    elif group_by == "Monthly":
        df["period"] = s.dt.to_period("M").apply(lambda p: p.start_time)
        df["period_label"] = s.dt.strftime("%b %Y")
    elif group_by == "Quarterly":
        df["period"] = s.dt.to_period("Q").apply(lambda p: p.start_time)
        df["period_label"] = s.dt.to_period("Q").astype(str)
    elif group_by == "Yearly":
        df["period"] = s.dt.to_period("Y").apply(lambda p: p.start_time)
        df["period_label"] = s.dt.strftime("%Y")
    else:
        df["period"] = s.dt.to_period("M").apply(lambda p: p.start_time)
        df["period_label"] = s.dt.strftime("%b %Y")
    return df


def get_snapshots(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return sorted list of distinct snapshot dates."""
    if "snapshot_date" not in df.columns:
        return []
    raw = pd.to_datetime(df["snapshot_date"], errors="coerce").dropna().unique()
    return sorted([pd.Timestamp(s) for s in raw])


def latest_snapshot_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows from the most recent snapshot only."""
    snaps = get_snapshots(df)
    if not snaps:
        return df
    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    return df2[df2["snapshot_date"] == snaps[-1]].copy()


def snapshot_view(df: pd.DataFrame, snap: pd.Timestamp) -> pd.DataFrame:
    """Return rows for a specific snapshot date."""
    df2 = df.copy()
    df2["snapshot_date"] = pd.to_datetime(df2["snapshot_date"], errors="coerce")
    return df2[df2["snapshot_date"] == snap].copy()


def rolling_kpis(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for col in ["occupancy_pct", "adr", "revpar", "revenue"]:
        if col in df.columns:
            df[f"{col}_r{window}"] = df[col].rolling(window, min_periods=1).mean()
    return df


def weekday_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "day_of_week" not in df.columns:
        return pd.DataFrame()
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    out = rm_aggregate(df, ["day_of_week"])
    out["day_name"] = out["day_of_week"].map(day_map)
    return out.sort_values("day_of_week")


def monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "month" not in df.columns:
        return pd.DataFrame()
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    out = rm_aggregate(df, ["month"])
    out["month_name"] = out["month"].map(month_map)
    return out.sort_values("month")


def aggregate_period(df: pd.DataFrame, group_by: str = "Monthly") -> pd.DataFrame:
    df = add_period_col(df, "date", group_by)
    out = rm_aggregate(df, ["period", "period_label"])
    return out.sort_values("period")


def compression_nights(df: pd.DataFrame, threshold: float = 85.0) -> pd.DataFrame:
    if df.empty or "occupancy_pct" not in df.columns:
        return pd.DataFrame()
    return df[df["occupancy_pct"] >= threshold].copy()
