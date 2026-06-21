"""Pickup and Pace analysis modules."""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PICKUP
# ═══════════════════════════════════════════════════════════════════════════

def compute_pickup(
    df: pd.DataFrame,
    windows: list[int] = [1, 7, 14, 30],
    metrics: list[str] = ["rooms_sold", "revenue", "adr"],
) -> pd.DataFrame:
    """
    Calculate pickup (change in booked rooms/revenue) over rolling windows.

    Assumes df has columns: date, hotel_name, and metric columns.
    Uses snapshot_date if available to simulate true pickup snapshots;
    otherwise derives pickup from daily data differences.

    Returns DataFrame with pickup columns appended.
    """
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    has_snapshot = "snapshot_date" in df.columns
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return pd.DataFrame()

    if has_snapshot:
        return _pickup_from_snapshots(df, windows, available_metrics)
    else:
        return _pickup_from_daily(df, windows, available_metrics)


def _pickup_from_daily(
    df: pd.DataFrame,
    windows: list[int],
    metrics: list[str],
) -> pd.DataFrame:
    """Derive pickup as day-over-day change in daily data."""
    group_keys = ["date"]
    if "hotel_name" in df.columns:
        group_keys = ["hotel_name", "date"]

    agg = df.groupby(group_keys)[metrics].sum(min_count=1).reset_index()
    agg = agg.sort_values(group_keys)

    if "hotel_name" in agg.columns:
        parts = []
        for hotel, hdf in agg.groupby("hotel_name"):
            hdf = hdf.sort_values("date").copy()
            for m in metrics:
                for w in windows:
                    hdf[f"{m}_pickup_{w}d"] = hdf[m].diff(w)
            parts.append(hdf)
        return pd.concat(parts, ignore_index=True)
    else:
        agg = agg.sort_values("date").copy()
        for m in metrics:
            for w in windows:
                agg[f"{m}_pickup_{w}d"] = agg[m].diff(w)
        return agg


def _pickup_from_snapshots(
    df: pd.DataFrame,
    windows: list[int],
    metrics: list[str],
) -> pd.DataFrame:
    """Compute pickup between consecutive snapshots for each arrival date."""
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["snapshot_date", "date"])

    group_keys = ["date", "snapshot_date"]
    if "hotel_name" in df.columns:
        group_keys = ["hotel_name"] + group_keys

    agg = df.groupby(group_keys)[metrics].sum(min_count=1).reset_index()
    agg = agg.sort_values(group_keys)

    rows = []
    for key, grp in agg.groupby([c for c in group_keys if c != "snapshot_date"]):
        grp = grp.sort_values("snapshot_date").reset_index(drop=True)
        for m in metrics:
            for w in windows:
                grp[f"{m}_pickup_{w}d"] = grp[m].diff(w)
        rows.append(grp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pickup_summary(pickup_df: pd.DataFrame, metrics: list[str] = ["rooms_sold", "revenue"]) -> pd.DataFrame:
    """Summarise pickup by hotel and window."""
    if pickup_df.empty:
        return pd.DataFrame()

    rows = []
    hotels = pickup_df["hotel_name"].unique() if "hotel_name" in pickup_df.columns else ["All"]
    for hotel in hotels:
        hdf = pickup_df[pickup_df["hotel_name"] == hotel] if "hotel_name" in pickup_df.columns else pickup_df
        for m in metrics:
            for w in [1, 7, 14, 30]:
                col = f"{m}_pickup_{w}d"
                if col in hdf.columns:
                    rows.append({
                        "hotel_name": hotel,
                        "metric": m,
                        "window_days": w,
                        "avg_pickup": round(float(hdf[col].mean()), 2),
                        "total_pickup": round(float(hdf[col].sum()), 2),
                        "max_pickup": round(float(hdf[col].max()), 2),
                        "min_pickup": round(float(hdf[col].min()), 2),
                    })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# PACE
# ═══════════════════════════════════════════════════════════════════════════

def compute_pace(
    df: pd.DataFrame,
    benchmark: str = "last_year",
    metrics: list[str] = ["rooms_sold", "revenue", "adr", "occupancy_pct"],
) -> pd.DataFrame:
    """
    Compute pace (current vs benchmark) for each hotel-date combination.

    Parameters
    ----------
    benchmark : str
        'last_year' | 'budget' | 'forecast' | 'portfolio_avg'
    """
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return pd.DataFrame()

    if benchmark == "last_year":
        return _pace_vs_last_year(df, available_metrics)
    elif benchmark == "budget":
        return _pace_vs_budget(df, available_metrics)
    elif benchmark == "forecast":
        return _pace_vs_forecast(df, available_metrics)
    elif benchmark == "portfolio_avg":
        return _pace_vs_portfolio_avg(df, available_metrics)
    else:
        logger.warning("Unknown benchmark '%s'; defaulting to last_year", benchmark)
        return _pace_vs_last_year(df, available_metrics)


def _pace_vs_last_year(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())
    if len(years) < 2:
        logger.warning("Pace vs last year requires at least 2 years of data.")
        return pd.DataFrame()

    hotel_col = "hotel_name" if "hotel_name" in df.columns else None
    group = ["month_day"] + ([hotel_col] if hotel_col else [])

    rows = []
    for yr_idx in range(1, len(years)):
        cy = years[yr_idx]
        py = years[yr_idx - 1]
        cy_df = df[df["year"] == cy]
        py_df = df[df["year"] == py]

        cy_agg = cy_df.groupby(group)[metrics].mean()
        py_agg = py_df.groupby(group)[metrics].mean()

        for m in metrics:
            if m in cy_agg.columns and m in py_agg.columns:
                joined = cy_agg[[m]].join(py_agg[[m]], rsuffix="_ly").reset_index()
                joined["year"] = cy
                joined["metric"] = m
                joined["current"] = joined[m]
                joined["benchmark"] = joined[f"{m}_ly"]
                joined["pace_var"] = joined["current"] - joined["benchmark"]
                joined["pace_pct"] = np.where(
                    joined["benchmark"] != 0,
                    (joined["pace_var"] / joined["benchmark"].abs()) * 100,
                    np.nan,
                )
                joined = joined.drop(columns=[m, f"{m}_ly"])
                rows.append(joined)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pace_vs_budget(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    budget_map = {"rooms_sold": "budget_rooms", "revenue": "budget_revenue", "adr": "budget_adr"}
    rows = []
    for m in metrics:
        bm = budget_map.get(m)
        if bm and bm in df.columns:
            tmp = df[["date"] + (["hotel_name"] if "hotel_name" in df.columns else []) + [m, bm]].copy()
            tmp["metric"] = m
            tmp["current"] = tmp[m]
            tmp["benchmark"] = tmp[bm]
            tmp["pace_var"] = tmp["current"] - tmp["benchmark"]
            tmp["pace_pct"] = np.where(
                tmp["benchmark"] != 0,
                (tmp["pace_var"] / tmp["benchmark"].abs()) * 100,
                np.nan,
            )
            rows.append(tmp.drop(columns=[m, bm]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pace_vs_forecast(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    fcast_map = {
        "rooms_sold": "forecast_rooms",
        "revenue": "forecast_revenue",
        "adr": "forecast_adr",
        "occupancy_pct": "forecast_occupancy_pct",
    }
    rows = []
    for m in metrics:
        fm = fcast_map.get(m)
        if fm and fm in df.columns:
            tmp = df[["date"] + (["hotel_name"] if "hotel_name" in df.columns else []) + [m, fm]].copy()
            tmp["metric"] = m
            tmp["current"] = tmp[m]
            tmp["benchmark"] = tmp[fm]
            tmp["pace_var"] = tmp["current"] - tmp["benchmark"]
            tmp["pace_pct"] = np.where(
                tmp["benchmark"] != 0,
                (tmp["pace_var"] / tmp["benchmark"].abs()) * 100,
                np.nan,
            )
            rows.append(tmp.drop(columns=[m, fm]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pace_vs_portfolio_avg(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if "hotel_name" not in df.columns:
        return pd.DataFrame()

    portfolio_avg = df.groupby("date")[metrics].mean()
    rows = []
    for hotel, hdf in df.groupby("hotel_name"):
        hdf = hdf.set_index("date")
        for m in metrics:
            if m not in hdf.columns or m not in portfolio_avg.columns:
                continue
            tmp = hdf[[m]].join(portfolio_avg[[m]], rsuffix="_portfolio").reset_index()
            tmp["hotel_name"] = hotel
            tmp["metric"] = m
            tmp["current"] = tmp[m]
            tmp["benchmark"] = tmp[f"{m}_portfolio"]
            tmp["pace_var"] = tmp["current"] - tmp["benchmark"]
            tmp["pace_pct"] = np.where(
                tmp["benchmark"] != 0,
                (tmp["pace_var"] / tmp["benchmark"].abs()) * 100,
                np.nan,
            )
            rows.append(tmp.drop(columns=[m, f"{m}_portfolio"]))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def booking_window_distribution(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute lead-time (booking window) distribution.
    Requires booking_date and date columns.
    """
    if "booking_date" not in df.columns or "date" not in df.columns:
        return None

    df = df.copy()
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["lead_days"] = (df["date"] - df["booking_date"]).dt.days
    df = df.dropna(subset=["lead_days"])
    df = df[df["lead_days"] >= 0]

    if df.empty:
        return None

    group = ["hotel_name"] if "hotel_name" in df.columns else []
    agg_cols = group + ["lead_days"]
    out = df[agg_cols].copy()
    return out
