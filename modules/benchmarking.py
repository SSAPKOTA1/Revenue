"""Benchmarking: rank hotels, compute variance, contribution analysis."""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def rank_hotels(
    df: pd.DataFrame,
    metric: str = "revpar",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank all hotels by a given KPI metric.

    Returns sorted DataFrame with rank column.
    """
    if df.empty or "hotel_name" not in df.columns or metric not in df.columns:
        return pd.DataFrame()

    agg = df.groupby("hotel_name").agg(
        rooms_sold=("rooms_sold", "sum") if "rooms_sold" in df.columns else (metric, "count"),
        rooms_available=("rooms_available", "sum") if "rooms_available" in df.columns else (metric, "count"),
        revenue=("revenue", "sum") if "revenue" in df.columns else (metric, "count"),
    )

    # Derived KPIs from totals
    if "rooms_sold" in agg.columns and "rooms_available" in agg.columns:
        agg["occupancy_pct"] = (agg["rooms_sold"] / agg["rooms_available"].replace(0, np.nan)) * 100
    if "revenue" in agg.columns and "rooms_sold" in agg.columns:
        agg["adr"] = agg["revenue"] / agg["rooms_sold"].replace(0, np.nan)
    if "revenue" in agg.columns and "rooms_available" in agg.columns:
        agg["revpar"] = agg["revenue"] / agg["rooms_available"].replace(0, np.nan)

    agg = agg.reset_index()

    if metric not in agg.columns:
        # Fall back to mean of the requested metric
        fallback = df.groupby("hotel_name")[metric].mean().reset_index()
        agg = agg.merge(fallback, on="hotel_name", how="left")

    agg = agg.sort_values(metric, ascending=ascending).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    return agg


def portfolio_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each hotel's share of portfolio totals for key metrics.
    """
    if df.empty or "hotel_name" not in df.columns:
        return pd.DataFrame()

    metrics = [m for m in ["revenue", "rooms_sold", "rooms_available"] if m in df.columns]
    if not metrics:
        return pd.DataFrame()

    hotel_totals = df.groupby("hotel_name")[metrics].sum(min_count=1).reset_index()
    portfolio_totals = hotel_totals[metrics].sum()

    for m in metrics:
        total = portfolio_totals[m]
        hotel_totals[f"{m}_share_pct"] = (
            (hotel_totals[m] / total * 100).round(2) if total > 0 else np.nan
        )

    hotel_totals = hotel_totals.sort_values(metrics[0], ascending=False).reset_index(drop=True)
    return hotel_totals


def variance_analysis(
    df: pd.DataFrame,
    compare_col: str = "budget_revenue",
    actual_col: str = "revenue",
) -> pd.DataFrame:
    """
    Variance between actual and budget (or any reference column) by hotel.
    """
    if df.empty:
        return pd.DataFrame()
    if actual_col not in df.columns or compare_col not in df.columns:
        return pd.DataFrame()

    group = ["hotel_name"] if "hotel_name" in df.columns else []
    agg = df.groupby(group)[[actual_col, compare_col]].sum(min_count=1).reset_index()
    agg["variance"] = agg[actual_col] - agg[compare_col]
    agg["variance_pct"] = np.where(
        agg[compare_col] != 0,
        (agg["variance"] / agg[compare_col].abs()) * 100,
        np.nan,
    )
    agg = agg.sort_values("variance", ascending=False).reset_index(drop=True)
    return agg


def portfolio_summary(df: pd.DataFrame) -> dict[str, float]:
    """Compute portfolio-level KPI totals."""
    result: dict[str, float] = {}
    if df.empty:
        return result

    if "rooms_sold" in df.columns:
        result["rooms_sold"] = float(df["rooms_sold"].sum())
    if "rooms_available" in df.columns:
        result["rooms_available"] = float(df["rooms_available"].sum())
    if "revenue" in df.columns:
        result["revenue"] = float(df["revenue"].sum())

    if "rooms_sold" in result and "rooms_available" in result and result["rooms_available"] > 0:
        result["occupancy_pct"] = result["rooms_sold"] / result["rooms_available"] * 100
    if "revenue" in result and result.get("rooms_sold", 0) > 0:
        result["adr"] = result["revenue"] / result["rooms_sold"]
    if "revenue" in result and result.get("rooms_available", 0) > 0:
        result["revpar"] = result["revenue"] / result["rooms_available"]

    n_hotels = df["hotel_name"].nunique() if "hotel_name" in df.columns else 1
    result["hotel_count"] = float(n_hotels)

    return result
