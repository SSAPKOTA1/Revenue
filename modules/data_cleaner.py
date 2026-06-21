"""Data cleaning: remove duplicates, cap outliers, impute, normalise."""

import logging

import pandas as pd
import numpy as np

from config.settings import VALIDATION_RULES, NUMERIC_COLS

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Steps
    -----
    1. Drop rows without a date.
    2. Remove duplicate (date, hotel_name) pairs — keep last.
    3. Cap out-of-range values to valid bounds (do not drop).
    4. Clip negative revenues/rooms to 0.
    5. Re-derive occupancy_pct, ADR, RevPAR where possible.
    6. Convert hotel_name to consistent title-case.
    7. Add derived time columns (year, month, quarter, day_of_week).
    """
    if df.empty:
        return df

    original_len = len(df)

    # 1. Drop rows without date
    if "date" in df.columns:
        df = df.dropna(subset=["date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    # 2. Dedup — include snapshot_date so each (hotel, arrival_date, snapshot) is unique
    dedup_keys = [c for c in ["date", "hotel_name", "snapshot_date"] if c in df.columns]
    if dedup_keys:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
        logger.info("Dedup removed %d rows", before - len(df))

    # 3. Cap range violations (don't drop — just cap)
    for col, rules in VALIDATION_RULES.items():
        if col not in df.columns:
            continue
        lo, hi = rules.get("min", -np.inf), rules.get("max", np.inf)
        before_cap = df[col].copy()
        df[col] = df[col].clip(lower=lo, upper=hi)
        n_capped = (df[col] != before_cap).sum()
        if n_capped:
            logger.debug("Capped %d values in '%s' to [%s, %s]", n_capped, col, lo, hi)

    # 4. Clip negative numeric values
    for col in ["revenue", "rooms_sold", "rooms_available", "adr", "revpar"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # 5. Re-derive KPIs
    df = _rederive(df)

    # 6. Normalise hotel names
    if "hotel_name" in df.columns:
        df["hotel_name"] = (
            df["hotel_name"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.title()
        )

    # 7. Time dimension columns
    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")
        df["quarter"] = df["date"].dt.quarter
        df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon
        df["day_name"] = df["date"].dt.strftime("%a")
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["day_of_year"] = df["date"].dt.dayofyear
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

    logger.info(
        "Cleaning complete: %d → %d rows (%.1f%% retained)",
        original_len, len(df), len(df) / original_len * 100 if original_len else 0,
    )
    return df.reset_index(drop=True)


def _rederive(df: pd.DataFrame) -> pd.DataFrame:
    has = lambda c: c in df.columns  # noqa: E731

    if has("rooms_sold") and has("rooms_available"):
        mask = df["rooms_available"] > 0
        df.loc[mask, "occupancy_pct"] = (
            df.loc[mask, "rooms_sold"] / df.loc[mask, "rooms_available"] * 100
        ).clip(0, 100)

    if has("revenue") and has("rooms_sold"):
        mask = df["rooms_sold"] > 0
        df.loc[mask, "adr"] = (df.loc[mask, "revenue"] / df.loc[mask, "rooms_sold"]).clip(0)

    if has("revenue") and has("rooms_available"):
        mask = df["rooms_available"] > 0
        df.loc[mask, "revpar"] = (df.loc[mask, "revenue"] / df.loc[mask, "rooms_available"]).clip(0)
    elif has("adr") and has("occupancy_pct"):
        df["revpar"] = (df["adr"] * df["occupancy_pct"] / 100).clip(0)

    return df
