"""Anomaly detection using Z-score and rolling deviation methods."""

import logging

import pandas as pd
import numpy as np

from config.settings import ANOMALY_ZSCORE_THRESHOLD, ANOMALY_ROLLING_WINDOW

logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    metrics: list[str] = ["revenue", "occupancy_pct", "adr", "revpar"],
    method: str = "both",
) -> pd.DataFrame:
    """
    Flag anomalous rows using Z-score and/or rolling deviation.

    Parameters
    ----------
    method : 'zscore' | 'rolling' | 'both'

    Returns a copy of df with additional columns:
        <metric>_zscore
        <metric>_is_anomaly
        anomaly_flag  (True if any metric is anomalous)
        anomaly_reason (text description)
    """
    if df.empty or "date" not in df.columns:
        return df

    result = df.copy()
    result["anomaly_flag"] = False
    result["anomaly_reason"] = ""

    available = [m for m in metrics if m in result.columns]
    if not available:
        return result

    for metric in available:
        series = result[metric].copy()

        # ── Z-score ────────────────────────────────────────────────────────
        if method in ("zscore", "both"):
            mean = series.mean()
            std = series.std()
            if std > 0:
                zscore = (series - mean) / std
                result[f"{metric}_zscore"] = zscore.round(3)
                is_anom = zscore.abs() > ANOMALY_ZSCORE_THRESHOLD
                result[f"{metric}_zscore_anomaly"] = is_anom
                result.loc[is_anom, "anomaly_flag"] = True
                result.loc[is_anom, "anomaly_reason"] += f"{metric}(z-score) "
            else:
                result[f"{metric}_zscore"] = 0.0
                result[f"{metric}_zscore_anomaly"] = False

        # ── Rolling deviation ──────────────────────────────────────────────
        if method in ("rolling", "both"):
            win = ANOMALY_ROLLING_WINDOW
            rolling_mean = series.rolling(win, min_periods=3, center=True).mean()
            rolling_std = series.rolling(win, min_periods=3, center=True).std()
            rolling_std = rolling_std.replace(0, np.nan)
            roll_z = ((series - rolling_mean) / rolling_std).abs()
            is_roll_anom = roll_z > ANOMALY_ZSCORE_THRESHOLD
            result[f"{metric}_roll_anomaly"] = is_roll_anom
            result.loc[is_roll_anom, "anomaly_flag"] = True
            result.loc[is_roll_anom, "anomaly_reason"] += f"{metric}(rolling) "

    result["anomaly_reason"] = result["anomaly_reason"].str.strip()
    n = result["anomaly_flag"].sum()
    logger.info("Anomaly detection: %d anomalies flagged out of %d rows", n, len(result))
    return result


def anomaly_summary(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """Return a concise table of anomalous rows."""
    if flagged_df.empty or "anomaly_flag" not in flagged_df.columns:
        return pd.DataFrame()

    anom = flagged_df[flagged_df["anomaly_flag"]].copy()
    if anom.empty:
        return pd.DataFrame()

    display_cols = [c for c in [
        "date", "hotel_name", "occupancy_pct", "adr", "revpar", "revenue",
        "anomaly_reason",
    ] if c in anom.columns]

    return anom[display_cols].sort_values("date").reset_index(drop=True)
