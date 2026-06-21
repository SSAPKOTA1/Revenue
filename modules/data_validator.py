"""Data validation: check KPI ranges, generate quality score and report."""

import logging
from typing import Any

import pandas as pd
import numpy as np

from config.settings import VALIDATION_RULES

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run validation checks on the master DataFrame.

    Returns
    -------
    dict with keys:
        score        : float  0-100 data quality score
        issues       : list[dict]  per-column issue summaries
        total_rows   : int
        valid_rows   : int
        summary      : str
    """
    if df.empty:
        return {
            "score": 0.0,
            "issues": [],
            "total_rows": 0,
            "valid_rows": 0,
            "summary": "No data loaded.",
        }

    total = len(df)
    issues: list[dict] = []
    penalty = 0.0

    # ── Missing date ───────────────────────────────────────────────────────
    if "date" in df.columns:
        n_missing_date = df["date"].isna().sum()
        if n_missing_date:
            pct = n_missing_date / total * 100
            issues.append({
                "column": "date",
                "check": "missing",
                "count": int(n_missing_date),
                "pct": round(pct, 2),
                "severity": "high",
            })
            penalty += min(pct * 0.5, 20)
    else:
        issues.append({
            "column": "date",
            "check": "missing_column",
            "count": total,
            "pct": 100.0,
            "severity": "high",
        })
        penalty += 20

    # ── Range checks ──────────────────────────────────────────────────────
    for col, rules in VALIDATION_RULES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        lo, hi = rules.get("min", -np.inf), rules.get("max", np.inf)
        out_of_range = ((series < lo) | (series > hi)).sum()
        if out_of_range:
            pct = out_of_range / total * 100
            severity = "high" if pct > 10 else "medium" if pct > 2 else "low"
            issues.append({
                "column": col,
                "check": f"out_of_range [{lo}, {hi}]",
                "count": int(out_of_range),
                "pct": round(pct, 2),
                "severity": severity,
            })
            penalty += min(pct * 0.2, 10)

        # Missing values
        n_missing = df[col].isna().sum()
        if n_missing:
            pct = n_missing / total * 100
            issues.append({
                "column": col,
                "check": "missing",
                "count": int(n_missing),
                "pct": round(pct, 2),
                "severity": "medium" if pct > 5 else "low",
            })
            penalty += min(pct * 0.1, 5)

    # ── Duplicate rows ─────────────────────────────────────────────────────
    dup_cols = [c for c in ["date", "hotel_name"] if c in df.columns]
    if dup_cols:
        dupes = df.duplicated(subset=dup_cols).sum()
        if dupes:
            pct = dupes / total * 100
            issues.append({
                "column": "+".join(dup_cols),
                "check": "duplicate_rows",
                "count": int(dupes),
                "pct": round(pct, 2),
                "severity": "medium",
            })
            penalty += min(pct * 0.3, 15)

    # ── Missing hotel_name ─────────────────────────────────────────────────
    if "hotel_name" in df.columns:
        n_no_hotel = df["hotel_name"].isna().sum()
        if n_no_hotel:
            pct = n_no_hotel / total * 100
            issues.append({
                "column": "hotel_name",
                "check": "missing",
                "count": int(n_no_hotel),
                "pct": round(pct, 2),
                "severity": "high",
            })
            penalty += min(pct * 0.5, 15)

    score = max(0.0, round(100.0 - penalty, 1))

    # Count rows that have at minimum a date and one KPI
    key_cols = [c for c in ["date", "rooms_sold", "revenue", "adr"] if c in df.columns]
    valid_rows = int(df[key_cols].dropna(how="all").shape[0]) if key_cols else total

    # Issues sorted by severity
    sev_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: sev_order.get(x["severity"], 3))

    summary = _build_summary(score, issues, total, valid_rows)
    logger.info("Validation complete — score %.1f, %d issues", score, len(issues))

    return {
        "score": score,
        "issues": issues,
        "total_rows": total,
        "valid_rows": valid_rows,
        "summary": summary,
    }


def _build_summary(score: float, issues: list[dict], total: int, valid: int) -> str:
    high = sum(1 for i in issues if i["severity"] == "high")
    med = sum(1 for i in issues if i["severity"] == "medium")
    low = sum(1 for i in issues if i["severity"] == "low")
    grade = "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Poor"
    return (
        f"Data Quality: {grade} ({score}/100) | "
        f"{total:,} total rows, {valid:,} valid | "
        f"{high} high / {med} medium / {low} low severity issues"
    )
