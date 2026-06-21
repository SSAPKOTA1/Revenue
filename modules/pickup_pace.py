"""
Pickup and Pace analysis using snapshot_date.

KEY CONCEPTS
------------
snapshot_date : the date the file was exported / the booking-on-books was recorded
                (derived from the folder structure: year/month/day)
date          : the arrival / stay date (the future date the booking is for)

PICKUP
------
For a given (hotel, arrival_date), pickup is the change in rooms/revenue/ADR
between consecutive snapshot dates:

    pickup_Nd(hotel, arrival_date, snapshot) =
        value(snapshot) - value(snapshot - N days)

PACE
----
Pace compares the current snapshot's on-books position for a future date
against a benchmark (last year, budget, forecast, portfolio average):

    pace_var = current_value - benchmark_value
    pace_pct = pace_var / |benchmark_value| × 100
"""

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
    Compute pickup (change in on-books) between consecutive snapshot dates.

    If snapshot_date is present (multi-snapshot data):
        For each (hotel, arrival_date) pair, compute the change in each
        metric between snapshot T and snapshot T-N days.

    If snapshot_date is absent (single-snapshot / daily data):
        Fall back to day-over-day differences in the daily time series.

    Returns the input DataFrame with additional pickup columns:
        <metric>_pickup_<N>d
    """
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()

    available = [m for m in metrics if m in df.columns]
    if not available:
        return pd.DataFrame()

    if 'snapshot_date' in df.columns and df['snapshot_date'].notna().any():
        return _pickup_from_snapshots(df, windows, available)
    else:
        return _pickup_from_daily(df, windows, available)


def _pickup_from_snapshots(
    df: pd.DataFrame,
    windows: list[int],
    metrics: list[str],
) -> pd.DataFrame:
    """
    True pickup: how much did on-books change between snapshots?

    For each (hotel, arrival_date), we have a time series of snapshots.
    Pickup_Nd at snapshot T = value(T) - value(T - N days).
    """
    df = df.copy()
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], errors='coerce')
    df['date']          = pd.to_datetime(df['date'],          errors='coerce')
    df = df.dropna(subset=['snapshot_date', 'date'])

    hotel_col = 'hotel_name' if 'hotel_name' in df.columns else None
    group_keys = (['hotel_name'] if hotel_col else []) + ['date']

    parts = []
    groups = df.groupby(group_keys)

    for key, grp in groups:
        grp = grp.sort_values('snapshot_date').reset_index(drop=True)
        grp = grp.set_index('snapshot_date')

        for m in metrics:
            if m not in grp.columns:
                continue
            series = grp[m]
            for w in windows:
                lag_idx = grp.index - pd.Timedelta(days=w)
                lag_vals = series.reindex(lag_idx).values
                grp[f'{m}_pickup_{w}d'] = series.values - lag_vals

        parts.append(grp.reset_index())

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    logger.info(
        "Pickup computed from %d snapshots across %d (hotel, date) pairs",
        df['snapshot_date'].nunique(), len(groups),
    )
    return out


def _pickup_from_daily(
    df: pd.DataFrame,
    windows: list[int],
    metrics: list[str],
) -> pd.DataFrame:
    """Fallback: pickup as day-over-day difference in the daily time series."""
    group_keys = ['hotel_name'] if 'hotel_name' in df.columns else []
    agg_keys   = group_keys + ['date']
    agg = df.groupby(agg_keys)[metrics].sum(min_count=1).reset_index()
    agg = agg.sort_values(agg_keys)

    if group_keys:
        parts = []
        for hotel, hdf in agg.groupby('hotel_name'):
            hdf = hdf.sort_values('date').copy()
            for m in metrics:
                for w in windows:
                    hdf[f'{m}_pickup_{w}d'] = hdf[m].diff(w)
            parts.append(hdf)
        return pd.concat(parts, ignore_index=True)
    else:
        agg = agg.sort_values('date').copy()
        for m in metrics:
            for w in windows:
                agg[f'{m}_pickup_{w}d'] = agg[m].diff(w)
        return agg


def pickup_summary(
    pickup_df: pd.DataFrame,
    metrics: list[str] = ['rooms_sold', 'revenue'],
) -> pd.DataFrame:
    """Summarise average/total pickup by hotel and window."""
    if pickup_df.empty:
        return pd.DataFrame()

    hotels = pickup_df['hotel_name'].unique() if 'hotel_name' in pickup_df.columns else ['Portfolio']
    rows = []
    for hotel in hotels:
        hdf = pickup_df[pickup_df['hotel_name'] == hotel] if 'hotel_name' in pickup_df.columns else pickup_df
        for m in metrics:
            for w in [1, 7, 14, 30]:
                col = f'{m}_pickup_{w}d'
                if col in hdf.columns:
                    s = hdf[col].dropna()
                    rows.append({
                        'hotel_name':   hotel,
                        'metric':       m,
                        'window_days':  w,
                        'avg_pickup':   round(float(s.mean()), 2) if len(s) else np.nan,
                        'total_pickup': round(float(s.sum()),  2) if len(s) else np.nan,
                        'max_pickup':   round(float(s.max()),  2) if len(s) else np.nan,
                        'min_pickup':   round(float(s.min()),  2) if len(s) else np.nan,
                    })
    return pd.DataFrame(rows)


def pickup_curve(
    df: pd.DataFrame,
    hotel: str,
    arrival_date: pd.Timestamp,
    metric: str = 'rooms_sold',
) -> pd.DataFrame:
    """
    For a specific hotel and arrival date, return the pickup curve:
    how the on-books position built up across snapshot dates.

    Returns DataFrame with columns: snapshot_date, <metric>, days_to_arrival
    """
    if 'snapshot_date' not in df.columns:
        return pd.DataFrame()

    mask = (df['hotel_name'] == hotel) & (df['date'] == arrival_date) if 'hotel_name' in df.columns \
        else (df['date'] == arrival_date)
    sub = df[mask].copy()
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()

    sub['snapshot_date']   = pd.to_datetime(sub['snapshot_date'])
    sub['days_to_arrival'] = (pd.Timestamp(arrival_date) - sub['snapshot_date']).dt.days
    sub = sub[sub['days_to_arrival'] >= 0].sort_values('snapshot_date')
    return sub[['snapshot_date', 'days_to_arrival', metric]].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# PACE
# ═══════════════════════════════════════════════════════════════════════════

def compute_pace(
    df: pd.DataFrame,
    benchmark: str = 'last_year',
    metrics: list[str] = ['rooms_sold', 'revenue', 'adr', 'occupancy_pct'],
    as_of_snapshot: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Pace: current on-books vs benchmark for each future arrival date.

    If snapshot_date is present, uses the most recent snapshot (or
    as_of_snapshot if specified) as 'current'.

    Parameters
    ----------
    benchmark       : 'last_year' | 'budget' | 'forecast' | 'portfolio_avg'
    as_of_snapshot  : use this specific snapshot date as 'current';
                      defaults to the latest available snapshot
    """
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()

    available = [m for m in metrics if m in df.columns]
    if not available:
        return pd.DataFrame()

    # Select the 'current' view: latest snapshot (or specified one)
    if 'snapshot_date' in df.columns and df['snapshot_date'].notna().any():
        df = df.copy()
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], errors='coerce')
        if as_of_snapshot:
            current_df = df[df['snapshot_date'] == as_of_snapshot]
        else:
            latest = df['snapshot_date'].max()
            current_df = df[df['snapshot_date'] == latest]
        logger.info("Pace using snapshot: %s", df['snapshot_date'].max())
    else:
        current_df = df

    if current_df.empty:
        return pd.DataFrame()

    if benchmark == 'last_year':
        return _pace_vs_last_year(df, current_df, available)
    elif benchmark == 'budget':
        return _pace_vs_budget(current_df, available)
    elif benchmark == 'forecast':
        return _pace_vs_forecast(current_df, available)
    elif benchmark == 'portfolio_avg':
        return _pace_vs_portfolio_avg(current_df, available)
    else:
        return _pace_vs_last_year(df, current_df, available)


def _pace_vs_last_year(
    full_df: pd.DataFrame,
    current_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Pace vs same period last year.

    For snapshot-based data: compare latest snapshot's on-books
    to the equivalent snapshot from 365 days ago.
    For single-snapshot data: compare this year's actuals to last year's.
    """
    has_snapshots = 'snapshot_date' in full_df.columns and full_df['snapshot_date'].notna().any()

    if has_snapshots:
        # Find a snapshot from ~1 year ago
        latest_snap    = pd.to_datetime(full_df['snapshot_date']).max()
        target_ly_snap = latest_snap - pd.DateOffset(years=1)
        # Pick the closest available snapshot to last year same date
        snaps = pd.to_datetime(full_df['snapshot_date']).dropna().unique()
        if len(snaps) < 2:
            logger.warning("Not enough snapshots for LY pace comparison.")
            return pd.DataFrame()
        diffs  = abs(pd.DatetimeIndex(snaps) - target_ly_snap)
        ly_snap = snaps[diffs.argmin()]
        ly_df   = full_df[pd.to_datetime(full_df['snapshot_date']) == ly_snap]
    else:
        # Use calendar year split
        current_df = current_df.copy()
        current_df['year'] = pd.to_datetime(current_df['date']).dt.year
        years = sorted(current_df['year'].unique())
        if len(years) < 2:
            logger.warning("Need ≥2 years of data for LY pace.")
            return pd.DataFrame()
        cy = years[-1]
        py = years[-2]
        current_df = current_df[current_df['year'] == cy]
        ly_df      = full_df[pd.to_datetime(full_df['date']).dt.year == py].copy()
        ly_df['date'] = pd.to_datetime(ly_df['date']) + pd.DateOffset(years=1)

    return _build_pace_df(current_df, ly_df, metrics, label='Last Year')


def _build_pace_df(
    current: pd.DataFrame,
    benchmark: pd.DataFrame,
    metrics: list[str],
    label: str = 'Benchmark',
) -> pd.DataFrame:
    hotel_col = 'hotel_name' if 'hotel_name' in current.columns else None
    group     = (['hotel_name'] if hotel_col else []) + ['date']

    c_agg = current.groupby(group)[metrics].mean()
    b_agg = benchmark.groupby(group)[metrics].mean()

    rows = []
    for m in metrics:
        if m not in c_agg.columns or m not in b_agg.columns:
            continue
        joined = c_agg[[m]].join(b_agg[[m]], rsuffix='_bm').reset_index()
        joined['metric']     = m
        joined['benchmark_label'] = label
        joined['current']    = joined[m]
        joined['benchmark']  = joined[f'{m}_bm']
        joined['pace_var']   = joined['current'] - joined['benchmark']
        joined['pace_pct']   = np.where(
            joined['benchmark'] != 0,
            joined['pace_var'] / joined['benchmark'].abs() * 100,
            np.nan,
        )
        rows.append(joined.drop(columns=[m, f'{m}_bm']))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pace_vs_budget(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    bmap = {'rooms_sold': 'budget_rooms', 'revenue': 'budget_revenue', 'adr': 'budget_adr'}
    rows = []
    hotel_col = ['hotel_name'] if 'hotel_name' in df.columns else []
    for m in metrics:
        bm = bmap.get(m)
        if bm and bm in df.columns:
            tmp = df[hotel_col + ['date', m, bm]].copy()
            tmp['metric']          = m
            tmp['benchmark_label'] = 'Budget'
            tmp['current']         = tmp[m]
            tmp['benchmark']       = tmp[bm]
            tmp['pace_var']        = tmp['current'] - tmp['benchmark']
            tmp['pace_pct']        = np.where(
                tmp['benchmark'] != 0,
                tmp['pace_var'] / tmp['benchmark'].abs() * 100, np.nan)
            rows.append(tmp.drop(columns=[m, bm]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pace_vs_forecast(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    fmap = {
        'rooms_sold':    'forecast_rooms',
        'revenue':       'forecast_revenue',
        'adr':           'forecast_adr',
        'occupancy_pct': 'forecast_occupancy_pct',
    }
    rows = []
    hotel_col = ['hotel_name'] if 'hotel_name' in df.columns else []
    for m in metrics:
        fm = fmap.get(m)
        if fm and fm in df.columns:
            tmp = df[hotel_col + ['date', m, fm]].copy()
            tmp['metric']          = m
            tmp['benchmark_label'] = 'Forecast'
            tmp['current']         = tmp[m]
            tmp['benchmark']       = tmp[fm]
            tmp['pace_var']        = tmp['current'] - tmp['benchmark']
            tmp['pace_pct']        = np.where(
                tmp['benchmark'] != 0,
                tmp['pace_var'] / tmp['benchmark'].abs() * 100, np.nan)
            rows.append(tmp.drop(columns=[m, fm]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pace_vs_portfolio_avg(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if 'hotel_name' not in df.columns:
        return pd.DataFrame()
    portfolio_avg = df.groupby('date')[metrics].mean()
    rows = []
    for hotel, hdf in df.groupby('hotel_name'):
        hdf_idx = hdf.set_index('date')
        for m in metrics:
            if m not in hdf_idx.columns or m not in portfolio_avg.columns:
                continue
            joined = hdf_idx[[m]].join(portfolio_avg[[m]], rsuffix='_p').reset_index()
            joined['hotel_name']      = hotel
            joined['metric']          = m
            joined['benchmark_label'] = 'Portfolio Avg'
            joined['current']         = joined[m]
            joined['benchmark']       = joined[f'{m}_p']
            joined['pace_var']        = joined['current'] - joined['benchmark']
            joined['pace_pct']        = np.where(
                joined['benchmark'] != 0,
                joined['pace_var'] / joined['benchmark'].abs() * 100, np.nan)
            rows.append(joined.drop(columns=[m, f'{m}_p']))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# BOOKING WINDOW
# ═══════════════════════════════════════════════════════════════════════════

def booking_window_distribution(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute lead-time from snapshot_date → arrival date.
    snapshot_date acts as the proxy for booking date when
    no explicit booking_date column exists.
    """
    date_col = None
    if 'booking_date' in df.columns:
        date_col = 'booking_date'
    elif 'snapshot_date' in df.columns:
        date_col = 'snapshot_date'

    if date_col is None or 'date' not in df.columns:
        return None

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    out['date']   = pd.to_datetime(out['date'],   errors='coerce')
    out['lead_days'] = (out['date'] - out[date_col]).dt.days
    out = out[out['lead_days'] >= 0].dropna(subset=['lead_days'])

    if out.empty:
        return None

    keep = (['hotel_name'] if 'hotel_name' in out.columns else []) + ['lead_days']
    return out[keep].copy()
