"""Forecasting engine: Prophet, Exponential Smoothing, Moving Average."""

import logging
import warnings
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def _prep_series(df: pd.DataFrame, metric: str) -> Optional[pd.DataFrame]:
    """Return clean (date, value) series for a metric, sorted by date."""
    if metric not in df.columns or "date" not in df.columns:
        return None
    s = df[["date", metric]].dropna().copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.groupby("date")[metric].mean().reset_index()
    s = s.sort_values("date").reset_index(drop=True)
    if len(s) < 10:
        return None
    return s


def _make_future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")


# ── Prophet ────────────────────────────────────────────────────────────────

def forecast_prophet(
    df: pd.DataFrame,
    metric: str = "revpar",
    horizon: int = 90,
) -> Optional[pd.DataFrame]:
    """Forecast using Facebook Prophet."""
    try:
        from prophet import Prophet  # type: ignore
    except ImportError:
        logger.warning("Prophet not installed; skipping.")
        return None

    series = _prep_series(df, metric)
    if series is None:
        return None

    prophet_df = series.rename(columns={"date": "ds", metric: "y"})

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(prophet_df)

        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)

        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out.columns = ["date", "forecast", "lower_80", "upper_80"]
        out["date"] = pd.to_datetime(out["date"])
        out["method"] = "Prophet"
        out["metric"] = metric

        # Mark historical vs future
        last_hist = series["date"].max()
        out["is_forecast"] = out["date"] > last_hist

        logger.info("Prophet forecast complete for '%s', horizon=%d", metric, horizon)
        return out

    except Exception as e:
        logger.error("Prophet forecast failed for '%s': %s", metric, e)
        return None


# ── Exponential Smoothing ──────────────────────────────────────────────────

def forecast_exp_smoothing(
    df: pd.DataFrame,
    metric: str = "revpar",
    horizon: int = 90,
) -> Optional[pd.DataFrame]:
    """Forecast using Holt-Winters Exponential Smoothing."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except ImportError:
        logger.warning("statsmodels not installed; skipping.")
        return None

    series = _prep_series(df, metric)
    if series is None:
        return None

    y = series[metric].values

    try:
        if len(y) >= 2 * 7:
            model = ExponentialSmoothing(
                y, trend="add", seasonal="add", seasonal_periods=7, damped_trend=True
            )
        elif len(y) >= 14:
            model = ExponentialSmoothing(y, trend="add", damped_trend=True)
        else:
            model = ExponentialSmoothing(y, trend="add")

        fit = model.fit(optimized=True, remove_bias=True)
        fcast_values = fit.forecast(horizon)

        # Naive confidence interval: ±1.96 * residual std
        resid_std = float(fit.resid.std()) if hasattr(fit, "resid") else float(np.std(y) * 0.1)
        lower = fcast_values - 1.96 * resid_std
        upper = fcast_values + 1.96 * resid_std

        future_dates = _make_future_dates(series["date"].max(), horizon)

        # Historical fitted
        hist = pd.DataFrame({
            "date": series["date"],
            "forecast": fit.fittedvalues,
            "lower_80": fit.fittedvalues - resid_std,
            "upper_80": fit.fittedvalues + resid_std,
            "is_forecast": False,
        })

        fcast = pd.DataFrame({
            "date": future_dates,
            "forecast": np.maximum(fcast_values, 0),
            "lower_80": np.maximum(lower, 0),
            "upper_80": np.maximum(upper, 0),
            "is_forecast": True,
        })

        out = pd.concat([hist, fcast], ignore_index=True)
        out["method"] = "Exponential Smoothing"
        out["metric"] = metric
        logger.info("ExpSmoothing forecast complete for '%s', horizon=%d", metric, horizon)
        return out

    except Exception as e:
        logger.error("Exponential smoothing failed for '%s': %s", metric, e)
        return None


# ── Moving Average ─────────────────────────────────────────────────────────

def forecast_moving_average(
    df: pd.DataFrame,
    metric: str = "revpar",
    horizon: int = 90,
    window: int = 7,
) -> Optional[pd.DataFrame]:
    """Forecast using a simple moving average with linear trend extrapolation."""
    series = _prep_series(df, metric)
    if series is None:
        return None

    y = series[metric].values
    ma = pd.Series(y).rolling(window, min_periods=1).mean().values

    # Linear trend from last 30 data points
    n_trend = min(30, len(y))
    x = np.arange(n_trend)
    y_trend = y[-n_trend:]
    if len(x) >= 2:
        coeffs = np.polyfit(x, y_trend, 1)
        slope, intercept = coeffs
    else:
        slope, intercept = 0, float(y[-1])

    last_val = float(ma[-1])
    trend_factor = slope
    std_dev = float(np.std(y[-30:])) if len(y) >= 30 else float(np.std(y))

    future_dates = _make_future_dates(series["date"].max(), horizon)
    fcast_values = np.array([
        max(0, last_val + trend_factor * (i + 1))
        for i in range(horizon)
    ])

    hist = pd.DataFrame({
        "date": series["date"],
        "forecast": ma,
        "lower_80": ma - std_dev,
        "upper_80": ma + std_dev,
        "is_forecast": False,
    })
    fcast = pd.DataFrame({
        "date": future_dates,
        "forecast": fcast_values,
        "lower_80": np.maximum(fcast_values - 1.96 * std_dev, 0),
        "upper_80": fcast_values + 1.96 * std_dev,
        "is_forecast": True,
    })

    out = pd.concat([hist, fcast], ignore_index=True)
    out["method"] = "Moving Average"
    out["metric"] = metric
    logger.info("Moving Average forecast complete for '%s', horizon=%d", metric, horizon)
    return out


# ── Dispatcher ─────────────────────────────────────────────────────────────

def run_forecast(
    df: pd.DataFrame,
    method: str = "Prophet",
    metric: str = "revpar",
    horizon: int = 90,
) -> Optional[pd.DataFrame]:
    """Dispatch to the appropriate forecasting method."""
    dispatch = {
        "Prophet": forecast_prophet,
        "Exponential Smoothing": forecast_exp_smoothing,
        "Moving Average": forecast_moving_average,
    }
    func = dispatch.get(method)
    if func is None:
        logger.warning("Unknown forecast method '%s'", method)
        return None
    return func(df, metric=metric, horizon=horizon)


def compute_forecast_accuracy(
    actuals: pd.DataFrame,
    forecast_df: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    """
    Compare actual vs forecasted values on overlapping dates.
    Returns MAPE, MAE, RMSE.
    """
    if actuals.empty or forecast_df is None or forecast_df.empty:
        return {}

    actual_s = actuals[["date", metric]].dropna()
    actual_s["date"] = pd.to_datetime(actual_s["date"])
    fcast_s = forecast_df[["date", "forecast"]].copy()
    fcast_s["date"] = pd.to_datetime(fcast_s["date"])

    merged = actual_s.merge(fcast_s, on="date")
    merged = merged[merged[metric] != 0]
    if merged.empty:
        return {}

    a = merged[metric].values
    f = merged["forecast"].values
    mape = float(np.mean(np.abs((a - f) / np.abs(a))) * 100)
    mae = float(np.mean(np.abs(a - f)))
    rmse = float(np.sqrt(np.mean((a - f) ** 2)))

    return {"MAPE_%": round(mape, 2), "MAE": round(mae, 2), "RMSE": round(rmse, 2)}
