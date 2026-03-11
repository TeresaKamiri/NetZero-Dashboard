"""Forecasting and calibration utilities for historical emissions/energy series."""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning


Z80 = 1.2815515655446004
Z95 = 1.959963984540054


def _fit_arima_quiet(series_like):
    """Fit ARIMA with fallback orders while suppressing noisy expected warnings."""
    # These warnings are common during ARIMA initialization on short/noisy series.
    # They do not invalidate fitting and produce noisy logs in Streamlit.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Non-invertible starting MA parameters found.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Non-stationary starting autoregressive parameters found.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="An unsupported index was provided.*",
            category=ValueWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="No supported index is available.*",
            category=ValueWarning,
        )
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Primary spec first, then simpler fallbacks for short/noisy slices.
        candidate_orders = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (0, 1, 0)]
        model = None
        for order in candidate_orders:
            try:
                candidate = ARIMA(
                    series_like,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method_kwargs={"maxiter": 300})
                converged = bool(getattr(candidate, "mle_retvals", {}).get("converged", True))
                model = candidate
                if converged:
                    break
            except Exception:
                continue
        if model is None:
            raise RuntimeError("ARIMA fit failed for all fallback orders.")
    return model


def build_forecast_long(ts_df: pd.DataFrame, value_col: str, forecast_to: int = 2050) -> tuple[pd.DataFrame, object]:
    """Return long-format historical + forecast table and fitted ARIMA model."""
    out = pd.DataFrame(columns=["Year", "Value", "Type", "Metric"])
    model = None
    if ts_df.empty or value_col not in ts_df.columns:
        return out, model

    hist = ts_df[["Year", value_col]].dropna().sort_values("Year").copy()
    if hist.empty:
        return out, model

    hist_out = hist.rename(columns={value_col: "Value"})
    hist_out["Type"] = "Historical"
    hist_out["Metric"] = value_col

    last_year = int(hist["Year"].max())
    steps = int(forecast_to - last_year)
    if len(hist) < 5 or steps <= 0:
        return hist_out, model

    try:
        # Use a supported annual PeriodIndex so statsmodels forecasting is stable.
        y = hist[value_col].astype(float).copy()
        y.index = pd.PeriodIndex(hist["Year"].astype(int), freq="Y")
        model = _fit_arima_quiet(y)
        fc = model.forecast(steps=steps)
        fc_years = np.arange(last_year + 1, forecast_to + 1)
        fc_out = pd.DataFrame(
            {"Year": fc_years, "Value": fc.values, "Type": "Forecast", "Metric": value_col}
        )
        return pd.concat([hist_out, fc_out], ignore_index=True), model
    except Exception:
        return hist_out, model


def walk_forward_backtest(ts_df: pd.DataFrame, value_col: str, min_train_size: int = 6) -> pd.DataFrame:
    """Run expanding-window one-step backtest with interval diagnostics."""
    hist = ts_df[["Year", value_col]].dropna().sort_values("Year").copy()
    if len(hist) <= min_train_size:
        return pd.DataFrame()

    y = hist[value_col].values
    years = hist["Year"].values
    rows = []
    for i in range(min_train_size, len(hist)):
        train = y[:i]
        actual = float(y[i])
        year = int(years[i])
        try:
            model = _fit_arima_quiet(train)
            pred = float(model.forecast(steps=1)[0])
            sigma = float(np.std(model.resid)) if len(model.resid) > 1 else np.nan
            lo80 = pred - (Z80 * sigma) if np.isfinite(sigma) else np.nan
            hi80 = pred + (Z80 * sigma) if np.isfinite(sigma) else np.nan
            lo95 = pred - (Z95 * sigma) if np.isfinite(sigma) else np.nan
            hi95 = pred + (Z95 * sigma) if np.isfinite(sigma) else np.nan
            err = actual - pred
            ape = abs(err) / actual if actual != 0 else np.nan
            rows.append(
                {
                    "Year": year,
                    "Actual": actual,
                    "Predicted": pred,
                    "Sigma_model": sigma,
                    "PI80_lower": lo80,
                    "PI80_upper": hi80,
                    "PI95_lower": lo95,
                    "PI95_upper": hi95,
                    "Covered80": bool(np.isfinite(lo80) and lo80 <= actual <= hi80),
                    "Covered95": bool(np.isfinite(lo95) and lo95 <= actual <= hi95),
                    "Error": err,
                    "AbsError": abs(err),
                    "APE": ape,
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["RMSE_running"] = np.sqrt((out["Error"] ** 2).expanding().mean())
    out["MAE_running"] = out["AbsError"].expanding().mean()
    out["MAPE_running"] = (out["APE"].expanding().mean()) * 100
    return out


def backtest_calibration_summary(bt_df: pd.DataFrame) -> dict:
    """Summarize empirical interval coverage and average interval widths."""
    if bt_df.empty:
        return {}
    cov80 = float(bt_df["Covered80"].mean()) if "Covered80" in bt_df else np.nan
    cov95 = float(bt_df["Covered95"].mean()) if "Covered95" in bt_df else np.nan
    width80 = (
        float((bt_df["PI80_upper"] - bt_df["PI80_lower"]).mean())
        if {"PI80_upper", "PI80_lower"}.issubset(bt_df.columns)
        else np.nan
    )
    width95 = (
        float((bt_df["PI95_upper"] - bt_df["PI95_lower"]).mean())
        if {"PI95_upper", "PI95_lower"}.issubset(bt_df.columns)
        else np.nan
    )
    return {
        "coverage_80": cov80,
        "coverage_95": cov95,
        "coverage_gap_80": cov80 - 0.80 if np.isfinite(cov80) else np.nan,
        "coverage_gap_95": cov95 - 0.95 if np.isfinite(cov95) else np.nan,
        "mean_interval_width_80": width80,
        "mean_interval_width_95": width95,
    }
