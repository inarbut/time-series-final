"""Produce NARBUT_project.csv from train.csv, weather_forecast.csv and the backtest (backtest_final.csv).

    python make_forecast.py

Steps: fit the two stacked LightGBMs on the whole history -> mean = point forecast; 95% interval from the
backtest residuals (scaled by the forecast and by the recent 14-day volatility); MAPE-aware shrink of small
forecasts (fitted on the backtest); format check.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_lib as fl

SURNAME = "Narbut"


def check_submission(sub, future):
    assert list(sub.columns) == ["ds", "forecast", "lower_95", "upper_95"], "wrong columns"
    assert len(sub) == fl.H, f"need {fl.H} rows, got {len(sub)}"
    assert (pd.to_datetime(sub["ds"]).to_numpy() == future.to_numpy()).all(), "the hours do not match FUTURE"
    assert sub[["forecast", "lower_95", "upper_95"]].notna().all().all(), "missing values"
    assert (sub["lower_95"] >= 0).all(), "a lower bound is negative"
    assert (sub["lower_95"] <= sub["forecast"]).all() and (sub["forecast"] <= sub["upper_95"]).all(), \
        "the forecast must lie inside its interval"
    print("format OK")


def calibrate(bt, data):
    """Interval coefficients from all backtest origins; the shrink (fl.RAMP) and the interval width (fl.WIDTH)
    are the values that minimise the simulated leaderboard rank (notebook, Q3)."""
    bt = fl.add_vol(fl.add_groups(bt), data.y, data.daylight)
    coef = fl.fit_interval(bt.final, bt.actual - bt.final, bt.group, bt.vol)
    return coef, fl.RAMP


def forecast(data, weather_fc, bt, verbose=True):
    ctx = fl.Ctx(data)
    E = ctx.n
    Xhor = ctx.horizon_X(E, weather_fc.reset_index(drop=True))
    designs = [fl.Design(ctx, "raw"), fl.Design(ctx, "ratio")]   # same settings as in the backtest
    preds = [d.fit_predict(E, Xhor) for d in designs]
    final = np.mean(preds, axis=0)
    coef, (smin, f0) = calibrate(bt, data)
    vday, vnight = fl.origin_vol(data.y, data.daylight, E)
    vol = np.where(weather_fc.daylight.to_numpy() > 0, vday, vnight)
    groups = fl.group_of(weather_fc.daylight, np.arange(1, fl.H + 1))
    lo, hi = fl.apply_interval(final, groups, vol, coef)
    point = np.clip(fl.ramp(final, smin, f0), lo, hi)
    if verbose:
        print(f"shrink: smin={smin:.2f}, f0={f0:.0f}; interval width x{fl.WIDTH}; "
              f"origin volatility day={vday:.0f}, dark={vnight:.0f}")
        for g, ((al, bl), (ah, bh)) in coef.items():
            print(f"  {g:9s} lower = f - ({al:.2f}*vol + {bl:.2f}*f)   upper = f + ({ah:.2f}*vol + {bh:.2f}*f)")
    detail = pd.DataFrame({"ds": weather_fc.ds, "raw": preds[0], "ratio": preds[1], "final": final,
                           "forecast": point, "lower_95": lo, "upper_95": hi})
    return detail


def main():
    data = pd.read_csv("train.csv", parse_dates=["ds"])
    weather_fc = pd.read_csv("weather_forecast.csv", parse_dates=["ds"])
    if not Path("backtest_final.csv").exists():
        import run_backtest
        run_backtest.main()
    bt = pd.read_csv("backtest_final.csv", parse_dates=["ds"])
    detail = forecast(data, weather_fc, bt)
    sub = pd.DataFrame({"ds": weather_fc.ds.dt.strftime("%Y-%m-%d %H:%M:%S"), "forecast": detail.forecast,
                        "lower_95": detail.lower_95, "upper_95": detail.upper_95})
    check_submission(sub, weather_fc.ds)
    file = f"{SURNAME.upper()}_project.csv"
    sub.to_csv(file, index=False)
    print("saved", file)
    return detail


if __name__ == "__main__":
    main()
