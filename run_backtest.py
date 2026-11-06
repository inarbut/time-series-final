"""Regenerate backtest_final.csv: 150 daily forecast origins (midnight), 48 hours each, with simulated
weather-forecast noise on the horizon. Columns: the new model (raw, ratio, final = their mean), the
Session 1 model (A1, A0, B -> orig) and the seasonal-naive benchmark (sn, sn_lo, sn_hi).

    python run_backtest.py            # about 40-60 minutes on 8 cores

The notebook reads backtest_final.csv; delete it (or set RUN_BACKTEST = True there) to recompute.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import forecast_lib as fl

# ---- the Session 1 model, kept only as a reference in the backtest ----------------------------------
S1_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_estimators=600, random_state=0, n_jobs=2)
S1_A_COLS = ["hour", "doy", "dow", "t"] + fl.W + ["clear_day", "wind3"]
S1_B_COLS = ["hour", "doy"] + fl.W + ["clear_day", "wind3", "h", "lvl24", "lvl6", "lvl72", "last"]


def session1_A(ctx, E, Xhor, use_lags):
    yy = pd.Series(ctx.y)
    X = ctx.Xh[S1_A_COLS].copy()
    F = Xhor[S1_A_COLS].copy()
    if use_lags:
        X["lag48"] = yy.shift(48).to_numpy(); X["lag168"] = yy.shift(168).to_numpy()
        X["roll48_mean"] = yy.shift(48).rolling(24).mean().to_numpy()
        s = pd.Series(np.r_[ctx.y[:E], np.full(fl.H, np.nan)])
        F["lag48"] = s.shift(48).to_numpy()[E:]; F["lag168"] = s.shift(168).to_numpy()[E:]
        F["roll48_mean"] = s.shift(48).rolling(24).mean().to_numpy()[E:]
    m = lgb.LGBMRegressor(**S1_PARAMS).fit(X.iloc[200:E], ctx.y[200:E])
    return np.clip(m.predict(F), 0, None)


def session1_B(ctx, E, Xhor):
    S = ctx.stack(aug=0.0)
    T = S[S["_t"].to_numpy() < E]
    m = lgb.LGBMRegressor(**S1_PARAMS).fit(T[S1_B_COLS], T["_y"])
    F = Xhor.copy()
    for c in ["lvl24", "lvl6", "lvl72", "last"]:
        F[c] = ctx.OF[c].iloc[E - 1]
    F["h"] = np.arange(1, fl.H + 1)
    return np.clip(m.predict(F[S1_B_COLS]), 0, None)


def main(n_origins=150, out="backtest_final.csv", with_session1=True):
    data = pd.read_csv("train.csv", parse_dates=["ds"])
    ctx = fl.Ctx(data)
    d_raw, d_ratio = fl.Design(ctx, "raw"), fl.Design(ctx, "ratio")
    models = {"raw": d_raw.fit_predict, "ratio": d_ratio.fit_predict}
    if with_session1:
        ctx.stack(aug=0.0)
        models["A1"] = lambda E, X: session1_A(ctx, E, X, True)
        models["A0"] = lambda E, X: session1_A(ctx, E, X, False)
        models["B"] = lambda E, X: session1_B(ctx, E, X)
    bt = fl.backtest(ctx, fl.origin_list(ctx.n, n_origins, 24), models)
    bt["final"] = (bt.raw + bt.ratio) / 2
    if with_session1:
        bt["orig"] = ((bt.A1 + bt.A0) / 2 + bt.B) / 2
    bt.to_csv(out, index=False)
    return bt


if __name__ == "__main__":
    main()
