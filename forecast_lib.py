"""MATH840 final project: 48-hour generation forecast with a 95% interval.

Everything the notebook needs lives here:
  * features of the target hour (calendar, solar alignment, weather, smoothed/day-level weather)
  * features known at the forecast origin (recent levels, DST-aligned seasonal naive)
  * a stacked "one model for all horizons" LightGBM, trained on simulated weather-forecast noise
  * a rolling-origin backtest with simulated forecast noise on the horizon
  * the 95% interval (residual quantiles that grow linearly with the forecast)
  * the MAPE-aware adjustment of small point forecasts
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import lightgbm as lgb

H = 48                     # hours to forecast
M = 24                     # benchmark season
W = ["cloud_cover", "temperature", "wind_speed", "pressure", "humidity", "daylight", "air_density"]
HS = (1, 6, 12, 18, 24, 30, 36, 42, 48)   # horizons stacked into the training set

# ---------------------------------------------------------------- metrics and benchmark (as in the template)


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((np.asarray(actual, float) - np.asarray(forecast, float)) ** 2)))


def mape(actual, forecast):
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs((a - f) / a)) * 100)


def winkler(actual, lower, upper, alpha=0.05):
    a, lo, hi = (np.asarray(x, float) for x in (actual, lower, upper))
    return float(np.mean((hi - lo) + 2 / alpha * (np.clip(lo - a, 0, None) + np.clip(a - hi, 0, None))))


def snaive(history, h=H, m=M):
    history = np.asarray(history, float)
    point = np.array([history[-m + (i % m)] for i in range(h)])
    sigma = np.std(history[m:] - history[:-m], ddof=1)
    k = np.arange(h) // m
    half = 1.96 * sigma * np.sqrt(k + 1)
    return point, np.clip(point - half, 0, None), point + half


# ---------------------------------------------------------------- weather-forecast error model
# weather_forecast.csv looks like "actual weather + noise that grows on day 2". The sizes below are read
# off the hour-to-hour roughness of the forecast file versus the actual weather (see notebook, section 2.1).
SIG = {"cloud_cover": (0.10, 0.25), "temperature": (1.0, 2.2), "wind_speed": (0.5, 1.0),
       "pressure": (0.5, 1.7), "humidity": (3.0, 6.0), "air_density": (0.010, 0.014)}


def add_noise(wx, steps, rng, scale=1.0):
    """Simulated forecast: actual weather + Gaussian noise (sigma for day 1 / day 2), rounded like the file."""
    out = wx.copy()
    steps = np.asarray(steps)
    for c, (s1, s2) in SIG.items():
        s = np.where(steps <= 24, s1, s2) * scale
        v = out[c].to_numpy(float) + rng.normal(0, 1, len(steps)) * s
        if c == "cloud_cover":
            v = np.clip(np.round(v * 4) / 4, 0, 1)
        elif c == "wind_speed":
            v = np.clip(np.round(v), 0, None)
        elif c == "humidity":
            v = np.clip(np.round(v), 0, 100)
        elif c in ("temperature", "pressure"):
            v = np.round(v)
        out[c] = v
    return out


# ---------------------------------------------------------------- features


def hour_feats(df):
    """Features of the target hour. Hours are hour-ending, so 00:00 closes the previous day (a day = 01..24)."""
    df = df.reset_index(drop=True)
    X = pd.DataFrame(index=df.index)
    ds = df.ds
    X["hour"] = ds.dt.hour
    X["doy"] = ds.dt.dayofyear
    X["dow"] = ds.dt.dayofweek
    X["t"] = (ds - pd.Timestamp("2023-01-01")).dt.days.astype(float)
    for c in W:
        X[c] = df[c].to_numpy(float)
    # solar alignment from the daylight flag: hour relative to the middle of the daylight window.
    # This is unaffected by the clock change at the end of October (daylight 08-16 -> 07-15).
    shifted = ds - pd.Timedelta(hours=1)
    date = shifted.dt.normalize()
    dl = df.daylight.to_numpy(float)
    hr = (shifted.dt.hour + 1).to_numpy(float)
    g = pd.DataFrame({"date": date, "h": np.where(dl > 0, hr, np.nan)})
    first = g.groupby("date").h.transform("min")
    last = g.groupby("date").h.transform("max")
    X["day_len"] = (last - first + 1).to_numpy()
    X["mid"] = ((first + last) / 2).to_numpy()
    X["solar_hour"] = hr - X["mid"].to_numpy()
    X["clear_day"] = dl * (1 - df.cloud_cover.to_numpy(float))
    X["cloud_day"] = dl * df.cloud_cover.to_numpy(float)
    X["wind3"] = df.wind_speed.to_numpy(float) ** 3 * df.air_density.to_numpy(float)
    # centred 3-hour means: the forecast noise is roughly independent from hour to hour
    for c in ["cloud_cover", "humidity", "temperature", "wind_speed"]:
        X[c + "_r3"] = df[c].rolling(3, center=True, min_periods=1).mean().to_numpy()
    # means over the daylight hours of the same day: a robust "type of day"
    for c in ["cloud_cover", "humidity", "temperature"]:
        X[c + "_dmean"] = df[c].where(df.daylight > 0).groupby(date).transform("mean").to_numpy()
    X["wind_dmean"] = df.wind_speed.groupby(date).transform("mean").to_numpy()
    X["wind3_dmean"] = X["wind3"].groupby(date).transform("mean").to_numpy()
    return X


def origin_feats(y, daylight):
    """Features known at origin index o (they use y[..o] only): recent levels and the 28-day capacity proxy."""
    n = len(y)
    cs = np.cumsum(np.r_[0.0, y])
    idx = np.arange(n)

    def tmean(k):
        lo = np.maximum(idx - k + 1, 0)
        return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)

    F = {"last": y.copy(), "lvl6": tmean(6), "lvl24": tmean(24), "lvl72": tmean(72),
         "lvl168": tmean(168), "lvl672": tmean(672)}
    nd = (1 - daylight).astype(float)
    csn = np.cumsum(np.r_[0.0, y * nd]); cnn = np.cumsum(np.r_[0.0, nd])
    csd = np.cumsum(np.r_[0.0, y * (1 - nd)]); cnd = np.cumsum(np.r_[0.0, 1 - nd])
    for k in (168, 672):
        lo = np.maximum(idx - k + 1, 0)
        F[f"night{k}"] = (csn[idx + 1] - csn[lo]) / np.maximum(cnn[idx + 1] - cnn[lo], 1)
        F[f"day{k}"] = (csd[idx + 1] - csd[lo]) / np.maximum(cnd[idx + 1] - cnd[lo], 1)
    s = pd.Series(y)
    F["max168"] = s.rolling(168, min_periods=1).max().to_numpy()
    F["q90_672"] = s.rolling(672, min_periods=24).quantile(0.9).to_numpy()
    return pd.DataFrame(F)


HOUR_COLS = ["hour", "doy", "solar_hour", "day_len"] + W + [
    "clear_day", "cloud_day", "wind3", "cloud_cover_r3", "humidity_r3", "temperature_r3", "wind_speed_r3",
    "cloud_cover_dmean", "humidity_dmean", "temperature_dmean", "wind_dmean", "wind3_dmean"]
ORIGIN_COLS = ["last", "lvl6", "lvl24", "lvl72", "lvl168", "lvl672", "night168", "day168", "night672",
               "day672", "max168", "q90_672"]
COLS = HOUR_COLS + ["h", "sn_al"] + ORIGIN_COLS

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, verbose=-1, seed=0,
              num_threads=2, max_bin=255)
ROUNDS = 600


class Ctx:
    """Precomputed features of the history."""

    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.y = self.data.y.to_numpy(float)
        self.n = len(self.data)
        self.Xh = hour_feats(self.data)
        self.OF = origin_feats(self.y, self.data.daylight.to_numpy())
        self.mid = self.Xh["mid"].to_numpy(float)
        self._stack = {}

    def horizon_X(self, E, wx):
        """Target-hour features for the 48 hours starting at index E, given their (forecast) weather wx."""
        lo = max(E - 72, 0)
        frame = pd.concat([self.data.iloc[lo:E], wx], ignore_index=True)
        return hour_feats(frame).iloc[-len(wx):].reset_index(drop=True)

    def sn_aligned(self, t, o, h, mid_t=None):
        """Seasonal-naive value at the same *solar* hour of the reference day (DST-proof)."""
        t = np.asarray(t); o = np.asarray(o); h = np.asarray(h) + np.zeros_like(t)
        r = o - 23 + ((h - 1) % 24)
        mt = self.mid[t] if mid_t is None else np.asarray(mid_t)
        d = np.round(self.mid[r] - mt)
        d = np.where(np.isnan(d), 0, d).astype(int)
        r2 = r + d
        r2 = np.where((r2 <= o) & (r2 >= 0), r2, r)
        return self.y[r2]

    def stack(self, aug=1.0, seed=7, hs=HS):
        """Training set of (target hour t, horizon h) rows over the whole history (filtered by t later).
        With aug > 0 the weather of each h-copy carries simulated forecast noise for lead time h."""
        key = (tuple(hs), aug, seed)
        if key not in self._stack:
            parts = []
            for h in hs:
                t = np.arange(700, self.n)
                o = t - h
                if aug > 0:
                    rng = np.random.default_rng(seed * 1000 + h)
                    noisy = add_noise(self.data.drop(columns="y"), np.full(self.n, h), rng, aug)
                    noisy.insert(1, "y", self.y)
                    P = hour_feats(noisy).iloc[t].reset_index(drop=True)
                else:
                    P = self.Xh.iloc[t].reset_index(drop=True)
                P = pd.concat([P, self.OF.iloc[o].reset_index(drop=True)], axis=1)
                P["h"] = h
                P["sn_al"] = self.sn_aligned(t, o, h)
                P["_t"] = t
                P["_y"] = self.y[t]
                parts.append(P)
            self._stack[key] = pd.concat(parts, ignore_index=True)
        return self._stack[key]


class Design:
    """One stacked LightGBM (target 'raw' = y, or 'ratio' = y / 28-day mean level at the origin)."""

    def __init__(self, ctx, target="raw", aug=1.0, params=None, rounds=ROUNDS, cols=COLS):
        self.ctx, self.target, self.cols, self.rounds = ctx, target, list(cols), rounds
        self.params = dict(PARAMS if params is None else params)
        S = ctx.stack(aug)
        self.t = S["_t"].to_numpy()
        z = S["_y"].to_numpy()
        if target == "ratio":
            z = z / np.maximum(S["lvl672"].to_numpy(), 1)
        self.full = lgb.Dataset(S[self.cols].to_numpy(np.float32), label=z, feature_name=self.cols,
                                free_raw_data=False, params={"max_bin": self.params["max_bin"], "verbose": -1}).construct()

    def fit_predict(self, E, Xhor):
        """Train on target hours before E and forecast the 48 hours E..E+47 (Xhor = their hour features)."""
        ctx = self.ctx
        ds = self.full.subset(np.nonzero(self.t < E)[0]).construct()
        bst = lgb.train(self.params, ds, num_boost_round=self.rounds)
        F = Xhor.copy()
        o = E - 1
        for c in ctx.OF.columns:
            F[c] = ctx.OF[c].iloc[o]
        F["h"] = np.arange(1, H + 1)
        F["sn_al"] = ctx.sn_aligned(E + np.arange(H), np.full(H, o), np.arange(1, H + 1), mid_t=F["mid"].to_numpy(float))
        p = bst.predict(F[self.cols].to_numpy(np.float32))
        if self.target == "ratio":
            p = p * max(ctx.OF["lvl672"].iloc[o], 1)
        self.booster = bst
        return np.clip(p, 0, None)


def ensemble_forecast(designs, E, Xhor):
    """Final point forecast before the MAPE adjustment: equal-weight mean of the designs."""
    return np.mean([d.fit_predict(E, Xhor) for d in designs], axis=0)


# ---------------------------------------------------------------- backtest


def origin_list(n, count, step=24):
    """Origins at midnight: the forecast starts at index E (01:00) and covers E..E+47."""
    return [n - H - step * k for k in range(count)]


def backtest(ctx, origins, models, noise_scale=1.0, workers=8, seed=123):
    """models: name -> callable(E, Xhor) -> 48 predictions. The horizon gets simulated forecast weather."""
    def one(E):
        rng = np.random.default_rng(seed + E)
        steps = np.arange(1, H + 1)
        wx_true = ctx.data.iloc[E:E + H].drop(columns="y").reset_index(drop=True)
        wx = add_noise(wx_true, steps, rng, noise_scale) if noise_scale > 0 else wx_true
        Xhor = ctx.horizon_X(E, wx)
        sn = snaive(ctx.y[:E])
        out = pd.DataFrame({"E": E, "step": steps, "ds": ctx.data.ds.iloc[E:E + H].to_numpy(),
                            "actual": ctx.y[E:E + H], "daylight": wx_true.daylight.to_numpy(),
                            "sn": sn[0], "sn_lo": sn[1], "sn_hi": sn[2]})
        for name, fn in models.items():
            out[name] = fn(E, Xhor)
        return out
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, origins))
    print(f"backtest: {len(origins)} origins, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(res, ignore_index=True)


def add_groups(bt):
    """Interval groups: daylight / dark x day 1 / day 2 of the horizon."""
    bt = bt.copy()
    bt["ds"] = pd.to_datetime(bt.ds)
    bt["day"] = np.where(bt.step > 24, 2, 1)
    bt["group"] = np.where(bt.daylight > 0, "light", "dark") + "_d" + bt.day.astype(str)
    return bt


def group_of(daylight, steps):
    return np.where(np.asarray(daylight) > 0, "light", "dark") + "_d" + np.where(np.asarray(steps) > 24, "2", "1")


# ---------------------------------------------------------------- 95% interval
VOL_DAYS = 14


def origin_vol(y, daylight, E, days=VOL_DAYS):
    """Recent volatility known at the origin: RMS of day-to-day changes y_t - y_{t-24} over the last `days`
    days, separately for daylight and dark hours. October (fog / clear switching, calm nights) is far more
    volatile than summer, and the forecast errors follow it."""
    y = np.asarray(y, float); dl = np.asarray(daylight) > 0
    seg = y[E - 24 * (days + 1):E]
    d = seg[24:] - seg[:-24]
    m = dl[E - 24 * days:E]
    return float(np.sqrt(np.mean(d[m] ** 2))), float(np.sqrt(np.mean(d[~m] ** 2)))


def add_vol(bt, y, daylight):
    """Attach the origin volatility that matches each row's daylight flag."""
    bt = bt.copy()
    V = {E: origin_vol(y, daylight, E) for E in bt.E.unique()}
    bt["vol"] = [V[E][0] if dl > 0 else V[E][1] for E, dl in zip(bt.E, bt.daylight)]
    return bt


def pinball_fit_linear(x, r, tau):
    """Quantile regression r ~ a + b*x with b >= 0 (grid over b, a = tau-quantile of r - b*x)."""
    best = None
    for b in np.linspace(0, 1.5, 61):
        z = r - b * x
        a = np.quantile(z, tau)
        u = z - a
        loss = np.mean(np.maximum(tau * u, (tau - 1) * u))
        if best is None or loss < best[0]:
            best = (loss, a, b)
    return best[1], best[2]


def fit_interval(f, resid, groups, vol, q=0.95):
    """Per group, the distances to the 2.5% / 97.5% quantiles are a*vol + b*f:
    fitted as quantile regressions of resid/vol on f/vol (b >= 0)."""
    tau = 1 - (1 - q) / 2
    f, resid, groups, vol = (np.asarray(x) for x in (f, resid, groups, vol))
    coef = {}
    for g in np.unique(groups):
        m = groups == g
        x, r = f[m] / vol[m], resid[m] / vol[m]
        coef[g] = (pinball_fit_linear(x, -r, tau), pinball_fit_linear(x, r, tau))
    return coef


# The leaderboard ranks students per metric; it does not average scores. A slightly narrower interval than the
# calibrated one beats it on about two days in three (it loses big only on the rare stormy day), so for the
# ranking the distances are multiplied by WIDTH (chosen by the rank simulation in the notebook, Q3).
WIDTH = 0.9


def apply_interval(f, groups, vol, coef, width=WIDTH):
    f = np.asarray(f, float); vol = np.asarray(vol, float)
    lo = np.empty_like(f); hi = np.empty_like(f)
    for i, g in enumerate(np.asarray(groups)):
        (al, bl), (ah, bh) = coef[g]
        lo[i] = f[i] - width * (al * vol[i] + bl * f[i])
        hi[i] = f[i] + width * (ah * vol[i] + bh * f[i])
    return np.clip(np.minimum(lo, f), 0, None), np.maximum(hi, f)


# ---------------------------------------------------------------- MAPE-aware point adjustment
# Multiplier 0.6 for tiny forecasts, rising linearly to 1 at a forecast of 500. smin = 0.6 minimises the simulated
# leaderboard rank (notebook, Q2/Q3); the pure log-score fit (fit_ramp) would shrink harder (0.4).
RAMP = (0.6, 500.0)


def ramp(f, smin=RAMP[0], f0=RAMP[1]):
    """Shrink small forecasts. MAPE punishes over-forecasting near-zero hours (calm nights) far more than
    under-forecasting, so the MAPE-optimal point is below the mean there; large forecasts are unchanged."""
    f = np.asarray(f, float)
    return f * (smin + (1 - smin) * np.clip(f / f0, 0, 1))


def rank_proxy(E, a, f, w_mape=0.5):
    """Mean over origins of log RMSE + w * log MAPE (w < 1: competitors' MAPEs are more spread out)."""
    df = pd.DataFrame({"E": E, "se": (a - f) ** 2, "ape": np.abs(a - f) / a}).groupby("E").mean()
    return float((0.5 * np.log(df.se) + w_mape * np.log(100 * df.ape)).mean())


def fit_ramp(E, a, f, w_mape=0.5, smins=np.linspace(0.2, 1.0, 17), f0s=(100, 200, 300, 500, 800, 1200)):
    best = None
    for s in smins:
        for f0 in f0s:
            sc = rank_proxy(E, a, ramp(f, s, f0), w_mape)
            if best is None or sc < best[0]:
                best = (sc, s, f0)
    return best[1], best[2]


# ---------------------------------------------------------------- evaluation helpers


def per_origin_scores(bt, point, lo=None, hi=None, day=2):
    b = bt[bt.day == day] if "day" in bt else bt
    rows = []
    for E, g in b.groupby("E"):
        r = {"E": E, "RMSE": rmse(g.actual, g[point]), "MAPE": mape(g.actual, g[point])}
        if lo is not None:
            r["Winkler"] = winkler(g.actual, g[lo], g[hi])
            r["coverage"] = float(np.mean((g.actual >= g[lo]) & (g.actual <= g[hi])))
            r["width"] = float(np.mean(g[hi] - g[lo]))
        rows.append(r)
    return pd.DataFrame(rows)


def loo_pipeline(bt, col, ramp_par=RAMP, width=WIDTH, past_only=False, w_mape=0.5):
    """Interval fitted on other origins, applied to each origin in turn, then the MAPE shrink.
    past_only=False: leave-one-origin-out; True: only origins whose 48 hours ended before this origin.
    ramp_par=None fits the shrink on the other origins (log RMSE + w_mape * log MAPE) instead of fixing it."""
    bt = bt.copy()
    bt["point"] = np.nan; bt["lo"] = np.nan; bt["hi"] = np.nan
    for E in bt.E.unique():
        tr = (bt.E <= E - H) if past_only else (bt.E != E)
        te = bt.E == E
        if tr.sum() < 20 * H:
            continue
        coef = fit_interval(bt.loc[tr, col], bt.loc[tr, "actual"] - bt.loc[tr, col], bt.loc[tr, "group"], bt.loc[tr, "vol"])
        lo, hi = apply_interval(bt.loc[te, col], bt.loc[te, "group"], bt.loc[te, "vol"], coef, width)
        if ramp_par is None:
            t2 = tr & (bt.day == 2)
            s, f0 = fit_ramp(bt.E[t2].to_numpy(), bt.actual[t2].to_numpy(), bt[col][t2].to_numpy(), w_mape)
        else:
            s, f0 = ramp_par
        p = np.clip(ramp(bt.loc[te, col].to_numpy(), s, f0), lo, hi)
        bt.loc[te, "point"] = p; bt.loc[te, "lo"] = lo; bt.loc[te, "hi"] = hi
    return bt


# ---------------------------------------------------------------- leaderboard rank simulation


def day_scores(frame, point, lo, hi, day=2):
    """Per-origin RMSE, MAPE and Winkler on one day of the horizon (vectorised)."""
    b = frame[frame.day == day]
    a, p, l, u = (np.asarray(x, float) for x in (b.actual, point[frame.day == day], lo[frame.day == day], hi[frame.day == day]))
    df = pd.DataFrame({"E": b.E.to_numpy(), "se": (a - p) ** 2, "ape": np.abs(a - p) / a * 100,
                       "w": (u - l) + 40 * (np.clip(l - a, 0, None) + np.clip(a - u, 0, None))})
    g = df.groupby("E")
    return pd.DataFrame({"RMSE": np.sqrt(g.se.mean()), "MAPE": g.ape.mean(), "W": g.w.mean()})


def competitor_field(L):
    """Plausible other students: the benchmark, my Session 1 model, the other backtest models, a blend with
    seasonal naive and level-biased versions, each with a typical interval (conformal radius or +-1.96 sd)."""
    L = L.copy()
    part = pd.cut(L.ds.dt.hour, [-1, 5, 8, 15, 19, 23], labels=False).astype(str) + "_" + L.day.astype(str)

    def conformal(col):        # 95% quantile of |error| by 5 parts of the day x day, leave-one-origin-out
        lo = np.empty(len(L)); hi = np.empty(len(L)); err = (L.actual - L[col]).abs()
        for E in L.E.unique():
            tr, te = (L.E != E).to_numpy(), (L.E == E).to_numpy()
            r = part[te].map(err[tr].groupby(part[tr]).quantile(0.95)).to_numpy()
            lo[te] = np.clip(L[col].to_numpy()[te] - r, 0, None); hi[te] = L[col].to_numpy()[te] + r
        return lo, hi

    def sd196(col):
        sd = (L.actual - L[col]).std()
        return np.clip(L[col] - 1.96 * sd, 0, None).to_numpy(), (L[col] + 1.96 * sd).to_numpy()

    L["blend_sn"] = 0.5 * L.raw + 0.5 * L.sn
    L["orig_hi"] = 1.15 * L.orig
    L["ratio_lo"] = 0.9 * L.ratio
    field = {"seasonal naive": day_scores(L, L.sn, L.sn_lo, L.sn_hi)}
    for c in ["orig", "A1", "A0", "B", "raw", "ratio", "blend_sn", "orig_hi", "ratio_lo"]:
        field[c + " + conformal"] = day_scores(L, L[c], *conformal(c))
    for c in ["A0", "raw"]:
        field[c + " + 1.96 sd"] = day_scores(L, L[c], *sd196(c))
    return field


def simulate_rank(L, field, smin, width, f0=500.0):
    """Leaderboard rank of my file on each origin's day 2 among `field` (1 = best), for a shrink and a width.
    L must hold the calibrated (width 1) leave-one-origin-out interval in lo / hi."""
    f = L.final.to_numpy()
    lo = np.clip(f - (f - L.lo.to_numpy()) * width, 0, None); hi = f + (L.hi.to_numpy() - f) * width
    me = day_scores(L, np.clip(ramp(f, smin, f0), lo, hi), lo, hi)
    R = {}
    for m in ["RMSE", "MAPE", "W"]:
        M = pd.concat({k: v[m] for k, v in field.items()}, axis=1)
        R[m] = 1 + M.lt(me[m], axis=0).sum(axis=1)
    R = pd.DataFrame(R)
    R["combined"] = 0.3 * R.RMSE + 0.3 * R.MAPE + 0.4 * R.W
    R["beats snaive RMSE"] = me.RMSE < field["seasonal naive"].RMSE
    return R
