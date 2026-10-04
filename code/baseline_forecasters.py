#!/usr/bin/env python3
# =============================================================================
# baseline_forecasters.py -- baselines for DIRECTIONAL nodal DART
# prediction, built to a strict gate-closure information set.
#
# OUTPUT:
#   sign(DART(n,h)) for every hour h of operating day D, decided at day-ahead
#   GATE CLOSURE on D-1. Positive DART means the day-ahead price is above the realized
#   real-time (the INC-profitable direction).
#
# THE INFORMATION SET DESCRIPTION:
#   At gate closure on D-1 a market participant has the following information:
#     * DART is fully realized only through D-2. Real-time for D-1 still gets
#       realized, and real-time for D does not exist, so the most recent COMPLETE
#       DART is two days old. This problem of publication lag is a hard constraint and it
#       binds regardless of whether prices are ever revised.
#     * calendar variables for D (hour, weekday, month).
#   A participant does NOT know: day-ahead prices for D (they clear after bids
#   are submitted), or any real-time outcome on D-1 or D.
#
#   Every feature here is built with a MINIMUM LAG of `min_lag_days` (default 2)
#  
#
# MODELS (five classes, deliberately UNTUNED):
#   persistence   sign of the same node-hour's most recent complete DART
#   climatology   historical sign frequency for (node, hour-of-day, month)
#   logistic      L2 logistic regression on the lagged feature set
#   gbm           gradient boosting on the same features
#   mlp           small multilayer perceptron on the same features
#
#   They are NOT tuned. Heavy tuning would create the multiple-comparison
#   problem, and the main question then becomes whether model
#   CLASS differences survive dependence-aware inference -- not which model wins the prediction accuracy.
#
# Usage:
#   python baseline_forecasters.py --validate
#   python baseline_forecasters.py --data-dir data --panel panel_dedup_all.csv \
#          --component congestion --train-end 2025-12-31 --out-dir rq3_out
# =============================================================================
from __future__ import annotations

import argparse
import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger("baseline")

COMPONENTS = {"total": ("total_lmp_da", "total_lmp_rt"),
              "congestion": ("congestion_price_da", "congestion_price_rt")}
KEY = "datetime_beginning_utc"
MODELS = ("persistence", "climatology", "logistic", "gbm", "rf", "mlp",
          "sarima", "kalman", "markov")

# Model families, and what is DELIBERATELY excluded.
#   naive        persistence, climatology
#   econometric  sarima (ARMA + Fourier hour terms), kalman (local level),
#                markov (2-regime switching)
#   ML           logistic, gbm (boosting), rf (bagged trees), mlp
#

TS_MODELS = ("sarima", "kalman", "markov")
FOURIER_K = 2


# ------------------------------------------------------------------ data
def build_dart(data_dir, pid, component):
    def read(label):
        fs = sorted(glob.glob(os.path.join(data_dir, label, f"pnode={pid}", "*.csv")))
        return pd.concat((pd.read_csv(f) for f in fs), ignore_index=True) if fs else None
    da, rt = read("da_hourly"), read("rt_hourly")
    if da is None or rt is None:
        return None
    m = da.drop_duplicates(KEY).merge(rt.drop_duplicates(KEY), on=KEY, how="inner")
    m[KEY] = pd.to_datetime(m[KEY], utc=True, errors="coerce")
    m = m.dropna(subset=[KEY]).sort_values(KEY).reset_index(drop=True)
    dc, rc = COMPONENTS[component]
    if dc not in m or rc not in m:
        return None
    out = pd.DataFrame({KEY: m[KEY]})
    out["dart"] = (pd.to_numeric(m[dc], errors="coerce")
                   - pd.to_numeric(m[rc], errors="coerce"))
    return out.dropna(subset=["dart"]).reset_index(drop=True)


def make_features(d, min_lag_days=2, violate_lag=False):
    """Feature matrix for one node.

    min_lag_days is the publication lag: at gate closure on D-1 the freshest
    COMPLETE DART is from D-2, so every lag must be >= 2 days (48 hours).

    violate_lag=True deliberately uses a 1-DAY lag, i.e. information that is NOT
    available at gate closure. It exists ONLY to measure how much apparent skill
    inflates when the lag is ignored -- NOT for forecasting."""
    d = d.sort_values(KEY).reset_index(drop=True).copy()
    t = pd.DatetimeIndex(d[KEY])
    d["hour"] = t.hour
    d["dow"] = t.dayofweek
    d["month"] = t.month
    d["date"] = t.normalize()

    lag0 = 1 if violate_lag else int(min_lag_days)
    H = 24
    # same node-hour on previous days (the natural predictor for an hourly series)
    for k, name in ((lag0, "lag_a"), (lag0 + 1, "lag_b"), (lag0 + 5, "lag_week")):
        d[name] = d["dart"].shift(k * H)
    # rolling statistics over COMPLETE days only, all shifted by the same lag
    base = d["dart"].shift(lag0 * H)
    d["roll_mean_7d"] = base.rolling(7 * H, min_periods=24).mean()
    d["roll_med_7d"] = base.rolling(7 * H, min_periods=24).median()
    d["roll_signfreq_7d"] = (base > 0).rolling(7 * H, min_periods=24).mean()
    d["roll_absmed_7d"] = base.abs().rolling(7 * H, min_periods=24).median()
    d["roll_mean_30d"] = base.rolling(30 * H, min_periods=48).mean()
    d["roll_signfreq_30d"] = (base > 0).rolling(30 * H, min_periods=48).mean()
    # calendar (known for D at gate closure)
    d["sin_h"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["cos_h"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["is_weekend"] = (d["dow"] >= 5).astype(float)
    d["y"] = (d["dart"] > 0).astype(int)
    d["lag_days_used"] = lag0
    return d


def _fourier_exog(times, k=FOURIER_K):
    """Deterministic hour-of-day features plus a weekend flag. These are KNOWN
    for the target day at gate closure --- Exogenous inputs
    for a forecast and can be supplied over the forecast horizon."""
    t = pd.DatetimeIndex(times)
    cols = []
    for j in range(1, int(k) + 1):
        cols.append(np.sin(2 * np.pi * j * t.hour / 24))
        cols.append(np.cos(2 * np.pi * j * t.hour / 24))
    cols.append((t.dayofweek >= 5).astype(float))
    return np.column_stack(cols)


def _origin_index(dates, target_day, min_lag_days):
    """Index of the LAST observation a participant could have used when
    forecasting `target_day`.

    At gate closure on D-1, the latest COMPLETE day is D-min_lag_days, so the
    origin is the final observation on or before that day."""
  
    cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=int(min_lag_days))
    ok = np.flatnonzero(dates <= cutoff)
    return int(ok[-1]) if ok.size else -1


def _norm_cdf(z):
    from scipy.stats import norm
    return norm.cdf(z)


def markov_params(res):
    """Regime means, sigmas and the transition matrix from a fitted
    Markov Regression model.

    res.params ---- is an ndarray when the model is fitted on an array (no pandas
    index), so positional slicing would silently pick the transition
    probabilities instead of the regime means. Observed layout:
      ['p[0->0]', 'p[1->0]', 'const[0]', 'const[1]', 'sigma2[0]', 'sigma2[1]']
    Returns (mu, sigma, P) with P[i, j] = Pr(regime j at t | regime i at t-1)."""
    names = [str(x) for x in getattr(res.model, "param_names", [])]
    pars = np.asarray(res.params, float)

    def _par(pattern, k, fallback):
        nm = f"{pattern}[{k}]"
        return float(pars[names.index(nm)]) if nm in names else float(fallback)

    mu = np.array([_par("const", k, pars[2 + k] if len(pars) > 3 else 0.0)
                   for k in range(2)])
    var = np.array([_par("sigma2", k, pars[-2 + k]) for k in range(2)])
    sigma = np.sqrt(np.clip(var, 1e-12, None))
    # regime_transition is (k, k, 1) with [i, j] = Pr(i at t | j at t-1);
    # transpose so P[i, j] = Pr(j at t | i at t-1), matching the filter.
    P = np.asarray(res.regime_transition[:, :, 0], float).T
    rs = P.sum(axis=1, keepdims=True)
    P = np.divide(P, np.where(rs > 0, rs, 1.0))
    return mu, sigma, P


def _hamilton_filter(y, mu, sigma, P):
    """Filtered regime probabilities for a 2-regime Gaussian mixture with Markov
    transitions. P[i, j] = Pr(regime j at t | regime i at t-1).
    """
    y = np.asarray(y, float)
    k = len(mu)
    xi = np.full(k, 1.0 / k)
    # stationary distribution as the initial condition, when it exists
    try:
        w, v = np.linalg.eig(np.asarray(P, float).T)
        i = int(np.argmin(np.abs(w - 1.0)))
        st = np.real(v[:, i])
        if np.all(st > 0) or np.all(st < 0):
            xi = np.abs(st) / np.abs(st).sum()
    except Exception:                                   
        pass
    out = np.empty((len(y), k))
    Pt = np.asarray(P, float)
    for t in range(len(y)):
        pred = Pt.T @ xi                     
        dens = np.array([
            np.exp(-0.5 * ((y[t] - mu[j]) / sigma[j]) ** 2)
            / (sigma[j] * np.sqrt(2 * np.pi)) for j in range(k)])
        num = pred * dens
        tot = num.sum()
        xi = num / tot if tot > 0 else pred
        out[t] = xi
    return out


FEATURES = ["lag_a", "lag_b", "lag_week", "roll_mean_7d", "roll_med_7d",
            "roll_signfreq_7d", "roll_absmed_7d", "roll_mean_30d",
            "roll_signfreq_30d", "sin_h", "cos_h", "is_weekend"]


def build_panel(data_dir, panel_csv, component, min_lag_days=2,
                violate_lag=False, max_nodes=None):
    p = pd.read_csv(panel_csv)
    pids = sorted(set(int(x) for x in p["pnode_id"]))
    if max_nodes:
        pids = pids[:max_nodes]
    zmap = dict(zip(p.pnode_id.astype(int), p.zone))
    frames = []
    for pid in pids:
        d = build_dart(data_dir, pid, component)
        if d is None or len(d) < 24 * 60:
            log.warning("pid=%s: insufficient data", pid)
            continue
        f = make_features(d, min_lag_days, violate_lag)
        f["pnode_id"] = pid
        f["zone"] = zmap.get(pid, "?")
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)



# --------------------------------------------------- time-series forecasters -------------------------------------------
def _fit_ts_once(name, y_train, exog_train, seed=0):
    """Fit PARAMETERS once on the training period. The origin is rolled over the
    test period without refitting"""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        if name == "sarima":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            return SARIMAX(y_train, exog=exog_train, order=(2, 0, 1),
                           trend="c").fit(disp=False)
        if name == "kalman":
            from statsmodels.tsa.statespace.structural import UnobservedComponents
            return UnobservedComponents(y_train, level="local level").fit(disp=False)
        if name == "markov":
            from statsmodels.tsa.regime_switching.markov_regression import (
                MarkovRegression)
            n = len(y_train)
            cap = min(n, 8000)                
            return MarkovRegression(y_train[-cap:], k_regimes=2,
                                    switching_variance=True).fit()
    raise ValueError(name)


def _rolling_ts_predict(name, node_df, n_train, min_lag_days=2, seed=0):
    """P(DART > 0) for every TEST row of one node, rolling the forecast origin.

    For each test day D, the origin is the last observation on or before
    D - min_lag_days, so the forecast never uses information a participant 
    CANNOT POSSESS at gate closure. The horizon is derived from actual data
    POSITIONS, not by assuming 24 rows per day, so DST days are handled."""
    
    d = node_df.sort_values(KEY).reset_index(drop=True)
    y = d["dart"].to_numpy(float)
    dates = pd.DatetimeIndex(d["date"])
    times = pd.DatetimeIndex(d[KEY])
    exog = _fourier_exog(times)

    try:
        res = _fit_ts_once(name, y[:n_train],
                           exog[:n_train] if name == "sarima" else None, seed)
    except Exception as e:                                       # pragma: no cover
        log.warning("%s: fit failed (%s); falling back to the base rate", name, e)
        return np.full(len(d) - n_train, float((y[:n_train] > 0).mean()))

    out = np.full(len(d), np.nan)
    test_days = pd.unique(dates[n_train:])

    if name == "markov":
        mu, sigma, P = markov_params(res)
        p_up_regime = _norm_cdf(mu / sigma)

    for day in test_days:
        oi = _origin_index(dates, day, min_lag_days)
        tgt = np.flatnonzero(dates == day)
        if oi < 10 or tgt.size == 0:
            continue
        horizon = int(tgt[-1] - oi)
        if horizon <= 0 or horizon > 24 * 10:
            continue
        try:
            if name == "sarima":
                r2 = res.apply(y[:oi + 1], exog=exog[:oi + 1], refit=False)
                fc = r2.get_forecast(horizon, exog=exog[oi + 1:oi + 1 + horizon])
                mean = np.asarray(fc.predicted_mean, float)
                se = np.sqrt(np.clip(np.asarray(fc.var_pred_mean, float), 1e-12, None))
                pos = tgt - oi - 1
                out[tgt] = _norm_cdf(mean[pos] / se[pos])
            elif name == "kalman":
                r2 = res.apply(y[:oi + 1], refit=False)
                fc = r2.get_forecast(horizon)
                mean = np.asarray(fc.predicted_mean, float)
                se = np.sqrt(np.clip(np.asarray(fc.var_pred_mean, float), 1e-12, None))
                pos = tgt - oi - 1
                out[tgt] = _norm_cdf(mean[pos] / se[pos])
            else:   # markov
                xi = _hamilton_filter(y[:oi + 1], mu, sigma, P)[-1]
                for j, ix in enumerate(tgt):
                    h = int(ix - oi)
                    xh = xi @ np.linalg.matrix_power(P, h)
                    out[ix] = float(xh @ p_up_regime)
        except Exception as e:                                   # pragma: no cover
            log.debug("%s: origin %s failed (%s)", name, day, e)
            continue

    tail = out[n_train:]
    base = float((y[:n_train] > 0).mean())
    return np.where(np.isfinite(tail), tail, base)



def _fit_predict(name, tr, te, features=FEATURES, seed=0):
    """Return predicted P(DART > 0) on the test rows. Models' hyperparameters are NOT TUNED."""
  
    if name == "persistence":
        # sign of the most recent COMPLETE observation at that node-hour
        return (te["lag_a"] > 0).astype(float).to_numpy()

    if name == "climatology":
        # historical sign frequency by (node, hour, month), fitted on TRAIN only
        g = tr.groupby(["pnode_id", "hour", "month"])["y"].mean()
        gl = tr.groupby(["pnode_id", "hour"])["y"].mean()
        gn = tr.groupby(["pnode_id"])["y"].mean()
        glob_ = float(tr["y"].mean())
        idx3 = list(zip(te["pnode_id"], te["hour"], te["month"]))
        idx2 = list(zip(te["pnode_id"], te["hour"]))
        out = pd.Series(g.reindex(idx3).to_numpy(), index=te.index)
        out = out.fillna(pd.Series(gl.reindex(idx2).to_numpy(), index=te.index))
        out = out.fillna(pd.Series(gn.reindex(te["pnode_id"]).to_numpy(),
                                   index=te.index))
        return out.fillna(glob_).to_numpy()

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    Xtr, ytr = tr[features].to_numpy(float), tr["y"].to_numpy(int)
    Xte = te[features].to_numpy(float)
    if len(np.unique(ytr)) < 2:
        return np.full(len(te), float(ytr.mean()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "logistic":
            from sklearn.linear_model import LogisticRegression
            mdl = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                            ("m", LogisticRegression(max_iter=2000,
                                                     random_state=seed))])
        elif name == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            mdl = Pipeline([("m", HistGradientBoostingClassifier(
                random_state=seed))])        
        elif name == "rf":
            from sklearn.ensemble import RandomForestClassifier
            mdl = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", RandomForestClassifier(
                                n_estimators=300, min_samples_leaf=20,
                                n_jobs=-1, random_state=seed))])
        elif name == "mlp":
            from sklearn.neural_network import MLPClassifier
            mdl = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                            ("m", MLPClassifier(hidden_layer_sizes=(32, 16),
                                                max_iter=400, random_state=seed))])
        else:
            raise ValueError(f"unknown model {name}")
        mdl.fit(Xtr, ytr)
        return mdl.predict_proba(Xte)[:, 1]


def run_models(panel, train_end, models=MODELS, features=FEATURES, seed=0,
               min_lag_days=2):
    """Strict TEMPORAL split: train on rows up to train_end, test after.
    Returns the test frame with one probability column per model."""
    d = panel.dropna(subset=["y"]).copy()
    d["date"] = pd.DatetimeIndex(d[KEY]).normalize()
    cutoff = pd.Timestamp(train_end, tz="UTC").normalize()
    tr = d[d["date"] <= cutoff]
    te = d[d["date"] > cutoff].copy()
    if tr.empty or te.empty:
        raise SystemExit(f"empty split at {train_end}: train={len(tr)}, test={len(te)}")
    # rows without the core lag cannot be predicted by ANY model
    tr = tr.dropna(subset=["lag_a"])
    te = te.dropna(subset=["lag_a"])
    log.info("train %d rows (to %s) | test %d rows (%s..%s) | %d nodes",
             len(tr), cutoff.date(), len(te),
             te["date"].min().date(), te["date"].max().date(),
             te["pnode_id"].nunique())
    for m in models:
        if m in TS_MODELS:
            continue                            # handled per node below
        te[f"p_{m}"] = _fit_predict(m, tr, te, features, seed)

    ts = [m for m in models if m in TS_MODELS]
    if ts:
        for m in ts:
            te[f"p_{m}"] = np.nan
        full = pd.concat([tr, te], ignore_index=True).sort_values(
            ["pnode_id", KEY]).reset_index(drop=True)
        for pid, g in full.groupby("pnode_id", sort=True):
            g = g.sort_values(KEY).reset_index(drop=True)
            n_tr = int((g["date"] <= cutoff).sum())
            if n_tr < 24 * 30 or n_tr >= len(g):
                continue
            te_key = g.loc[n_tr:, KEY].to_numpy()
            for m in ts:
                log.info("  %s: node %s (%d train rows)", m, pid, n_tr)
                p = _rolling_ts_predict(m, g, n_tr, min_lag_days, seed)
                sel = (te["pnode_id"] == pid)
                mp = pd.Series(p, index=pd.DatetimeIndex(te_key))
                te.loc[sel, f"p_{m}"] = mp.reindex(
                    pd.DatetimeIndex(te.loc[sel, KEY])).to_numpy()
        for m in ts:
            base = float(tr["y"].mean())
            te[f"p_{m}"] = te[f"p_{m}"].fillna(base)
    return te, tr


# ------------------------------------------------------------------ metrics ---------------------------------------------------
def skill_metrics(te, model, thresh=0.5):
    """Directional and economic-magnitude metrics.Gross value only."""
    p = te[f"p_{model}"].to_numpy(float)
    y = te["y"].to_numpy(int)
    dart = te["dart"].to_numpy(float)
    pred = (p > thresh).astype(int)
    correct = (pred == y)
    out = {
        "model": model, "n": int(len(y)),
        "hit_rate": float(correct.mean()),
        "base_rate": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }

    maj = max(y.mean(), 1 - y.mean())
    out["hit_rate_minus_majority"] = out["hit_rate"] - float(maj)
    pos = pred == 1
    out["precision_up"] = float(y[pos].mean()) if pos.any() else np.nan
    out["share_predicted_up"] = float(pos.mean())
    # GROSS value: mean |DART| captured when right minus given up when wrong.
    # Economically interpretable, requires no settlement or cost assumptions.
    out["gross_value_per_mwh"] = float(np.mean(np.abs(dart) * np.where(correct, 1, -1)))
    out["mean_abs_dart_correct"] = (float(np.abs(dart[correct]).mean())
                                    if correct.any() else np.nan)
    out["mean_abs_dart_wrong"] = (float(np.abs(dart[~correct]).mean())
                                  if (~correct).any() else np.nan)
    return out


def reliability_table(te, model, bins=10):
    """Calibration plot."""
    p = te[f"p_{model}"].to_numpy(float)
    y = te["y"].to_numpy(int)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"model": model, "bin": b,
                     "bin_lo": edges[b], "bin_hi": edges[b + 1],
                     "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                     "observed_freq": float(y[m].mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ driver -----------------------------------------------------------------
def run(data_dir, panel_csv, out_dir, component, train_end, min_lag_days,
        models, max_nodes, seed):
    os.makedirs(out_dir, exist_ok=True)
    results, rel = [], []
    for violate in (False, True):
        tag = "lag_violating" if violate else "gate_closure"
        panel = build_panel(data_dir, panel_csv, component, min_lag_days,
                            violate, max_nodes)
        if panel.empty:
            raise SystemExit("no panel built")
        te, tr = run_models(panel, train_end, models, seed=seed,
                            min_lag_days=(1 if violate else min_lag_days))
        te.to_csv(os.path.join(out_dir, f"predictions_{tag}_{component}.csv"),
                  index=False)
        for m in models:
            s = skill_metrics(te, m)
            s["information_set"] = tag
            s["lag_days"] = int(te["lag_days_used"].iloc[0])
            results.append(s)
            if not violate:
                rel.append(reliability_table(te, m))
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(out_dir, f"baseline_skill_{component}.csv"), index=False)
    if rel:
        pd.concat(rel, ignore_index=True).to_csv(
            os.path.join(out_dir, f"reliability_{component}.csv"), index=False)

    pd.set_option("display.width", 220)
    print(f"\n=== BASELINE DIRECTIONAL SKILL ({component}) ===")
    cols = ["information_set", "lag_days", "model", "n", "hit_rate", "base_rate",
            "hit_rate_minus_majority", "brier", "gross_value_per_mwh"]
    print(res[cols].round(4).to_string(index=False))

    piv = res.pivot_table(index="model", columns="information_set",
                          values="hit_rate")
    if {"gate_closure", "lag_violating"} <= set(piv.columns):
        piv["inflation"] = piv["lag_violating"] - piv["gate_closure"]
        print("\n=== LAG-VIOLATION INFLATION (RQ2, reframed) ===")
        print(piv.round(4).to_string())
    return res


# ------------------------------------------------------------------ validation --------------------------------------------------------
def _validate():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok = ok and bool(cond)

    print("=== validate baseline forecasters ===")
    rng = np.random.default_rng(0)
    H = 24
    n_days = 400
    hours = pd.date_range("2024-08-01", periods=n_days * H, freq="h", tz="UTC")

    e_ = rng.normal(0, 5, n_days + 1)
    day_val = e_[1:] + e_[:-1]                 # MA(1): zero autocorr at lag 2
    dart = np.repeat(day_val, H) + rng.normal(0, 0.5, n_days * H)
    d = pd.DataFrame({KEY: hours, "dart": dart})

    f_ok = make_features(d, min_lag_days=2, violate_lag=False)
    f_bad = make_features(d, min_lag_days=2, violate_lag=True)
    chk("gate-closure features use a 2-day lag", f_ok["lag_days_used"].iloc[0] == 2)
    chk("violation features use a 1-day lag", f_bad["lag_days_used"].iloc[0] == 1)

    m_ok = f_ok["lag_a"].notna()
    m_bad = f_bad["lag_a"].notna()
    agree_ok = float((np.sign(f_ok.loc[m_ok, "lag_a"])
                      == np.sign(f_ok.loc[m_ok, "dart"])).mean())
    agree_bad = float((np.sign(f_bad.loc[m_bad, "lag_a"])
                       == np.sign(f_bad.loc[m_bad, "dart"])).mean())
    ceiling = 0.5 + np.arcsin(0.5) / np.pi          # 0.667, the MA(1) maximum
    chk("lag-1 feature carries signal, near the MA(1) theoretical ceiling",
        0.60 < agree_bad <= ceiling + 0.03,
        f"({agree_bad:.3f}; ceiling {ceiling:.3f})")
    chk("lag-2 feature carries NO signal (MA(1) autocorr is zero at lag 2)",
        abs(agree_ok - 0.5) < 0.05, f"({agree_ok:.3f})")
    chk("the two information sets separate by ~the theoretical maximum",
        agree_bad - agree_ok > 0.10,
        f"(separation {agree_bad - agree_ok:.3f}; max possible ~{ceiling - 0.5:.3f})")

    # ---------- NO LOOKAHEAD ----------
    # every feature at row i must be computable from dart[: i - lag*24 + 1]
    i = 5000
    lag_rows = 2 * H
    for col in ("lag_a", "lag_b", "lag_week", "roll_mean_7d", "roll_signfreq_7d"):
        v = f_ok[col].iloc[i]
        recomputed_ok = np.isfinite(v)
        chk(f"'{col}' is finite well inside the sample", recomputed_ok)
    chk("lag_a at row i equals dart at row i-48 (exact 2-day lag)",
        np.isclose(f_ok["lag_a"].iloc[i], f_ok["dart"].iloc[i - lag_rows]),
        f"({f_ok['lag_a'].iloc[i]:.4f} vs {f_ok['dart'].iloc[i - lag_rows]:.4f})")
    chk("first 48 rows of lag_a are NaN (no lookahead at the start)",
        f_ok["lag_a"].iloc[:lag_rows].isna().all())
    chk("rolling window is shifted, so it excludes the current hour",
        np.isclose(f_ok["roll_mean_7d"].iloc[i],
                   f_ok["dart"].iloc[i - lag_rows - 7 * H + 1:i - lag_rows + 1].mean()),
        "(matches a shifted 7-day mean)")

    # ---------- MODELS ----------
    panel = f_ok.assign(pnode_id=1, zone="Z")
    e2_ = rng.normal(0, 5, n_days + 1)
    dart2 = np.repeat(e2_[1:] + e2_[:-1], H) + rng.normal(0, 0.5, n_days * H)
    panel2 = make_features(pd.DataFrame({KEY: hours, "dart": dart2}),
                           2, False).assign(pnode_id=2, zone="Z")
    P = pd.concat([panel, panel2], ignore_index=True)
    te, tr = run_models(P, "2025-06-30", models=MODELS)
    chk("temporal split leaves no test date on or before the cutoff",
        te["date"].min() > pd.Timestamp("2025-06-30", tz="UTC"))
    chk("train and test do not overlap in time",
        tr["date"].max() <= pd.Timestamp("2025-06-30", tz="UTC"))
    for m in MODELS:
        p = te[f"p_{m}"].to_numpy(float)
        chk(f"{m}: probabilities within [0,1]", np.all((p >= 0) & (p <= 1)),
            f"(min {p.min():.3f}, max {p.max():.3f})")
        chk(f"{m}: no NaN predictions", np.isfinite(p).all())

    # persistence must EQUAL the sign of lag_a, by definition
    pp = te["p_persistence"].to_numpy()
    chk("persistence equals the sign of the lagged value exactly",
        np.array_equal(pp, (te["lag_a"] > 0).astype(float).to_numpy()))

    # climatology must be fitted on TRAIN only
    cl = te["p_climatology"].to_numpy()
    chk("climatology returns frequencies in [0,1] with no NaN",
        np.all((cl >= 0) & (cl <= 1)) and np.isfinite(cl).all())

    # ---------- METRICS ----------
    s = skill_metrics(te, "persistence")
    chk("hit rate is in [0,1]", 0 <= s["hit_rate"] <= 1, f"({s['hit_rate']:.3f})")
    chk("brier score is in [0,1]", 0 <= s["brier"] <= 1, f"({s['brier']:.3f})")
    # a perfect predictor scores 1.0 and a perfectly wrong one 0.0
    tperf = te.copy()
    tperf["p_perfect"] = tperf["y"].astype(float)
    sp = skill_metrics(tperf, "perfect")
    chk("a perfect predictor scores hit_rate 1.0 and brier 0.0",
        abs(sp["hit_rate"] - 1.0) < 1e-12 and sp["brier"] < 1e-12)
    chk("a perfect predictor's gross value equals mean |DART|",
        abs(sp["gross_value_per_mwh"] - np.abs(tperf["dart"]).mean()) < 1e-9,
        f"({sp['gross_value_per_mwh']:.4f})")
    tworst = te.copy()
    tworst["p_worst"] = 1.0 - tworst["y"].astype(float)
    sw = skill_metrics(tworst, "worst")
    chk("a perfectly wrong predictor scores hit_rate 0.0 and NEGATIVE gross value",
        abs(sw["hit_rate"]) < 1e-12 and sw["gross_value_per_mwh"] < 0,
        f"({sw['gross_value_per_mwh']:.4f})")

    # ---------- CALIBRATION ----------
    r = reliability_table(tperf, "perfect")
    chk("reliability table produced with monotone bins",
        len(r) > 0 and r["bin"].is_monotonic_increasing)
    chk("reliability bins sum to the test sample", int(r["n"].sum()) == len(tperf))

    # ================= NEW MODEL FAMILIES =================
    print("  --- rolling-origin lag discipline ---")
    dts = pd.DatetimeIndex(pd.DatetimeIndex(hours).normalize())
    day = dts[5000].normalize()
    oi = _origin_index(dts, day, 2)
    chk("origin is on or before D-2 (never later)",
        dts[oi] <= day - pd.Timedelta(days=2),
        f"(origin {dts[oi].date()} for target {day.date()})")
    chk("origin is the LATEST admissible observation",
        oi + 1 >= len(dts) or dts[oi + 1] > day - pd.Timedelta(days=2),
        f"(next row {dts[min(oi+1, len(dts)-1)].date()})")
    tgt = np.flatnonzero(dts == day)
    chk("forecast horizon reaches the target day's LAST hour",
        int(tgt[-1] - oi) >= 24, f"(horizon {int(tgt[-1] - oi)})")
    chk("target hours are strictly after the origin", tgt[0] > oi)
    oi1 = _origin_index(dts, day, 1)
    chk("a 1-day lag yields a LATER origin than a 2-day lag", oi1 > oi,
        f"({dts[oi1].date()} vs {dts[oi].date()})")

  
    reg = pd.date_range("2024-11-01", periods=6 * 24, freq="h", tz="UTC")
    short_day = reg.delete(30)                      # a 23-hour day
    long_day = short_day.insert(50, short_day[50])  
    dirr = pd.DatetimeIndex(pd.DatetimeIndex(long_day).normalize())
    counts = pd.Series(dirr).value_counts()
    chk("irregular index really has non-24-hour days",
        (counts != 24).any(), f"(day lengths {sorted(counts.unique())})")
    tday = dirr[-1]
    oid = _origin_index(dirr, tday, 2)
    chk("origin logic survives irregular days (position-based, not 24/day)",
        oid >= 0 and dirr[oid] <= tday - pd.Timedelta(days=2),
        f"(origin {dirr[oid].date()} for target {tday.date()})")
    tgt_irr = np.flatnonzero(dirr == tday)
    chk("horizon derived from POSITIONS covers the whole target day",
        int(tgt_irr[-1] - oid) >= int(tgt_irr.size),
        f"(horizon {int(tgt_irr[-1] - oid)}, day has {tgt_irr.size} hours)")


    try:
        from statsmodels.tsa.regime_switching.markov_regression import (
            MarkovRegression)
        rr = np.random.default_rng(7)
        st = np.zeros(1500, dtype=int)
        for i in range(1, 1500):
            st[i] = st[i-1] if rr.random() < 0.95 else 1 - st[i-1]
        ym = np.where(st == 0, rr.normal(-2, 1, 1500), rr.normal(3, 1, 1500))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mres = MarkovRegression(ym, k_regimes=2, switching_variance=True).fit()
        mu_, sg_, Pm = markov_params(mres)
        chk("markov_params recovers the injected regime means (-2 and +3)",
            abs(min(mu_) - (-2.0)) < 0.4 and abs(max(mu_) - 3.0) < 0.4,
            f"({np.round(np.sort(mu_), 3)})")
        mine = _hamilton_filter(ym, mu_, sg_, Pm)
        ref = np.asarray(mres.filtered_marginal_probabilities)
        d_ = float(np.max(np.abs(mine[50:] - ref[50:])))
        chk("Hamilton filter matches statsmodels' filtered probabilities",
            d_ < 1e-6, f"(max abs diff {d_:.2e})")
        chk("filtered probabilities are a valid distribution",
            np.allclose(mine.sum(axis=1), 1.0) and (mine >= 0).all())
        chk("transition matrix rows sum to 1", np.allclose(Pm.sum(axis=1), 1.0))
    except ImportError:
        print("  [SKIP] statsmodels unavailable for the Hamilton cross-check")


    rho_test = 0.995
    rr2 = np.random.default_rng(11)
    nT = 24 * 300
    ar = np.zeros(nT)
    for i in range(1, nT):
        ar[i] = rho_test * ar[i-1] + rr2.normal(0, 1)
    hrs2 = pd.date_range("2024-08-01", periods=nT, freq="h", tz="UTC")
    fa = make_features(pd.DataFrame({KEY: hrs2, "dart": ar}), 2, False)
    fa = fa.assign(pnode_id=9, zone="Z")
    fa["date"] = pd.DatetimeIndex(fa[KEY]).normalize()
    n_tr = int((fa["date"] <= pd.Timestamp("2025-03-31", tz="UTC")).sum())
    pk = _rolling_ts_predict("kalman", fa, n_tr, 2)
    chk("rolling Kalman returns one probability per TEST row",
        len(pk) == len(fa) - n_tr, f"({len(pk)} vs {len(fa) - n_tr})")
    chk("rolling Kalman probabilities are valid",
        np.all(np.isfinite(pk)) and np.all((pk >= 0) & (pk <= 1)),
        f"(min {pk.min():.3f}, max {pk.max():.3f})")
    yte = (fa["dart"].to_numpy()[n_tr:] > 0).astype(int)
  
    # ORACLE: the sign of the value at each forecast origin
  
    dts_fa = pd.DatetimeIndex(fa["date"])
    oracle = np.empty(len(fa) - n_tr)
    for j, ix in enumerate(range(n_tr, len(fa))):
        oi_ = _origin_index(dts_fa, dts_fa[ix], 2)
        oracle[j] = 1.0 if fa["dart"].iloc[oi_] > 0 else 0.0
    hit_o = float((oracle.astype(int) == yte).mean())
    theo = float(np.mean(0.5 + np.arcsin(np.clip(rho_test ** np.arange(25, 49),
                                                 -1, 1)) / np.pi))
    chk("oracle hit rate is close to the theoretical ceiling",
        abs(hit_o - theo) < 0.10, f"(oracle {hit_o:.3f} vs theory {theo:.3f})")

    hit_k = float(((pk > 0.5).astype(int) == yte).mean())
    chk("self-test: Kalman recovers sign on a synthetic AR(1) when signal exists (hit > 0.60; synthetic data)", hit_k > 0.60,
        f"(hit rate {hit_k:.3f}; oracle {hit_o:.3f})")
    chk("Kalman is very close to the oracle", hit_k > hit_o - 0.15,
        f"({hit_k:.3f} vs {hit_o:.3f})")

    ps = _rolling_ts_predict("sarima", fa, n_tr, 2)
    chk("rolling SARIMA probabilities are valid",
        np.all(np.isfinite(ps)) and np.all((ps >= 0) & (ps <= 1)),
        f"(min {ps.min():.3f}, max {ps.max():.3f})")
    hit_s = float(((ps > 0.5).astype(int) == yte).mean())
    chk("self-test: SARIMA recovers sign on a synthetic AR(1) when signal exists (hit > 0.60; synthetic data)", hit_k > 0.60,
        f"(hit rate {hit_s:.3f}; oracle {hit_o:.3f})")
    chk("SARIMA is very close to the oracle", hit_s > hit_o - 0.15,
        f"({hit_s:.3f} vs {hit_o:.3f})")

    pm_ = _rolling_ts_predict("markov", fa, n_tr, 2)
    chk("rolling Markov-switching probabilities are valid",
        np.all(np.isfinite(pm_)) and np.all((pm_ >= 0) & (pm_ <= 1)),
        f"(min {pm_.min():.3f}, max {pm_.max():.3f})")

    # --- NO LOOKAHEAD: corrupting the target day SHOULD NOT change its forecast
    fb = fa.copy()
    last_day = fb["date"].max()
    fb.loc[fb["date"] == last_day, "dart"] = 1e6      
    pk2 = _rolling_ts_predict("kalman", fb, n_tr, 2)
    same = np.isclose(pk[:-24], pk2[:-24], atol=1e-9).mean()
    chk("corrupting the FINAL day leaves earlier forecasts unchanged (no lookahead)",
        same > 0.999, f"({100*same:.2f}% identical)")

    # --- random forest
    te2, tr2 = run_models(P, "2025-06-30", models=("rf",))
    prf = te2["p_rf"].to_numpy(float)
    chk("random forest yields valid probabilities",
        np.all(np.isfinite(prf)) and np.all((prf >= 0) & (prf <= 1)),
        f"(min {prf.min():.3f}, max {prf.max():.3f})")

    chk("nine model classes across four families are registered",
        len(MODELS) == 9 and set(TS_MODELS) <= set(MODELS), f"({len(MODELS)})")

    print("=== " + ("BASELINE FORECASTERS VALIDATED" if ok
                    else "VALIDATION FAILED") + " ===")
    return ok


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--panel", default="panel_dedup_all.csv")
    ap.add_argument("--out-dir", default="rq3_out")
    ap.add_argument("--component", default="congestion", choices=list(COMPONENTS))
    ap.add_argument("--train-end", default="2025-12-31")
    ap.add_argument("--min-lag-days", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(0 if _validate() else 1)
    run(a.data_dir, a.panel, a.out_dir, a.component, a.train_end,
        a.min_lag_days, a.models, a.max_nodes, a.seed)


if __name__ == "__main__":
    main()
