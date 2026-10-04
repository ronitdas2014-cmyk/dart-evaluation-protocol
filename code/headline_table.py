#!/usr/bin/env python3
"""Build Table ----- (economic value on the full spread, for both PJM and CAISO markets).

Goal:
    From the full-spread predictions produced by baseline_forecasters.py
    (--component total), compute the passive-INC (always-long) premium and each
    forecaster's directional accuracy, gross winsorised value, and paired
    forecast-minus-INC increment, with five-day block-bootstrap 95% intervals.
    The signed-spread model is refit here (a GB regressor on the pipeline's own
    features).

Inputs:
    --pred PRED.csv        full-spread predictions (dart, p_<model>, features,
                           datetime_beginning_utc, pnode_id, zone), i.e. the
                           output of  baseline_forecasters.py --component total.
    --data_dir, --panel, --train_end : to rebuild the panel for the signed-spread
                           refit via baseline_forecasters.build_panel.

Outputs:
    --out CSV : one row per (method): market, hit, value [CI], increment [CI].

Usage
    python3 headline_table.py --pred headline_out/predictions_gate_closure_total.csv \
        --data_dir ./pjm_raw --panel panel_dedup_all.csv --train_end 2025-12-31 \
        --market PJM --out tab_headline_PJM.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

SEED, N_BOOT, BLOCK_DAYS, WINSOR_Q = 20260725, 2000, 5, 0.99


# Table 2 holds SEVEN forecasters: these six probability classifiers (bid the sign of
# P(spread>0)) PLUS one signed-spread GradientBoostingRegressor added below in this source file. The
# regressor is the economically aligned model of Prop. 3 -- it estimates the sign of
# the conditional MEAN spread (the payoff-optimal direction), which a probability
# classifier cannot target. 
CLASSIFIERS = ["persistence", "climatology", "logistic", "gbm", "rf", "mlp"]


def block_resamples(day_id):
    nd = int(day_id.max()) + 1
    rows = [np.where(day_id == d)[0] for d in range(nd)]
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(nd / BLOCK_DAYS))
    return [np.concatenate([rows[k] for k in
            ((rng.integers(0, nd, size=nb)[:, None] + np.arange(BLOCK_DAYS)) % nd).ravel()])
            for _ in range(N_BOOT)]


def main() -> None:
    import baseline_forecasters as bf 
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--panel", required=True); ap.add_argument("--train_end", required=True)
    ap.add_argument("--market", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    te = pd.read_csv(a.pred); te["datetime_beginning_utc"] = pd.to_datetime(
        te["datetime_beginning_utc"], utc=True)
    dart = te["dart"].to_numpy(float)
    cap = np.quantile(np.abs(dart), WINSOR_Q)
    win = np.clip(dart, -cap, cap)  # passive-INC payoff (direction +1)
    ts = np.where(dart > 0, 1, -1)

    # signed-spread refit on own panel/features
    panel = bf.build_panel(a.data_dir, a.panel, "total", 2, False, None)
    cutoff = pd.Timestamp(a.train_end, tz="UTC").normalize()
    tr = panel[panel["date"] <= cutoff].dropna(subset=bf.FEATURES)
    ct = np.quantile(np.abs(tr["dart"]), WINSOR_Q)
    gr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=0).fit(
        tr[bf.FEATURES], np.clip(tr["dart"], -ct, ct))

    directions = {"passive INC": np.ones(len(te), int)}
    for m in CLASSIFIERS:
        directions[m] = np.where(te[f"p_{m}"].to_numpy() > 0.5, 1, -1)
    directions["signed-spread"] = np.where(gr.predict(te[bf.FEATURES].fillna(0)) >= 0, 1, -1)

    day = pd.factorize(te["datetime_beginning_utc"].dt.floor("D").to_numpy())[0]
    BP = block_resamples(day)
    def ci(v): d = np.array([v[i].mean() for i in BP]); return np.percentile(d, [2.5, 97.5])

    rows = []
    for name, d in directions.items():
        pay = d * win
        lo, hi = ci(pay)
        inc = (d - 1) * win  # paired forecast-minus-INC increment
        ilo, ihi = (None, None) if name == "passive INC" else ci(inc)
        rows.append(dict(market=a.market, method=name, hit=round((d == ts).mean(), 3),
                         value=round(pay.mean(), 2), value_lo=round(lo, 2), value_hi=round(hi, 2),
                         increment=(None if name == "passive INC" else round(inc.mean(), 2)),
                         inc_lo=(None if ilo is None else round(ilo, 2)),
                         inc_hi=(None if ihi is None else round(ihi, 2))))
    out = pd.DataFrame(rows); out.to_csv(a.out, index=False)
    print(out.to_string(index=False)); print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
