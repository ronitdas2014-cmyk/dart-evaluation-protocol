#!/usr/bin/env python3
"""Build Table CAISO (external replication on CAISO both-legs congestion).

Goal:
    From the CAISO congestion predictions produced by baseline_forecasters.py
    (--component congestion, on the CAISO reformat), compute directional
    accuracy A, the winsorised-mass-weighted accuracy hbar_w, the excess over
    the node-conditional (per-node majority) benchmark Delta_nc, and the
    node-balanced gross winsorised payoff, each with a five-day block-bootstrap
    95% interval.

Preliminaries:
    A            = mean(hit),  hit = [ (p>0.5) == y ]
    hbar_w       = sum(min(|dart|,cap) * hit) / sum(min(|dart|,cap))
    Delta_nc     = A - A_nc,  A_nc = accuracy of the per-node majority sign;
                   the 95% CI is the PAIRED (hit - benchmark_correct) bootstrap
    payoff       = node-balanced mean of  direction * clip(dart, +/-cap)
    cap          = 99th percentile of |dart| on the Out-Of-Sample (OOS) panel

Inputs:
    --pred PRED.csv : CAISO congestion predictions (dart, y, p_<model>, pnode_id,
        datetime_beginning_utc), i.e. baseline_forecasters.py --component congestion
        on the caiso_reformat output.

Outputs:
    --out CSV : one row per forecaster (A, hbar_w, Delta_nc [CI], payoff [CI]).

Usage
    python3 caiso_table.py --pred caiso_cong_out/predictions_gate_closure_congestion.csv \
        --out tab_caiso.csv

"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SEED, N_BOOT, BLOCK_DAYS, WINSOR_Q = 20260725, 2000, 5, 0.99
FORECASTERS = {"persistence": "Persistence", "climatology": "Seasonal frequency",
               "logistic": "Logistic", "gbm": "Gradient boosting"}


def block_resamples(day_id):
    nd = int(day_id.max()) + 1
    rows = [np.where(day_id == d)[0] for d in range(nd)]
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(nd / BLOCK_DAYS))
    return [np.concatenate([rows[k] for k in
            ((rng.integers(0, nd, size=nb)[:, None] + np.arange(BLOCK_DAYS)) % nd).ravel()])
            for _ in range(N_BOOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True); ap.add_argument("--out", default="tab_caiso.csv")
    a = ap.parse_args()

    d = pd.read_csv(a.pred)
    y = d["y"].to_numpy(int); dart = d["dart"].to_numpy(float)
    cap = np.quantile(np.abs(dart), WINSOR_Q); w = np.minimum(np.abs(dart), cap)
    nodes = d["pnode_id"].to_numpy(); unodes = np.unique(nodes)
    up = {z: d.loc[d["pnode_id"] == z, "y"].mean() for z in unodes}
    bench = (d["pnode_id"].map({z: (1 if up[z] >= 0.5 else 0) for z in up}).to_numpy() == y).astype(float)
    A_nc = bench.mean()
    day = pd.factorize(pd.to_datetime(d["datetime_beginning_utc"]).dt.floor("D").to_numpy())[0]
    BP = block_resamples(day)

    def node_balanced(v, idx):
        s = nodes[idx]
        return float(np.mean([v[idx[s == z]].mean() for z in unodes if (s == z).any()]))

    rows = []
    for key, label in FORECASTERS.items():
        p = d[f"p_{key}"].to_numpy(); direction = np.where(p > 0.5, 1, -1)
        hit = ((p > 0.5).astype(int) == y).astype(float)
        A = hit.mean(); hbar_w = (w * hit).sum() / w.sum(); dnc = A - A_nc
        pay_row = direction * np.clip(dart, -cap, cap)
        payoff = node_balanced(pay_row, np.arange(len(d)))
        dnc_lo, dnc_hi = np.percentile([hit[i].mean() - bench[i].mean() for i in BP], [2.5, 97.5])
        p_lo, p_hi = np.percentile([node_balanced(pay_row, i) for i in BP], [2.5, 97.5])
        rows.append(dict(forecaster=label, A=round(A, 3), hbar_w=round(hbar_w, 3),
                         delta_nc=round(dnc, 3), dnc_lo=round(dnc_lo, 3), dnc_hi=round(dnc_hi, 3),
                         payoff=round(payoff, 2), pay_lo=round(p_lo, 2), pay_hi=round(p_hi, 2)))
    out = pd.DataFrame(rows); out.to_csv(a.out, index=False)
    print(f"A_nc (node-conditional) = {A_nc:.4f}   cap = ${cap:.0f}   n = {len(d)}")
    print(out.to_string(index=False)); print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
