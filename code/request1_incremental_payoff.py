#!/usr/bin/env python3
"""Incremental payoff of each forecaster relative to the passive INC benchmark.

Purpose
    For each forecasting method, compute the PAIRED per-period difference
        forecast payoff - passive-INC payoff
    on the full settlement spread, with a dependence-aware moving-block
    bootstrap. Report the point estimate and 95% CI, a one-sided test of a
    positive increment with Holm control across the family, and save the raw
    bootstrap draws.

Definitions
    Passive INC = always-long: submit an increment offer every hour (direction
    +1), which settles the full DA-RT spread. Payoffs are GROSS and winsorised
    at the 99th percentile of |spread| (the paper's headline convention; the
    fee sensitivity is handled separately in the cost section). The paired
    difference for method k in hour t is
        Delta_{k,t} = (d_{k,t} - 1) * winsor(spread_t),
    which equals the method payoff minus the passive-INC payoff on the same
    observation; it also equals the "Delta vs always-long" column of the
    headline table, so the two must agree when computed from the same predictions.

Inputs
    --pred PRED.csv  full-spread predictions with columns:
        dt, dart_total, and one direction column per method named d_<method>
        with values in {-1, +1} (+1 = INC/long, -1 = DEC/short).

Outputs
    --out-prefix PREFIX (default request1_pjm):
        PREFIX_summary.csv  one row per method: mean diff, 95% CI, bootstrap SE,
                            one-sided p (increment > 0), Holm-adjusted p, and
                            whether the 95% CI excludes zero.
        PREFIX_draws.csv    the B bootstrap draws of the mean difference, one
                            column per method (raw output, not just the summary).

Usage
    python3 request1_incremental_payoff.py --pred pred_total_spread.csv \
        --out-prefix request1_pjm
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SEED = 20260725
N_BOOT = 2000
BLOCK_DAYS = 5
WINSOR_Q = 0.99


def block_resamples(day_id: np.ndarray) -> list[np.ndarray]:
    """Pre-draw N_BOOT moving-block row-index resamples (5-day day-blocks)."""
    n_days = int(day_id.max()) + 1
    rows_by_day = [np.where(day_id == d)[0] for d in range(n_days)]
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n_days / BLOCK_DAYS))
    resamples = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n_days, size=n_blocks)[:, None]
        block_days = ((starts + np.arange(BLOCK_DAYS)) % n_days).ravel()
        resamples.append(np.concatenate([rows_by_day[d] for d in block_days]))
    return resamples


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm (1979) step-down familywise-adjusted p-values."""
    order = np.argsort(p_values)
    k = p_values.size
    adjusted = np.empty(k)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="full-spread predictions CSV")
    parser.add_argument("--out-prefix", default="request1_pjm", help="output prefix")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred, parse_dates=["dt"])
    spread = pred["dart_total"].to_numpy()
    cap = np.quantile(np.abs(spread), WINSOR_Q)
    winsorised = np.clip(spread, -cap, cap)  # passive-INC payoff (direction +1)

    methods = [c[2:] for c in pred.columns if c.startswith("d_")]
    day_id = pd.factorize(pred["dt"].dt.floor("D").to_numpy())[0]
    resamples = block_resamples(day_id)

    summary_rows = []
    draws_frame = {}
    for method in methods:
        direction = pred[f"d_{method}"].to_numpy().astype(float)
        if not np.all(np.isin(direction, (-1.0, 1.0))):
            raise ValueError(f"direction column d_{method} must be in {{-1,+1}}")
        diff = (direction - 1.0) * winsorised          # paired per-period difference
        point = float(diff.mean())
        draws = np.array([diff[idx].mean() for idx in resamples])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        se = float(draws.std(ddof=1))
        p_one_sided = float((1 + np.sum(draws <= 0.0)) / (N_BOOT + 1))  # H1: increment > 0
        draws_frame[method] = draws
        summary_rows.append({
            "method": method,
            "passive_inc_payoff": float(winsorised.mean()),
            "mean_increment": point,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "bootstrap_se": se,
            "p_onesided_beats_inc": p_one_sided,
            "ci_excludes_zero": not (lo <= 0.0 <= hi),
        })

    summary = pd.DataFrame(summary_rows)
    summary["p_holm"] = holm(summary["p_onesided_beats_inc"].to_numpy())
    summary["beats_inc_holm_5pct"] = summary["p_holm"] < 0.05
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    pd.DataFrame(draws_frame).to_csv(f"{args.out_prefix}_draws.csv", index=False)

    print(f"Incremental payoff over passive INC (full spread, gross)")
    print(f"passive INC (always-long) payoff = {winsorised.mean():+.2f}/MWh "
          f"(winsorised at Q{WINSOR_Q:.2f} = ${cap:.0f}); n = {spread.size}")
    print(f"{'method':14}{'increment':>10}{'95% CI':>18}{'p_1sided':>10}{'p_Holm':>9}{'excl 0':>8}")
    for r in summary_rows:
        h = summary.loc[summary['method'] == r['method'], 'p_holm'].iloc[0]
        print(f"  {r['method']:12}{r['mean_increment']:>+10.2f}"
              f"   [{r['ci_lo']:>+6.2f},{r['ci_hi']:>+6.2f}]"
              f"{r['p_onesided_beats_inc']:>10.3f}{h:>9.3f}{str(r['ci_excludes_zero']):>8}")
    n_beat = int(summary["beats_inc_holm_5pct"].sum())
    print(f"\nMethods beating passive INC (Holm, 5%): {n_beat} of {len(methods)}")
    print(f"Saved {args.out_prefix}_summary.csv and {args.out_prefix}_draws.csv")


if __name__ == "__main__":
    main()
