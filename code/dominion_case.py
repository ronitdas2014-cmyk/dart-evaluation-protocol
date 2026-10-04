#!/usr/bin/env python3
"""Dominion (DOM) Case verification.

Goal:
    Independently reproduce every number used in the Dominion running example
    (Introduction, Model Setup, Results), all on the congestion spread:
        - share of the panel's total absolute-spread economic mass
        - positive-spread frequency (p_up)
        - mean signed spread (mu_c) and its 95% block-bootstrap CI
        - mean magnitudes conditional on positive / negative spreads
        - the frequency-odds (O), magnitude ratio (kappa), and asymmetry O*kappa
        - the accuracy-optimal vs payoff-optimal direction and whether they diverge

Inputs:
    --data DATA_DIR/predictions_gate_closure_congestion.csv
        Out-of-sample node-hours with realized spread `dart` and `zone`.

Outputs:
    stdout summary and --out CSV (default dominion_case.csv), one row per
    quantity, with value, 95% CI where applicable, and a source note.

Usage
    python3 dominion_case.py --data ./data --out dominion_case.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

ZONE = "DOM"
SEED = 20260725
N_BOOT = 2000
BLOCK_DAYS = 5


def block_bootstrap_mean_ci(values: np.ndarray, day_id: np.ndarray) -> tuple[float, float]:
    """Return the 95% moving-block bootstrap CI for the mean of `values`.
    Days are the dependence unit; whole days are resampled in 5-day blocks.
    """
    n_days = day_id.max() + 1
    order = np.argsort(day_id, kind="stable")
    sorted_days = day_id[order]
    start = np.searchsorted(sorted_days, np.arange(n_days), "left")
    stop = np.searchsorted(sorted_days, np.arange(n_days), "right")
    day_sum = np.array([values[order][start[i]:stop[i]].sum() for i in range(n_days)])
    day_count = (stop - start).astype(float)
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n_days / BLOCK_DAYS))
    estimates = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, n_days, size=n_blocks)[:, None]
        blocks = ((starts + np.arange(BLOCK_DAYS)) % n_days).ravel()
        estimates[b] = day_sum[blocks].sum() / day_count[blocks].sum()
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data", help="directory with input CSVs")
    parser.add_argument("--out", default="dominion_case.csv", help="output CSV path")
    args = parser.parse_args()

    gc = pd.read_csv(f"{args.data}/predictions_gate_closure_congestion.csv")
    total_mass = np.abs(gc["dart"].to_numpy()).sum()

    dom = gc[gc["zone"] == ZONE].copy()
    spread = dom["dart"].to_numpy()
    dom_day = pd.factorize(
        pd.to_datetime(dom["datetime_beginning_utc"]).dt.floor("D").to_numpy()
    )[0]

    share = 100.0 * np.abs(spread).sum() / total_mass
    p_up = float((spread > 0).mean())
    mu_c = float(spread.mean())
    e_up = float(spread[spread > 0].mean())
    e_down = float(-spread[spread < 0].mean())
    odds = p_up / (1.0 - p_up)
    kappa = e_up / e_down
    asymmetry = odds * kappa
    mu_lo, mu_hi = block_bootstrap_mean_ci(spread, dom_day)
    acc_dir = "+" if p_up > 0.5 else "-"
    pay_dir = "+" if mu_c > 0 else "-"

    rows = [
        ("share_abs_spread_mass_pct", share, None, None, "share of 18-node panel |spread| mass"),
        ("positive_spread_frequency", p_up, None, None, "P(DA>RT)"),
        ("mean_signed_spread", mu_c, mu_lo, mu_hi, "mu_c with 95% block-bootstrap CI"),
        ("mean_magnitude_up", e_up, None, None, "E[spread | spread>0]"),
        ("mean_magnitude_down", e_down, None, None, "E[|spread| | spread<0]"),
        ("odds", odds, None, None, "p_up/(1-p_up)"),
        ("magnitude_ratio_kappa", kappa, None, None, "E[up]/E[|down|]"),
        ("asymmetry_O_times_kappa", asymmetry, None, None, "diverges if <1 when p_up>0.5"),
        ("accuracy_optimal_direction", acc_dir, None, None, "sign(p_up-0.5)"),
        ("payoff_optimal_direction", pay_dir, None, None, "sign(mu_c)"),
        ("directions_diverge", int(acc_dir != pay_dir), None, None, "1 = accuracy and payoff disagree"),
        ("n_obs", float(spread.size), None, None, "DOM out-of-sample node-hours"),
    ]
    table = pd.DataFrame(rows, columns=["quantity", "value", "ci_lo", "ci_hi", "note"])
    table.to_csv(args.out, index=False)

    print(f"Dominion (DOM) running-case, congestion spread, n={spread.size}")
    for name, value, lo, hi, note in rows:
        ci = f"  [{lo:+.2f}, {hi:+.2f}]" if lo is not None else ""
        shown = f"{value:+.3f}" if isinstance(value, float) else str(value)
        print(f"    {name:32} {shown}{ci}   ({note})")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
