#!/usr/bin/env python3
"""Data-snooping-robust test that no forecaster has positive economic value.

Purpose
    Apply White's (2000) Reality Check and Hansen's (2005) Superior
    Predictive Ability (SPA) test to the nine directional forecasters, with a
    benchmark of zero economic value. Reproduces the in-text Reality Check
    p-value of 0.59 for the nine-forecaster family.

Inputs
    --data DATA_DIR/predictions_gate_closure_congestion.csv
        78,264 out-of-sample node-hours: forecaster probabilities `p_<model>`,
        realised spread `dart`, sign outcome `y`, and `zone`.

Outputs (stdout)
    Best strategy value ($/MWh) and Reality Check / SPA p-values.

Usage
    python3 reality_check_spa.py --data ./data

Note
    The manuscript also reports a 73-strategy extended family (nine forecasters
    x four confidence thresholds x two winsorisation caps, plus always-long).
    That extension requires the paper's exact confidence-ranking and tie
    handling for the hard 0/1 forecasters (persistence), which is not
    reconstructed here; only the nine-forecaster family is recomputed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SEED = 20260725
N_BOOT = 2000
BLOCK_DAYS = 5
WINSOR_Q = 0.99
MODELS = [
    "persistence", "climatology", "logistic", "gbm", "rf",
    "mlp", "sarima", "kalman", "markov",
]


def _block_indices(day_id: np.ndarray) -> list[np.ndarray]:
    """Pre-draw N_BOOT moving-block resamples of row indices (5-day blocks)."""
    n_days = day_id.max() + 1
    rows_by_day = [np.where(day_id == d)[0] for d in range(n_days)]
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n_days / BLOCK_DAYS))
    resamples = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n_days, size=n_blocks)[:, None]
        block_days = ((starts + np.arange(BLOCK_DAYS)) % n_days).ravel()
        resamples.append(np.concatenate([rows_by_day[d] for d in block_days]))
    return resamples


def strategy_values(gc: pd.DataFrame) -> np.ndarray:
    """Per-period value ($/MWh) of each forecaster; benchmark = 0.

    Direction is sign(p - 0.5); the payoff is the realised spread winsorised at
    the 99th percentile of |DART|. Column k holds strategy k's per-period value.
    """
    dart = gc["dart"].to_numpy()
    cap = np.quantile(np.abs(dart), WINSOR_Q)
    winsorised = np.clip(dart, -cap, cap)
    columns = []
    for model in MODELS:
        direction = np.where(gc[f"p_{model}"].to_numpy() > 0.5, 1.0, -1.0)
        columns.append(direction * winsorised)
    return np.column_stack(columns)


def reality_check(values: np.ndarray, resamples: list[np.ndarray]) -> float:
    """White (2000) Reality Check p-value (block bootstrap, recentred)."""
    n = values.shape[0]
    mean_value = values.mean(axis=0)
    stat = np.sqrt(n) * mean_value.max()
    boot = np.empty(len(resamples))
    for b, idx in enumerate(resamples):
        boot[b] = np.sqrt(n) * (values[idx].mean(axis=0) - mean_value).max()
    return float((boot >= stat).mean())


def spa(values: np.ndarray, resamples: list[np.ndarray]) -> float:
    """Hansen (2005) consistent SPA p-value (studentised, block bootstrap)."""
    n = values.shape[0]
    mean_value = values.mean(axis=0)
    boot_means = np.array([values[idx].mean(axis=0) for idx in resamples])
    sd = np.where(boot_means.std(axis=0) < 1e-12, 1e-12, boot_means.std(axis=0))
    stat = np.maximum(0.0, np.sqrt(n) * mean_value / sd).max()
    threshold = -np.sqrt(2.0 * np.log(np.log(n)))
    recentre = mean_value * (np.sqrt(n) * mean_value / sd >= threshold)
    boot = np.empty(len(resamples))
    for b, idx in enumerate(resamples):
        z = np.sqrt(n) * (values[idx].mean(axis=0) - recentre) / sd
        boot[b] = np.maximum(0.0, z).max()
    return float((boot >= stat).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data", help="directory with input CSVs")
    args = parser.parse_args()
    gc = pd.read_csv(f"{args.data}/predictions_gate_closure_congestion.csv")
    day_id = pd.factorize(
        pd.to_datetime(gc["datetime_beginning_utc"]).dt.floor("D").to_numpy()
    )[0]
    resamples = _block_indices(day_id)
    values = strategy_values(gc)
    print("White Reality Check / Hansen SPA (nine-forecaster family, benchmark 0)")
    print(f"    best strategy value : {values.mean(axis=0).max():+.3f}/MWh")
    print(f"    Reality Check p     : {reality_check(values, resamples):.2f}"
          "   (manuscript 0.59)")
    print(f"    Hansen SPA p        : {spa(values, resamples):.2f}")


if __name__ == "__main__":
    main()
