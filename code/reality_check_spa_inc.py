#!/usr/bin/env python3
"""Reality Check / SPA against passive INC (and zero), saving raw output.

Purpose
    Test, robustly to data snooping, whether any of the nine directional
    forecasters has superior economic value relative to a benchmark. Two
    benchmarks are run so the versions can be compared:
        zero : the strategy's own payoff (existing paper version; RC p=0.59)
        inc  : passive INC, i.e. always-long (Jian's requested version)
    All payoffs are on the congestion spread, winsorised at the 99th percentile
    of |spread|. The passive-INC benchmark makes this a congestion notional; the
    full-spread version requires the persisted full-spread predictions.

Inputs
    --data DATA_DIR/predictions_gate_closure_congestion.csv
        Out-of-sample node-hours: `p_<model>`, `dart`, `datetime_beginning_utc`.

Outputs
    stdout summary, plus for each benchmark:
        <prefix>_<benchmark>_summary.csv  -- per-forecaster mean excess and p-values
        <prefix>_<benchmark>_draws.csv    -- the B bootstrap draws of the RC and SPA
                                             statistics (raw output for comparison)

Usage
    python3 reality_check_spa_inc.py --data ./data --prefix rc_spa
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


def block_resamples(day_id: np.ndarray) -> list[np.ndarray]:
    """Pre-draw N_BOOT moving-block row-index resamples (5-day day-blocks)."""
    n_days = day_id.max() + 1
    rows_by_day = [np.where(day_id == d)[0] for d in range(n_days)]
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(n_days / BLOCK_DAYS))
    out = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n_days, size=n_blocks)[:, None]
        block_days = ((starts + np.arange(BLOCK_DAYS)) % n_days).ravel()
        out.append(np.concatenate([rows_by_day[d] for d in block_days]))
    return out


def excess_matrix(gc: pd.DataFrame, benchmark: str) -> np.ndarray:
    """Per-period excess payoff of each forecaster over the benchmark (N x 9)."""
    dart = gc["dart"].to_numpy()
    cap = np.quantile(np.abs(dart), WINSOR_Q)
    winsorised = np.clip(dart, -cap, cap)
    base = winsorised if benchmark == "inc" else 0.0  # INC = always-long (dir +1)
    columns = []
    for model in MODELS:
        direction = np.where(gc[f"p_{model}"].to_numpy() > 0.5, 1.0, -1.0)
        columns.append(direction * winsorised - base)
    return np.column_stack(columns)


def reality_check(excess: np.ndarray, resamples: list[np.ndarray]) -> tuple[float, np.ndarray, float]:
    """White (2000) Reality Check: p-value, bootstrap draws, and statistic."""
    n = excess.shape[0]
    mean_excess = excess.mean(axis=0)
    stat = np.sqrt(n) * mean_excess.max()
    draws = np.array([
        np.sqrt(n) * (excess[idx].mean(axis=0) - mean_excess).max() for idx in resamples
    ])
    return float((draws >= stat).mean()), draws, float(stat)


def spa(excess: np.ndarray, resamples: list[np.ndarray]) -> tuple[float, np.ndarray, float]:
    """Hansen (2005) consistent SPA: p-value, bootstrap draws, and statistic."""
    n = excess.shape[0]
    mean_excess = excess.mean(axis=0)
    boot_means = np.array([excess[idx].mean(axis=0) for idx in resamples])
    sd = boot_means.std(axis=0)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    stat = np.maximum(0.0, np.sqrt(n) * mean_excess / sd).max()
    threshold = -np.sqrt(2.0 * np.log(np.log(n)))
    recentre = mean_excess * (np.sqrt(n) * mean_excess / sd >= threshold)
    draws = np.array([
        np.maximum(0.0, np.sqrt(n) * (excess[idx].mean(axis=0) - recentre) / sd).max()
        for idx in resamples
    ])
    return float((draws >= stat).mean()), draws, float(stat)


def run_benchmark(gc: pd.DataFrame, resamples: list[np.ndarray], benchmark: str,
                  prefix: str) -> None:
    """Run RC and SPA for one benchmark and save the summary and raw draws."""
    excess = excess_matrix(gc, benchmark)
    mean_excess = excess.mean(axis=0)
    p_rc, rc_draws, t_rc = reality_check(excess, resamples)
    p_spa, spa_draws, t_spa = spa(excess, resamples)

    summary = pd.DataFrame({"model": MODELS, "mean_excess_per_mwh": mean_excess})
    summary["benchmark"] = benchmark
    summary["reality_check_p"] = p_rc
    summary["spa_p"] = p_spa
    summary.to_csv(f"{prefix}_{benchmark}_summary.csv", index=False)
    pd.DataFrame({"rc_draw": rc_draws, "spa_draw": spa_draws}).to_csv(
        f"{prefix}_{benchmark}_draws.csv", index=False
    )

    label = "passive INC (always-long)" if benchmark == "inc" else "zero value"
    print(f"Benchmark = {label}")
    print(f"    best forecaster excess : {mean_excess.max():+.3f}/MWh")
    print(f"    Reality Check p        : {p_rc:.2f}")
    print(f"    Hansen SPA p           : {p_spa:.2f}")
    print(f"    saved {prefix}_{benchmark}_summary.csv and {prefix}_{benchmark}_draws.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data", help="directory with input CSVs")
    parser.add_argument("--prefix", default="rc_spa", help="output filename prefix")
    args = parser.parse_args()

    gc = pd.read_csv(f"{args.data}/predictions_gate_closure_congestion.csv")
    day_id = pd.factorize(
        pd.to_datetime(gc["datetime_beginning_utc"]).dt.floor("D").to_numpy()
    )[0]
    resamples = block_resamples(day_id)

    print("Reality Check / SPA, nine-forecaster family, congestion spread\n")
    run_benchmark(gc, resamples, "zero", args.prefix)
    print()
    run_benchmark(gc, resamples, "inc", args.prefix)


if __name__ == "__main__":
    main()
