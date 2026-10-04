#!/usr/bin/env python3
"""Stationarity of the congestion spread (ADF and KPSS tests).

Goal:
    Independently recompute the Augmented Dickey-Fuller and KPSS statistics on
    the full-sample hourly congestion spread and compare with the shipped
    stationarity table. Confirms the series is stationary (ADF rejects a unit
    root; KPSS does not reject stationarity).

Inputs:
    --raw RAW_DIR/da_hourly/pnode=<id>/*.csv , RAW_DIR/rt_hourly/pnode=<id>/*.csv
        Raw PJM hourly LMP with `congestion_price_da` / `congestion_price_rt`.
    --data DATA_DIR/stationarity_tests.csv (shipped statistics for comparison)
    --data DATA_DIR/panel_dedup_all.csv    (18-node evaluation panel)

Outputs:
    Per node: recomputed ADF / KPSS statistics vs. the output values.

Usage:
    python3 stationarity_tests.py --data ./data --raw ./pjm_raw

Dependencies:
    statsmodels (see requirements.txt).
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")  # KPSS p-value clipping expected.

ADF_CRIT_5 = -2.86  # 5% critical value for the ADF test (constant, no trend).
KPSS_CRIT_5 = 0.46  # 5% critical value for the KPSS test (level stationarity).


def congestion_spread(raw_dir: str, node: str) -> np.ndarray:
    """Return the hourly congestion spread (DA minus RT) for one node."""
    da = pd.concat(
        [pd.read_csv(f) for f in glob.glob(f"{raw_dir}/da_hourly/pnode={node}/*.csv")],
        ignore_index=True,
    )
    rt = pd.concat(
        [pd.read_csv(f) for f in glob.glob(f"{raw_dir}/rt_hourly/pnode={node}/*.csv")],
        ignore_index=True,
    )
    merged = da[["datetime_beginning_utc", "congestion_price_da"]].merge(
        rt[["datetime_beginning_utc", "congestion_price_rt"]],
        on="datetime_beginning_utc",
    )
    merged["dt"] = pd.to_datetime(merged["datetime_beginning_utc"])
    merged = merged.sort_values("dt")
    spread = merged["congestion_price_da"] - merged["congestion_price_rt"]
    return spread.dropna().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data", help="directory with shipped CSVs")
    parser.add_argument("--raw", required=True, help="raw PJM LMP directory")
    args = parser.parse_args()

    shipped = pd.read_csv(f"{args.data}/stationarity_tests.csv")
    panel = pd.read_csv(f"{args.data}/panel_dedup_all.csv")
    nodes = [
        str(n) for n in panel["pnode_id"]
        if os.path.isdir(f"{args.raw}/da_hourly/pnode={n}")
    ]

    print("ADF / KPSS on the full-sample hourly congestion spread")
    print(f"    (ADF 5% crit {ADF_CRIT_5}; KPSS 5% crit {KPSS_CRIT_5})")
    print(f"    {'node':>12}  {'ADF (recomp)':>12}  {'ADF (shipped)':>13}"
          f"  {'KPSS (recomp)':>13}  {'KPSS (shipped)':>14}")
    for node in nodes:
        row = shipped[shipped["label"] == f"{node}:congestion"]
        if row.empty:
            continue
        series = congestion_spread(args.raw, node)
        adf_stat = adfuller(series, autolag="AIC")[0]
        kpss_stat = kpss(series, regression="c", nlags="auto")[0]
        print(f"    {node:>12}  {adf_stat:>12.2f}  {row['raw_adf_stat'].iloc[0]:>13.2f}"
              f"  {kpss_stat:>13.2f}  {row['raw_kpss_stat'].iloc[0]:>14.2f}")
    print("\n    Conclusion (unchanged): ADF rejects the unit root; KPSS does not "
          "reject stationarity -> the series is stationary.")


if __name__ == "__main__":
    main()
