#!/usr/bin/env python3
"""Net-of-value calculation of the always-long premium (PJM virtual fee).

Purpose
    Replicate and validate the claim that the PJM virtual-transaction fee is
    heavy-tailed (median ~$0.44/MWh, mean ~$1.63/MWh over the evaluation
    window) and that it erodes the full-spread always-long premium.

Inputs
    --data DATA_DIR/or_rates_rto_2025_26.csv
        PJM daily operating-reserve and virtual-fee rates; the column
        `virtual_fee_da_plus_rto_dev_$per_MWh` is the round-trip virtual fee.

Outputs (stdout)
    Fee distribution over the January-June 2026 out-of-sample window and the
    always-long premium net of the mean fee.

Usage
    python3 cost_erosion.py --data ./data

"""
from __future__ import annotations

import argparse

import pandas as pd

# Out-of-sample evaluation window (UTC) and the verified always-long premium
# on the full spread, winsorised at the 99th percentile of |DART| ($/MWh).
OOS_START = "2026-01-01"
OOS_END = "2026-07-01"
PJM_ALWAYS_LONG_WIN99 = 2.58
FEE_COLUMN = "virtual_fee_da_plus_rto_dev_$per_MWh"


def load_fee(data_dir: str) -> pd.Series:
    """Return the daily virtual fee restricted to the OOS window ($/MWh)."""
    rates = pd.read_csv(f"{data_dir}/or_rates_rto_2025_26.csv")
    rates["date"] = pd.to_datetime(rates["operating_reserve_date"])
    in_window = (rates["date"] >= OOS_START) & (rates["date"] < OOS_END)
    return rates.loc[in_window, FEE_COLUMN]


def report(fee: pd.Series) -> None:
    """Print the fee distribution and the net-of-fee always-long premium."""
    print("PJM virtual fee, January-June 2026 out-of-sample window")
    print(f"    n days       : {fee.size}")
    print(f"    median       : ${fee.median():.2f}/MWh   (manuscript $0.44)")
    print(f"    mean         : ${fee.mean():.2f}/MWh   (manuscript $1.63)")
    print(f"    95th pct     : ${fee.quantile(0.95):.2f}/MWh")
    print(f"    maximum      : ${fee.max():.2f}/MWh   (heavy-tailed)")
    net = PJM_ALWAYS_LONG_WIN99 - fee.mean()
    print("\nAlways-long premium net of the mean fee")
    print(f"    gross (win@Q99): +${PJM_ALWAYS_LONG_WIN99:.2f}/MWh")
    print(f"    net            : {net:+.2f}/MWh (eroded; not demonstrable)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data", help="directory with input CSVs")
    args = parser.parse_args()
    report(load_fee(args.data))


if __name__ == "__main__":
    main()
