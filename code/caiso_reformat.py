#!/usr/bin/env python3
"""Reformat raw CAISO OASIS data into the pipeline's PJM-style layout.


Purpose
    baseline_forecasters.py expects, per node, hourly day-ahead and real-time
    files with total_lmp_* and congestion_price_* columns (PJM layout). This
    script converts the raw CAISO DAM (hourly) and RTPD/FMM (15-minute) files
    for the eight both-legs nodes into that layout. Real-time is averaged from
    the four FMM intervals to the hour.

Inputs
    --caiso_raw DIR/dam/*.csv , DIR/rtpd/*.csv  (columns: node, INTERVALSTARTTIME_GMT,
        total, congestion, energy, loss, ghg)

Outputs
    --out DIR/{da_hourly,rt_hourly}/pnode=<id>/all.csv  (datetime_beginning_utc,
        total_lmp_da/rt, congestion_price_da/rt) and a panel CSV (--panel_out).

Usage
    python3 caiso_reformat.py --caiso_raw ./caiso_data --out ./caiso_reformatted \
        --panel_out ./caiso_panel.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

# The eight both-legs nodes (material DA and RT congestion), with synthetic
# integer ids.
NODES = {
    "POD_CONTRL_1_POOLE-APND": 90000001, "CALGEN_1_UNITS-APND": 90000002,
    "DRUM_7_PL1X2-APND": 90000003, "SPAULD_6_UNIT12-APND": 90000004,
    "SUMMIT_ASR-APND": 90000005, "PGCC_1_PDRP34-APND": 90000006,
    "ELKHRN_1_EESX3-APND": 90000007, "MOSSLD_2_PSP1-APND": 90000008,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caiso_raw", required=True, help="raw CAISO dir (dam/, rtpd/)")
    parser.add_argument("--out", required=True, help="output pipeline-layout dir")
    parser.add_argument("--panel_out", default="caiso_panel.csv", help="panel CSV path")
    args = parser.parse_args()

    dam = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{args.caiso_raw}/dam/*.csv"))],
                    ignore_index=True)
    rt = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{args.caiso_raw}/rtpd/*.csv"))],
                   ignore_index=True)
    dam = dam[dam["node"].isin(NODES)]
    rt = rt[rt["node"].isin(NODES)]
    dam["hour"] = pd.to_datetime(dam["INTERVALSTARTTIME_GMT"], utc=True)
    rt["hour"] = pd.to_datetime(rt["INTERVALSTARTTIME_GMT"], utc=True).dt.floor("h")

    da = dam.groupby(["node", "hour"]).agg(
        total_lmp_da=("total", "mean"), congestion_price_da=("congestion", "mean")).reset_index()
    rtp = rt.groupby(["node", "hour"]).agg(
        total_lmp_rt=("total", "mean"), congestion_price_rt=("congestion", "mean")).reset_index()

    for name, pid in NODES.items():
        for leg, frame, cols in [
            ("da_hourly", da, ["total_lmp_da", "congestion_price_da"]),
            ("rt_hourly", rtp, ["total_lmp_rt", "congestion_price_rt"]),
        ]:
            out = frame[frame["node"] == name][["hour"] + cols].rename(
                columns={"hour": "datetime_beginning_utc"}).sort_values("datetime_beginning_utc")
            node_dir = f"{args.out}/{leg}/pnode={pid}"
            os.makedirs(node_dir, exist_ok=True)
            out.to_csv(f"{node_dir}/all.csv", index=False)

    pd.DataFrame({"pnode_id": list(NODES.values()), "zone": "CAISO",
                  "pnode_name": list(NODES)}).to_csv(args.panel_out, index=False)
    print(f"Reformatted {len(NODES)} CAISO nodes into {args.out}; panel -> {args.panel_out}")


if __name__ == "__main__":
    main()
