"""


This file builds the panel built through the validated forecasting paradigm
(baseline_forecasters.py), the temporal split, and the
shared prediction schema every model writes.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

# --- paths: package-relative defaults; override via environment variables ------
_CODE = os.path.dirname(os.path.abspath(__file__))   
_ROOT = os.path.dirname(_CODE)                        
HARNESS_DIR = os.environ.get("PAPERA_HARNESS", _CODE)


def _find_data_dir():

    if os.environ.get("PAPERA_DATA"):
        return os.environ["PAPERA_DATA"]
    for cand in (os.path.join(_ROOT, "pjm_raw"),
                 os.path.join(_ROOT, "data", "data"),
                 os.path.join(_ROOT, "data"),
                 _ROOT):
        if os.path.isdir(os.path.join(cand, "da_hourly")):
            return cand
    return os.path.join(_ROOT, "pjm_raw")   # default: place the raw PJM LMP here


DATA_DIR = _find_data_dir()
PANEL_CSV = os.environ.get("PAPERA_PANEL", os.path.join(_ROOT, "data", "panel_dedup_all.csv"))
PREDS_DIR = os.environ.get("PAPERA_PREDS", os.path.join(_ROOT, "preds"))

# --- protocol constants ------------------------
TRAIN_END = "2025-12-31"      # final-fit boundary; OOS = Jan--Jun 2026
MIN_LAG_DAYS = 2             
COMPONENT = "congestion"
SEED = 20260725

sys.path.insert(0, HARNESS_DIR)
import baseline_forecasters as bf  

FEATURES = list(bf.FEATURES)       
KEY = bf.KEY


def build_s0_panel(data_dir=DATA_DIR, panel_csv=PANEL_CSV):
    """Full panel with features + target, honest lag. Reused by every model."""
    return bf.build_panel(data_dir, panel_csv, COMPONENT,
                          min_lag_days=MIN_LAG_DAYS, violate_lag=False)


def split(panel, train_end=TRAIN_END):
    """Return (train, oos) by date. The OOS set is scored once."""
    d = panel.copy()
    d["date"] = pd.DatetimeIndex(d[KEY]).tz_convert("UTC").normalize()
    cut = pd.Timestamp(train_end, tz="UTC").normalize()
    return d[d["date"] <= cut].copy(), d[d["date"] > cut].copy()


# --- shared prediction schema -------------------------------------------------
SCHEMA = ["datetime_beginning_utc", "pnode_id", "zone", "dart", "y", "p"]


def preds_dir_for(feature_set):
    """Output directory by feature set (S0 for Paper A)."""
    return os.path.join(_ROOT, "preds" if feature_set == "S0" else "preds_" + str(feature_set))


def write_predictions(oos, model_name, p, yhat=None, preds_dir=None):
    """Write one model's OOS predictions in the shared schema.

    oos: OOS frame carrying KEY, pnode_id, zone, dart, y;  p: P(dart>0).
    """
    preds_dir = preds_dir or PREDS_DIR
    os.makedirs(preds_dir, exist_ok=True)
    out = pd.DataFrame({
        "datetime_beginning_utc": pd.DatetimeIndex(oos[KEY]).tz_convert("UTC"),
        "pnode_id": oos["pnode_id"].values,
        "zone": oos["zone"].values,
        "dart": oos["dart"].values,
        "y": oos["y"].astype(int).values,
        "p": pd.Series(p).clip(1e-6, 1 - 1e-6).values,
    })
    if yhat is not None:
        out["yhat"] = pd.Series(yhat).values
    path = os.path.join(preds_dir, f"{model_name}.csv")
    out.to_csv(path, index=False)
    return path


def load_all_predictions(models=None, preds_dir=None):
    """Merge per-model prediction files into one wide-p frame for evaluation."""
    preds_dir = preds_dir or PREDS_DIR
    files = [f for f in os.listdir(preds_dir) if f.endswith(".csv")]
    if models is not None:
        files = [f"{m}.csv" for m in models if f"{m}.csv" in files]
    merged = None
    for f in sorted(files):
        m = f[:-4]
        d = pd.read_csv(os.path.join(preds_dir, f))
        keep = d[["datetime_beginning_utc", "pnode_id", "zone", "dart", "y"]]
        pcol = d[["p"]].rename(columns={"p": f"p_{m}"})
        if "yhat" in d.columns:
            pcol[f"yhat_{m}"] = d["yhat"]
        piece = pd.concat([keep.reset_index(drop=True), pcol.reset_index(drop=True)], axis=1)
        if merged is None:
            merged = piece
        else:
            merged = merged.merge(
                piece[["datetime_beginning_utc", "pnode_id"]
                      + [c for c in piece.columns if c.startswith(("p_", "yhat_"))]],
                on=["datetime_beginning_utc", "pnode_id"], how="inner")
    return merged
