# How to reproduce every number in the paper

This manual gives you the steps to replicate all the 7 tables in the paper.

Everything is deterministic: fixed seed `20260725`, 5-day moving-block bootstrap,
B = 2000. 

---

## 0. Requirements 
```bash
pip install -r requirements.txt
export PYTHONPATH=code                       # Windows PowerShell: $env:PYTHONPATH="code"
python code/verify_all_tables.py --data data # reproduces Tables 1, 3-6
```
That alone reproduces 5 of the 7 tables from the prediction files already committed
to the repo. Steps 5–6 add the two tables that need the Zenodo raw prices of both the PJM and CAISO LMP price spread prices data.

---

## 1. Prerequisites
- **Python 3.10+** and **git**. Check: `python --version` and `git --version`.
- ~2 GB free disk for the raw price data once downloaded and unzipped.
- macOS/Linux: a normal shell. Windows: use **WSL** or **Git Bash** to run
  `run_all.sh`; or run the explicit `python ...` commands in Step 6, which work in
  any shell including PowerShell.

## 2. Get the code
```bash
git clone https://github.com/ronitdas2014-cmyk/dart-evaluation-protocol.git
cd dart-evaluation-protocol
```

## 3. Set up a clean environment
```bash
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
`requirements.txt` fixes numpy, pandas, scipy, scikit-learn, and statsmodels. The fixing-step matters: the gradient-boosted rows shift by ~$0.2/MWh across scikit-learn versions.

## 4. Verify the 5 tables

The repo already ships the forecaster prediction files these tables are built from, so data downloading of prices is needed from Zenodo.
```bash
export PYTHONPATH=code          # Windows PowerShell: $env:PYTHONPATH="code"
python code/verify_all_tables.py --data data
```
It prints a PASS/verdict line for **Table 1, Table 3, Table 4, Table 5,
Table 6**, comparing the recomputed value to the values published in the paper for every cell.

Two more that also need no raw data:
```bash
python code/reality_check_spa_inc.py --data data --prefix rc_spa   # Reality Check / SPA 
python code/dominion_case.py         --data data --out dominion_case.csv  # Dominion running case
```

## 5. Get the raw price data from Zenodo
Download the two raw archives from the Zenodo record for this paper:

>  **Zenodo record (DOI): https://doi.org/10.5281/zenodo.22866985**
>  *"DA – RT Price Spread Evaluation Protocol", CC-BY-4.0, published 2026-09-21.*
>  - PJM raw LMP archive     →  `pjm_raw.zip`   (23.6 MB)
>  - CAISO raw OASIS archive →  `caiso_data.zip` (30.9 MB)
>  - `replication-code-and-data.zip` (29.7 MB) is a full code+data mirror of this
>    repo — you do **not** need it if you cloned from GitHub.

Download from the web page ("Download" beside each file) or on the command line:
```bash
curl -L -o pjm_raw.zip    "https://zenodo.org/records/22866985/files/pjm_raw.zip?download=1"
curl -L -o caiso_data.zip "https://zenodo.org/records/22866985/files/caiso_data.zip?download=1"
unzip pjm_raw.zip
unzip caiso_data.zip
```

Unzip them **into the repo root** so you end up with exactly this layout (the folder
names `pjm_raw` and `caiso_data` are what the commands below expect — rename if your
archives unzip to something else):

```
dart-evaluation-protocol/
├── pjm_raw/
│   ├── da_hourly/pnode=<id>/*.csv        # columns incl. total_lmp_da, congestion_price_da
│   └── rt_hourly/pnode=<id>/*.csv        # columns incl. total_lmp_rt, congestion_price_rt
└── caiso_data/
    ├── dam/*.csv                         # columns: node, INTERVALSTARTTIME_GMT, total, congestion, ...
    └── rtpd/*.csv
```
Verify the layout before running anything else:
```bash
ls pjm_raw/da_hourly | head      # should list  pnode=... directories
ls caiso_data/dam    | head      # should list  dated .csv files
```
If `ls pjm_raw/da_hourly` does not show `pnode=...` folders, the data is not where
the pipeline looks — fix the folder nesting until it does.

## 6. Reproduce the two full-spread tables (Tables 2 and 7)
Option A — one command (macOS/Linux/WSL/Git Bash):
```bash
bash run_all.sh pjm_raw caiso_data
```
Option B — explicit commands (any OS, run from the repo root):
```bash
export PYTHONPATH=code          # Windows PowerShell: $env:PYTHONPATH="code"

# Table 2  (PJM full DA-RT spread; re-fits the 6 forecasters, then builds the table)
python code/baseline_forecasters.py --data-dir pjm_raw --panel data/panel_dedup_all.csv \
       --component total --train-end 2025-12-31 \
       --models persistence climatology logistic gbm rf mlp --out-dir headline_out
python code/headline_table.py --pred headline_out/predictions_gate_closure_total.csv \
       --data_dir pjm_raw --panel data/panel_dedup_all.csv --train_end 2025-12-31 \
       --market PJM --out tab_headline_PJM.csv

# Table 7  (reformat CAISO OASIS -> pipeline layout, re-fit on congestion, build the table)
python code/caiso_reformat.py --caiso_raw caiso_data --out caiso_reformatted --panel_out caiso_panel.csv
python code/baseline_forecasters.py --data-dir caiso_reformatted --panel caiso_panel.csv \
       --component congestion --train-end 2025-06-30 \
       --models persistence climatology logistic gbm rf mlp --out-dir caiso_cong_out
python code/caiso_table.py --pred caiso_cong_out/predictions_gate_closure_congestion.csv --out tab_caiso.csv

# supplementary in-text numbers that need the PJM raw prices
python code/fee_sensitivity.py            --data data
python code/tail_exponents.py      --data data --raw pjm_raw
python code/stationarity_tests_verify.py --data data --raw pjm_raw
```
Expected time needed to complete the tests: the two `baseline_forecasters.py` runs each fit SARIMA / Kalman
/ Markov and take a few minutes; the whole of Step 6 is roughly 10–30 minutes on a desktop.

## 7. Which script produces which paper object
| Paper object | Command | Needs raw data? |
|---|---|---|
| Table 1, Table 3, Table 4, Table 5, Table 6 | `verify_all_tables.py --data data` | No (uses committed predictions) |
| Table 2 (PJM & CAISO) | `baseline_forecasters.py --component total` + `headline_table.py` | Yes (PJM/CAISO raw) |
| Table 7 | `caiso_reformat.py` + `baseline_forecasters.py --component congestion` + `caiso_table.py` | Yes (CAISO raw) |
| Reality Check / SPA vs INC | `reality_check_spa_inc.py --data data` | No |
| Dominion running case | `dominion_case.py --data data` | No |
| transaction-cost sensitivity | `fee_sensitivity.py --data data` | No |
| Tail index (Hill) | `tail_exponents.py --data data --raw pjm_raw` | Yes (PJM raw) |
| Stationarity (ADF/KPSS) | `stationarity_tests_verify.py --data data --raw pjm_raw` | Yes (PJM raw) |
| In-text: tail concentration, pooled→zone collapse, ECE, PT, block-SE, n_eff | `verify_intext_numbers.py --data data [--pjm_raw pjm_raw]` | Partly |

## 8. What "Reproducibility" stands for in this case?
- **Deterministic columns reproduce exactly**: accuracy, PT, Brier, payoff points,
  panel composition, verdicts, and the always-long premium (+$2.58 PJM, +$2.75 CAISO).
- **Gradient-boosted rows** (Histogram-based Gradient Boosted Classifier, and the signed-spread regressor) can differ by
  ~$0.2/MWh across scikit-learn versions; the **conclusion is invariant** (no
  forecaster beats passive INC; every paired CI spans zero). Fix scikit-learn to the
  version in `requirements.txt` for digit-level precision.

## 9. Troubleshooting
- `ModuleNotFoundError: baseline_forecasters` → you didn't set `PYTHONPATH=code`
  (or run the script from inside `code/`).
- `FileNotFoundError ... da_hourly` → `--data-dir` does not contain `da_hourly/` +
  `rt_hourly/`; re-check the Step-5 layout with `ls`.
- SARIMA/Kalman/Markov or the stationarity script errors on import → `statsmodels`
  is not installed; re-run `pip install -r requirements.txt`.
- Windows: `bash run_all.sh` won't run in PowerShell — use WSL/Git Bash, or run the
  explicit commands in Step 6 Option B.
- Numbers off to a very small degree on gbm/signed rows → see Step 8; not an error.
