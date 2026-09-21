# DA - RT Price Spread Evaluation Protocol

Manuscript on directional forecasting of PJM day-ahead / real-time congestion
spreads. Every table and headline result maps
below to a named dataset and a runnable script. All scripts are deterministic
(seed 20260725; 5-day moving-block bootstrap; B=2000; `baseline_forecasters.py`
uses `--seed 0`), self-contained, and PEP 8. See `requirements.txt` for pinned
versions.

## 1. Core DA and RT LMP Datasets
### Raw (large; provide separately, not bundled)
| name | contents | source |
|---|---|---|
| `pjm_raw/` | `da_hourly/pnode=<id>/*.csv`, `rt_hourly/pnode=<id>/*.csv` with `total_lmp_da/rt`, `congestion_price_da/rt` | PJM Data Miner 2 |
| `caiso_data/` | `dam/*.csv`, `rtpd/*.csv` with `node, INTERVALSTARTTIME_GMT, total, congestion, energy, loss, ghg` | CAISO OASIS |

### Processed Data (`data/`)
| name | contents |
|---|---|
| `panel_dedup_all.csv` | 18-node PJM evaluation panel (zone, role, cong_std) |
| `caiso_node_characterization.csv` | CAISO DA/RT congestion activity (both-legs selection) |
| `predictions_gate_closure_congestion.csv` | PJM congestion predictions: 9 forecasters + `dart`/`y`/`zone` + features (output of the pipeline; bundled for convenience) |
| `pred_Sbid_9model.csv` | PJM participant-observable (S_bid) predictions |
| `rq1_by_node.csv` | RQ1 tail-index estimates (Hill, CIs, `n_exceed`) |
| `stationarity_tests.csv` | ADF/KPSS statistics |
| `or_rates_rto_2025_26.csv` | PJM daily operating-reserve / virtual-fee rates |

## 2. Prediction Paradigm (single source of every forecaster prediction)
`code/baseline_forecasters.py` (authors' pipeline). Validated: re-running
`--component congestion` reproduces the shipped predictions exactly (persistence,
logistic, gbm, mlp identical; rf corr 0.986).
```
# PJM congestion (drives tab:main/decision/naive/infoset/panel)
python code/baseline_forecasters.py --data-dir pjm_raw --panel data/panel_dedup_all.csv \
       --component congestion --train-end 2025-12-31 --out-dir rq3_out
# PJM full spread (drives tab:headline PJM)
python code/baseline_forecasters.py --data-dir pjm_raw --panel data/panel_dedup_all.csv \
       --component total --train-end 2025-12-31 --out-dir headline_out
# CAISO: reformat, then run the same pipeline
python code/caiso_reformat.py --caiso_raw caiso_data --out caiso_reformatted --panel_out caiso_panel.csv
python code/baseline_forecasters.py --data-dir caiso_reformatted --panel caiso_panel.csv \
       --component total      --train-end 2025-06-30 --out-dir caiso_headline_out   # tab:headline CAISO
python code/baseline_forecasters.py --data-dir caiso_reformatted --panel caiso_panel.csv \
       --component congestion --train-end 2025-06-30 --out-dir caiso_cong_out       # tab:caiso
```

## 3. Table -> datasets -> code -> status
| Table | Datasets | Code (command) | Status |
|---|---|---|---|
| `tab:panel` | `panel_dedup_all.csv`, congestion preds | `verify_all_tables.py --data data` | 
| `tab:main` | congestion preds | `verify_all_tables.py --data data` | 
| `tab:decision` | congestion preds | `verify_all_tables.py --data data` | 
| `tab:naive` | congestion preds | `verify_all_tables.py --data data` | 
| `tab:infoset` | `pred_Sbid_9model.csv` + congestion preds | `verify_all_tables.py --data data` | 
| `tab:headline` | `pjm_raw` + `caiso_data` -> total preds | `baseline_forecasters.py --component total`; then `headline_table.py --pred <total> --market {PJM,CAISO}` | 
| `tab:caiso` | `caiso_data` -> caiso congestion preds | `caiso_reformat.py`; `baseline_forecasters.py --component congestion`; then `caiso_table.py --pred <caiso cong>` | 
| `tab:notation` | none | (definitions only) |

## 4. Headline results -> code
| Result | Code | Reproduced value |
|---|---|---|
| Passive-INC premium (full spread) | `headline_table.py` (passive INC row) | 
| Request 1: paired forecast-minus-INC increments | `request1_incremental_payoff.py --pred <total dir-format>` (or `headline_table.py` increment cols) |
| Request 2: Reality Check / SPA vs INC | `reality_check_spa_inc.py --data data` | 
| Request 3: Dominion running case | `dominion_case.py --data data` | 
| RQ1 tail index (Hill) | `rq1_tail_exponents.py --data data --raw pjm_raw` | 
| Stationarity (ADF/KPSS) | `stationarity_tests_verify.py --data data --raw pjm_raw` | 
| Cost erosion (virtual fee) | `cost_erosion.py --data data` | 
| Tuned-GBM comparator | `gbm_tuned.py` | 
| In-text numbers (tail conc., collapse, ECE, PT, block-SE, n_eff) | `verify_intext_numbers.py --data data [--pjm_raw pjm_raw]` |

## 5. One command
`bash run_all.sh pjm_raw caiso_data`  runs the pipeline and all table/result scripts in order.

