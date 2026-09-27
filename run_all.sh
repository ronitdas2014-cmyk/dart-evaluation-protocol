#!/usr/bin/env bash
# Usage: bash run_all.sh PJM_RAW_DIR CAISO_RAW_DIR
set -e
PJM="${1:?pass pjm_raw dir}"; CAISO="${2:?pass caiso_data dir}"; export PYTHONPATH=code
echo "== mechanism tables (congestion preds bundled) =="
python3 code/verify_all_tables.py --data data
echo "== Requests 2 and 3 =="
python3 code/reality_check_spa_inc.py --data data --prefix rc_spa
python3 code/dominion_case.py --data data --out dominion_case.csv
echo "== supplementary (need raw) =="
python3 code/fee_sensitivity.py --data data
python3 code/rq1_tail_exponents.py --data data --raw "$PJM"
python3 code/stationarity_tests_verify.py --data data --raw "$PJM"
echo "== full-spread pipeline + headline/caiso tables =="
python3 code/baseline_forecasters.py --data-dir "$PJM" --panel data/panel_dedup_all.csv --component total --train-end 2025-12-31 --models persistence climatology logistic gbm rf mlp --out-dir headline_out
python3 code/headline_table.py --pred headline_out/predictions_gate_closure_total.csv --data_dir "$PJM" --panel data/panel_dedup_all.csv --train_end 2025-12-31 --market PJM --out tab_headline_PJM.csv
python3 code/caiso_reformat.py --caiso_raw "$CAISO" --out caiso_reformatted --panel_out caiso_panel.csv
python3 code/baseline_forecasters.py --data-dir caiso_reformatted --panel caiso_panel.csv --component congestion --train-end 2025-06-30 --models persistence climatology logistic gbm rf mlp --out-dir caiso_cong_out
python3 code/caiso_table.py --pred caiso_cong_out/predictions_gate_closure_congestion.csv --out tab_caiso.csv
echo "DONE."
