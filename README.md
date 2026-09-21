# DA - RT Price Spread Evaluation Protocol

Manuscript on directional forecasting of PJM day-ahead / real-time congestion
spreads. Every table and headline result maps
below to a named dataset and a runnable script. All scripts are deterministic
(seed 20260725; 5-day moving-block bootstrap; B=2000; `baseline_forecasters.py`
uses `--seed 0`), self-contained, and PEP 8. See `requirements.txt` for pinned
versions.



## Execution Command
`bash run_all.sh pjm_raw caiso_data`  runs the pipeline and all table/result scripts in order.

