#!/bin/bash
cd "/c/Users/ongs6/vs_code/DWTA/after_review_ieee_access/code_DWTA"
configs="5M_5N_5T.xlsx 5M_7N_5T.xlsx 10M_15N_5T.xlsx 15M_15N_5T.xlsx 15M_20N_5T.xlsx 20M_30N_5T.xlsx 30M_30N_10T.xlsx 30M_40N_10T.xlsx 40M_50N_10T.xlsx 50M_50N_15T.xlsx 50M_70N_15T.xlsx 70M_100N_15T.xlsx"
> result/parallel_postfix_raw.txt
for c in $configs; do
    "../venv/Scripts/python.exe" eval_parallel_one_config.py "$c" 2>>result/parallel_postfix_errors.log | tee -a result/parallel_postfix_raw.txt
done
echo "ALL DONE"
