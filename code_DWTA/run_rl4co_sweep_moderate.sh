#!/bin/bash
# Sequential AM + POMO training across all 12 tiered-benchmark scales, on the
# MODERATE temporal-dilemma curriculum. One process at a time (shared GPU).
# SCIP is deferred (slow); Greedy is already computed separately
# (eval_greedy_moderate_all_configs.py). Logs land in result/rl4co_sweep/.
set -e
cd "$(dirname "$0")"
PY="../venv/Scripts/python.exe"
OUT="result/rl4co_sweep"
mkdir -p "$OUT"

run_config() {
    local M=$1 N=$2 T=$3
    local am_bs=$4 am_ep=$5
    local pomo_bs=$6 pomo_ep=$7 pomo_starts=$8

    echo "=== [$(date)] ${M}M_${N}N_${T}T : AM start ===" | tee -a "$OUT/sweep_progress.log"
    "$PY" train_rl4co_am.py --curriculum moderate --num_weapon $M --num_target $N --max_time $T \
        --batch_size $am_bs --train_data_size $((am_bs*40)) --val_data_size $((am_bs*4)) \
        --max_epochs $am_ep --eval_every 5 \
        > "$OUT/am_moderate_${M}M_${N}N_${T}T.log" 2>&1 || echo "AM FAILED for ${M}M_${N}N_${T}T" | tee -a "$OUT/sweep_progress.log"
    echo "=== [$(date)] ${M}M_${N}N_${T}T : AM done ===" | tee -a "$OUT/sweep_progress.log"

    echo "=== [$(date)] ${M}M_${N}N_${T}T : POMO start (num_starts=$pomo_starts) ===" | tee -a "$OUT/sweep_progress.log"
    starts_arg=""
    if [ "$pomo_starts" != "none" ]; then
        starts_arg="--num_starts $pomo_starts"
    fi
    "$PY" train_rl4co_pomo.py --curriculum moderate --num_weapon $M --num_target $N --max_time $T \
        --batch_size $pomo_bs --train_data_size $((pomo_bs*40)) --val_data_size $((pomo_bs*4)) \
        --max_epochs $pomo_ep --eval_every 5 $starts_arg \
        > "$OUT/pomo_moderate_${M}M_${N}N_${T}T.log" 2>&1 || echo "POMO FAILED for ${M}M_${N}N_${T}T" | tee -a "$OUT/sweep_progress.log"
    echo "=== [$(date)] ${M}M_${N}N_${T}T : POMO done ===" | tee -a "$OUT/sweep_progress.log"
}

# Small
run_config 5 5 5     256 100   64 100 none
run_config 5 7 5     256 100   64 100 none
run_config 10 15 5   256 100   64 100 none
# Medium
run_config 15 15 5   128 80    32 80  16
run_config 15 20 5   128 80    32 80  16
run_config 20 30 5   128 80    32 80  16
# Large
run_config 30 30 10  64  60    16 60  16
run_config 30 40 10  64  60    16 60  16
run_config 40 50 10  64  60    16 60  16
# Battlefield
run_config 50 50 15  32  40    8  40  10
run_config 50 70 15  32  40    8  40  10
run_config 70 100 15 32  40    8  40  10

echo "=== [$(date)] SWEEP COMPLETE ===" | tee -a "$OUT/sweep_progress.log"
