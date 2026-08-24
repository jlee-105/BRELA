"""SCIP on the MODERATE temporal-dilemma curriculum, across all 12
tiered-benchmark scales -- mirrors eval_scip_benchmark.py (same TIME_LIMIT,
same result-CSV schema) but generates instances via
common/temporal_dilemma_generator_moderate.py (same seed=123 generator used
to validate the RL+Auction hybrid and the 5x5x5 reference) instead of
reading TEST_INSTANCE/*.xlsx. Deferred earlier because it's slow; run
overnight. Long-running: up to TIME_LIMIT seconds per instance, 10 instances
per config, 12 configs.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from opt.SCIP import solve_wta_scip
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance

RESULT_DIR = "./result"
N_EVAL = 10
SEED = 123
TIME_LIMIT = 600  # 10 min per instance (matches eval_scip_benchmark.py's standard-benchmark cap)

os.makedirs(RESULT_DIR, exist_ok=True)
PROGRESS_LOG = os.path.join(RESULT_DIR, "scip_moderate_benchmark_progress.log")


def _log(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(msg + "\n")


def _append_row(csv_path, row):
    header = not os.path.exists(csv_path)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=header, index=False)


def run_config(M, N, T):
    csv_path = os.path.join(RESULT_DIR, f"scip_moderate_{M}M_{N}N_{T}T_{TIME_LIMIT}s.csv")
    rng = np.random.default_rng(SEED)  # same seed/order as common.rl4co_eval's moderate loader

    rows = []
    for i in range(N_EVAL):
        V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)

        _log(f"[{M}M_{N}N_{T}T] instance {i}: solving (time_limit={TIME_LIMIT}s)...")
        start = time.time()
        obj_value, _, _, status, gap = solve_wta_scip(
            M, N, T, V, np.asarray(P), A=AMM, W=PREP, tw=TW, Time_Limit=TIME_LIMIT
        )
        elapsed = time.time() - start

        obj_norm = obj_value / sum(V) if obj_value is not None else None
        result = {
            "instance": i,
            "objective_raw": obj_value,
            "objective_norm": obj_norm,
            "gap": gap,
            "status": str(status),
            "solve_time": elapsed,
        }
        rows.append(result)
        _append_row(csv_path, result)
        _log(f"[{M}M_{N}N_{T}T] instance {i}: status={status}, gap={gap:.4f}, "
             f"obj_norm={obj_norm if obj_norm is None else round(obj_norm, 4)}, time={elapsed:.1f}s")

    solved = [r for r in rows if r["objective_norm"] is not None]
    n_optimal = sum(1 for r in rows if r["status"] == "optimal")
    obj_arr = np.array([r["objective_norm"] for r in solved])
    gap_arr = np.array([r["gap"] for r in rows])

    _log("=" * 60)
    _log(f"[{M}M_{N}N_{T}T] SUMMARY: objective_norm mean={obj_arr.mean():.4f}" if len(obj_arr) else "  no solved instances")
    _log(f"  gap: mean={gap_arr.mean():.4f} max={gap_arr.max():.4f}")
    _log(f"  optimal: {n_optimal}/{len(rows)}")
    _log("=" * 60)
    return rows


ALL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_index', type=int, default=None,
                         help='run only ALL_CONFIGS[config_index] (for one-config-per-process '
                              'invocation, so each config gets its own completion notification); '
                              'omit to run all 12 sequentially in one process')
    args = parser.parse_args()

    configs = ALL_CONFIGS if args.config_index is None else [ALL_CONFIGS[args.config_index]]
    for M, N, T in configs:
        _log(f"\n{'='*20} Starting {M}M_{N}N_{T}T (moderate) {'='*20}")
        run_config(M, N, T)
        _log(f"\n{'='*20} DONE {M}M_{N}N_{T}T (moderate) {'='*20}")
    if args.config_index is None:
        _log("\nALL MODERATE CONFIGS DONE")
