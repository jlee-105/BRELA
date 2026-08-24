import json
import os
import time

import numpy as np
import pandas as pd

from opt.SCIP import solve_wta_scip

TEST_DIR = "./TEST_INSTANCE"
RESULT_DIR = "./result"
N_EVAL = 10  # same instances (rows 0..9) used by eval_tiered_benchmark.py
TIME_LIMIT = 600  # 10 min: generous "time-relaxed" cap for the best-found reference,
                   # beyond which a battlefield decision wouldn't be usable anyway

os.makedirs(RESULT_DIR, exist_ok=True)
PROGRESS_LOG = os.path.join(RESULT_DIR, "scip_benchmark_progress.log")


def _log(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(msg + "\n")


def _append_row(csv_path, row):
    header = not os.path.exists(csv_path)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=header, index=False)


def run_config(fname, M, N, T):
    df = pd.read_excel(f"{TEST_DIR}/{fname}")
    csv_path = os.path.join(RESULT_DIR, f"scip_{M}M_{N}N_{T}T_{TIME_LIMIT}s.csv")

    rows = []
    for i in range(min(N_EVAL, len(df))):
        row = df.iloc[i]
        V = json.loads(row["V"])
        P = np.array(json.loads(row["P"]))
        TW = json.loads(row["TW"])
        amm = json.loads(row["AMM"])
        prep = json.loads(row["PREP"])

        _log(f"[{fname}] instance {i}: solving (time_limit={TIME_LIMIT}s)...")
        start = time.time()
        obj_value, _, _, status, gap = solve_wta_scip(
            M, N, T, V, P, A=amm, W=prep, tw=TW, Time_Limit=TIME_LIMIT
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
        _log(f"[{fname}] instance {i}: status={status}, gap={gap:.4f}, "
             f"obj_norm={obj_norm if obj_norm is None else round(obj_norm, 4)}, time={elapsed:.1f}s")

    solved = [r for r in rows if r["objective_norm"] is not None]
    n_optimal = sum(1 for r in rows if r["status"] == "optimal")
    n_timelimit = sum(1 for r in rows if r["status"] == "timelimit")
    n_infeasible = sum(1 for r in rows if r["status"] == "infeasible")

    obj_arr = np.array([r["objective_norm"] for r in solved])
    gap_arr = np.array([r["gap"] for r in rows])

    _log("=" * 60)
    _log(f"[{fname}] SUMMARY over {len(rows)} instances (time_limit={TIME_LIMIT}s)")
    _log(f"  objective_norm: mean={obj_arr.mean():.4f} std={obj_arr.std():.4f}" if len(obj_arr) else "  no solved instances")
    _log(f"  gap: mean={gap_arr.mean():.4f} max={gap_arr.max():.4f}")
    _log(f"  optimal: {n_optimal}/{len(rows)}  timelimit: {n_timelimit}/{len(rows)}  infeasible: {n_infeasible}/{len(rows)}")
    _log("=" * 60)
    return rows


ALL_CONFIGS = [
    # already done: ("5M_5N_5T.xlsx", 5, 5, 5)
    ("5M_7N_5T.xlsx", 5, 7, 5),
    ("10M_15N_5T.xlsx", 10, 15, 5),
    ("15M_15N_5T.xlsx", 15, 15, 5),
    ("15M_20N_5T.xlsx", 15, 20, 5),
    ("20M_30N_5T.xlsx", 20, 30, 5),
    ("30M_30N_10T.xlsx", 30, 30, 10),
    ("30M_40N_10T.xlsx", 30, 40, 10),
    ("40M_50N_10T.xlsx", 40, 50, 10),
    ("50M_50N_15T.xlsx", 50, 50, 15),
    ("50M_70N_15T.xlsx", 50, 70, 15),
    ("70M_100N_15T.xlsx", 70, 100, 15),
]

if __name__ == "__main__":
    for fname, M, N, T in ALL_CONFIGS:
        _log(f"\n{'='*20} Starting {fname} ({M}x{N}x{T}) {'='*20}")
        run_config(fname, M=M, N=N, T=T)
    _log("\nALL CONFIGS DONE")
