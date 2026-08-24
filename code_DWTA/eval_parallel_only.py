"""One-off: re-evaluate SCoPE (parallel decoder) only, using the GNN_TRAIN_ABL_no_critic_seed*
checkpoints -- the ones actually promoted to canonical "SCoPE" per the 2026-08-05 decision
(critic+shaping hurt held-out generalization; see Section V-C / Prop discussion). The
eval_tiered_benchmark.py SEED_DIRS mapping still points at the OLDER GNN_TRAIN_20260803/
GNN_TRAIN_seed1-4 checkpoints (pre-ablation-decision, WITH critic) -- that mismatch is why
the first parallel-only rerun matched the old numbers exactly and still lost to Greedy.
Sequential is skipped here (not needed for this comparison)."""
import json
import time

import numpy as np
import pandas as pd
import torch

from common.DWTA_GNN import create_gnn_actor
from common.TORCH_OBJECTS import DEVICE
from eval_tiered_benchmark import N_EVAL, TEST_DIR, TIERS, eval_instance_parallel

RESULTS_CSV = "result/tiered_benchmark_results_parallel_nocritic.csv"
PROGRESS_LOG = "result/parallel_nocritic_progress.log"

NOCRITIC_DIRS = {i: f"GNN_TRAIN_ABL_no_critic_seed{i}" for i in range(6)}


def load_nocritic_actor(seed):
    path = f"result/{NOCRITIC_DIRS[seed]}/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"
    actor = create_gnn_actor().to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    actor.eval()
    return actor


def _log(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(msg + "\n")


def _append_row(row_out, wrote_header):
    pd.DataFrame([row_out]).to_csv(RESULTS_CSV, mode="a", header=not wrote_header, index=False)


def run():
    import os
    open(PROGRESS_LOG, "w").close()
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
    wrote_header = False

    total_jobs = 1 * sum(len(files) for files in TIERS.values())
    job_i = 0

    for seed in [5]:
        _log(f"=== parallel (no_critic) seed {seed} ===")
        actor = load_nocritic_actor(seed)

        for tier, files in TIERS.items():
            for fname in files:
                job_i += 1
                df = pd.read_excel(f"{TEST_DIR}/{fname}")
                nw, nt, mt = int(df.iloc[0]["M"]), int(df.iloc[0]["N"]), int(df.iloc[0]["T"])

                metrics = []
                config_t0 = time.time()
                for i in range(min(N_EVAL, len(df))):
                    row = df.iloc[i]
                    V = json.loads(row["V"])
                    P = np.array(json.loads(row["P"]))
                    TW = json.loads(row["TW"])
                    amm = json.loads(row["AMM"])
                    prep = json.loads(row["PREP"])
                    cost = json.loads(row["COST"])
                    result = eval_instance_parallel(actor, V, P, TW, nw, nt, mt, amm, prep, cost)
                    metrics.append(result)

                obj = np.array([m["objective"] for m in metrics])
                destr = np.array([m["destruction"] for m in metrics])
                fire = np.array([m["fire_rate"] for m in metrics])
                tsec = np.array([m["time_s"] for m in metrics])
                disp = np.array([m["dispersion"] for m in metrics if m["dispersion"] is not None])
                waste = np.array([m["wasteful_redundancy"] for m in metrics if m["wasteful_redundancy"] is not None])

                row_out = {
                    "decoder": "parallel_no_critic", "seed": seed, "tier": tier,
                    "config": fname.replace(".xlsx", ""), "M": nw, "N": nt, "T": mt,
                    "n_instances": len(metrics),
                    "objective_mean": obj.mean(), "objective_std": obj.std(),
                    "destruction_mean": destr.mean(), "destruction_std": destr.std(),
                    "fire_rate_mean": fire.mean(),
                    "time_s_mean": tsec.mean(), "time_s_std": tsec.std(),
                    "dispersion_mean": disp.mean() if len(disp) else None,
                    "wasteful_redundancy_mean": waste.mean() if len(waste) else None,
                }
                _append_row(row_out, wrote_header)
                wrote_header = True
                _log(f"  [{job_i}/{total_jobs}] {fname}: obj={obj.mean():.4f}+-{obj.std():.4f} "
                     f"destr={destr.mean():.2%} time={tsec.mean():.3f}s (config took {time.time()-config_t0:.1f}s)")

    _log(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    run()
