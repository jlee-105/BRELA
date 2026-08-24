"""
Evaluate the T3 module-wise ablation checkpoints (no_critic, no_shaping,
no_reward_to_go; 5 seeds each) against the full-model parallel baseline,
across the same 4-tier x 3-config held-out benchmark used in
eval_tiered_benchmark.py.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN import create_gnn_actor
import eval_tiered_benchmark as E

N_EVAL = E.N_EVAL
TEST_DIR = E.TEST_DIR
TIERS = E.TIERS

ABLATIONS = ["no_critic", "no_shaping", "no_reward_to_go"]
RESULTS_CSV = "result/ablation_benchmark_results.csv"
PROGRESS_LOG = "result/ablation_benchmark_progress.log"


def _log(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(msg + "\n")


def load_ablation_actor(ablation, seed):
    path = f"result/GNN_TRAIN_ABL_{ablation}_seed{seed}/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"
    actor = create_gnn_actor().to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    actor.eval()
    return actor


def run():
    import os
    open(PROGRESS_LOG, "w").close()
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
    wrote_header = False

    total_jobs = len(ABLATIONS) * 5 * sum(len(files) for files in TIERS.values())
    job_i = 0

    for ablation in ABLATIONS:
        for seed in range(5):
            _log(f"=== ablation={ablation} seed={seed} ===")
            actor = load_ablation_actor(ablation, seed)

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
                        result = E.eval_instance_parallel(actor, V, P, TW, nw, nt, mt, amm, prep, cost)
                        metrics.append(result)

                    obj = np.array([m["objective"] for m in metrics])
                    destr = np.array([m["destruction"] for m in metrics])
                    disp = np.array([m["dispersion"] for m in metrics if m["dispersion"] is not None])
                    waste = np.array([m["wasteful_redundancy"] for m in metrics if m["wasteful_redundancy"] is not None])

                    row_out = {
                        "ablation": ablation, "seed": seed, "tier": tier,
                        "config": fname.replace(".xlsx", ""), "M": nw, "N": nt, "T": mt,
                        "n_instances": len(metrics),
                        "objective_mean": obj.mean(), "objective_std": obj.std(),
                        "destruction_mean": destr.mean(), "destruction_std": destr.std(),
                        "dispersion_mean": disp.mean() if len(disp) else None,
                        "wasteful_redundancy_mean": waste.mean() if len(waste) else None,
                    }
                    pd.DataFrame([row_out]).to_csv(RESULTS_CSV, mode="a", header=not wrote_header, index=False)
                    wrote_header = True
                    _log(f"  [{job_i}/{total_jobs}] {ablation}/seed{seed}/{fname}: "
                         f"obj={obj.mean():.4f}+-{obj.std():.4f} destr={destr.mean():.2%} "
                         f"(config took {time.time()-config_t0:.1f}s)")

    _log(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    run()
