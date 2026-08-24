"""
Evaluate the post-Bug-4-fix Parallel checkpoint (seed5, no_reward_to_go,
GNN_TRAIN_ABL_no_reward_to_go_seed5(2), epoch 200) across all 12 tiered-
benchmark configs, and merge with the existing (still-valid, Bug-4-doesn't-
affect-Greedy) Greedy results into one comparison table.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from common.DWTA_GNN import create_gnn_actor
from common.TORCH_OBJECTS import DEVICE
from eval_tiered_benchmark import eval_instance_parallel, TIERS, TEST_DIR, N_EVAL

CKPT = "result/GNN_TRAIN_ABL_no_reward_to_go_seed5(2)/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"
GREEDY_CSV = "result/greedy_benchmark_results.csv"
OUT_CSV = "result/parallel_postfix_vs_greedy.csv"

actor = create_gnn_actor().to(DEVICE)
actor.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
actor.eval()

greedy_df = pd.read_csv(GREEDY_CSV).set_index("config")

rows = []
for tier, files in TIERS.items():
    for fname in files:
        config = fname.replace(".xlsx", "")
        df = pd.read_excel(f"{TEST_DIR}/{fname}")
        nw, nt, mt = int(df.iloc[0]["M"]), int(df.iloc[0]["N"]), int(df.iloc[0]["T"])

        objs = []
        t0 = time.time()
        for i in range(min(N_EVAL, len(df))):
            row = df.iloc[i]
            V = json.loads(row["V"]); P = np.array(json.loads(row["P"])); TW = json.loads(row["TW"])
            amm = json.loads(row["AMM"]); prep = json.loads(row["PREP"]); cost = json.loads(row["COST"])
            with torch.no_grad():
                result = eval_instance_parallel(actor, V, P, TW, nw, nt, mt, amm, prep, cost)
            objs.append(result["objective"])
        elapsed = time.time() - t0

        par_mean = float(np.mean(objs))
        greedy_mean = float(greedy_df.loc[config, "objective_mean"])
        gap = par_mean - greedy_mean
        row_out = {
            "tier": tier, "config": config, "M": nw, "N": nt, "T": mt,
            "greedy_objective": greedy_mean,
            "parallel_postfix_objective": par_mean,
            "gap": gap,
            "winner": "Parallel" if gap < 0 else "Greedy",
        }
        rows.append(row_out)
        print(f"[{tier}] {config}: Greedy={greedy_mean:.4f} Parallel={par_mean:.4f} "
              f"gap={gap:+.4f} ({'Parallel' if gap < 0 else 'Greedy'} ahead) [{elapsed:.1f}s]", flush=True)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to {OUT_CSV}")
print(out_df.to_string(index=False))
