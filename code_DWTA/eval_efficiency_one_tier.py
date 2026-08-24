"""Per-tier subprocess (avoids the mysterious slow-accumulation issue found
earlier tonight when eval_instance_* is looped many times in one process)."""
import json
import sys

import numpy as np
import pandas as pd
import torch

from common.DWTA_GNN import create_gnn_actor
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.TORCH_OBJECTS import DEVICE
from eval_tiered_benchmark import eval_instance_parallel, eval_instance_sequential, TIERS, TEST_DIR, N_EVAL

PAR_CKPT = "result/GNN_TRAIN_ABL_no_reward_to_go_seed5(2)/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"
SEQ_CKPT = "result/GNN_TRAIN_SEQ_ABL_no_reward_to_go_seed5(4)/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"

tier = sys.argv[1]
files = TIERS[tier]

par_actor = create_gnn_actor().to(DEVICE)
par_actor.load_state_dict(torch.load(PAR_CKPT, map_location=DEVICE, weights_only=False))
par_actor.eval()

seq_actor = create_gnn_actor_sequential().to(DEVICE)
seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
seq_actor.eval()

par_times, seq_times = [], []
for fname in files:
    df = pd.read_excel(f"{TEST_DIR}/{fname}")
    nw, nt, mt = int(df.iloc[0]["M"]), int(df.iloc[0]["N"]), int(df.iloc[0]["T"])
    for i in range(min(N_EVAL, len(df))):
        row = df.iloc[i]
        V = json.loads(row["V"]); P = np.array(json.loads(row["P"])); TW = json.loads(row["TW"])
        amm = json.loads(row["AMM"]); prep = json.loads(row["PREP"]); cost = json.loads(row["COST"])
        with torch.no_grad():
            par_times.append(eval_instance_parallel(par_actor, V, P, TW, nw, nt, mt, amm, prep, cost)["time_s"])
            seq_times.append(eval_instance_sequential(seq_actor, V, P, TW, nw, nt, mt, amm, prep, cost)["time_s"])

par_mean, seq_mean = float(np.mean(par_times)), float(np.mean(seq_times))
print(f"TIER_RESULT {tier} {seq_mean:.6f} {par_mean:.6f} {seq_mean/par_mean:.2f}", flush=True)
