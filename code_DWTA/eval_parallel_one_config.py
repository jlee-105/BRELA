"""Evaluate the post-fix Parallel checkpoint on ONE config, called as a fresh
subprocess per config to avoid whatever accumulates/slows down across many
repeated Environment/actor calls within a single long-running process."""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.DWTA_GNN import create_gnn_actor
from common.TORCH_OBJECTS import DEVICE
from eval_tiered_benchmark import eval_instance_parallel, TEST_DIR, N_EVAL

CKPT = sys.argv[2] if len(sys.argv) > 2 else "result/GNN_TRAIN_ABL_no_reward_to_go_seed5/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"

fname = sys.argv[1]
actor = create_gnn_actor().to(DEVICE)
actor.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
actor.eval()

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
print(f"RESULT {fname} {par_mean:.6f} {elapsed:.2f}", flush=True)
