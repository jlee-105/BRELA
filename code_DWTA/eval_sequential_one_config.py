"""Evaluate a Sequential checkpoint on ONE config, fresh subprocess per config
(mirrors eval_parallel_one_config.py's subprocess-isolation workaround)."""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.TORCH_OBJECTS import DEVICE
from eval_tiered_benchmark import eval_instance_sequential, TEST_DIR, N_EVAL

CKPT = sys.argv[2] if len(sys.argv) > 2 else "result/GNN_TRAIN_SEQ_ABL_no_reward_to_go_seed5/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"

fname = sys.argv[1]
actor = create_gnn_actor_sequential().to(DEVICE)
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
        result = eval_instance_sequential(actor, V, P, TW, nw, nt, mt, amm, prep, cost)
    objs.append(result["objective"])
elapsed = time.time() - t0

seq_mean = float(np.mean(objs))
print(f"RESULT {fname} {seq_mean:.6f} {elapsed:.2f}", flush=True)
