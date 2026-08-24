"""Evaluate auction WITH must_fire (respecting the trained policy's own
fire/hold decision) on ONE config, fresh subprocess per config."""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.DWTA_GNN import create_gnn_actor
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from eval_tiered_benchmark import patch_globals, TEST_DIR, N_EVAL

CKPT = "result/GNN_TRAIN_ABL_no_reward_to_go_seed5/CheckPoint_epoch00190/GNN_ACTOR_state_dic.pt"
actor = create_gnn_actor().to(DEVICE)
actor.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
actor.eval()


@torch.no_grad()
def eval_instance_auction_mustfire(V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :nw, :].argmax(dim=-1)
        must_fire = policy_choice < nt
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


fname = sys.argv[1]
df = pd.read_excel(f"{TEST_DIR}/{fname}")
nw, nt, mt = int(df.iloc[0]["M"]), int(df.iloc[0]["N"]), int(df.iloc[0]["T"])

objs = []
t0 = time.time()
for i in range(min(N_EVAL, len(df))):
    row = df.iloc[i]
    V = json.loads(row["V"]); P = np.array(json.loads(row["P"])); TW = json.loads(row["TW"])
    amm = json.loads(row["AMM"]); prep = json.loads(row["PREP"]); cost = json.loads(row["COST"])
    objs.append(eval_instance_auction_mustfire(V, P, TW, nw, nt, mt, amm, prep, cost))
elapsed = time.time() - t0

mean_obj = float(np.mean(objs))
print(f"RESULT {fname} {mean_obj:.6f} {elapsed:.2f}", flush=True)
