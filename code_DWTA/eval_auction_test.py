"""Quick test: does the auction refinement (applied to the Parallel epoch190
checkpoint's own prob/mask tensors each round) help vs. plain greedy-argmax
decoding, on real (20,30,5) instances?"""
import json
import time

import numpy as np
import pandas as pd
import torch

from common.DWTA_GNN import create_gnn_actor
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from eval_tiered_benchmark import patch_globals, eval_instance_parallel

CKPT = "result/GNN_TRAIN_ABL_no_reward_to_go_seed5/CheckPoint_epoch00190/GNN_ACTOR_state_dic.pt"
actor = create_gnn_actor().to(DEVICE)
actor.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
actor.eval()


@torch.no_grad()
def eval_instance_auction(V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]  # [1,1,N]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]  # [1,1,M,N]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0  # [1,1,M,N]
        action = auction_round_action(remaining_value, prob, legal_mask)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


df = pd.read_excel("TEST_INSTANCE/20M_30N_5T.xlsx")
nw, nt, mt = 20, 30, 5

naive_objs, auction_objs = [], []
t0 = time.time()
for i in range(10):
    row = df.iloc[i]
    V = json.loads(row["V"]); P = np.array(json.loads(row["P"])); TW = json.loads(row["TW"])
    amm = json.loads(row["AMM"]); prep = json.loads(row["PREP"]); cost = json.loads(row["COST"])

    naive = eval_instance_parallel(actor, V, P, TW, nw, nt, mt, amm, prep, cost)["objective"]
    auc = eval_instance_auction(V, P, TW, nw, nt, mt, amm, prep, cost)
    naive_objs.append(naive)
    auction_objs.append(auc)
    print(f"instance {i}: naive={naive:.4f} auction={auc:.4f}", flush=True)

print(f"\nMEAN naive={np.mean(naive_objs):.4f} auction={np.mean(auction_objs):.4f} "
      f"(elapsed {time.time()-t0:.1f}s)")
