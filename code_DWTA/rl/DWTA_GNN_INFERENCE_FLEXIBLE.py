import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import sys
import pandas as pd
import ast
import numpy as np
import json
import time
import torch
from torch import no_grad

# Add path for parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_Instance_generation import *
from rl_rollout.DWTA_Simulator_rollout import Environment
from common.DWTA_GNN import create_gnn_actor
from common.TORCH_OBJECTS import DEVICE


def main():
    # Load actor (GNN baseline)
    actor = create_gnn_actor().to(DEVICE)
    actor_path = '../TRAIN/GNN_TRAIN_20250809(10)/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt'
    actor.load_state_dict(torch.load(actor_path, map_location=torch.device(DEVICE), weights_only=False))
    actor.eval()

    # Data file
    file_name = '../TEST_INSTANCE/30M_30N.xlsx'
    df = pd.read_excel(file_name)

    results = []
    start_v = []

    with no_grad():
        for i in range(10):
            print(f"index---- {i}")

            V = ast.literal_eval(df.loc[i]['V'])
            df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
            P = df['P'][i]
            TW = ast.literal_eval(df.loc[i]['TW'])
            TW = np.array(TW)

            # Generate inputs
            assignment_encoding, weapon_to_target_prob = input_generation(
                NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS,
                value=V, prob=P, TW=TW, max_time=MAX_TIME, batch_size=1
            )

            # Expand to [batch=1, para=VAL_PARA, ...]
            assignment_encoding = assignment_encoding[:, None, :, :].expand(1, VAL_PARA, NUM_TARGETS * NUM_WEAPONS + 1, assignment_encoding.size(-1))
            weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(1, VAL_PARA, NUM_WEAPONS, NUM_TARGETS)

            # Environment (baseline simulator)
            env = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)

            # Pure GNN rollout (greedy)
            for t in range(MAX_TIME):
                for w in range(NUM_WEAPONS):
                    mask = env.mask.clone()
                    if (mask > 0).any():
                        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask)
                        selected_action = policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).argmax(dim=1).view(env.assignment_encoding.size(0), env.assignment_encoding.size(1))
                    else:
                        selected_action = torch.tensor([NUM_WEAPONS * NUM_TARGETS], device=DEVICE)
                        selected_action = selected_action[None, :].expand(1, VAL_PARA)

                    env.update_internal_variables(selected_action=selected_action)
                env.time_update()

            obj_value = (env.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
            result = obj_value.min().item()
            results.append(result)
            print(f"GNN Result: {result:.3f}")

            start_v.append(env.original_target_value[0, :, 0:NUM_TARGETS].sum())

    print("GNN Results:", [f"{r:.3f}" for r in results])
    print("GNN Average:", sum(results) / len(results))

    # Optional: save CSV
    try:
        out_df = pd.DataFrame({'obj': results})
        out_df.to_csv('GNN_BASELINE_30-30-5.csv', index=False)
        print("Saved GNN baseline results to GNN_BASELINE_30-30-5.csv")
    except Exception as e:
        print(f"CSV save skipped: {e}")

