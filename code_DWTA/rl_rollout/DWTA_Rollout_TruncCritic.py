import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import sys
import argparse
import pandas as pd
import torch
from torch import no_grad
import ast
import numpy as np
import json
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_Instance_generation import *
from rl_rollout.DWTA_Simulator_rollout import Environment
from common.DWTA_GNN import create_gnn_actor, create_gnn_critic
from common.utilities import Average_Meter, Get_Logger
from rl_rollout.BEAM_WITH_SIMULATION_TRUNC_CRITIC import Beam_Search_Trunc, batch_dimension_resize_trunc
from common.TORCH_OBJECTS import DEVICE


def main(beta=2, to_go_weight=1.0, use_greedy_only=False):
    actor = create_gnn_actor().to(DEVICE)
    critic = create_gnn_critic().to(DEVICE)

    # Load weights (update paths as needed)
    actor_path = '../TRAIN/GNN_TRAIN_20250809(10)/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt'
    critic_path = '../TRAIN/GNN_TRAIN_20250809(10)/CheckPoint_epoch00200/GNN_CRITIC_state_dic.pt'

    actor.load_state_dict(torch.load(actor_path, map_location=torch.device(DEVICE), weights_only=False))
    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path, map_location=torch.device(DEVICE), weights_only=False))
    actor.eval()
    critic.eval()

    # Data file
    file_name = '../TEST_INSTANCE/30M_30N.xlsx'
    df = pd.read_excel(file_name)

    # Rollout params: beta=None or <0 => full rollout; use_greedy_only=True => argmax only
    obj = []

    with no_grad():
        for i in range(1):
            print("index----", i)

            V = ast.literal_eval(df.loc[i]['V'])
            df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
            P = df['P'][i]
            TW = ast.literal_eval(df.loc[i]['TW'])
            TW = np.array(TW)

            assignment_encoding, weapon_to_target_prob = input_generation(
                NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=V, prob=P, TW=TW, max_time=MAX_TIME, batch_size=1
            )
            assignment_encoding = assignment_encoding[:, None, :, :].expand(1, VAL_PARA, NUM_TARGETS * NUM_WEAPONS + 1, assignment_encoding.size(-1))
            weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(1, VAL_PARA, NUM_WEAPONS, NUM_TARGETS)

            env = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)

            for time_clock in range(MAX_TIME):
                for index in range(NUM_WEAPONS):
                    possible_actions = env.mask.clone()
                    if (possible_actions > 0).any():
                        beam = Beam_Search_Trunc(env=env, actor=actor, value=critic, available_actions=possible_actions, beta=beta, to_go_weight=to_go_weight, use_greedy_only=use_greedy_only)
                        beam.reset()
                        expanded = beam.expand_actions()
                        b_idx, g_idx = beam.do_beam_simulation(possible_node_index=expanded, time=time_clock, w_index=index)
                        selected_action = g_idx.unsqueeze(1)
                        env = batch_dimension_resize_trunc(env=env, batch_index=b_idx, group_index=g_idx)
                    else:
                        selected_action = torch.tensor([NUM_WEAPONS*NUM_TARGETS], device=DEVICE)[None, :].expand(1, VAL_PARA)
                    env.update_internal_variables(selected_action=selected_action)
                env.time_update()

            obj_value = (env.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
            min_obj = obj_value.min()
            print("Objective:", float(min_obj))
            obj.append(min_obj)

    print("Average:", sum(obj)/len(obj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with rollout (Beam_Search_Trunc)")
    parser.add_argument("--beta", type=int, default=2,
                        help="Rollout depth; use -1 for full rollout to end of episode (Bertsekas)")
    parser.add_argument("--to_go_weight", type=float, default=1.0)
    parser.add_argument("--use_greedy_only", action="store_true",
                        help="Use only greedy (argmax) policy for rollout; else multi-policy")
    args = parser.parse_args()
    beta = None if args.beta < 0 else args.beta
    main(beta=beta, to_go_weight=args.to_go_weight, use_greedy_only=args.use_greedy_only) 