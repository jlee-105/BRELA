"""
Rollout Algorithm - Following DWTA_INFERENCE_TREE.py exactly
Using same simulator, same tensor operations, same everything
"""

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import pandas as pd
from torch import no_grad
import sys
import copy

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dynamic_Instance_generation import *
from DWTA_Simulator_rollout import Environment
from DWTA_GNN import create_gnn_actor
from utilities import Average_Meter, Get_Logger
import ast
import numpy as np
import json
import time
from BEAM_WITH_SIMULATION_NOT_TRUNC import *
from TORCH_OBJECTS import DEVICE

# Load actor
actor = create_gnn_actor().to(DEVICE)
actor_path = '../TRAIN/GNN_TRAIN_20250809(10)/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt'
actor.load_state_dict(torch.load(actor_path, map_location=torch.device(DEVICE), weights_only=False))
actor.eval()

start = time.time()
# data file name
file_name='../TEST_INSTANCE/30M_30N.xlsx'
df = pd.read_excel(file_name)

########################################
# EVALUATION
########################################

obj = list()
start_time = time.time()
start_v = list()
with no_grad():
    for i in range(1):
        start = time.time()
        print("index----", i)
        i = i

        V = ast.literal_eval(df.loc[i]['V'])
        df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
        P= df['P'][i]
        TW = ast.literal_eval(df.loc[i]['TW'])
        TW = np.array(TW)

        # initial data generation - START WITH SINGLE PATH
        assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=NUM_WEAPONS, NUM_TARGET=NUM_TARGETS, value=V, prob=P, TW=TW, max_time=MAX_TIME, batch_size=1)
        print(assignment_encoding.shape)
        
        # Initial expansion like working code
        assignment_encoding = assignment_encoding[:, None, :, :].expand(1, VAL_PARA, NUM_TARGETS * NUM_WEAPONS + 1, 9)
        weapon_to_target_prob = weapon_to_target_prob[:, None, :, :].expand(1, VAL_PARA, NUM_WEAPONS, NUM_TARGETS)
        
        # Start with single path environment
        env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=MAX_TIME)

        for time_clock in range(MAX_TIME):
            for index in range(NUM_WEAPONS):
                #check all possible expansions
                possible_actions = env_e.mask.clone()

                if (possible_actions > 0).any():
                    # Parallel rollout using Beam_Search to evaluate all actions, keep single best
                    beam_search = Beam_Search(actor=actor, value=None, env=env_e, available_actions=possible_actions)
                    beam_search.reset()
                    expanded_node_index = beam_search.expand_actions()
                    selected_batch_index, selected_group_index = beam_search.do_beam_simulation(
                        possible_node_index=expanded_node_index, time=time_clock, w_index=index
                    )
                    selected_action = selected_group_index.unsqueeze(dim=1)
                    env_e = batch_dimension_resize(env=env_e, batch_index=selected_batch_index, group_index=selected_group_index)
                 
                else:
                    selected_action = torch.tensor([NUM_WEAPONS*NUM_TARGETS]).to(DEVICE)
                    selected_action = selected_action[None, :].expand(1, VAL_PARA)

                env_e.update_internal_variables(selected_action=selected_action)

            env_e.time_update()

        obj_value = (env_e.current_target_value[:, :, 0:NUM_TARGETS]).sum(2)
        min_obj = obj_value.min()
        obj.append(min_obj)

        start_v.append(env_e.original_target_value[0, :, 0:NUM_TARGETS].sum())

print("Results:", [f'{r:.3f}' for r in obj])
print("Average:", sum(obj) / len(obj)) 