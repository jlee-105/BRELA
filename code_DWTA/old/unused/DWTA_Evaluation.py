import copy
import torch
from Dynamic_HYPER_PARAMETER import *
from TORCH_OBJECTS import *
from Dynamic_Instance_generation import input_generation
from DWTA_Simulator import Environment  # Required for evaluation
#from DWTA_GNN import EdgeAwareGNN_ACTOR, EdgeAwareGNN_CRITIC  # Import GNN models
#import DWTA_BHGT as MCTS_MODEL  # Not used in evaluation_pure
import time
from utilities import Get_Logger
from utilities import Average_Meter
import os
from datetime import datetime

########################################
# EVALUATION
########################################


def evaluation_pure(model, value, prob, TW, episode):

    TRAIN_BATCH = 1
    NUM_PAR = 1

    model.eval()
    
    # Detect problem size from input data (size-agnostic)
    actual_num_weapons, actual_num_targets = prob.shape
    actual_max_time = MAX_TIME  # Use current global value
    
    # data generation (size-agnostic)
    assignment_encoding, weapon_to_target_prob = input_generation(NUM_WEAPON=actual_num_weapons, NUM_TARGET=actual_num_targets, value=value, prob=prob, TW=TW, max_time=actual_max_time, batch_size=VAL_BATCH)
    # expand 대신 repeat + contiguous로 메모리 공유 제거
    assignment_encoding = assignment_encoding.unsqueeze(1).repeat(1, VAL_PARA, 1, 1).contiguous()
    weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1).repeat(1, VAL_PARA, 1, 1).contiguous()
    env_e = Environment(assignment_encoding=assignment_encoding, weapon_to_target_prob=weapon_to_target_prob, max_time=actual_max_time)

    # Print initial target values for first 10 instances

    for time_clock in range(actual_max_time):

        for index in range(actual_num_weapons):
            policy, _ = model(assignment_embedding=env_e.assignment_encoding.detach().clone(), 
                            prob=weapon_to_target_prob.clone(), 
                            mask=env_e.mask.clone())
            
            # Greedy action selection (argmax over probabilities)
            flat_policy = policy.view(-1, actual_num_weapons * actual_num_targets + 1)
            action_index = torch.argmax(flat_policy, dim=1).view(VAL_BATCH, VAL_PARA)
            
            # Removed per-step action logging (including 'No Action')
            
            env_e.update_internal_variables(selected_action=action_index)

        env_e.time_update()


    # Size-agnostic objective calculation
    obj_value = (env_e.current_target_value[:, :, 0:actual_num_targets]).sum(2)
    #print("ob", obj_value)
    obj_value_ = obj_value.squeeze()
    obj_values = torch.min(obj_value_)
    # print("obj_values", obj_values)
    # a = input()

    return obj_values
