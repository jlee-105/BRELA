import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *

class Environment:

    def __init__(self, assignment_encoding, weapon_to_target_prob, max_time):

        # assignment encoding
        self.assignment_encoding = assignment_encoding.clone()
        # weapon-to-target probability (static)
        self.weapon_to_target_prob = weapon_to_target_prob.clone()
        # record weapon to target (dynamic)
        self.weapon_to_target_assign = torch.full(size=(assignment_encoding.size(0), assignment_encoding.size(1), NUM_WEAPONS), fill_value=-1).to(DEVICE)
        # target value initialization
        self.current_target_value = assignment_encoding.clone()[:, :,  :-1, -2]*MAX_TARGET_VALUE
        self.original_target_value = assignment_encoding.clone()[:, :, :-1, -2]*MAX_TARGET_VALUE
        # weapon status
        self.all_weapon_NOT_done = torch.tensor(True).to(DEVICE)
        # tracking possible weapons
        self.possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
        self.possible_weapons = self.possible_weapons[None, None, :].expand(assignment_encoding.size(0), assignment_encoding.size(1), NUM_WEAPONS)

        # tracking available actions without NO ACTION
        self.available_actions = torch.arange(0, NUM_WEAPONS*NUM_TARGETS).int().to(DEVICE)
        self.available_actions = self.available_actions[None, None, :].expand(assignment_encoding.size(0), assignment_encoding.size(1), NUM_WEAPONS*NUM_TARGETS)
        initial_available_target = (assignment_encoding.clone()[:, :, :-1, -3])
        self.available_actions = initial_available_target.bool()

        # mask initialization
        self.mask = torch.full(size=(assignment_encoding.size(0), assignment_encoding.size(1), NUM_WEAPONS*NUM_TARGETS+1,), fill_value=1.0).to(DEVICE)
        self.mask[:, :, -1] = 1.0

        # Not emerged one should be masked ----------------------------
        active_target = self.mask.clone()*assignment_encoding[:,:,:, -4]==0
        self.mask = self.mask.clone() * active_target.float()



        # weapon availability
        self.weapon_availability = torch.full(size=(assignment_encoding.size(0), assignment_encoding.size(1), NUM_WEAPONS), fill_value=1.0).to(DEVICE)
        amm_availability = torch.tensor(AMM[:NUM_WEAPONS]).to(DEVICE)
        self.amm_availability = amm_availability[None, None, :].expand(assignment_encoding.size(0), assignment_encoding.size(1), amm_availability.size(0))


        # waiting time for each weapon
        self.weapon_wait_time = torch.full(size=(assignment_encoding.size(0), assignment_encoding.size(1), NUM_WEAPONS), fill_value=0.0).to(DEVICE)
        # waiting time for each weapon
        self.time_left = torch.tensor(max_time).to(DEVICE)

        # added part
        self.target_availability = torch.full(size=(assignment_encoding.size(0), assignment_encoding.size(1), NUM_WEAPONS), fill_value=1.0).to(DEVICE)
        # print(assignment_encoding)
        # a = input()
        initial_available_target = (assignment_encoding[:, :, :, -4][:, :, :NUM_TARGETS]==0)
        self.target_availability = initial_available_target.float()


        # waiting time for each weapon
        self.target_start_time = assignment_encoding.clone()[:, :, :, -4][:, :, :NUM_TARGETS]*MAX_TIME
        self.target_end_time = assignment_encoding.clone()[:, :, :, -3][:, :, :NUM_TARGETS]*MAX_TIME

        # print("AFafaf", self.target_active_window)
        # self.target_window = assignment_encoding.clone()[:, :, :, 0][:,:, :NUM_TARGETS]
        self.n_target_hit = torch.full(size=(assignment_encoding.size(0), assignment_encoding.size(1), NUM_TARGETS), fill_value=0.0).to(DEVICE)

        # clock
        self.clock = torch.tensor(0).to(DEVICE)
        # count fires
        self.n_fires = torch.tensor(0).to(DEVICE)
        self.mask[:, :, -1] = 1.0

    def update_internal_variables(self, selected_action):

        batch_size = selected_action.size(0)
        para_size = selected_action.size(1)

        if (selected_action < NUM_WEAPONS * NUM_TARGETS).any():

            selected_action_index = torch.where(selected_action<NUM_WEAPONS*NUM_TARGETS)

            # print(selected_action_index)
            # a = input()

            # batch index, parallel index
            batch_id = selected_action_index[0]
            par_id = selected_action_index[1]

            # weapon, target index
            weapon_index = selected_action[batch_id, par_id] // NUM_TARGETS
            target_index = selected_action[batch_id, par_id] % NUM_TARGETS

            self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
            # dim [batch, par, weapon, target]

            reduced_values = self.current_target_value[batch_id, par_id, weapon_index, target_index] * self.weapon_to_target_prob[batch_id, par_id,weapon_index,target_index]
            # shape: dim selected actions

            reduced_values = reduced_values[:, None].expand(reduced_values.size(0), NUM_WEAPONS)
            self.current_target_value[batch_id, par_id, :, target_index] = self.current_target_value[batch_id, par_id, :, target_index] - reduced_values.float()
            # reshape of current target value

            self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS*NUM_TARGETS)
            # additional weapon to target update
            self.weapon_to_target_assign[batch_id, par_id, weapon_index] = target_index

            # weapon wait_time -- update
            wait_time = torch.FloatTensor(PREPARATION_TIME[:NUM_WEAPONS]).to(DEVICE)
            wait_time = wait_time[None, None, :].expand(batch_size, para_size, wait_time.size(0))
            self.weapon_wait_time[batch_id, par_id, weapon_index] = wait_time[batch_id, par_id, weapon_index]

            # weapon availability update
            self.weapon_availability[(self.weapon_wait_time >0) | (self.amm_availability<=0)] = 0.0
            self.n_target_hit[batch_id, par_id, target_index] = self.n_target_hit.clone()[batch_id, par_id, target_index]+1


            # ammunition update
            self.amm_availability = self.amm_availability.clone()
            self.amm_availability[batch_id, par_id, weapon_index] -= 1

            # number of fires update
            self.n_fires = self.n_fires + 1

            # possible weapon update
            possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
            self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
            possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
            self.possible_weapons = possible_weapons_mask.long()

            # print("possible_weapon", self.possible_weapons.shape)
            # a = input()

            # update target availability
            target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
            self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
            # print("self.target_start_time", self.target_start_time)
            # print("self.target_end_time", self.target_end_time)
            # a = input()

            possible_target_mask = (self.target_start_time<= self.clock) & (self.target_end_time>= self.clock)
            self.target_availability = possible_target_mask.long()

            # print("possible_target", self.target_availability)

            # update available actions
            available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)
            available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)

            # print("available_actions", available_actions)
            # a = input()

            if torch.any(self.possible_weapons <= 0):
                unavailable_weapon = torch.where((self.possible_weapons <= 0))
                unavailable_weapon_batch_index = unavailable_weapon[0]
                unavailable_weapon_par_index = unavailable_weapon[1]
                unavailable_weapon_w_index = unavailable_weapon[2]
                available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index, :] = 0.0


            if torch.any((self.target_availability <= 0)):
                unavailable_target = torch.where((self.target_availability <= 0))
                unavailable_target_batch_index = unavailable_target[0]
                unavailable_target_par_index = unavailable_target[1]
                unavailable_target_t_index = unavailable_target[2]
                available_actions[unavailable_target_batch_index, unavailable_target_par_index, :, unavailable_target_t_index] = 0.0


            self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
            # print(self.available_actions)
            # a = input()

            self.mask[:, :, :-1] = self.available_actions.clone()

            """ factors : additiona,  wait_to_ready, Number_of_fired, time_target_start, active_time, target_value, probability"""
            # State Update
            assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1 , :].detach().clone()
            assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS, -1)

            # remaining ammunition update
            remaining_ammunition = self.amm_availability.clone()[:, :, :, None].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/max(AMM)
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 0] = remaining_ammunition[batch_id, par_id, weapon_index, :].float()

            # weapon availability update
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 1] = 0.0

            # number of fired
            n_target_hit = self.n_target_hit.clone()[:, :, None, :].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/(NUM_TARGETS*MAX_TIME)
            assignment_encoding_without_no_action[batch_id, par_id, :, target_index, 4] = n_target_hit[batch_id, par_id, :, target_index]

            # target value
            assignment_encoding_without_no_action[:, :, :, :, -2] = self.current_target_value.clone().reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/MAX_TARGET_VALUE
            # reshape
            self.assignment_encoding[:, :, :-1 , :] = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS,-1)

        else:
            self.n_fires = self.n_fires + 1

        self.mask[:, :, -1] = 1.0

        return 0


    def mask_probs(self, action_probs):

        mask_probs = action_probs * self.mask.to(DEVICE)
        sums = mask_probs.sum()
        normalized_probability = mask_probs / sums

        return normalized_probability

    def time_update(self):

        # move one clock
        self.clock = self.clock + 1
        # weapon_wait_time update
        self.weapon_wait_time[self.weapon_wait_time>0] -= 1
        # update leftover time
        self.time_left = MAX_TIME - self.clock
        # # update self.target_active_window
        # self.target_active_window[self.target_active_window>0] -= 1
        batch_size = self.assignment_encoding.size(0)
        para_size = self.assignment_encoding.size(1)

        # check new target emerge
        # if torch.any(self.target_start_time == self.clock):
        #     new_target_location = torch.where(self.target_start_time == self.clock)
        #     max_time_window = torch.FloatTensor(MAX_TIME_WINDOW).to(DEVICE)
        #     max_time_window = max_time_window[None, None, :].expand(batch_size, para_size, max_time_window.size(0))
        #     self.target_active_window[new_target_location[0], new_target_location[1], new_target_location[2]] = max_time_window[new_target_location[0], new_target_location[1], new_target_location[2]]
        # else:
        #     pass

        # update weapon availability
        self.weapon_availability[(self.weapon_wait_time == 0) & (self.amm_availability>0)] = 1.0
        possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
        self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
        possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
        self.possible_weapons = possible_weapons_mask.long()


        target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
        self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
        #if torch.any(self.target_active_window > 0):
        possible_target_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
        self.target_availability = possible_target_mask.long()


        # update available actions and mask based on the possible weapons and targets
        available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)
        available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)


        if torch.any((self.possible_weapons <= 0)):
            unavailable_weapon = torch.where((self.possible_weapons <= 0))
            unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index  = unavailable_weapon[0], unavailable_weapon[1], unavailable_weapon[2]
            available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index, :] = 0.0

        if torch.any((self.target_availability <= 0)):
            unavailable_target = torch.where((self.target_availability <= 0))
            unavailable_target_batch_index, unavailable_target_par_index, unavailable_target_t_index = unavailable_target[0], unavailable_target[1], unavailable_target[2]
            available_actions[unavailable_target_batch_index, unavailable_target_par_index, :,unavailable_target_t_index] = 0.0

        self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
        self.mask[:, :, :-1] = self.available_actions.clone()

        # if self.clock == MAX_TIME-1:
        #     self.mask[:, :, -1] = 0.0
        #     all_zeros = self.mask.sum(dim=-1) == 0
        #     for i in range(batch_size):
        #         for j in range(para_size):
        #             if all_zeros[i, j]:  # Direct checking
        #                 self.mask[i, j, -1] = 1.0
        # else:
        self.mask[:, :, -1] = 1.0

        # else:
        #     pass

        # state update ---------------------------------------------
        assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1, :].clone()
        assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS, -1)

        # available weapon update
        available_weapon_index = torch.where(self.weapon_availability==1)
        # Weapon availability update:
        assignment_encoding_without_no_action[available_weapon_index[0], available_weapon_index[1], available_weapon_index[2], :, 1] = 1.0

        # weapon wait time update
        updated_wait_time = self.weapon_wait_time.clone()[:, :, :,  None].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/max(PREPARATION_TIME)
        assignment_encoding_without_no_action[:, :, :, :, 3] = updated_wait_time

        # time left
        assignment_encoding_without_no_action[:, :, :, :, 2] = self.time_left/MAX_TIME
        self.assignment_encoding[:, :, :-1, :] = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS, -1)


        return 0

    def reward_calculation(self):
        reward = 1- self.current_target_value[0:NUM_TARGETS].sum()/self.original_target_value[0:NUM_TARGETS].sum()
        return reward, self.current_target_value

