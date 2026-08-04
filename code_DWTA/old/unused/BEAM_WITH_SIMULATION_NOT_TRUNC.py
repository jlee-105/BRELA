from Dynamic_HYPER_PARAMETER import *
from TORCH_OBJECTS import DEVICE
alpha = 1
import torch


def expansion():

    return 0

# class Beam_Search:
#
#     def __init__(self, env, actor, value, available_actions):
#
#         # assignment encoding
#         self.beam_env = env
#         #self.original_env = env
#         # weapon-to-target probability (static)
#         self.policy = actor
#         self.to_go_value = value
#         # record weapon to target (dynamic)
#         self.beam_available_actions = available_actions
#         self.beam_result = None
#
#         self.assignment_encoding = None
#         self.weapon_to_target_prob = None
#         self.weapon_to_target_assign = None
#         self.current_target_value = None
#         self.original_target_value = None
#         self.possible_weapons = None
#         self.available_actions = None
#         self.mask = None
#         self.weapon_availability = None
#         self.amm_availability = None
#         self.weapon_wait_time = None
#         self.time_left = None
#         self.target_availability = None
#         self.target_start_time = None
#         self.target_end_time = None
#         self.n_target_hit = None
#         self.clock = None
#         self.n_fires = None
#
#     def reset(self):
#
#         possible_action_index = torch.where(self.beam_available_actions>0)
#         batch_idx = possible_action_index[0]
#         expanded_group_index = possible_action_index[2]
#         unique_elements, counts = torch.unique(batch_idx, return_counts=True)
#         max_count = counts.max()
#
#         # reset all internal variables
#         # 1) encoding
#         self.assignment_encoding = self.beam_env.assignment_encoding.repeat(1, max_count, 1, 1)
#         # 2) weapon-to-target probability
#         self.weapon_to_target_prob = self.beam_env.weapon_to_target_prob.repeat(1, max_count, 1, 1)
#         # 3) weapon-to-target assign
#         self.weapon_to_target_assign = self.beam_env.weapon_to_target_assign.repeat(1, max_count, 1)
#         # 4) current_target_value
#         self.current_target_value = self.beam_env.current_target_value.repeat(1, max_count, 1)
#         # 5) current_target_value
#         self.original_target_value = self.beam_env.original_target_value.repeat(1, max_count, 1)
#         # 6) possible_weapons
#         self.possible_weapons = self.beam_env.possible_weapons.repeat(1, max_count, 1)
#         # 7) available actions
#         self.available_actions = self.beam_env.available_actions.repeat(1, max_count, 1)
#         # 8) mask
#         self.mask = self.beam_env.mask.repeat(1, max_count, 1)
#         # 9) weapon availability
#         self.weapon_availability = self.beam_env.weapon_availability.repeat(1, max_count, 1)
#         # 10) ammunition availability
#         self.amm_availability = self.beam_env.amm_availability.repeat(1, max_count, 1)
#         # 11) waiting time for each weapon
#         self.weapon_wait_time = self.beam_env.weapon_wait_time.repeat(1, max_count, 1)
#         # 12) left time
#         self.time_left = self.beam_env.time_left
#         # 13) target_availability
#         self.target_availability = self.beam_env.target_availability.repeat(1, max_count, 1)
#         # 14) target_start_time
#         self.target_start_time = self.beam_env.target_start_time.repeat(1, max_count, 1)
#         # 15) target_end time
#         self.target_end_time = self.beam_env.target_end_time.repeat(1, max_count, 1)
#         # 16) target hit
#         self.n_target_hit = self.beam_env.n_target_hit.repeat(1, max_count, 1)
#         # 17) clock
#         self.clock = self.beam_env.clock
#         # 18) n_fires
#         self.n_fires = self.beam_env.n_fires
#
#         # print("reset---available_actions", self.beam_available_actions)
#         # print("reset----bean__current--env", self.beam_env.current_target_value)
#         # print("reset----beam--reformatted--env", self.current_target_value)
#         # a = input()
#
#     def expand_actions(self):
#
#         # check possible index
#         possible_action_index = torch.where(self.beam_available_actions > 0)
#         # print("expanded_action_action", possible_action_index)
#         # alpha is batch
#         batch_idx = possible_action_index[0]
#         # print("expanded-----batch_idx", batch_idx)
#         # node is will be group
#         node_index = possible_action_index[2]
#         # print("expanded-----node_index", node_index)
#         # a = input()
#         unique_elements, counts = torch.unique(batch_idx, return_counts=True)
#         max_count = counts.max()
#         # print("unique_element", unique_elements)
#         # a = input()
#
#         selected_actions = torch.full(size=(self.beam_available_actions.size(0), max_count), fill_value=-1, dtype=torch.int64).to(DEVICE)
#         # print(selected_actions)
#         # a = input()
#
#         # Pad each batch's `node_index` to `max_count`
#         for batch in unique_elements:
#             batch_mask = batch_idx == batch
#             indices_for_batch = node_index[batch_mask]
#
#             # Pad the indices with the last value to match `max_count`
#             #
#             # if indices_for_batch.size(0) < max_count:
#             #     padding = indices_for_batch[-1:].repeat(max_count - indices_for_batch.size(0))
#             #     indices_for_batch = torch.cat((indices_for_batch, padding))
#
#             if indices_for_batch.size(0) < max_count:
#                 padding = indices_for_batch[:1].repeat(max_count - indices_for_batch.size(0))
#                 indices_for_batch = torch.cat((indices_for_batch, padding))
#
#             # if indices_for_batch.size(0) < max_count:
#             #     rand_idx = torch.randint(0, indices_for_batch.size(0),(1,), device=indices_for_batch.device)
#             #     padding_value = indices_for_batch[rand_idx]
#             #     padding = padding_value.repeat(max_count - indices_for_batch.size(0))
#             #     indices_for_batch = torch.cat((indices_for_batch, padding))
#
#             selected_actions[batch] = indices_for_batch
#
#         # print(selected_actions)
#         # a = input()
#
#         # print(self.current_target_value)
#         self.update_internal_variables(selected_action=selected_actions)
#         # print(self.current_target_value)
#         # a = input()
#
#         return selected_actions
#
#
#     def update_internal_variables(self, selected_action):
#
#         batch_size = selected_action.size(0)
#         para_size = selected_action.size(1)
#
#         if (selected_action < NUM_WEAPONS * NUM_TARGETS).any():
#
#             selected_action_index = torch.where(selected_action<NUM_WEAPONS*NUM_TARGETS)
#
#             # print(selected_action_index)
#             # a = input()
#
#             # batch index, parallel index
#             batch_id = selected_action_index[0]
#             par_id = selected_action_index[1]
#
#             # weapon, target index
#             weapon_index = selected_action[batch_id, par_id] // NUM_TARGETS
#             target_index = selected_action[batch_id, par_id] % NUM_TARGETS
#
#             # print("weapon_index", weapon_index)
#             # print("target_index", target_index)
#             # print("self_a", self.available_actions.shape)
#             # a = input()
#
#             self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
#             # dim [batch, par, weapon, target]
#             reduced_values = self.current_target_value[batch_id, par_id, weapon_index, target_index] * self.weapon_to_target_prob[batch_id, par_id,weapon_index,target_index]
#             # shape: dim selected actions
#             reduced_values = reduced_values[:, None].expand(reduced_values.size(0), NUM_WEAPONS)
#             self.current_target_value[batch_id, par_id, :, target_index] = self.current_target_value[batch_id, par_id, :, target_index] - reduced_values.float()
#             # reshape of current target value
#             self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS*NUM_TARGETS)
#             # additional weapon to target update
#             self.weapon_to_target_assign[batch_id, par_id, weapon_index] = target_index
#
#             # weapon wait_time -- update
#             wait_time = torch.FloatTensor(PREPARATION_TIME[:NUM_WEAPONS]).to(DEVICE)
#             wait_time = wait_time[None, None, :].expand(batch_size, para_size, wait_time.size(0))
#             self.weapon_wait_time[batch_id, par_id, weapon_index] = wait_time[batch_id, par_id, weapon_index]
#
#             # weapon availability update
#             self.weapon_availability[(self.weapon_wait_time >0) | (self.amm_availability<=0)] = 0.0
#             self.n_target_hit[batch_id, par_id, target_index] = self.n_target_hit.clone()[batch_id, par_id, target_index]+1
#
#             # ammunition update
#             self.amm_availability = self.amm_availability.clone()
#             self.amm_availability[batch_id, par_id, weapon_index] -= 1
#
#             # number of fires update
#             self.n_fires = self.n_fires + 1
#
#             # possible weapon update
#             possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
#             self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
#             possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
#             self.possible_weapons = possible_weapons_mask.long()
#
#             # print("possible_weapon", self.possible_weapons.shape)
#             # a = input()
#
#             # update target availability
#             target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
#             self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
#             # print("self.target_start_time", self.target_start_time)
#             # print("self.target_end_time", self.target_end_time)
#             # a = input()
#
#             possible_target_mask = (self.target_start_time<= self.clock) & (self.target_end_time>= self.clock)
#             self.target_availability = possible_target_mask.long()
#
#             # print("possible_target", self.target_availability)
#
#             # update available actions
#             available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)
#             available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)
#
#             # print("available_actions", available_actions)
#             # a = input()
#
#             if torch.any(self.possible_weapons <= 0):
#                 unavailable_weapon = torch.where((self.possible_weapons <= 0))
#                 unavailable_weapon_batch_index = unavailable_weapon[0]
#                 unavailable_weapon_par_index = unavailable_weapon[1]
#                 unavailable_weapon_w_index = unavailable_weapon[2]
#                 available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index, :] = 0.0
#
#
#             if torch.any((self.target_availability <= 0)):
#                 unavailable_target = torch.where((self.target_availability <= 0))
#                 unavailable_target_batch_index = unavailable_target[0]
#                 unavailable_target_par_index = unavailable_target[1]
#                 unavailable_target_t_index = unavailable_target[2]
#                 available_actions[unavailable_target_batch_index, unavailable_target_par_index, :, unavailable_target_t_index] = 0.0
#
#
#             self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
#             # print(self.available_actions)
#             # a = input()
#
#             self.mask[:, :, :-1] = self.available_actions.clone()
#
#             """ factors : additiona,  wait_to_ready, Number_of_fired, time_target_start, active_time, target_value, probability"""
#             # State Update
#             assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1 , :].detach().clone()
#             assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS, -1)
#
#
#             # remaining ammunition update
#             remaining_ammunition = self.amm_availability.clone()[:, :, :, None].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/max(AMM)
#             assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 0] = remaining_ammunition[batch_id, par_id, weapon_index, :].float()
#
#             # weapon availability update
#             assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 1] = 0.0
#
#             # number of fired
#             n_target_hit = self.n_target_hit.clone()[:, :, None, :].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/(NUM_TARGETS*MAX_TIME)
#             assignment_encoding_without_no_action[batch_id, par_id, :, target_index, 4] = n_target_hit[batch_id, par_id, :, target_index]
#
#             # target value
#             assignment_encoding_without_no_action[:, :, :, :, -2] = self.current_target_value.clone().reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/MAX_TARGET_VALUE
#
#             # reshape
#             self.assignment_encoding[:, :, :-1 , :] = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS,-1)
#
#         else:
#             self.n_fires = self.n_fires + 1
#
#         self.mask[:, :, -1] =1.0
#
#         # print("updated ---- status --- weapon-to-target-assign", self.weapon_to_target_assign)
#         # print("updated ---- status --- current-value", self.current_target_value)
#         # print("updated ---- status --- mask", self.mask)
#         # a = input()
#
#     def time_update(self):
#
#
#         # print("hey time updated")
#         # a = input()
#
#         # move one clock
#         self.clock = self.clock + 1
#         # weapon_wait_time update
#         self.weapon_wait_time[self.weapon_wait_time>0] -= 1
#         # update leftover time
#         self.time_left = MAX_TIME - self.clock
#         # # update self.target_active_window
#         # self.target_active_window[self.target_active_window>0] -= 1
#         batch_size = self.assignment_encoding.size(0)
#         para_size = self.assignment_encoding.size(1)
#
#         # check new target emerge
#         # if torch.any(self.target_start_time == self.clock):
#         #     new_target_location = torch.where(self.target_start_time == self.clock)
#         #     max_time_window = torch.FloatTensor(MAX_TIME_WINDOW).to(DEVICE)
#         #     max_time_window = max_time_window[None, None, :].expand(batch_size, para_size, max_time_window.size(0))
#         #     self.target_active_window[new_target_location[0], new_target_location[1], new_target_location[2]] = max_time_window[new_target_location[0], new_target_location[1], new_target_location[2]]
#         # else:
#         #     pass
#
#         # update weapon availability
#         self.weapon_availability[(self.weapon_wait_time == 0) & (self.amm_availability>0)] = 1.0
#         possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
#         self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
#         possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
#         self.possible_weapons = possible_weapons_mask.long()
#
#
#         target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
#         self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
#         #if torch.any(self.target_active_window > 0):
#         possible_target_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
#         self.target_availability = possible_target_mask.long()
#
#
#         # update available actions and mask based on the possible weapons and targets
#         available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)
#         available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)
#
#
#         if torch.any((self.possible_weapons <= 0)):
#             unavailable_weapon = torch.where((self.possible_weapons <= 0))
#             unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index  = unavailable_weapon[0], unavailable_weapon[1], unavailable_weapon[2]
#             available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index, :] = 0.0
#
#         if torch.any((self.target_availability <= 0)):
#             unavailable_target = torch.where((self.target_availability <= 0))
#             unavailable_target_batch_index, unavailable_target_par_index, unavailable_target_t_index = unavailable_target[0], unavailable_target[1], unavailable_target[2]
#             available_actions[unavailable_target_batch_index, unavailable_target_par_index, :,unavailable_target_t_index] = 0.0
#
#         self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
#         self.mask[:, :, :-1] = self.available_actions.clone()
#
#         # if self.clock == MAX_TIME:
#         self.mask[:, :, -1] = 1.0
#         # else:
#         #     pass
#
#         # state update ---------------------------------------------
#         assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1, :].clone()
#         assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS, -1)
#
#         # available weapon update
#         available_weapon_index = torch.where(self.weapon_availability==1)
#         # Weapon availability update:
#         assignment_encoding_without_no_action[available_weapon_index[0], available_weapon_index[1], available_weapon_index[2], :, 1] = 1.0
#
#         # weapon wait time update
#         updated_wait_time = self.weapon_wait_time.clone()[:, :, :,  None].expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)/max(PREPARATION_TIME)
#         assignment_encoding_without_no_action[:, :, :, :, 3] = updated_wait_time
#
#         # time left
#         assignment_encoding_without_no_action[:, :, :, :, 2] = self.time_left/MAX_TIME
#
#         self.assignment_encoding[:, :, :-1, :] = assignment_encoding_without_no_action.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS, -1)
#
#
#         return 0

class Beam_Search:

    def __init__(self, env, actor, value, available_actions):

        # assignment encoding
        self.beam_env = env
        # self.original_env = env
        # weapon-to-target probability (static)
        self.policy = actor
        self.to_go_value = value
        # record weapon to target (dynamic)
        self.beam_available_actions = available_actions
        self.beam_result = None

        self.assignment_encoding = None
        self.weapon_to_target_prob = None
        self.weapon_to_target_assign = None
        self.current_target_value = None
        self.original_target_value = None
        self.possible_weapons = None
        self.available_actions = None
        self.mask = None
        self.weapon_availability = None
        self.amm_availability = None
        self.weapon_wait_time = None
        self.time_left = None
        self.target_availability = None
        self.target_start_time = None
        self.target_end_time = None
        self.n_target_hit = None
        self.clock = None
        self.n_fires = None

        self._selected_actions_buffer = torch.empty(
            self.beam_available_actions.size(0),
            NUM_WEAPONS * NUM_TARGETS + 1,
            dtype=torch.int64,
            device=DEVICE
        )

        self.prep_time_buffer = torch.tensor(PREPARATION_TIME[:NUM_WEAPONS], dtype=torch.float, device=DEVICE)

        self._available_actions_buffer = torch.empty(
            (alpha, NUM_WEAPONS * NUM_TARGETS + 1, NUM_WEAPONS, NUM_TARGETS),
            dtype=torch.float, device=DEVICE
        )

        self.weapons_arange = torch.arange(NUM_WEAPONS, device=DEVICE).int()
        self.target_arange = torch.arange(0, NUM_TARGETS).int().to(DEVICE)

    def reset(self):

        possible_action_index = torch.where(self.beam_available_actions > 0)
        batch_idx = possible_action_index[0]
        expanded_group_index = possible_action_index[2]
        unique_elements, counts = torch.unique(batch_idx, return_counts=True)
        max_count = counts.max()

        # reset all internal variables
        # 1) encoding
        # self.assignment_encoding = self.beam_env.assignment_encoding.repeat(1, max_count, 1, 1)
        # # 2) weapon-to-target probability
        # self.weapon_to_target_prob = self.beam_env.weapon_to_target_prob.repeat(1, max_count, 1, 1)
        # # 3) weapon-to-target assign
        # self.weapon_to_target_assign = self.beam_env.weapon_to_target_assign.repeat(1, max_count, 1)
        # # 4) current_target_value
        # self.current_target_value = self.beam_env.current_target_value.repeat(1, max_count, 1)
        # # 5) current_target_value
        # self.original_target_value = self.beam_env.original_target_value.repeat(1, max_count, 1)
        # # 6) possible_weapons
        # self.possible_weapons = self.beam_env.possible_weapons.repeat(1, max_count, 1)
        # # 7) available actions
        # self.available_actions = self.beam_env.available_actions.repeat(1, max_count, 1)
        # # 8) mask
        # self.mask = self.beam_env.mask.repeat(1, max_count, 1)
        # # 9) weapon availability
        # self.weapon_availability = self.beam_env.weapon_availability.repeat(1, max_count, 1)
        # # 10) ammunition availability
        # self.amm_availability = self.beam_env.amm_availability.repeat(1, max_count, 1)
        # # 11) waiting time for each weapon
        # self.weapon_wait_time = self.beam_env.weapon_wait_time.repeat(1, max_count, 1)
        # # 12) left time
        # self.time_left = self.beam_env.time_left
        # # 13) target_availability
        # self.target_availability = self.beam_env.target_availability.repeat(1, max_count, 1)
        # # 14) target_start_time
        # self.target_start_time = self.beam_env.target_start_time.repeat(1, max_count, 1)
        # # 15) target_end time
        # self.target_end_time = self.beam_env.target_end_time.repeat(1, max_count, 1)
        # # 16) target hit
        # self.n_target_hit = self.beam_env.n_target_hit.repeat(1, max_count, 1)
        # # 17) clock
        # self.clock = self.beam_env.clock
        # # 18) n_fires
        # self.n_fires = self.beam_env.n_fires

        self.assignment_encoding = self.beam_env.assignment_encoding.expand(-1, max_count, -1, -1).clone()
        # 2) weapon-to-target probability
        self.weapon_to_target_prob = self.beam_env.weapon_to_target_prob.expand(-1, max_count, -1, -1).clone()
        # 3) weapon-to-target assign
        self.weapon_to_target_assign = self.beam_env.weapon_to_target_assign.expand(-1, max_count, -1).clone()
        # 4) current_target_value
        self.current_target_value = self.beam_env.current_target_value.expand(-1, max_count, -1).clone()
        # 5) current_target_value
        self.original_target_value = self.beam_env.original_target_value.expand(-1, max_count, -1).clone()
        # 6) possible_weapons
        self.possible_weapons = self.beam_env.possible_weapons.expand(-1, max_count, -1).clone()
        # 7) available actions
        self.available_actions = self.beam_env.available_actions.expand(-1, max_count, -1).clone()
        # 8) mask
        self.mask = self.beam_env.mask.expand(-1, max_count, -1).clone()
        # 9) weapon availability
        self.weapon_availability = self.beam_env.weapon_availability.expand(-1, max_count, -1).clone()
        # 10) ammunition availability
        self.amm_availability = self.beam_env.amm_availability.expand(-1, max_count, -1).clone()
        # 11) waiting time for each weapon
        self.weapon_wait_time = self.beam_env.weapon_wait_time.expand(-1, max_count, -1).clone()
        # 12) left time
        self.time_left = self.beam_env.time_left.clone()
        # 13) target_availability
        self.target_availability = self.beam_env.target_availability.expand(-1, max_count, -1).clone()
        # 14) target_start_time
        self.target_start_time = self.beam_env.target_start_time.expand(-1, max_count, -1).clone()
        # 15) target_end time
        self.target_end_time = self.beam_env.target_end_time.expand(-1, max_count, -1).clone()
        # 16) target hit
        self.n_target_hit = self.beam_env.n_target_hit.expand(-1, max_count, -1).clone()
        # 17) clock
        self.clock = self.beam_env.clock.clone()
        # 18) n_fires
        self.n_fires = self.beam_env.n_fires.clone()

    def expand_actions(self):

        # check possible index
        possible_action_index = torch.where(self.beam_available_actions > 0)
        # print("expanded_action_action", possible_action_index)
        # alpha is batch
        batch_idx = possible_action_index[0]
        # print("expanded-----batch_idx", batch_idx)
        # node is will be group
        node_index = possible_action_index[2]
        # print("expanded-----node_index", node_index)
        # a = input()
        unique_elements, counts = torch.unique(batch_idx, return_counts=True)
        max_count = counts.max()
        # print("unique_element", unique_elements)
        # a = input()

        self._selected_actions_buffer.fill_(-1)

        # We'll use only [:, :max_count] for the current step
        selected_actions = self._selected_actions_buffer[:, :max_count]

        # selected_actions = torch.full(size=(self.beam_available_actions.size(0), max_count), fill_value=-1, dtype=torch.int64).to(DEVICE)
        # print(selected_actions)
        # a = input()

        # Pad each batch's `node_index` to `max_count`
        for batch in unique_elements:
            batch_mask = batch_idx == batch
            indices_for_batch = node_index[batch_mask]

            # Pad the indices with the last value to match `max_count`
            #
            # if indices_for_batch.size(0) < max_count:
            #     padding = indices_for_batch[-1:].repeat(max_count - indices_for_batch.size(0))
            #     indices_for_batch = torch.cat((indices_for_batch, padding))

            if indices_for_batch.size(0) < max_count:
                padding = indices_for_batch[:1].repeat(max_count - indices_for_batch.size(0))
                indices_for_batch = torch.cat((indices_for_batch, padding))

            # if indices_for_batch.size(0) < max_count:
            #     rand_idx = torch.randint(0, indices_for_batch.size(0),(1,), device=indices_for_batch.device)
            #     padding_value = indices_for_batch[rand_idx]
            #     padding = padding_value.repeat(max_count - indices_for_batch.size(0))
            #     indices_for_batch = torch.cat((indices_for_batch, padding))

            selected_actions[batch] = indices_for_batch

        # print(selected_actions)
        # a = input()

        # print(self.current_target_value)
        self.update_internal_variables(selected_action=selected_actions)
        # print(self.current_target_value)
        # a = input()

        return selected_actions

    def expand_actions_(self):

        # check possible index
        possible_action_index = torch.where(self.beam_available_actions > 0)
        # print("expanded_action_action", possible_action_index)
        # alpha is batch

        # print(possible_action_index)
        # a = input()

        batch_idx = possible_action_index[0]
        # print("expanded-----batch_idx", batch_idx)
        # node is will be group
        node_index = possible_action_index[2]
        # print("expanded-----node_index", node_index)

        # selected_actions = torch.full( size=(self.beam_available_actions.size(0), max_count),  fill_value=-1, dtype=torch.int64).to(DEVICE)

        # a = input()
        unique_elements, counts = torch.unique(batch_idx, return_counts=True)
        max_count = counts.max()
        # print("unique_element", unique_elements)
        # a = input()

        # Fill buffer with -1
        # self._selected_actions_buffer.fill_(-1)        #
        # # We'll only use a slice for the current max_count
        # selected_actions = self._selected_actions_buffer[:, :max_count]

        selected_actions = torch.full(size=(self.beam_available_actions.size(0), max_count), fill_value=-1,
                                      dtype=torch.int64).to(DEVICE)
        #
        # selected_actions = torch.full(size=(self.beam_available_actions.size(0), max_count), fill_value=-1, dtype=torch.int64).to(DEVICE)
        # print(selected_actions)
        # a = input()

        # Pad each batch's `node_index` to `max_count`
        for batch in unique_elements:
            batch_mask = batch_idx == batch
            indices_for_batch = node_index[batch_mask]

            # Pad the indices with the last value to match `max_count`
            #
            # if indices_for_batch.size(0) < max_count:
            #     padding = indices_for_batch[-1:].repeat(max_count - indices_for_batch.size(0))
            #     indices_for_batch = torch.cat((indices_for_batch, padding))

            if indices_for_batch.size(0) < max_count:
                padding = indices_for_batch[:1].repeat(max_count - indices_for_batch.size(0))
                indices_for_batch = torch.cat((indices_for_batch, padding))

            # if indices_for_batch.size(0) < max_count:
            #     rand_idx = torch.randint(0, indices_for_batch.size(0),(1,), device=indices_for_batch.device)
            #     padding_value = indices_for_batch[rand_idx]
            #     padding = padding_value.repeat(max_count - indices_for_batch.size(0))
            #     indices_for_batch = torch.cat((indices_for_batch, padding))

            selected_actions[batch] = indices_for_batch

        # print(selected_actions)
        # a = input()

        # print(self.current_target_value)
        self.update_internal_variables(selected_action=selected_actions)

        # print(self.current_target_value)
        # a = input()

        return selected_actions

    def update_internal_variables(self, selected_action):

        batch_size = selected_action.size(0)
        para_size = selected_action.size(1)

        if (selected_action < NUM_WEAPONS * NUM_TARGETS).any():

            selected_action_index = torch.where(selected_action < NUM_WEAPONS * NUM_TARGETS)

            # print(selected_action_index)
            # a = input()

            # batch index, parallel index
            batch_id = selected_action_index[0]
            par_id = selected_action_index[1]

            # weapon, target index
            weapon_index = selected_action[batch_id, par_id] // NUM_TARGETS
            target_index = selected_action[batch_id, par_id] % NUM_TARGETS

            # print("weapon_index", weapon_index)
            # print("target_index", target_index)
            # print("self_a", self.available_actions.shape)
            # a = input()

            self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS,
                                                                          NUM_TARGETS)
            # dim [batch, par, weapon, target]
            reduced_values = self.current_target_value[batch_id, par_id, weapon_index, target_index] * \
                             self.weapon_to_target_prob[batch_id, par_id, weapon_index, target_index]
            # shape: dim selected actions
            reduced_values = reduced_values[:, None].expand(reduced_values.size(0), NUM_WEAPONS)
            self.current_target_value[batch_id, par_id, :, target_index] = self.current_target_value[batch_id,
                                                                           par_id, :,
                                                                           target_index] - reduced_values.float()
            # reshape of current target value
            self.current_target_value = self.current_target_value.reshape(batch_size, para_size,
                                                                          NUM_WEAPONS * NUM_TARGETS)
            # additional weapon to target update
            self.weapon_to_target_assign[batch_id, par_id, weapon_index] = target_index

            # weapon wait_time -- update
            # wait_time = torch.FloatTensor(PREPARATION_TIME[:NUM_WEAPONS]).to(DEVICE)
            wait_time = self.prep_time_buffer.view(1, 1, -1).expand(batch_size, para_size, -1)
            # wait_time = wait_time[None, None, :].expand(batch_size, para_size, wait_time.size(0))
            self.weapon_wait_time[batch_id, par_id, weapon_index] = wait_time[batch_id, par_id, weapon_index]

            # weapon availability update
            self.weapon_availability[(self.weapon_wait_time > 0) | (self.amm_availability <= 0)] = 0.0
            self.n_target_hit[batch_id, par_id, target_index] += 1
            # self.n_target_hit[batch_id, par_id, target_index] = self.n_target_hit.clone()[batch_id, par_id, target_index]+1

            # ammunition update
            # self.amm_availability = self.amm_availability.clone()
            self.amm_availability[batch_id, par_id, weapon_index] -= 1

            # number of fires update
            # self.n_fires = self.n_fires +1
            self.n_fires += 1

            # possible weapon update
            # possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
            self.possible_weapons = self.weapons_arange.view(1, 1, -1).expand(batch_size, para_size, -1)
            # self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
            possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
            self.possible_weapons = possible_weapons_mask.long()

            # print("possible_weapon", self.possible_weapons.shape)
            # a = input()

            # update target availability
            # target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
            self.target_availability = self.target_arange.view(1, 1, -1).expand(batch_size, para_size, -1)
            # self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
            # print("self.target_start_time", self.target_start_time)
            # print("self.target_end_time", self.target_end_time)
            # a = input()

            possible_target_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
            self.target_availability = possible_target_mask.long()

            # print("possible_target", self.target_availability)

            # update available actions
            # available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)
            # available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)
            # In update_internal_variables:
            self._available_actions_buffer[:batch_size, :para_size].fill_(1.0)
            available_actions = self._available_actions_buffer[:batch_size, :para_size]

            # print("available_actions", available_actions)
            # a = input()

            if torch.any(self.possible_weapons <= 0):
                unavailable_weapon = torch.where((self.possible_weapons <= 0))
                unavailable_weapon_batch_index = unavailable_weapon[0]
                unavailable_weapon_par_index = unavailable_weapon[1]
                unavailable_weapon_w_index = unavailable_weapon[2]
                available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index,
                unavailable_weapon_w_index, :] = 0.0

            if torch.any((self.target_availability <= 0)):
                unavailable_target = torch.where((self.target_availability <= 0))
                unavailable_target_batch_index = unavailable_target[0]
                unavailable_target_par_index = unavailable_target[1]
                unavailable_target_t_index = unavailable_target[2]
                available_actions[unavailable_target_batch_index, unavailable_target_par_index, :,
                unavailable_target_t_index] = 0.0

            # print(self.available_actions.shape)
            # print(available_actions.shape)

            # self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
            # print(self.available_actions)
            # a = input()

            # self.mask[:, :, :-1] = self.available_actions.clone()

            self.available_actions = available_actions.view(batch_size, para_size, -1)
            self.mask[:, :, :-1] = self.available_actions

            """ factors : additiona,  wait_to_ready, Number_of_fired, time_target_start, active_time, target_value, probability"""
            # State Update
            assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1, :].detach().clone()
            assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size,
                                                                                                  para_size,
                                                                                                  NUM_WEAPONS,
                                                                                                  NUM_TARGETS, -1)

            # remaining ammunition update
            remaining_ammunition = self.amm_availability.clone()[:, :, :, None].expand(batch_size, para_size,
                                                                                       NUM_WEAPONS,
                                                                                       NUM_TARGETS) / max(AMM)
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 0] = remaining_ammunition[
                                                                                          batch_id, par_id,
                                                                                          weapon_index, :].float()

            # weapon availability update
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 1] = 0.0

            # number of fired
            n_target_hit = self.n_target_hit.clone()[:, :, None, :].expand(batch_size, para_size, NUM_WEAPONS,
                                                                           NUM_TARGETS) / (NUM_TARGETS * MAX_TIME)
            assignment_encoding_without_no_action[batch_id, par_id, :, target_index, 4] = n_target_hit[batch_id,
                                                                                          par_id, :, target_index]

            # target value
            assignment_encoding_without_no_action[:, :, :, :, -2] = self.current_target_value.clone().reshape(
                batch_size, para_size, NUM_WEAPONS, NUM_TARGETS) / MAX_TARGET_VALUE

            # reshape
            self.assignment_encoding[:, :, :-1, :] = assignment_encoding_without_no_action.reshape(batch_size,
                                                                                                   para_size,
                                                                                                   NUM_WEAPONS * NUM_TARGETS,
                                                                                                   -1)

        else:
            self.n_fires = self.n_fires + 1

        self.mask[:, :, -1] = 1.0

    def update_internal_variables_(self, selected_action):

        batch_size = selected_action.size(0)
        para_size = selected_action.size(1)

        if (selected_action < NUM_WEAPONS * NUM_TARGETS).any():

            selected_action_index = torch.where(selected_action < NUM_WEAPONS * NUM_TARGETS)

            # print(selected_action_index)
            # a = input()

            # batch index, parallel index
            batch_id = selected_action_index[0]
            par_id = selected_action_index[1]

            # weapon, target index
            weapon_index = selected_action[batch_id, par_id] // NUM_TARGETS
            target_index = selected_action[batch_id, par_id] % NUM_TARGETS

            # print("weapon_index", weapon_index)
            # print("target_index", target_index)
            # print("self_a", self.available_actions.shape)
            # a = input()

            self.current_target_value = self.current_target_value.reshape(batch_size, para_size, NUM_WEAPONS,
                                                                          NUM_TARGETS)
            # dim [batch, par, weapon, target]
            reduced_values = self.current_target_value[batch_id, par_id, weapon_index, target_index] * \
                             self.weapon_to_target_prob[batch_id, par_id, weapon_index, target_index]
            # shape: dim selected actions
            reduced_values = reduced_values.clone()[:, None].expand(reduced_values.size(0), NUM_WEAPONS)
            self.current_target_value.clone()[batch_id, par_id, :, target_index] = self.current_target_value[
                                                                                   batch_id, par_id, :,
                                                                                   target_index] - reduced_values.float()
            # reshape of current target value
            self.current_target_value = self.current_target_value.reshape(batch_size, para_size,
                                                                          NUM_WEAPONS * NUM_TARGETS)
            # additional weapon to target update
            self.weapon_to_target_assign.clone()[batch_id, par_id, weapon_index] = target_index

            # weapon wait_time -- update
            wait_time_3d = self.prep_time_buffer.view(1, 1, -1).expand(batch_size, para_size, -1)

            # Then assign as before:
            self.weapon_wait_time.clone()[batch_id, par_id, weapon_index] = wait_time_3d[
                batch_id, par_id, weapon_index]

            #
            # wait_time = torch.FloatTensor(PREPARATION_TIME[:NUM_WEAPONS]).to(DEVICE)
            # wait_time = wait_time[None, None, :].expand(batch_size, para_size, wait_time.size(0))
            # self.weapon_wait_time.clone()[batch_id, par_id, weapon_index] = wait_time[batch_id, par_id, weapon_index]

            # weapon availability update
            self.weapon_availability.clone()[(self.weapon_wait_time > 0) | (self.amm_availability <= 0)] = 0.0
            self.n_target_hit.clone()[batch_id, par_id, target_index] = self.n_target_hit.clone()[
                                                                            batch_id, par_id, target_index] + 1

            # ammunition update
            self.amm_availability = self.amm_availability.clone()
            self.amm_availability[batch_id, par_id, weapon_index] -= 1

            # number of fires update
            self.n_fires = self.n_fires + 1

            # possible weapon update
            possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
            self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size,
                                                                           possible_weapons.size(0))
            possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
            self.possible_weapons = possible_weapons_mask.long()

            # print("possible_weapon", self.possible_weapons.shape)
            # a = input()

            # update target availability
            target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
            self.target_availability = target_availability[None, None, :].expand(batch_size, para_size,
                                                                                 target_availability.size(0))
            # print("self.target_start_time", self.target_start_time)
            # print("self.target_end_time", self.target_end_time)
            # a = input()

            possible_target_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
            self.target_availability = possible_target_mask.long()

            # print("possible_target", self.target_availability)

            # update available actions
            available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS),
                                           fill_value=1.0).to(DEVICE)
            available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1),
                                                          NUM_WEAPONS, NUM_TARGETS)

            # print("available_actions", available_actions)
            # a = input()

            if torch.any(self.possible_weapons <= 0):
                unavailable_weapon = torch.where((self.possible_weapons <= 0))
                unavailable_weapon_batch_index = unavailable_weapon[0]
                unavailable_weapon_par_index = unavailable_weapon[1]
                unavailable_weapon_w_index = unavailable_weapon[2]
                available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index,
                unavailable_weapon_w_index, :] = 0.0

            if torch.any((self.target_availability <= 0)):
                unavailable_target = torch.where((self.target_availability <= 0))
                unavailable_target_batch_index = unavailable_target[0]
                unavailable_target_par_index = unavailable_target[1]
                unavailable_target_t_index = unavailable_target[2]
                available_actions[unavailable_target_batch_index, unavailable_target_par_index, :,
                unavailable_target_t_index] = 0.0

            self.available_actions = available_actions.reshape(batch_size, para_size, -1).clone()
            # print(self.available_actions)
            # a = input()

            self.mask.clone()[:, :, :-1] = self.available_actions.clone()

            """ factors : additiona,  wait_to_ready, Number_of_fired, time_target_start, active_time, target_value, probability"""
            # State Update
            assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1, :].detach().clone()
            assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size,
                                                                                                  para_size,
                                                                                                  NUM_WEAPONS,
                                                                                                  NUM_TARGETS, -1)

            # remaining ammunition update
            remaining_ammunition = self.amm_availability.clone()[:, :, :, None].expand(batch_size, para_size,
                                                                                       NUM_WEAPONS,
                                                                                       NUM_TARGETS) / max(AMM)
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 0] = remaining_ammunition[
                                                                                          batch_id, par_id,
                                                                                          weapon_index, :].float()

            # weapon availability update
            assignment_encoding_without_no_action[batch_id, par_id, weapon_index, :, 1] = 0.0

            # number of fired
            n_target_hit = self.n_target_hit.clone()[:, :, None, :].expand(batch_size, para_size, NUM_WEAPONS,
                                                                           NUM_TARGETS) / (NUM_TARGETS * MAX_TIME)
            assignment_encoding_without_no_action[batch_id, par_id, :, target_index, 4] = n_target_hit[batch_id,
                                                                                          par_id, :, target_index]

            # target value
            assignment_encoding_without_no_action[:, :, :, :, -2] = self.current_target_value.clone().reshape(
                batch_size, para_size, NUM_WEAPONS, NUM_TARGETS) / MAX_TARGET_VALUE

            # reshape
            self.assignment_encoding.clone()[:, :, :-1, :] = assignment_encoding_without_no_action.reshape(
                batch_size, para_size, NUM_WEAPONS * NUM_TARGETS, -1)

        else:
            self.n_fires = self.n_fires + 1

        self.mask.clone()[:, :, -1] = 1.0

    def time_update(self):

        # print("hey time updated")
        # a = input()

        # move one clock
        self.clock = self.clock + 1
        # weapon_wait_time update
        self.weapon_wait_time[self.weapon_wait_time > 0] -= 1
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
        self.weapon_availability[(self.weapon_wait_time == 0) & (self.amm_availability > 0)] = 1.0
        # possible_weapons = torch.arange(0, NUM_WEAPONS).int().to(DEVICE)
        self.possible_weapons = self.weapons_arange.view(1, 1, -1).expand(batch_size, para_size, -1)

        # self.possible_weapons = possible_weapons[None, None, :].expand(batch_size, para_size, possible_weapons.size(0))
        possible_weapons_mask = (self.weapon_wait_time <= 0) & (self.amm_availability > 0)
        self.possible_weapons = possible_weapons_mask.long()

        # self.possible_weapons = ((self.weapon_wait_time <= 0) & (self.amm_availability > 0)).long()

        # target_availability = torch.arange(0, NUM_TARGETS).int().to(DEVICE)
        self.target_availability = self.target_arange.view(1, 1, -1).expand(batch_size, para_size, -1)
        # self.target_availability = target_availability[None, None, :].expand(batch_size, para_size, target_availability.size(0))
        # if torch.any(self.target_active_window > 0):
        possible_target_mask = (self.target_start_time <= self.clock) & (self.target_end_time >= self.clock)
        self.target_availability = possible_target_mask.long()

        # update available actions and mask based on the possible weapons and targets

        available_actions = torch.full(size=(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS), fill_value=1.0).to(DEVICE)




        available_actions = available_actions.reshape(available_actions.size(0), available_actions.size(1), NUM_WEAPONS, NUM_TARGETS)

        if torch.any((self.possible_weapons <= 0)):
            unavailable_weapon = torch.where((self.possible_weapons <= 0))
            unavailable_weapon_batch_index, unavailable_weapon_par_index, unavailable_weapon_w_index = \
            unavailable_weapon[0], unavailable_weapon[1], unavailable_weapon[2]
            available_actions[unavailable_weapon_batch_index, unavailable_weapon_par_index,
            unavailable_weapon_w_index, :] = 0.0

        if torch.any((self.target_availability <= 0)):
            unavailable_target = torch.where((self.target_availability <= 0))
            unavailable_target_batch_index, unavailable_target_par_index, unavailable_target_t_index = \
            unavailable_target[0], unavailable_target[1], unavailable_target[2]
            available_actions[unavailable_target_batch_index, unavailable_target_par_index, :,
            unavailable_target_t_index] = 0.0

        self.available_actions = available_actions.reshape(batch_size, para_size, -1)
        self.mask[:, :, :-1] = self.available_actions

        # if self.clock == MAX_TIME:
        self.mask[:, :, -1] = 1.0
        # else:
        #     pass

        # state update ---------------------------------------------
        assignment_encoding_without_no_action = self.assignment_encoding[:, :, :-1, :].clone()
        assignment_encoding_without_no_action = assignment_encoding_without_no_action.reshape(batch_size, para_size,
                                                                                              NUM_WEAPONS,
                                                                                              NUM_TARGETS, -1)

        # available weapon update
        available_weapon_index = torch.where(self.weapon_availability == 1)
        # Weapon availability update:
        assignment_encoding_without_no_action[available_weapon_index[0], available_weapon_index[1],
        available_weapon_index[2], :, 1] = 1.0

        # weapon wait time update
        updated_wait_time = self.weapon_wait_time.clone()[:, :, :, None].expand(batch_size, para_size, NUM_WEAPONS,
                                                                                NUM_TARGETS) / max(PREPARATION_TIME)
        assignment_encoding_without_no_action[:, :, :, :, 3] = updated_wait_time

        # time left
        assignment_encoding_without_no_action[:, :, :, :, 2] = self.time_left / MAX_TIME

        self.assignment_encoding[:, :, :-1, :] = assignment_encoding_without_no_action.reshape(batch_size,      para_size,   NUM_WEAPONS * NUM_TARGETS,     -1)

        return 0

    def select_best_actions(self, selected_actions, initial):

        # print("selected_actions", selected_actions)
        # print("self---", self.current_target_value)
        # a = input()


        # Find unique rows and their inverse indices for mapping
        beam_results = self.beam_result.to('cpu')
        # checking unique rows and indices
        unique_rows, inverse_indices = torch.unique(beam_results, dim=0, return_inverse=True)

        # print("unique_rows", unique_rows)
        # print("inverse_index", inverse_indices)
        # a = input()

        # Extract the indices of the original tensor corresponding to the unique rows
        original_indices = []
        for unique_row in unique_rows:
            # Find the index of the first occurrence of each unique row in the original tensor
            index = (beam_results == unique_row).all(dim=1).nonzero(as_tuple=True)[0][0]
            original_indices.append(index)
        original_indices = torch.tensor(original_indices)
        intermediate_index = torch.sort(original_indices).values
        unique_tensor = torch.unique(intermediate_index)
        # print("unique_tensor", unique_tensor)
        # a = input()

        unique_rows = beam_results[unique_tensor, :]
        flattened_unique = unique_rows.flatten()

        # print("flattened_unique_tensor", flattened_unique)
        # print("unqie_rows", unique_rows)
        # a = input()

        # else:
        #     unique_tensor = torch.arange(len(beam_results))
        #     print("u-tensor", unique_tensor)
        #     unique_rows = beam_results.clone()
        #     print("unique_rows", unique_rows)
        #     flattened_unique = unique_rows.flatten()
        #     print("----------flattened_unique_tensor", flattened_unique)
        #     a = input()


        sorted_values, sorted_indices = flattened_unique.sort(stable=True)
        # print("sorted_values", sorted_values)
        # print("sorted_indices", sorted_indices)
        # a = input()
        # k = alpha
        # top_values = sorted_values[:k]
        # top_indices = sorted_indices[:k]
        # print("top_values", top_values)
        # print("top_indices", top_indices)

        # #top_indices = torch.tensor(top_indices)

        # --------------------------------------------------------------------------------------------------
        # # Remove duplicates and get indices
        # unique_sorted_values, inverse_indices = torch.unique(sorted_values, return_inverse=True, sorted=True)
        # print("unique_sorted_values", unique_sorted_values)
        # print("inverse-unique", inverse_indices)
        #
        # #
        # # # Find the first occurrence index for each unique value
        # first_occurrence_indices = []
        # for val in unique_sorted_values:
        #     idx = (sorted_values == val).nonzero(as_tuple=True)[0][0]
        #     first_occurrence_indices.append(sorted_indices[idx])
        #
        # # Convert list to tensor
        # first_occurrence_indices = torch.tensor(first_occurrence_indices)

        #----------------------------------------------------------------
        # 1) Get unique values (sorted)
        unique_vals = torch.unique(sorted_values, sorted=True)

        # 2) For each unique value, find the first occurrence index from 'sorted_indices'
        first_occ_indices = []
        for val in unique_vals:
            idx = (sorted_values == val).nonzero(as_tuple=True)[0][0]
            first_occ_indices.append(sorted_indices[idx].item())

        # 3) If fewer than 5 unique, replicate the earliest until we have 5
        max_slots = alpha
        if len(first_occ_indices) < max_slots:
            # Number needed to reach 5
            needed = max_slots - len(first_occ_indices)
            earliest_idx = first_occ_indices[0]
            # Insert the earliest index as many times as needed
            for _ in range(needed):
                # Insert right after the first element, so the earliest stays in front
                first_occ_indices.insert(1, earliest_idx)
        first_occ_indices = torch.tensor(first_occ_indices)

        k = alpha
        top_values = unique_vals[:k]
        top_indices = first_occ_indices[:k]



        # print("top_values", top_values)
        # print("top_indices", top_indices)
        # a = input()
        #------------------------------------------------------------------------------------------

        # Take the top-k smallest unique values
        # k = alpha
        # top_values = unique_sorted_values[:k]
        # top_indices = first_occurrence_indices[:k]
        #--------------------------------------------------------------------------------------------------
        # print("top_indices", top_indices)
        # print("top_values", top_values)
        # a = input()


        # print(top_values)
        # a = input()
        # Step 3: Convert flat indices to [batch, group] indices in unique rows
        unique_batch_indices = top_indices // unique_rows.size(1)
        unique_group_indices = top_indices % unique_rows.size(1)
        unique_batch_indices = unique_tensor[unique_batch_indices]

        final_batch_indices = unique_batch_indices
        final_group_indices = unique_group_indices

        # Ensure all results are back on the original device
        final_batch_indices = final_batch_indices.to(beam_results.device)
        final_group_indices = final_group_indices.to(beam_results.device)
        top_values = top_values.to(beam_results.device)


        # print("final_batch_index", final_batch_indices)
        # print("final_group_index", final_group_indices)
        # a = input()

        batch_indices = final_batch_indices
        group_indices = selected_actions[final_batch_indices, final_group_indices]

        # print("node_index", selected_actions)
        # print("batch", batch_indices)
        # print("group_index", group_indices)
        # a = input()

        return batch_indices, group_indices


    def do_beam_simulation(self, possible_node_index, time, w_index):
        """ This part main beam search part"""

        # print("sibal------", possible_node_index)
        # print("current_value ==", self.current_target_value[:, :, 0:NUM_TARGETS].sum(2))
        # a = input()

        # set-up for simulation
        if time + w_index == 0:
            initial = 'yes'
        else:
            initial = 'no'

        start_time = self.n_fires // NUM_WEAPONS
        start_weapon = self.n_fires % NUM_WEAPONS

        # print("start_weapon", start_weapon)
        # print("start_time", start_time)

        if start_weapon == 0:
            self.time_update()


        # left_over_step = NUM_WEAPONS*MAX_TIME - self.n_fires
        # limit = min(n_step, left_over_step)
        # iteration_count = 1

        for time_clock in range(start_time, MAX_TIME):
            # print("time_clock", time_clock)
            # if iteration_count >= limit:
            #     break
            # weapon_processed = 0

            for index in range(start_weapon, NUM_WEAPONS):
                # print("weapon_index", index)

                policy,  _ = self.policy(assignment_embedding=self.assignment_encoding,  prob=self.weapon_to_target_prob, mask=self.mask)
                #print(policy)
                action_index = policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).argmax(dim=1).view(self.assignment_encoding.size(0), self.assignment_encoding.size(1))
                self.update_internal_variables(selected_action=action_index)

            self.time_update()
            start_weapon = 0

        # value_traverse = 1 - self.current_target_value[:, :, 0:NUM_TARGETS].sum(2) / self.original_target_value[:, :, 0:NUM_TARGETS].sum(2)
        # to_go_value = self.to_go_value(state=self.assignment_encoding.clone().detach(), mask=self.mask.clone().detach())
        # q_value = -1 * (value_traverse + to_go_value)

        self.beam_result = self.current_target_value[:, :, 0:NUM_TARGETS].sum(2)
        #print("sofar-----------------", self.beam_result.min())
        batch, group = self.select_best_actions(selected_actions = possible_node_index, initial = initial)
        # a = input()


        # print("batch",batch)
        # print("group", group)
        # a =input()

        return batch, group


def batch_dimension_resize(env, batch_index, group_index):
    # 1) encoding
    env.assignment_encoding = env.assignment_encoding[batch_index, :, :, :].clone()
    # 2) weapon-to-target probability
    env.weapon_to_target_prob = env.weapon_to_target_prob[batch_index, :, :, :].clone()
    # 3) weapon-to-target assign
    env.weapon_to_target_assign = env.weapon_to_target_assign[batch_index, :, :].clone()
    # 4) current_target_value
    env.current_target_value = env.current_target_value[batch_index, :, :].clone()
    # 5) current_target_value
    env.original_target_value = env.original_target_value[batch_index, :, :].clone()
    # 6) possible_weapons
    env.possible_weapons = env.possible_weapons[batch_index, :, :].clone()
    # 7) available actions
    env.available_actions = env.available_actions[batch_index, :, :].clone()
    # 8) mask
    env.mask = env.mask[batch_index, :, :].clone()
    # 9) weapon availability
    env.weapon_availability = env.weapon_availability[batch_index, :, :].clone()
    # 10) ammunition availability
    env.amm_availability = env.amm_availability[batch_index, :, :].clone()
    # 11) waiting time for each weapon
    env.weapon_wait_time = env.weapon_wait_time[batch_index, :, :].clone()
    # 13) target_availability
    env.target_availability = env.target_availability[batch_index, :, :].clone()
    # 14) target_start_time
    env.target_start_time = env.target_start_time[batch_index, :, :].clone()
    # 15) target_end time
    env.target_end_time = env.target_end_time[batch_index, :, :].clone()
    # 16) target hit
    env.n_target_hit = env.n_target_hit[batch_index, :, :].clone()

    #
    # print("resize----", env.current_target_value)
    # a = input()




    return env






