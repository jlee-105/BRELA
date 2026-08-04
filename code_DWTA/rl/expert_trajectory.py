"""
Expert Trajectory Collection for ExIt (Expert Iteration).

Uses multi-policy truncated beam search to find improved actions at each
(time, weapon) decision point, then stores (state, expert_action, value)
tuples in a replay buffer for imitation learning.
"""
import os
import sys
import torch
import random
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation import input_generation
from rl_rollout.DWTA_Simulator_rollout import Environment
from rl_rollout.BEAM_WITH_SIMULATION_TRUNC_CRITIC import (
    Beam_Search_Trunc,
    batch_dimension_resize_trunc,
)
from .Dynamic_Sampling_GNN import (
    get_random_problem_size,
    patch_hyperparameters_for_epoch,
    restore_hyperparameters,
)


class ExpertBuffer:
    """Buffer of (state, wtp, mask, expert_action, value, nw, nt). Sample by same size. clear() each iter."""

    def __init__(self, capacity=EXIT_EXPERT_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def clear(self):
        self.buffer.clear()

    def push(self, state, wtp, mask, expert_action, value, nw, nt):
        self.buffer.append((
            state.detach().cpu(),
            wtp.detach().cpu(),
            mask.detach().cpu(),
            expert_action.detach().cpu(),
            value,
            int(nw),
            int(nt),
        ))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        by_size = {}
        for b in self.buffer:
            key = (b[5], b[6])
            if key not in by_size:
                by_size[key] = []
            by_size[key].append(b)
        candidates = [v for v in by_size.values() if len(v) >= batch_size]
        if not candidates:
            candidates = list(by_size.values())
        pool = random.choice(candidates) if candidates else []
        n = min(batch_size, len(pool))
        batch = random.sample(pool, n)
        states = torch.stack([b[0] for b in batch]).to(DEVICE)
        wtps = torch.stack([b[1] for b in batch]).to(DEVICE)
        masks = torch.stack([b[2] for b in batch]).to(DEVICE)
        actions = torch.stack([b[3] for b in batch]).to(DEVICE)
        values = torch.tensor([b[4] for b in batch], device=DEVICE, dtype=torch.float32)
        nw, nt = batch[0][5], batch[0][6]
        return states, wtps, masks, actions, values, nw, nt

    def __len__(self):
        return len(self.buffer)


@torch.no_grad()
def collect_expert_trajectories(
    actor,
    critic,
    num_instances=5,
    beta=2,
    to_go_weight=1.0,
    exit_iter=0,
    num_weapons=None,
    num_targets=None,
    max_time=None,
):
    """
    Generate instances with same multi-scale distribution as base RL (get_random_problem_size)
    and run beam search to collect expert (state, action) pairs.
    Returns list of (state, mask, expert_action, episode_value, nw, nt).
    """
    actor.eval()
    critic.eval()
    trajectories = []

    for inst_idx in range(num_instances):
        if num_weapons is not None and num_targets is not None and max_time is not None:
            nw, nt, mt = num_weapons, num_targets, max_time
            ep_amm_list = [2] * nw
            ep_prep_list = [1, 2] * ((nw // 2) + 1)
            ep_cost_list = [50] * nw
            original_hyperparams = patch_hyperparameters_for_epoch(
                nw, nt, mt, ep_amm_list, prep_list=ep_prep_list[:nw], cost_list=ep_cost_list,
            )
        else:
            nw, nt, mt, ep_amm_list, ep_prep_list, ep_cost_list = get_random_problem_size(exit_iter, inst_idx)
            original_hyperparams = patch_hyperparameters_for_epoch(
                nw, nt, mt, ep_amm_list,
                prep_list=ep_prep_list, cost_list=ep_cost_list,
            )
        try:
            ae, wtp = input_generation(
                NUM_WEAPON=nw, NUM_TARGET=nt,
                value=None, prob=None, TW=None,
                max_time=mt, batch_size=1,
            )
            ae = ae[:, None, :, :].expand(1, VAL_PARA, nt * nw + 1, NUM_FEATURES).contiguous()
            wtp = wtp[:, None, :, :].expand(1, VAL_PARA, nw, nt).contiguous()

            env = Environment(
                assignment_encoding=ae,
                weapon_to_target_prob=wtp,
                max_time=mt,
            )
            episode_pairs = []

            for t in range(mt):
                for w in range(nw):
                    state_snap = env.assignment_encoding.clone()
                    wtp_snap = env.weapon_to_target_prob.clone()
                    mask_snap = env.mask.clone()

                    if (mask_snap > 0).any():
                        beam = Beam_Search_Trunc(
                            env=env, actor=actor, value=critic,
                            available_actions=mask_snap,
                            beta=None, to_go_weight=0.0,
                            use_greedy_only=True,
                        )
                        beam.reset()
                        expanded = beam.expand_actions()
                        b_idx, g_idx = beam.do_beam_simulation(
                            possible_node_index=expanded, time=t, w_index=w,
                        )
                        selected_action = g_idx.unsqueeze(1)
                        env = batch_dimension_resize_trunc(
                            env=env, batch_index=b_idx, group_index=g_idx,
                        )
                    else:
                        selected_action = torch.tensor(
                            [nw * nt], device=DEVICE
                        )[None, :].expand(1, VAL_PARA)

                    episode_pairs.append((
                        state_snap.squeeze(0).squeeze(0),
                        wtp_snap.squeeze(0).squeeze(0),
                        mask_snap.squeeze(0).squeeze(0),
                        selected_action.squeeze(0).squeeze(0),
                    ))
                    env.update_internal_variables(selected_action=selected_action)
                env.time_update()

            final_value = env.current_target_value[:, :, 0:nt].sum().item()
            for (s, wtp, m, a) in episode_pairs:
                trajectories.append((s, wtp, m, a, final_value, nw, nt))
        finally:
            restore_hyperparameters(original_hyperparams)

    return trajectories
