"""
Multi-scale self-play training for the Sequential actor
(common/DWTA_GNN_sequential.py::create_gnn_actor_sequential) on the MODERATE
temporal-dilemma curriculum -- the Sequential-decoder counterpart to
rl/Dynamic_Sampling_GNN_moderate.py (Parallel/SCoPE), same rationale: fair
apples-to-apples comparison against AM/POMO/SCoPE, all trained M,N,T ~
U[5,7] multi-scale and zero-shot evaluated on 12 much-larger held-out
configs.

New file, does not modify rl/Dynamic_Sampling_GNN_sequential.py at all --
same REINFORCE math (reward-to-go, per-step POMO-para baseline
normalization, grad-norm clipping, no critic ever) and same ammo-utilization
terminal bonus added to the Parallel/SCoPE moderate variant (see that
file's docstring: SCIP-vs-model diagnostic found systematic under-firing,
not bad temporal judgment), only the instance generator and decode loop
structure (single flat W*T+1 action space, one edge decided at a time) are
Sequential-specific.
"""
import os
import random
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation_moderate import generate_moderate_training_instances
from common.DWTA_Simulator import Environment
from common.utilities import Average_Meter

from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters
from Dynamic_Sampling_GNN_moderate import get_random_moderate_problem_size


def self_play_gnn_sequential_moderate(actor, episode, epoch, logger=None):
    """Sequential-decoder counterpart to
    Dynamic_Sampling_GNN_moderate.py::self_play_gnn_moderate. No critic,
    ever."""
    try:
        actor.train()
        actor_losses = Average_Meter()
        entropy_coef = 1e-3 if 'ENTROPY_COEF' not in globals() else ENTROPY_COEF

        total_entropy = 0.0
        total_steps = 0
        total_objective = 0.0
        total_destruction_ratio = 0.0

        for ep in range(episode):
            ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list, ep_prep_list, ep_cost_list = get_random_moderate_problem_size()

            original_hyperparams = patch_hyperparameters_for_epoch(
                ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list,
                prep_list=ep_prep_list, cost_list=ep_cost_list,
            )

            try:
                assignment_encoding, weapon_to_target_prob = generate_moderate_training_instances(
                    batch_size=TRAIN_BATCH,
                    num_weapons=ep_num_weapons,
                    num_targets=ep_num_targets,
                    max_time=ep_max_time,
                    amm_list=ep_amm_list,
                )

                assignment_encoding = assignment_encoding.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
                weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()

                env = Environment(
                    assignment_encoding=assignment_encoding,
                    weapon_to_target_prob=weapon_to_target_prob,
                    max_time=ep_max_time,
                )

                log_probs = []
                entropies = []
                rewards = []
                total_fires = None  # [batch, para], accumulated across all decisions

                original_value = env.original_target_value[:, :, 0:ep_num_targets].sum(2)
                prev_value = original_value.clone()

                b_sz, p_sz = assignment_encoding.shape[:2]
                num_edges = ep_num_weapons * ep_num_targets
                num_actions = num_edges + 1

                for time_step in range(ep_max_time):
                    for decision in range(ep_num_weapons):
                        current_state = env.assignment_encoding.clone()
                        current_prob = env.weapon_to_target_prob.clone()

                        policy, _ = actor(
                            assignment_embedding=current_state,
                            prob=current_prob,
                            mask=env.mask.clone()
                        )

                        safe_policy = policy.clamp_min(1e-8)
                        step_entropy = -(safe_policy * safe_policy.log()).sum(dim=-1)
                        entropies.append(step_entropy)
                        total_entropy += step_entropy.mean().item()
                        total_steps += 1

                        flat_policy = policy.reshape(-1, num_actions)
                        action = torch.multinomial(flat_policy, 1).view(b_sz, p_sz)

                        log_prob = torch.log(
                            policy.gather(-1, action.unsqueeze(-1)).clamp_min(1e-8)
                        ).squeeze(-1)
                        log_probs.append(log_prob.unsqueeze(-1))

                        fires_this_decision = (action < num_edges).float()  # [batch, para]
                        total_fires = fires_this_decision if total_fires is None else total_fires + fires_this_decision

                        env.update_internal_variables(selected_action=action)

                        curr_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)
                        step_reward = (prev_value - curr_value) / (original_value + 1e-8)
                        rewards.append(step_reward)
                        prev_value = curr_value

                    env.time_update()

                # Same ammo-utilization terminal bonus as the Parallel/SCoPE
                # moderate variant -- see that file's docstring for why.
                total_ammo = float(sum(ep_amm_list))
                ammo_utilization = total_fires / max(total_ammo, 1.0)  # [batch, para]
                ammo_bonus_coef = 0.1
                rewards[-1] = rewards[-1] + ammo_bonus_coef * ammo_utilization

                final_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)
                destruction_ratio = 1 - (final_value / (original_value + 1e-8))

                gamma = 1.0
                T_steps = len(rewards)
                running = torch.zeros_like(rewards[0])
                shaped_returns = [None] * T_steps
                for t in reversed(range(T_steps)):
                    running = rewards[t] + gamma * running
                    shaped_returns[t] = running

                advantages = []
                for G_t in shaped_returns:
                    baseline_t = G_t.mean(dim=1, keepdim=True)
                    adv_t = G_t - baseline_t
                    adv_std_t = adv_t.std(dim=1, keepdim=True).clamp_min(1e-6)
                    advantages.append(adv_t / adv_std_t)

                actor_loss = 0
                for log_prob, adv_t in zip(log_probs, advantages):
                    actor_loss = actor_loss + (-(log_prob.squeeze(-1) * adv_t).mean())
                if entropies:
                    entropy_mean = torch.stack(entropies).mean()
                    actor_loss = actor_loss - entropy_coef * entropy_mean

                actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor.optimizer.step()

                actor_losses.push(torch.tensor(actor_loss.item()), 1)
                total_objective += final_value.mean().item()
                total_destruction_ratio += destruction_ratio.mean().item()

                if logger is not None:
                    logger.info(
                        f"Episode {ep+1}/{episode} ({ep_num_weapons}Mx{ep_num_targets}Nx{ep_max_time}T) "
                        f"| Actor Loss: {actor_loss.item():.6f}"
                    )
            finally:
                restore_hyperparameters(original_hyperparams)

        avg_entropy = total_entropy / max(total_steps, 1)
        return actor_losses.result(), {
            'objective': total_objective / episode,
            'destruction_ratio': total_destruction_ratio / episode,
            'entropy': avg_entropy,
        }

    except Exception as e:
        print(f"Error in self_play_gnn_sequential_moderate: {e}")
        raise
