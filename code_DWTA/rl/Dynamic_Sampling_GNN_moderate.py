"""
Multi-scale self-play training for the FULL N+1-way GNN actor
(common/DWTA_GNN.py::create_gnn_actor) on the MODERATE temporal-dilemma
curriculum -- for a fair, apples-to-apples comparison against RL4CO AM/POMO
(train_rl4co_am_multiscale.py / train_rl4co_pomo_multiscale.py), which are
trained the same way: M,N,T ~ U[5,7] resampled every episode, zero-shot
evaluated on 12 much-larger held-out configs.

New file, does not modify rl/Dynamic_Sampling_GNN.py (self_play_gnn) at all
-- same REINFORCE math (reward-to-go, per-step POMO-para baseline
normalization, grad-norm clipping, no critic ever), only the instance
generator is swapped for generate_moderate_training_instances (see
common/Dynamic_Instance_generation_moderate.py's docstring for why ammo/prep
are kept batch-shared rather than per-instance-random).
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

SCALE_LOW, SCALE_HIGH = 5, 7


def get_random_moderate_problem_size():
    num_weapons = random.randint(SCALE_LOW, SCALE_HIGH)
    num_targets = random.randint(SCALE_LOW, SCALE_HIGH)
    max_time = random.randint(SCALE_LOW, SCALE_HIGH)
    amm_list = [random.randint(1, max(2, max_time // 2 + 1)) for _ in range(num_weapons)]
    prep_list = [random.randint(0, 1) for _ in range(num_weapons)]
    cost_list = [random.randint(2, 10) * 10 for _ in range(num_weapons)]
    return (num_weapons, num_targets, max_time, amm_list, prep_list, cost_list)


def self_play_gnn_moderate(actor, episode, epoch, logger=None):
    """Same structure/REINFORCE math as Dynamic_Sampling_GNN.py::self_play_gnn
    with ablation='no_critic,no_reward_to_go' semantics permanently baked in
    -- no, actually reward-to-go IS kept here (matches the un-ablated
    default), only the instance generator differs. No critic, ever."""
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
                total_fires = None  # [batch, para, 1], accumulated across rounds

                original_value = env.original_target_value[:, :, 0:ep_num_targets].sum(2)
                prev_value = original_value.clone()

                for time_step in range(ep_max_time):
                    current_state = env.assignment_encoding.clone()
                    current_prob = env.weapon_to_target_prob.clone()

                    policy, _ = actor(
                        assignment_embedding=current_state,
                        prob=current_prob,
                        mask=env.mask_per_weapon.clone()
                    )

                    safe_policy = policy.clamp_min(1e-8)
                    step_entropy = -(safe_policy * safe_policy.log()).sum(dim=-1)
                    entropies.append(step_entropy)
                    total_entropy += step_entropy.mean().item()
                    total_steps += 1

                    b_sz, p_sz = current_state.shape[:2]
                    flat_policy = policy.reshape(-1, ep_num_targets + 1)
                    action = torch.multinomial(flat_policy, 1).view(b_sz, p_sz, ep_num_weapons)

                    per_weapon_log_prob = torch.log(
                        policy.gather(-1, action.unsqueeze(-1)).clamp_min(1e-8)
                    ).squeeze(-1)
                    joint_log_prob = per_weapon_log_prob.sum(dim=-1, keepdim=True)
                    log_probs.append(joint_log_prob)

                    env.update_internal_variables_parallel(selected_actions=action)

                    fires_this_round = (action < ep_num_targets).float().sum(dim=-1, keepdim=True)  # [batch, para, 1]
                    total_fires = fires_this_round if total_fires is None else total_fires + fires_this_round

                    curr_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)
                    step_reward = (prev_value - curr_value) / (original_value + 1e-8)
                    rewards.append(step_reward)
                    prev_value = curr_value

                    env.time_update()

                # Ammo-utilization bonus: leaving legal, worthwhile ammo unused
                # is already implicitly penalized via lower destruction, but
                # REINFORCE's credit assignment is weak (esp. this early in
                # training), so add a small explicit terminal bonus for
                # actually using available ammo. Added to the LAST step's
                # reward so reward-to-go propagates it to every earlier
                # decision too (see self_play_gnn_moderate's reward-to-go
                # loop below). NOT a hard constraint -- holding is still free
                # to be the better choice when no worthwhile target exists.
                total_ammo = float(sum(ep_amm_list))
                ammo_utilization = (total_fires.squeeze(-1) / max(total_ammo, 1.0))  # [batch, para], matches rewards[-1]'s shape
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
        print(f"Error in self_play_gnn_moderate: {e}")
        raise
