"""
Sequential (originally-published BReRLA) training loop -- the comparison
point for the parallel multi-pointer decoder trained in Dynamic_Sampling_GNN.py.

Same REINFORCE algorithm (reward-to-go + potential-based shaping, POMO-style
per-step advantage normalization) and same hyperparameters as the parallel
version -- the only difference is decoding structure: instead of one actor
forward call per time step deciding all weapons simultaneously, this calls
the actor once per (time_step, decision) pair -- M decision opportunities
per time step, each immediately updating environment state via the existing
single-edge `env.update_internal_variables`, matching the architecture
described in BReRLA_revision_plan.md:112.
"""

import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.utilities import Average_Meter
from rl.Dynamic_Sampling_GNN import (
    get_random_problem_size,
    patch_hyperparameters_for_epoch,
    restore_hyperparameters,
)


def _flat_mask_to_critic_mask(flat_mask, num_weapons, num_targets):
    """
    Reshape the flat [batch, para, W*T+1] mask (env.mask) into the
    [batch, para, W, T+1] shape EdgeAwareGNN_CRITIC expects, so the existing
    critic can be reused unchanged. Only the first T columns per weapon row
    (edge legality, used for masked pooling) carry real information; the
    critic never reads the values in the last column, only its presence
    (to infer num_targets from mask.size(3)-1), so it's padded with ones.
    """
    batch_size, para_size = flat_mask.shape[:2]
    edge_part = flat_mask[:, :, :-1].view(batch_size, para_size, num_weapons, num_targets)
    noop_pad = torch.ones(batch_size, para_size, num_weapons, 1, device=flat_mask.device)
    return torch.cat([edge_part, noop_pad], dim=-1)


def self_play_gnn(old_actor, actor, critic, episode, temp, epoch, logger=None, ablation=None):
    """
    Multi-episodic REINFORCE training for the sequential GNN actor, with
    random multi-scale training. Mirrors Dynamic_Sampling_GNN.self_play_gnn
    exactly except for the inner decision loop (see module docstring).

    ablation: None (full model) or a comma-separated combination of
        'no_critic' (no critic, no shaping, plain reward-to-go/whole-episode
        REINFORCE) and 'no_reward_to_go' (whole-episode return applied to
        every step instead of per-step reward-to-go).
    """
    active_ablations = set(ablation.split(',')) if ablation else set()

    try:
        actor.train()
        critic.train()

        actor_losses = Average_Meter()
        critic_losses = Average_Meter()

        entropy_coef = 1e-3 if 'ENTROPY_COEF' not in globals() else ENTROPY_COEF

        total_entropy = 0.0
        total_steps = 0
        total_objective = 0.0
        total_destruction_ratio = 0.0

        for ep in range(episode):
            ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list, ep_prep_list, ep_cost_list = get_random_problem_size(epoch, ep)

            ep_alpha = 1.0

            original_hyperparams = patch_hyperparameters_for_epoch(
                ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list,
                prep_list=ep_prep_list, cost_list=ep_cost_list,
            )

            try:
                assignment_encoding, weapon_to_target_prob = input_generation(
                    NUM_WEAPON=ep_num_weapons,
                    NUM_TARGET=ep_num_targets,
                    value=None,
                    prob=None,
                    TW=None,
                    max_time=ep_max_time,
                    batch_size=TRAIN_BATCH,
                    alpha=ep_alpha,
                )

                assignment_encoding = assignment_encoding.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
                weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()

                env = Environment(
                    assignment_encoding=assignment_encoding,
                    weapon_to_target_prob=weapon_to_target_prob,
                    max_time=ep_max_time
                )

                if logger is not None:
                    try:
                        ts = env.target_start_time[0, 0, :ep_num_targets].detach().cpu().tolist()
                        te = env.target_end_time[0, 0, :ep_num_targets].detach().cpu().tolist()
                        logger.info(f"Target time windows (batch0): start={ts}, end={te}")
                    except Exception:
                        pass

                log_probs = []
                values = []
                entropies = []
                rewards = []  # per-decision immediate reward (one entry per single-edge pick)

                original_value = env.original_target_value[:, :, 0:ep_num_targets].sum(2)  # [batch, para]
                prev_value = original_value.clone()

                b_sz, p_sz = assignment_encoding.shape[:2]
                num_actions = ep_num_weapons * ep_num_targets + 1

                # Execute episode: M unrestricted decision opportunities per time
                # step (original sequential architecture -- see module docstring),
                # each immediately updating env state; time advances once per
                # time step, after all M decisions.
                for time_step in range(ep_max_time):
                    for decision in range(ep_num_weapons):
                        current_state = env.assignment_encoding.clone()
                        current_prob = env.weapon_to_target_prob.clone()

                        # Policy: [batch, para, W*T+1] -- ONE joint distribution
                        # over every (weapon,target) edge plus the single global
                        # no-op, computed from the CURRENT (already-updated) state.
                        policy, _ = actor(
                            assignment_embedding=current_state,
                            prob=current_prob,
                            mask=env.mask.clone()
                        )

                        safe_policy = policy.clamp_min(1e-8)
                        step_entropy = -(safe_policy * safe_policy.log()).sum(dim=-1)  # [batch, para]
                        entropies.append(step_entropy)
                        total_entropy += step_entropy.mean().item()
                        total_steps += 1

                        flat_policy = policy.reshape(-1, num_actions)
                        action = torch.multinomial(flat_policy, 1).view(b_sz, p_sz)  # [batch, para]

                        value = torch.zeros(b_sz, p_sz, 1, device=current_state.device, dtype=current_state.dtype)

                        log_prob = torch.log(
                            policy.gather(-1, action.unsqueeze(-1)).clamp_min(1e-8)
                        ).squeeze(-1)  # [batch, para]

                        values.append(value.clone())
                        log_probs.append(log_prob.unsqueeze(-1))  # [batch, para, 1] for shape parity below

                        # Env step: apply this ONE (weapon,target) decision only.
                        env.update_internal_variables(selected_action=action)

                        curr_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)
                        step_reward = (prev_value - curr_value) / (original_value + 1e-8)
                        rewards.append(step_reward)
                        prev_value = curr_value

                    # Advance time once per time step, after all M decisions.
                    env.time_update()

                final_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)  # [batch, para]

                destruction_ratio = 1 - (final_value / (original_value + 1e-8))
                returns = destruction_ratio

                # No critic, no potential-based shaping -- plain REINFORCE (Section
                # "NEVER EVER USE CRITIC IN THE REINFORCE" directive). Raw step reward used as-is.
                gamma = 1.0
                T_steps = len(rewards)
                shaped_rewards = rewards

                shaped_returns = [None] * T_steps
                if 'no_reward_to_go' in active_ablations:
                    # Whole-episode return applied uniformly to every step (POMO-style),
                    # instead of per-step reward-to-go -- mirrors Dynamic_Sampling_GNN.py's
                    # (parallel) no_reward_to_go ablation, added 2026-08-06 to allow a
                    # matched sequential-vs-parallel comparison under the same credit style.
                    total_return = sum(shaped_rewards)
                    for t in range(T_steps):
                        shaped_returns[t] = total_return
                else:
                    running = torch.zeros_like(shaped_rewards[0])
                    for t in reversed(range(T_steps)):
                        running = shaped_rewards[t] + gamma * running
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

                critic_loss = torch.tensor(0.0)

                actor_losses.push(torch.tensor(actor_loss.item()), 1)
                critic_losses.push(torch.tensor(critic_loss.item()), 1)

                total_objective += final_value.mean().item()
                total_destruction_ratio += returns.mean().item()

                if logger is not None:
                    logger.info(
                        f"Episode {ep+1}/{episode} | Actor Loss: {actor_loss.item():.6f} | "
                        f"Critic Loss: {critic_loss.item():.6f}"
                    )

            finally:
                restore_hyperparameters(original_hyperparams)

        avg_entropy = total_entropy / max(total_steps, 1)

        epoch_objective = total_objective / episode
        avg_destruction = total_destruction_ratio / episode

        return actor_losses.result(), critic_losses.result(), {
            'num_weapons': 'mixed',
            'num_targets': 'mixed',
            'max_time': 'mixed',
            'amm': 'mixed',
            'objective': epoch_objective,
            'destruction_ratio': avg_destruction,
        }

    except Exception as e:
        print(f"Error in self_play_gnn (sequential): {e}")
        raise
