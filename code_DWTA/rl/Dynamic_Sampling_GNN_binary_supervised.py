"""
Pure SUPERVISED training for the binary fire/hold actor -- no REINFORCE, no
log-prob, no advantage/baseline at all.

At each round, for each weapon with a legal target, compute a fire-vs-hold
label via short-horizon ROLLOUT: deepcopy the environment, force the
weapon's decision each way (all other weapons follow the actor's own
current greedy baseline), simulate `rollout_k` more rounds using the
actor's own greedy policy + the standard 1:1 auction, and compare final
remaining value. Train fire_prob via binary cross-entropy directly against
that label.

Motivation (2026-08-20 session): every REINFORCE-based attempt at a binary
fire/hold actor this session (and in prior sessions -- see
brerla_auction_train_inference_mismatch memory, Findings 5-6) hit training
instability -- collapse to degenerate always-fire or always-hold policies,
oscillation, or extreme sensitivity to initialization. Diagnosed as
bang-bang/binary action spaces being inherently harder to explore stably
under sparse-reward policy gradients than the N+1-way softmax spaces that
trained comparatively smoothly (CommSCoPE, Sequential, Sinkhorn). This file
sidesteps the credit-assignment problem entirely by replacing the RL reward
signal with a DENSE, DIRECTLY COMPUTED supervised target every single
decision (not a learned critic -- a real simulation, matching the project's
"never use a learned critic" standing rule).

New file -- does not modify Dynamic_Sampling_GNN_binary.py or any existing
training script/checkpoint.
"""
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.utilities import Average_Meter
from common.auction_refinement import auction_round_action

from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters


def get_random_temporal_problem_size():
    num_weapons = random.randint(5, 7)
    num_targets = random.randint(5, 7)
    max_time = random.randint(5, 7)
    return num_weapons, num_targets, max_time


@torch.no_grad()
def _rollout_forward(env, actor, nw, nt, k):
    """Advance env `k` more rounds using the actor's own greedy fire/hold
    decision + 1:1 auction. Used only to score rollout branches."""
    for _ in range(k):
        fire_prob, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = fire_prob > 0.5
        rv = env.current_target_value[:, :, 0:nt]
        pr = env.weapon_to_target_prob[:, :, :nw, :nt]
        lm = env.mask_per_weapon[:, :, :nw, :nt] > 0
        action = auction_round_action(rv, pr, lm, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()


@torch.no_grad()
def compute_rollout_labels(env, actor, must_fire_baseline, nw, nt, mt, t, rollout_k=2):
    """
    For each weapon with a legal target this round, branch fire vs hold
    (other weapons follow `must_fire_baseline`), simulate `rollout_k` more
    rounds with the actor's own greedy policy, and label 1.0 (fire) if the
    fire branch ends with LOWER remaining value (more destroyed) than the
    hold branch, else 0.0.

    Returns:
        labels: [B, P, M] float
        valid:  [B, P, M] bool -- weapons with no legal target this round
            have no meaningful label (excluded from the loss).
    """
    B, P, M = must_fire_baseline.shape
    device = must_fire_baseline.device
    labels = torch.zeros(B, P, M, device=device)

    legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
    valid = legal_mask.any(dim=-1)  # [B, P, M]

    k = max(0, min(rollout_k, mt - t - 1))

    for m in range(M):
        if not valid[:, :, m].any():
            continue

        env_fire = copy.deepcopy(env)
        mf_fire = must_fire_baseline.clone()
        mf_fire[:, :, m] = True
        rv = env_fire.current_target_value[:, :, 0:nt]
        pr = env_fire.weapon_to_target_prob[:, :, :nw, :nt]
        lm = env_fire.mask_per_weapon[:, :, :nw, :nt] > 0
        action = auction_round_action(rv, pr, lm, must_fire=mf_fire)
        env_fire.update_internal_variables_parallel(selected_actions=action)
        env_fire.time_update()
        _rollout_forward(env_fire, actor, nw, nt, k)
        val_fire = env_fire.current_target_value[:, :, 0:nt].sum(dim=2)  # [B, P]

        env_hold = copy.deepcopy(env)
        mf_hold = must_fire_baseline.clone()
        mf_hold[:, :, m] = False
        rv = env_hold.current_target_value[:, :, 0:nt]
        pr = env_hold.weapon_to_target_prob[:, :, :nw, :nt]
        lm = env_hold.mask_per_weapon[:, :, :nw, :nt] > 0
        action = auction_round_action(rv, pr, lm, must_fire=mf_hold)
        env_hold.update_internal_variables_parallel(selected_actions=action)
        env_hold.time_update()
        _rollout_forward(env_hold, actor, nw, nt, k)
        val_hold = env_hold.current_target_value[:, :, 0:nt].sum(dim=2)  # [B, P]

        labels[:, :, m] = (val_fire < val_hold).float()

    return labels, valid


def self_play_gnn_binary_supervised(actor, episode, epoch, generator_fn=None, rollout_k=2, logger=None):
    """Pure supervised training loop: BCE against rollout-computed labels,
    no REINFORCE, no advantage/baseline. `episode` counts full instance
    episodes (each with TRAIN_BATCH x NUM_PAR parallel copies, matching the
    REINFORCE version's batching so results stay comparable)."""
    if generator_fn is None:
        from common.temporal_dilemma_generator import generate_temporal_dilemma_instance
        generator_fn = generate_temporal_dilemma_instance

    actor.train()
    losses = Average_Meter()
    total_objective = 0.0
    total_destruction_ratio = 0.0

    for ep in range(episode):
        num_weapons, num_targets, max_time = get_random_temporal_problem_size()
        V, P, TW, amm_list, prep_list, cost_list = generator_fn(num_weapons, num_targets, max_time)

        original_hyperparams = patch_hyperparameters_for_epoch(
            num_weapons, num_targets, max_time, amm_list,
            prep_list=prep_list, cost_list=cost_list,
        )
        try:
            ae, wtp = input_generation(
                NUM_WEAPON=num_weapons, NUM_TARGET=num_targets,
                value=V, prob=np.array(P), TW=TW, max_time=max_time,
                batch_size=TRAIN_BATCH, amm=amm_list,
            )
            ae = ae.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
            wtp = wtp.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
            env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=max_time)

            original_value = env.original_target_value[:, :, 0:num_targets].sum(2)

            episode_loss = 0.0
            n_terms = 0

            for t in range(max_time):
                fire_prob, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
                must_fire_baseline = (fire_prob > 0.5)

                labels, valid = compute_rollout_labels(
                    env, actor, must_fire_baseline, num_weapons, num_targets, max_time, t,
                    rollout_k=rollout_k,
                )

                if valid.any():
                    bce = F.binary_cross_entropy(
                        fire_prob[valid].clamp(1e-6, 1 - 1e-6), labels[valid]
                    )
                    episode_loss = episode_loss + bce
                    n_terms += 1

                remaining_value = env.current_target_value[:, :, 0:num_targets]
                prob = env.weapon_to_target_prob[:, :, :num_weapons, :num_targets]
                legal_mask = env.mask_per_weapon[:, :, :num_weapons, :num_targets] > 0
                action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire_baseline)
                env.update_internal_variables_parallel(selected_actions=action)
                env.time_update()

            if n_terms > 0:
                episode_loss = episode_loss / n_terms
                actor.optimizer.zero_grad()
                episode_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor.optimizer.step()
                losses.push(torch.tensor(episode_loss.item()), 1)

            final_value = env.current_target_value[:, :, 0:num_targets].sum(2)
            destruction_ratio = 1 - (final_value / (original_value + 1e-8))
            total_objective += final_value.mean().item()
            total_destruction_ratio += destruction_ratio.mean().item()

            if logger is not None:
                logger.info(f"Episode {ep+1}/{episode} | BCE Loss: {episode_loss if n_terms else 0.0}")

        finally:
            restore_hyperparameters(original_hyperparams)

    return losses.result(), {
        'objective': total_objective / episode,
        'destruction_ratio': total_destruction_ratio / episode,
    }
