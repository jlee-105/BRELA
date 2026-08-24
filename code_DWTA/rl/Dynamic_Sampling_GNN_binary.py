"""
Training loop for the RL+Auction hybrid: a binary fire/hold policy
(common/DWTA_GNN_binary.py) trained with REINFORCE, with target assignment
handled entirely by the auction refinement (common/auction_refinement.py) --
not learned, not backpropagated through. Only the binary fire/hold decision
is part of the policy gradient.

Trains on temporal-dilemma instances (common/temporal_dilemma_generator.py)
specifically, since the main multi-scale curriculum was found to contain
almost no genuine hold-vs-fire structure (see memory
brerla_rl_auction_hybrid_decomposition.md).

New file -- does not modify rl/Dynamic_Sampling_GNN.py or any other existing
training code/results.
"""
import os
import random
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.utilities import Average_Meter
from common.temporal_dilemma_generator import generate_temporal_dilemma_instance
from common.auction_refinement import auction_round_action
from rl.Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters


def get_random_temporal_problem_size():
    """Multi-scale sampling for the temporal-dilemma curriculum -- same
    dimension ranges as the main curriculum (rl/Dynamic_Sampling_GNN.py's
    get_random_problem_size), but AMM/V/P/TW come from the temporal-dilemma
    generator instead, so amm_list/prep_list/cost_list here are placeholders
    only used to size the hyperparameter patch; the generator below
    overrides AMM per-instance."""
    num_weapons = random.randint(5, 7)
    num_targets = random.randint(5, 7)
    max_time = random.randint(5, 7)
    return num_weapons, num_targets, max_time


def self_play_gnn_binary(actor, episode, epoch, logger=None, generator_fn=None, epsilon=0.2, entropy_coef=1e-2,
                          auction_fn=None):
    """
    Multi-episodic REINFORCE for the binary fire/hold actor. No critic
    (per 2026-08-07 session directive: never use critic with REINFORCE),
    whole-episode return (per the reward-to-go finding earlier the same
    session -- the no-op/hold action's credit assignment specifically
    requires this).

    generator_fn: instance generator with signature (M, N, T) -> (V, P, TW,
        AMM, PREP, COST); defaults to the extreme temporal-dilemma generator.
        Pass common.temporal_dilemma_generator_moderate.generate_moderate_temporal_instance
        for the moderate-difficulty curriculum.
    epsilon: epsilon-greedy-style exploration floor, mixed into the sampled
        Bernoulli probability as fire_prob*(1-epsilon) + 0.5*epsilon. Added
        2026-08-07 after the moderate curriculum's entropy bonus alone
        proved insufficient -- observed fire_prob monotonically saturating
        toward 1.0 across training (0.90 -> 0.99 -> 0.998 at epochs 5/25/45)
        despite the entropy term, rather than being restrained by it. Unlike
        an entropy bonus (a soft penalty the return term can outweigh),
        epsilon-mixing is a hard floor: no matter how confident the network
        becomes, the actually-sampled probability can never leave
        [epsilon/2, 1-epsilon/2], guaranteeing a minimum exploration rate.
    """
    if generator_fn is None:
        generator_fn = generate_temporal_dilemma_instance
    if auction_fn is None:
        auction_fn = auction_round_action
    try:
        actor.train()
        actor_losses = Average_Meter()
        total_objective = 0.0
        total_destruction_ratio = 0.0

        for ep in range(episode):
            num_weapons, num_targets, max_time = get_random_temporal_problem_size()
            V, P, TW, amm_list, prep_list, cost_list = generator_fn(
                num_weapons, num_targets, max_time
            )

            original_hyperparams = patch_hyperparameters_for_epoch(
                num_weapons, num_targets, max_time, amm_list,
                prep_list=prep_list, cost_list=cost_list,
            )

            try:
                import numpy as np
                ae, wtp = input_generation(
                    NUM_WEAPON=num_weapons, NUM_TARGET=num_targets,
                    value=V, prob=np.array(P), TW=TW, max_time=max_time,
                    batch_size=TRAIN_BATCH, amm=amm_list,
                )
                ae = ae.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
                wtp = wtp.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()

                env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=max_time)

                log_probs = []
                rewards = []
                entropies = []
                original_value = env.original_target_value[:, :, 0:num_targets].sum(2)
                prev_value = original_value.clone()

                for t in range(max_time):
                    remaining_value = env.current_target_value[:, :, 0:num_targets]
                    prob = env.weapon_to_target_prob[:, :, :num_weapons, :num_targets]
                    legal_mask = env.mask_per_weapon[:, :, :num_weapons, :num_targets] > 0

                    fire_prob, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
                    # fire_prob: [batch, para, W], already 0 where no legal target exists
                    has_legal_for_eps = legal_mask.any(dim=-1)  # [batch, para, W]
                    mixed_prob = fire_prob * (1 - epsilon) + 0.5 * epsilon
                    sample_prob = torch.where(has_legal_for_eps, mixed_prob, fire_prob)
                    dist = torch.distributions.Bernoulli(probs=sample_prob.clamp(1e-6, 1 - 1e-6))
                    fire_decision = dist.sample()  # [batch, para, W] in {0,1}
                    log_prob = dist.log_prob(fire_decision).sum(dim=-1, keepdim=True)  # [batch, para, 1]
                    # Per-weapon Bernoulli entropy, masked to legal-target weapons only
                    # (a weapon with no legal target has fire_prob forced to 0 by the
                    # actor itself, not a real decision -- its "entropy" is not policy
                    # exploration and should not be optimized).
                    has_legal = legal_mask.any(dim=-1).float()  # [batch, para, W]
                    step_entropy = (dist.entropy() * has_legal).sum(dim=-1)  # [batch, para]

                    must_fire = fire_decision.bool()
                    action = auction_fn(remaining_value, prob, legal_mask, must_fire=must_fire)

                    log_probs.append(log_prob)
                    entropies.append(step_entropy)

                    env.update_internal_variables_parallel(selected_actions=action)
                    curr_value = env.current_target_value[:, :, 0:num_targets].sum(2)
                    step_reward = (prev_value - curr_value) / (original_value + 1e-8)
                    rewards.append(step_reward)
                    prev_value = curr_value

                    env.time_update()

                final_value = env.current_target_value[:, :, 0:num_targets].sum(2)
                destruction_ratio = 1 - (final_value / (original_value + 1e-8))

                # Whole-episode return, applied uniformly to every step (no
                # reward-to-go -- see brerla_simulator_bugs memory for why).
                total_return = sum(rewards)
                advantages = []
                baseline = total_return.mean(dim=1, keepdim=True)
                adv = total_return - baseline
                adv_std = adv.std(dim=1, keepdim=True).clamp_min(1e-6)
                adv = adv / adv_std

                actor_loss = 0
                for log_prob in log_probs:
                    actor_loss = actor_loss + (-(log_prob.squeeze(-1) * adv).mean())
                if entropies:
                    # entropy_coef now passed in by the caller (annealed over
                    # training -- see rl/DWTA_GNN_TRAIN_binary_moderate.py).
                    # A Bernoulli's max entropy (ln 2 ~ 0.69) is much smaller than an
                    # N+1-way softmax's (ln(N+1)), so a proportionally larger
                    # coefficient is needed for entropy to meaningfully compete
                    # against the return term and prevent early collapse to a
                    # near-deterministic always-fire policy (observed empirically
                    # on the moderate temporal-dilemma curriculum, 2026-08-07).
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
                    logger.info(f"Episode {ep+1}/{episode} | Actor Loss: {actor_loss.item():.6f}")

            finally:
                restore_hyperparameters(original_hyperparams)

        epoch_objective = total_objective / episode
        avg_destruction = total_destruction_ratio / episode

        return actor_losses.result(), {
            'objective': epoch_objective,
            'destruction_ratio': avg_destruction,
        }

    except Exception as e:
        print(f"Error in self_play_gnn_binary: {e}")
        raise
