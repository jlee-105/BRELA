"""
Training loop for the LEARNED-guidance RL+Auction hybrid
(common/DWTA_GNN_binary_guided.py): the actor outputs fire/hold, target
preference, AND a per-instance guide_weight for the capacitated auction --
all three are genuine stochastic actions with their own REINFORCE log-prob
terms, so nothing is hand-tuned (see that file's docstring for the full
design rationale and brerla_hybrid_policy_auction_framing memory for why a
fixed guide_weight was rejected).

Target-choice credit is computed AFTER the auction decides the REAL
executed target for each firing weapon (log(target_pref[executed target])
under the policy's CURRENT distribution) -- never the policy's own
hypothetical argmax/sample, which is what caused the train/inference
mismatch documented in brerla_auction_train_inference_mismatch memory. This
is what makes crediting target choice safe even though the auction (not
the policy) has final say over which target actually gets used.

New file -- does not modify rl/Dynamic_Sampling_GNN_binary.py or any other
existing training code/results.
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
from common.auction_refinement import auction_round_action_multifire_guided
from common.DWTA_GNN_binary_guided import GUIDE_WEIGHT_MAX, GUIDE_WEIGHT_STD
from rl.Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters


def get_random_temporal_problem_size():
    num_weapons = random.randint(5, 7)
    num_targets = random.randint(5, 7)
    max_time = random.randint(5, 7)
    return num_weapons, num_targets, max_time


def self_play_gnn_binary_guided(actor, episode, epoch, logger=None, generator_fn=None,
                                 epsilon=0.2, entropy_coef=1e-2, target_entropy_coef=1e-2):
    """Same REINFORCE structure as Dynamic_Sampling_GNN_binary.py::self_play_gnn_binary
    (no critic, whole-episode return, epsilon-mixed exploration floor on
    fire/hold), extended with target-choice and guide_weight log-prob terms."""
    if generator_fn is None:
        generator_fn = generate_temporal_dilemma_instance
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

                    fire_prob, target_pref, guide_mu = actor(
                        env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon
                    )

                    # --- fire/hold (Bernoulli), same epsilon-mixing as the plain binary actor ---
                    has_legal_for_eps = legal_mask.any(dim=-1)
                    mixed_prob = fire_prob * (1 - epsilon) + 0.5 * epsilon
                    sample_prob = torch.where(has_legal_for_eps, mixed_prob, fire_prob)
                    fire_dist = torch.distributions.Bernoulli(probs=sample_prob.clamp(1e-6, 1 - 1e-6))
                    fire_decision = fire_dist.sample()
                    fire_logprob = fire_dist.log_prob(fire_decision).sum(dim=-1, keepdim=True)
                    has_legal = legal_mask.any(dim=-1).float()
                    fire_entropy = (fire_dist.entropy() * has_legal).sum(dim=-1)

                    # --- guide_weight (Normal), sampled once per round per instance ---
                    guide_dist = torch.distributions.Normal(guide_mu, GUIDE_WEIGHT_STD)
                    guide_raw_sample = guide_dist.sample()
                    guide_logprob = guide_dist.log_prob(guide_raw_sample).unsqueeze(-1)  # [B,P,1]
                    guide_weight_for_auction = guide_raw_sample.clamp(0.0, GUIDE_WEIGHT_MAX)

                    must_fire = fire_decision.bool()
                    action = auction_round_action_multifire_guided(
                        remaining_value, prob, legal_mask, target_pref,
                        must_fire=must_fire, guide_weight=guide_weight_for_auction,
                    )

                    # --- target-choice credit: log-prob of the ACTUALLY EXECUTED
                    # target under the policy's current distribution (not a
                    # separately-sampled hypothetical target) ---
                    fired_mask = (action < num_targets).float()  # [B,P,W]
                    safe_action = action.clamp(max=num_targets - 1)
                    chosen_pref = target_pref.gather(-1, safe_action.unsqueeze(-1)).squeeze(-1)
                    target_logprob_per_weapon = torch.log(chosen_pref.clamp_min(1e-8)) * fired_mask
                    target_logprob = target_logprob_per_weapon.sum(dim=-1, keepdim=True)  # [B,P,1]

                    safe_target_pref = target_pref.clamp_min(1e-8)
                    target_entropy_per_weapon = -(safe_target_pref * safe_target_pref.log()).sum(dim=-1)
                    target_entropy = (target_entropy_per_weapon * fired_mask).sum(dim=-1)  # [B,P]

                    total_logprob = fire_logprob + target_logprob + guide_logprob
                    log_probs.append(total_logprob)
                    entropies.append(fire_entropy + target_entropy_coef * target_entropy)

                    env.update_internal_variables_parallel(selected_actions=action)
                    curr_value = env.current_target_value[:, :, 0:num_targets].sum(2)
                    step_reward = (prev_value - curr_value) / (original_value + 1e-8)
                    rewards.append(step_reward)
                    prev_value = curr_value

                    env.time_update()

                final_value = env.current_target_value[:, :, 0:num_targets].sum(2)
                destruction_ratio = 1 - (final_value / (original_value + 1e-8))

                total_return = sum(rewards)
                baseline = total_return.mean(dim=1, keepdim=True)
                adv = total_return - baseline
                adv_std = adv.std(dim=1, keepdim=True).clamp_min(1e-6)
                adv = adv / adv_std

                actor_loss = 0
                for log_prob in log_probs:
                    actor_loss = actor_loss + (-(log_prob.squeeze(-1) * adv).mean())
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
        print(f"Error in self_play_gnn_binary_guided: {e}")
        raise
