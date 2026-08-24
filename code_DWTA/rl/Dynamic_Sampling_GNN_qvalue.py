"""
Monte-Carlo Q-VALUE regression training for EdgeAwareGNN_ACTOR_COMM_QVALUE
(common/DWTA_GNN_comm_qvalue.py), fed through the Q-value-driven capacitated
auction (common/auction_refinement_qvalue.py::auction_round_action_multifire_qvalue).

NO REINFORCE, no log-prob, no sampled action anywhere. Each round, for each
weapon, the actor's own Q-value prediction at the ACTION THE AUCTION ACTUALLY
ASSIGNED (using the network's own current Q-values as the auction's ranking
criterion, REDA-style) is regressed via MSE toward that weapon's OWN
reward-to-go (Monte Carlo return, matching this project's established
whole-episode/reward-to-go conventions elsewhere -- see brerla_simulator_bugs
memory for why reward-to-go, not whole-episode-return, is the correct target
specifically for a genuine value function -- unlike REINFORCE's baseline
choice, which is a variance-reduction convenience, this is a real V(s,a)
regression target and must be the ACTUAL forward return from that decision).

New file -- does not modify Dynamic_Sampling_GNN_moderate.py or any existing
training script.
"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation_moderate import generate_moderate_training_instances
from common.DWTA_Simulator import Environment
from common.utilities import Average_Meter
from common.auction_refinement_qvalue import auction_round_action_multifire_qvalue

from Dynamic_Sampling_GNN_moderate import get_random_moderate_problem_size
from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters


def self_play_gnn_qvalue(actor, episode, epoch, max_per_target=2, explore_std=0.15, logger=None):
    try:
        actor.train()
        losses = Average_Meter()
        total_objective = 0.0
        total_destruction_ratio = 0.0

        for ep in range(episode):
            ep_nw, ep_nt, ep_mt, ep_amm, ep_prep, ep_cost = get_random_moderate_problem_size()

            original_hyperparams = patch_hyperparameters_for_epoch(
                ep_nw, ep_nt, ep_mt, ep_amm, prep_list=ep_prep, cost_list=ep_cost,
            )
            try:
                assignment_encoding, weapon_to_target_prob = generate_moderate_training_instances(
                    batch_size=TRAIN_BATCH, num_weapons=ep_nw, num_targets=ep_nt,
                    max_time=ep_mt, amm_list=ep_amm,
                )
                assignment_encoding = assignment_encoding.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
                weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()

                env = Environment(
                    assignment_encoding=assignment_encoding,
                    weapon_to_target_prob=weapon_to_target_prob,
                    max_time=ep_mt,
                )

                original_value = env.original_target_value[:, :, 0:ep_nt].sum(2)  # [B,P]

                q_preds = []
                rewards = []

                for t in range(ep_mt):
                    q_values, _ = actor(
                        assignment_embedding=env.assignment_encoding,
                        prob=env.weapon_to_target_prob,
                        mask=env.mask_per_weapon,
                    )  # [B,P,M,N+1], WITH grad

                    remaining_value = env.current_target_value[:, :, 0:ep_nt]
                    prob = env.weapon_to_target_prob[:, :, :ep_nw, :ep_nt]
                    legal_mask = env.mask_per_weapon[:, :, :ep_nw, :ep_nt] > 0

                    # Exploration noise added ONLY to the ranking signal fed
                    # to the auction, not to q_pred_selected below (which
                    # must regress the network's actual, un-noised belief).
                    # Without this, action selection is fully deterministic
                    # given the current Q-values -- if a bad init or an
                    # unlucky gradient step ever pushes no-op above firing
                    # for most weapons, the auction never fires again and
                    # never observes a reward to learn from (confirmed
                    # empirically as a real cold-start trap, 2026-08-20).
                    noisy_q = q_values.detach() + explore_std * torch.randn_like(q_values)
                    action, realized_value = auction_round_action_multifire_qvalue(
                        noisy_q, remaining_value, prob, legal_mask, max_per_target=max_per_target,
                    )

                    q_pred_selected = q_values.gather(-1, action.unsqueeze(-1)).squeeze(-1)  # [B,P,M], WITH grad
                    q_preds.append(q_pred_selected)

                    norm_reward = realized_value / (original_value.unsqueeze(-1) + 1e-8)  # [B,P,M]
                    rewards.append(norm_reward)

                    env.update_internal_variables_parallel(selected_actions=action)
                    env.time_update()

                # reward-to-go per weapon (gamma=1.0, matching this project's
                # established convention elsewhere)
                T_steps = len(rewards)
                running = torch.zeros_like(rewards[0])
                returns = [None] * T_steps
                for t in reversed(range(T_steps)):
                    running = rewards[t] + running
                    returns[t] = running.clone()

                mse_loss = 0.0
                for q_pred, G in zip(q_preds, returns):
                    mse_loss = mse_loss + ((q_pred - G) ** 2).mean()
                mse_loss = mse_loss / T_steps

                actor.optimizer.zero_grad()
                mse_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor.optimizer.step()

                losses.push(torch.tensor(mse_loss.item()), 1)

                final_value = env.current_target_value[:, :, 0:ep_nt].sum(2)
                destruction_ratio = 1 - (final_value / (original_value + 1e-8))
                total_objective += final_value.mean().item()
                total_destruction_ratio += destruction_ratio.mean().item()

                if logger is not None:
                    logger.info(f"Episode {ep+1}/{episode} | MSE Loss: {mse_loss.item():.6f}")

            finally:
                restore_hyperparameters(original_hyperparams)

        return losses.result(), {
            'objective': total_objective / episode,
            'destruction_ratio': total_destruction_ratio / episode,
        }

    except Exception as e:
        print(f"Error in self_play_gnn_qvalue: {e}")
        raise
