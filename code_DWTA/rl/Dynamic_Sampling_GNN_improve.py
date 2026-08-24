"""
Improvement-based (neural local search) training loop.

Paradigm shift from every other training file in this project: the policy
never constructs a solution. A FROZEN base pipeline (trained CommSinkhorn
actor + 1:1 auction -- the project's current best, mean 0.1357) produces a
complete schedule; the improvement policy then repeatedly picks ONE
(round, weapon) slot and FLIPS its fire/hold decision, and is rewarded by
the resulting change in the true objective.

Why this structurally avoids the failures documented in
brerla_sinkhorn_coordination_experiment memory:
  - DENSE reward: each edit's effect on the objective is measured directly
    and attributed to exactly that edit -- no sparse episode-level credit
    assignment, which is what killed every binary fire/hold attempt.
  - CANNOT REGRESS below the base: at inference we keep the best solution
    seen across edits, so the reported number is bounded by the base
    pipeline's own result.
  - The learned quantity is exactly the thing the auction provably lacks
    (temporal judgment about deferring/adding shots), not a target-choice
    signal the auction discards anyway.

No new auction function is needed: flipping is applied by XOR-ing the base
policy's own must_fire decision, then calling the existing
auction_round_action unchanged.

New file -- does not modify any existing training code.
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
from common.auction_refinement import auction_round_action

from Dynamic_Sampling_GNN_moderate import get_random_moderate_problem_size
from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters


@torch.no_grad()
def simulate_with_flips(base_actor, assignment_encoding, weapon_to_target_prob,
                         flip_mask, nw, nt, mt, return_fire_state=False, auction_fn=None):
    """Run one full episode. At each round the frozen base actor decides
    fire/hold, then flip_mask[:, :, t, :] XOR-inverts those decisions
    before the (unchanged) auction assigns targets.

    Args:
        flip_mask: [B, P, T, M] bool -- True where this slot's base
            decision should be inverted.
        return_fire_state: if True, also return the EFFECTIVE (post-flip)
            fire/hold decision per slot -- needed by the move operator,
            which must restrict "cancel" candidates to slots that actually
            fire and "add" candidates to slots that do not.
        auction_fn: target-assignment rule; defaults to the 1:1 eviction
            auction. Pass `auction_round_action_multifire` for the
            capacitated many-to-one variant, which measurement showed is
            both better at Large tier (it lifts the 1:1 auction's hard
            dispersion=1.0 constraint) and faster (no evict/re-bid loop).

    Returns:
        objective: [B, P] normalized remaining value (lower is better).
        states: list of T tuples (assignment_encoding, weapon_to_target_prob,
            mask_per_weapon) recorded BEFORE each round's decision, for the
            improvement policy to score afterwards.
        fire_state (only if return_fire_state): [B, P, T, M] bool.
    """
    if auction_fn is None:
        auction_fn = auction_round_action

    env = Environment(assignment_encoding=assignment_encoding.clone(),
                       weapon_to_target_prob=weapon_to_target_prob.clone(), max_time=mt)
    original_value = env.original_target_value[:, :, 0:nt].sum(2)
    states = []
    fire_states = []

    for t in range(mt):
        states.append((env.assignment_encoding.clone(),
                        env.weapon_to_target_prob.clone(),
                        env.mask_per_weapon.clone()))

        policy, _ = base_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire_base = policy[:, :, :nw, :].argmax(dim=-1) < nt  # [B,P,M]
        must_fire = must_fire_base ^ flip_mask[:, :, t, :]

        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        action = auction_fn(remaining_value, prob, legal_mask, must_fire=must_fire)

        if return_fire_state:
            # What the weapon ACTUALLY ends up doing, not merely what was
            # requested -- a weapon told to fire with no legal/available
            # target still ends up at no-op, and treating that as "firing"
            # would let the move operator try to cancel a shot that was
            # never taken.
            fire_states.append(action < nt)

        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    final_value = env.current_target_value[:, :, 0:nt].sum(2)
    objective = final_value / (original_value + 1e-8)

    if return_fire_state:
        return objective, states, torch.stack(fire_states, dim=2)  # [B,P,T,M]
    return objective, states


def score_all_slots(improve_actor, states, nw):
    """Run the improvement actor on every recorded round state.
    Returns flip_logits [B, P, T, M] WITH grad."""
    per_round = []
    for (ae, wtp, mask) in states:
        logit, _ = improve_actor(ae, wtp, mask)  # [B,P,M]
        per_round.append(logit[:, :, :nw])
    return torch.stack(per_round, dim=2)  # [B,P,T,M]


def self_play_gnn_improve(base_actor, improve_actor, episode, epoch,
                           n_edits=3, batch_size=4, para_size=8, logger=None):
    """REINFORCE over a sequence of n_edits improvement moves. Reward for
    each edit is the objective improvement it caused (dense, immediately
    attributable) -- not an episode-level return."""
    base_actor.eval()
    improve_actor.train()
    losses = Average_Meter()
    total_base_obj = 0.0
    total_final_obj = 0.0

    for ep in range(episode):
        ep_nw, ep_nt, ep_mt, ep_amm, ep_prep, ep_cost = get_random_moderate_problem_size()
        original_hyperparams = patch_hyperparameters_for_epoch(
            ep_nw, ep_nt, ep_mt, ep_amm, prep_list=ep_prep, cost_list=ep_cost,
        )
        try:
            ae, wtp = generate_moderate_training_instances(
                batch_size=batch_size, num_weapons=ep_nw, num_targets=ep_nt,
                max_time=ep_mt, amm_list=ep_amm,
            )
            ae = ae.unsqueeze(1).repeat(1, para_size, 1, 1).contiguous()
            wtp = wtp.unsqueeze(1).repeat(1, para_size, 1, 1).contiguous()

            B, P, M, T = batch_size, para_size, ep_nw, ep_mt
            flip_mask = torch.zeros(B, P, T, M, dtype=torch.bool, device=DEVICE)

            obj_prev, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, ep_nw, ep_nt, ep_mt)
            base_obj = obj_prev.clone()

            log_probs = []
            rewards = []

            for k in range(n_edits):
                logits = score_all_slots(improve_actor, states, ep_nw)  # [B,P,T,M] with grad
                flat_logits = logits.reshape(B, P, T * M)
                # Never re-select an already-flipped slot (flipping twice is a no-op).
                flat_logits = flat_logits.masked_fill(flip_mask.reshape(B, P, T * M), -1e9)
                probs = torch.softmax(flat_logits, dim=-1)

                idx = torch.multinomial(probs.reshape(-1, T * M), 1).reshape(B, P)  # [B,P]
                log_prob = torch.log(probs.gather(-1, idx.unsqueeze(-1)).clamp_min(1e-8)).squeeze(-1)

                new_flip = flip_mask.reshape(B, P, T * M).clone()
                new_flip.scatter_(-1, idx.unsqueeze(-1), True)
                flip_mask = new_flip.reshape(B, P, T, M)

                obj_new, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, ep_nw, ep_nt, ep_mt)
                rewards.append(obj_prev - obj_new)  # positive = objective improved
                log_probs.append(log_prob)
                obj_prev = obj_new

            # Per-edit advantage, baselined across the para dimension
            # (POMO-style, same convention as every other trainer here).
            actor_loss = 0.0
            for log_prob, reward in zip(log_probs, rewards):
                adv = reward - reward.mean(dim=1, keepdim=True)
                adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-6)
                actor_loss = actor_loss - (log_prob * adv).mean()
            actor_loss = actor_loss / max(len(log_probs), 1)

            improve_actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(improve_actor.parameters(), max_norm=1.0)
            improve_actor.optimizer.step()

            losses.push(torch.tensor(actor_loss.item()), 1)
            total_base_obj += base_obj.mean().item()
            total_final_obj += obj_prev.mean().item()

            if logger is not None:
                logger.info(f"Episode {ep+1}/{episode} | loss {actor_loss.item():.6f}")
        finally:
            restore_hyperparameters(original_hyperparams)

    return losses.result(), {
        'base_objective': total_base_obj / episode,
        'improved_objective': total_final_obj / episode,
    }
