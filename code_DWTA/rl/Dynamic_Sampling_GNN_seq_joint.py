"""
Single-process JOINT training of proposer + auditor on the SEQUENTIAL
decoder (rather than the parallel/auction pipeline).

Why switch base architectures (2026-08-23): the parallel pipeline stopped
improving. Even after the multifire auction fixed the diagnosed dispersion
bottleneck at Large tier (30M_30N_10T 0.0245 -> 0.0125, beating Sequential's
0.0162), Battlefield tier stayed behind and 70M_100N_15T got worse. Just as
importantly the speed premise is shaky: the auction is a Python loop over
weapons, so parallel+auction is not obviously cheaper than sequential
decoding, and every audit edit costs a full re-simulation on top. Sequential
is the strongest proposer available (beats the parallel pipeline on 8/12
configs), so putting the audit layer on the strongest base makes the
contribution the LEARNED IMPROVEMENT LAYER rather than the parallel decoder.

Edit-space mapping (the one design wrinkle): the parallel auditor flips a
(round, weapon) fire/hold decision, but the sequential decoder has no
per-weapon slot -- each round it makes M picks from one global (M*N+1)
softmax, and any pick may involve any still-eligible weapon. The slot grid
is therefore reinterpreted as (round t, decision index j): the j-th of the
M decisions taken in round t. Flipping a slot means
  - the decision was a FIRE  -> force it to no-op;
  - the decision was a NO-OP -> force it to the highest-scoring legal fire.
This preserves the parallel version's semantics (cancel a shot / add a
shot), keeps the grid at [T, M] so the existing auditor architecture is
reused unchanged, and needs no change to the sequential decoder itself.

Auditor states are recorded once per ROUND (not per decision), exactly as
in the parallel version: T forwards instead of T*M, and it keeps
`score_all_slots` reusable as-is. The auditor therefore scores a slot from
the round-start state rather than the mid-round state -- cheaper, at the
cost of not seeing intra-round evolution.

New file -- does not modify the sequential trainer or the parallel joint code.
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

from Dynamic_Sampling_GNN_moderate import get_random_moderate_problem_size
from Dynamic_Sampling_GNN_improve import score_all_slots
from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters

MIN_FIRE_PROB = 1e-6  # below this, a "forced fire" has no legal edge to take


@torch.no_grad()
def simulate_sequential_with_flips(base_actor, assignment_encoding, weapon_to_target_prob,
                                    flip_mask, nw, nt, mt):
    """Greedy (argmax) sequential rollout, with flip_mask[..., t, j] flipping
    the j-th decision of round t.

    Returns:
        objective: [B, P] normalized remaining value (lower is better).
        states:    list of T round-start (assignment_encoding, prob, mask)
                   tuples for the auditor to score.
    """
    env = Environment(assignment_encoding=assignment_encoding.clone(),
                       weapon_to_target_prob=weapon_to_target_prob.clone(), max_time=mt)
    original_value = env.original_target_value[:, :, 0:nt].sum(2)
    num_edges = nw * nt
    states = []

    for t in range(mt):
        states.append((env.assignment_encoding.clone(),
                        env.weapon_to_target_prob.clone(),
                        env.mask_per_weapon.clone()))

        for j in range(nw):
            policy, _ = base_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask.clone())
            action = policy.argmax(dim=-1)                      # [B, P]
            flip = flip_mask[:, :, t, j]                        # [B, P]

            if flip.any():
                edge_policy = policy[..., :num_edges]
                best_edge = edge_policy.argmax(dim=-1)          # [B, P]
                best_edge_prob = edge_policy.max(dim=-1).values
                has_legal_edge = best_edge_prob > MIN_FIRE_PROB

                was_fire = action < num_edges
                # fire -> no-op; no-op -> best legal fire (unchanged if none legal)
                forced = torch.where(was_fire,
                                      torch.full_like(action, num_edges),
                                      torch.where(has_legal_edge, best_edge, action))
                action = torch.where(flip, forced, action)

            env.update_internal_variables(selected_action=action)

        env.time_update()

    final_value = env.current_target_value[:, :, 0:nt].sum(2)
    return final_value / (original_value + 1e-8), states


def _train_proposer_sequential(actor, ae, wtp, nw, nt, mt, amm_list, entropy_coef=1e-3):
    """One REINFORCE update for the sequential proposer, mirroring
    Dynamic_Sampling_GNN_sequential_moderate.py's math exactly (reward-to-go,
    per-step para-baselined advantage, ammo bonus, entropy bonus, no critic)."""
    env = Environment(assignment_encoding=ae.clone(),
                       weapon_to_target_prob=wtp.clone(), max_time=mt)

    log_probs, entropies, rewards = [], [], []
    total_fires = None
    original_value = env.original_target_value[:, :, 0:nt].sum(2)
    prev_value = original_value.clone()

    b_sz, p_sz = ae.shape[:2]
    num_edges = nw * nt
    num_actions = num_edges + 1

    for _ in range(mt):
        for _ in range(nw):
            policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask.clone())

            safe_policy = policy.clamp_min(1e-8)
            entropies.append(-(safe_policy * safe_policy.log()).sum(dim=-1))

            action = torch.multinomial(policy.reshape(-1, num_actions), 1).view(b_sz, p_sz)
            log_probs.append(torch.log(
                policy.gather(-1, action.unsqueeze(-1)).clamp_min(1e-8)).squeeze(-1).unsqueeze(-1))

            fires = (action < num_edges).float()
            total_fires = fires if total_fires is None else total_fires + fires

            env.update_internal_variables(selected_action=action)

            curr_value = env.current_target_value[:, :, 0:nt].sum(2)
            rewards.append((prev_value - curr_value) / (original_value + 1e-8))
            prev_value = curr_value

        env.time_update()

    total_ammo = float(sum(amm_list))
    rewards[-1] = rewards[-1] + 0.1 * (total_fires / max(total_ammo, 1.0))

    running = torch.zeros_like(rewards[0])
    returns = [None] * len(rewards)
    for t in reversed(range(len(rewards))):
        running = rewards[t] + running
        returns[t] = running.clone()

    loss = 0
    for log_prob, G_t in zip(log_probs, returns):
        adv = G_t - G_t.mean(dim=1, keepdim=True)
        adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-6)
        loss = loss + (-(log_prob.squeeze(-1) * adv).mean())
    loss = loss - entropy_coef * torch.stack(entropies).mean()

    actor.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    actor.optimizer.step()

    final_value = env.current_target_value[:, :, 0:nt].sum(2)
    destruction = (1 - final_value / (original_value + 1e-8)).mean().item()
    return loss.item(), destruction


def _train_auditor_sequential(base_actor, improve_actor, ae, wtp, nw, nt, mt, n_edits):
    """REINFORCE over a sequence of n_edits flips on the sequential
    proposer's completed schedule; reward per edit is the objective delta."""
    B, P = ae.shape[:2]
    flip_mask = torch.zeros(B, P, mt, nw, dtype=torch.bool, device=DEVICE)

    obj_prev, states = simulate_sequential_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
    base_obj = obj_prev.clone()

    log_probs, rewards = [], []
    n_slots = mt * nw

    for _ in range(n_edits):
        logits = score_all_slots(improve_actor, states, nw).reshape(B, P, n_slots)
        logits = logits.masked_fill(flip_mask.reshape(B, P, n_slots), -1e9)
        probs = torch.softmax(logits, dim=-1)

        idx = torch.multinomial(probs.reshape(-1, n_slots), 1).reshape(B, P)
        log_probs.append(torch.log(probs.gather(-1, idx.unsqueeze(-1)).clamp_min(1e-8)).squeeze(-1))

        new_flip = flip_mask.reshape(B, P, n_slots).clone()
        new_flip.scatter_(-1, idx.unsqueeze(-1), True)
        flip_mask = new_flip.reshape(B, P, mt, nw)

        obj_new, states = simulate_sequential_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
        rewards.append(obj_prev - obj_new)
        obj_prev = obj_new

    loss = 0.0
    for log_prob, reward in zip(log_probs, rewards):
        adv = reward - reward.mean(dim=1, keepdim=True)
        adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-6)
        loss = loss - (log_prob * adv).mean()
    loss = loss / max(len(log_probs), 1)

    improve_actor.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(improve_actor.parameters(), max_norm=1.0)
    improve_actor.optimizer.step()

    return loss.item(), base_obj.mean().item(), obj_prev.mean().item()


def self_play_joint_sequential(base_actor, improve_actor, episode, epoch,
                                n_edits=2, batch_size=8, para_size=8, entropy_coef=1e-3,
                                train_proposer_this_step=True, problem_size_fn=None, logger=None):
    """One joint step on the sequential base. Exactly one agent is updated
    per call when alternating; see the parallel counterpart
    (Dynamic_Sampling_GNN_joint.self_play_joint) for the rationale."""
    if problem_size_fn is None:
        problem_size_fn = get_random_moderate_problem_size

    base_actor.train()
    improve_actor.train()
    prop_losses, aud_losses = Average_Meter(), Average_Meter()
    tot_destr = tot_base = tot_audited = 0.0

    for ep in range(episode):
        nw, nt, mt, amm, prep, cost = problem_size_fn()
        original_hyperparams = patch_hyperparameters_for_epoch(
            nw, nt, mt, amm, prep_list=prep, cost_list=cost)
        try:
            ae, wtp = generate_moderate_training_instances(
                batch_size=batch_size, num_weapons=nw, num_targets=nt,
                max_time=mt, amm_list=amm)
            ae = ae.unsqueeze(1).repeat(1, para_size, 1, 1).contiguous()
            wtp = wtp.unsqueeze(1).repeat(1, para_size, 1, 1).contiguous()

            if train_proposer_this_step:
                p_loss, destr = _train_proposer_sequential(
                    base_actor, ae, wtp, nw, nt, mt, amm, entropy_coef)
                prop_losses.push(torch.tensor(p_loss), 1)
                tot_destr += destr
            else:
                base_actor.eval()
                a_loss, base_obj, audited_obj = _train_auditor_sequential(
                    base_actor, improve_actor, ae, wtp, nw, nt, mt, n_edits)
                base_actor.train()
                aud_losses.push(torch.tensor(a_loss), 1)
                tot_base += base_obj
                tot_audited += audited_obj

            if logger is not None:
                who = 'proposer' if train_proposer_this_step else 'auditor'
                logger.info(f"Episode {ep+1}/{episode} | updated {who}")
        finally:
            restore_hyperparameters(original_hyperparams)

    p_out = prop_losses.result() if train_proposer_this_step else 0.0
    a_out = 0.0 if train_proposer_this_step else aud_losses.result()
    return p_out, a_out, {
        'destruction_ratio': tot_destr / episode,
        'base_objective': tot_base / episode,
        'audited_objective': tot_audited / episode,
    }
