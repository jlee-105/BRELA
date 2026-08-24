"""
MOVE-operator improvement policy: instead of flipping ONE (round, weapon)
fire/hold decision, each edit atomically RESCHEDULES one weapon's shot --
cancel it at round t1, fire it at round t2 instead.

Why this operator (2026-08-21): the single-flip operator
(Dynamic_Sampling_GNN_improve.py) cannot express temporal rescheduling in
one move. "Hold at round 2 so the shot is available at round 4" requires
TWO flips, and the first one alone (cancelling a shot) looks purely harmful
to a greedy edit selector, so the pair is never discovered. Since the whole
premise of the learned component is that it supplies the temporal judgment
the auction structurally lacks, the edit space should contain that move
atomically.

Each edit is factorized into two conditional choices:
  1. CANCEL: pick a (round, weapon) slot that currently FIRES.
  2. ADD:    pick a different round for THAT SAME weapon that currently
             does not fire.
Log-probability of the edit is the sum of the two, so REINFORCE credits
both halves of the move with the objective delta it produced.

New file -- does not modify Dynamic_Sampling_GNN_improve.py (whose
single-flip operator produced the current project best, 0.1291).
"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation_moderate import generate_moderate_training_instances
from common.utilities import Average_Meter

from Dynamic_Sampling_GNN_moderate import get_random_moderate_problem_size
from Dynamic_Sampling_GNN_improve import simulate_with_flips, score_all_slots
from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters

NEG = -1e9


def _masked_sample(logits, mask, greedy=False):
    """Sample (or argmax) over the last dim under a boolean mask.

    Returns (idx [.., ], log_prob [..], valid [..]). Rows with no valid
    entry get idx=0, log_prob=0, valid=False -- callers must drop those
    from both the applied edit and the loss.
    """
    valid = mask.any(dim=-1)
    safe_mask = mask | (~valid).unsqueeze(-1)  # avoid all -inf rows -> NaN softmax
    masked = logits.masked_fill(~safe_mask, NEG)
    probs = torch.softmax(masked, dim=-1)

    flat = probs.reshape(-1, probs.shape[-1])
    if greedy:
        idx = flat.argmax(dim=-1, keepdim=True)
    else:
        idx = torch.multinomial(flat, 1)
    idx = idx.reshape(probs.shape[:-1])

    log_prob = torch.log(probs.gather(-1, idx.unsqueeze(-1)).clamp_min(1e-8)).squeeze(-1)
    log_prob = log_prob * valid.float()
    return idx, log_prob, valid


def propose_move(improve_actor, states, fire_state, nw, mt, greedy=False):
    """One atomic move edit. Returns (t1, t2, weapon, log_prob, valid)."""
    B, P = fire_state.shape[:2]
    n_slots = mt * nw

    logits = score_all_slots(improve_actor, states, nw)            # [B,P,T,M], with grad
    flat_logits = logits.reshape(B, P, n_slots)

    # --- 1. cancel: must currently fire ---
    cancel_mask = fire_state.reshape(B, P, n_slots)
    idx1, lp1, valid1 = _masked_sample(flat_logits, cancel_mask, greedy)
    t1 = idx1 // nw
    weapon = idx1 % nw

    # --- 2. add: same weapon, a round it does not currently fire ---
    weapon_onehot = torch.zeros(B, P, nw, dtype=torch.bool, device=fire_state.device)
    weapon_onehot.scatter_(-1, weapon.unsqueeze(-1), True)
    same_weapon = weapon_onehot.unsqueeze(2).expand(B, P, mt, nw)
    add_mask = (same_weapon & ~fire_state).reshape(B, P, n_slots)
    idx2, lp2, valid2 = _masked_sample(flat_logits, add_mask, greedy)
    t2 = idx2 // nw

    valid = valid1 & valid2
    log_prob = (lp1 + lp2) * valid.float()
    return t1, t2, weapon, log_prob, valid


def apply_move(flip_mask, t1, t2, weapon, valid):
    """Toggle the two slots of the move (XOR semantics: toggling a slot
    inverts its effective fire/hold decision)."""
    B, P, T, M = flip_mask.shape
    new_mask = flip_mask.clone()
    b_idx = torch.arange(B, device=flip_mask.device).view(B, 1).expand(B, P)
    p_idx = torch.arange(P, device=flip_mask.device).view(1, P).expand(B, P)

    v = valid
    if v.any():
        bi, pi = b_idx[v], p_idx[v]
        new_mask[bi, pi, t1[v], weapon[v]] ^= True
        new_mask[bi, pi, t2[v], weapon[v]] ^= True
    return new_mask


def self_play_gnn_move(base_actor, improve_actor, episode, epoch,
                        n_edits=3, batch_size=4, para_size=8, logger=None):
    base_actor.eval()
    improve_actor.train()
    losses = Average_Meter()
    tot_base = tot_final = 0.0

    for ep in range(episode):
        nw, nt, mt, amm, prep, cost = get_random_moderate_problem_size()
        original_hyperparams = patch_hyperparameters_for_epoch(
            nw, nt, mt, amm, prep_list=prep, cost_list=cost)
        try:
            ae, wtp = generate_moderate_training_instances(
                batch_size=batch_size, num_weapons=nw, num_targets=nt,
                max_time=mt, amm_list=amm)
            ae = ae.unsqueeze(1).repeat(1, para_size, 1, 1).contiguous()
            wtp = wtp.unsqueeze(1).repeat(1, para_size, 1, 1).contiguous()

            B, P = batch_size, para_size
            flip_mask = torch.zeros(B, P, mt, nw, dtype=torch.bool, device=DEVICE)

            obj_prev, states, fire_state = simulate_with_flips(
                base_actor, ae, wtp, flip_mask, nw, nt, mt, return_fire_state=True)
            base_obj = obj_prev.clone()

            log_probs, rewards, valids = [], [], []
            for _ in range(n_edits):
                t1, t2, weapon, log_prob, valid = propose_move(
                    improve_actor, states, fire_state, nw, mt)
                flip_mask = apply_move(flip_mask, t1, t2, weapon, valid)

                obj_new, states, fire_state = simulate_with_flips(
                    base_actor, ae, wtp, flip_mask, nw, nt, mt, return_fire_state=True)

                log_probs.append(log_prob)
                rewards.append(obj_prev - obj_new)
                valids.append(valid)
                obj_prev = obj_new

            loss = 0.0
            for log_prob, reward, valid in zip(log_probs, rewards, valids):
                adv = reward - reward.mean(dim=1, keepdim=True)
                adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-6)
                loss = loss - (log_prob * adv * valid.float()).mean()
            loss = loss / max(len(log_probs), 1)

            improve_actor.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(improve_actor.parameters(), max_norm=1.0)
            improve_actor.optimizer.step()

            losses.push(torch.tensor(loss.item()), 1)
            tot_base += base_obj.mean().item()
            tot_final += obj_prev.mean().item()

            if logger is not None:
                logger.info(f"Episode {ep+1}/{episode} | loss {loss.item():.6f}")
        finally:
            restore_hyperparameters(original_hyperparams)

    return losses.result(), {
        'base_objective': tot_base / episode,
        'improved_objective': tot_final / episode,
    }
