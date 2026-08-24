"""
COMBINED edit operator: each edit is either a single FLIP or a paired MOVE,
chosen by the policy itself.

Motivation (2026-08-21): the single-flip operator
(Dynamic_Sampling_GNN_improve.py) reached the project best (0.1291), while
the move-only operator (Dynamic_Sampling_GNN_move.py) did WORSE (0.1330 at
its first eval, and far worse at Large tier: 30M_40N_10T 0.0632 vs flip's
0.0415). Diagnosis: move-only does not ENLARGE the neighborhood, it
SWITCHES to a different one -- it forces every edit to pair a cancel with an
add, so it cannot express the plain "just cancel this shot" or "just add
this shot" edits that the flip operator found valuable. This file makes the
neighborhood the genuine UNION of the two.

Each edit is factorized:
  Stage 1  pick any slot A = (t1, m) over all T*M slots.
  Stage 2  pick a partner for A from {NONE} u {rounds t != t1 of weapon m}.
             NONE  -> plain flip of A  (recovers the 0.1291 operator)
             t2    -> toggle both A and (t2, m), i.e. reschedule weapon m's
                      shot between rounds (recovers the move operator)

Because NONE is always available, this operator's neighborhood strictly
CONTAINS the single-flip neighborhood, so the policy can always fall back
to what already worked -- the failure mode of move-only cannot recur by
construction.

New file -- does not modify Dynamic_Sampling_GNN_improve.py or
Dynamic_Sampling_GNN_move.py.
"""
import os
import sys

import torch
import torch.nn as nn

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


class CombinedEditPolicy(nn.Module):
    """Wraps the slot-scoring improvement actor and adds the single extra
    parameter this operator needs: the logit of choosing NONE (i.e. keep
    the edit a plain flip rather than pairing it into a move).

    Kept as a wrapper rather than a field on EdgeAwareGNN_ACTOR_IMPROVE so
    that existing checkpoints of that class (including the one behind the
    current project best) still load with strict=True.
    """

    def __init__(self, slot_actor, none_init=2.0):
        # none_init biases the operator toward plain FLIPs at initialization.
        # With T round-options competing against a single NONE option, an
        # unbiased init makes ~83% of edits moves (measured: move_rate 0.94
        # at step 0), i.e. training would start far from the flip-only
        # behavior that produced the current project best. Starting near
        # flip-only and letting the policy learn to introduce moves where
        # they actually pay is the same conservative pattern used for the
        # Sinkhorn bonus scale (initialized to 0).
        super().__init__()
        self.slot_actor = slot_actor
        self.none_logit = nn.Parameter(torch.tensor(float(none_init)))

    def forward(self, *args, **kwargs):
        return self.slot_actor(*args, **kwargs)


def legal_slots_from_states(states, nw, nt):
    """[B,P,T,M] bool -- did weapon m have ANY legal target at round t in
    the rollout these states came from?

    Constraints (ammo, reload/preparation, target time windows) are already
    enforced inside the environment and encoded in mask_per_weapon, so this
    is read off directly rather than re-derived. Used to avoid spending the
    edit budget on edits that would be silently absorbed: telling a weapon
    to fire when it has no legal target leaves the rollout unchanged, which
    costs one edit and yields exactly zero reward.

    This is a proxy, not an oracle -- legality after an edit can differ from
    legality before it (an earlier cancel frees ammo/reload capacity later).
    The rollout is always re-simulated from scratch, so correctness never
    depends on this mask; only edit-budget efficiency does.
    """
    per_round = [mask[:, :, :nw, :nt].bool().any(dim=-1) for (_, _, mask) in states]
    return torch.stack(per_round, dim=2)  # [B,P,T,M]


def _sample_masked(logits, mask, greedy):
    """Softmax-sample over the last dim under `mask`; rows with nothing
    valid fall back to unmasked so the distribution stays well-defined."""
    valid_row = mask.any(dim=-1, keepdim=True)
    safe = logits.masked_fill(~(mask | ~valid_row), NEG)
    probs = torch.softmax(safe, dim=-1)
    n = probs.shape[-1]
    if greedy:
        idx = probs.argmax(dim=-1)
    else:
        idx = torch.multinomial(probs.reshape(-1, n), 1).reshape(probs.shape[:-1])
    lp = torch.log(probs.gather(-1, idx.unsqueeze(-1)).clamp_min(1e-8)).squeeze(-1)
    return idx, lp


def propose_combined(policy, states, nw, mt, fire_state, legal_state, greedy=False):
    """One combined edit. Returns (t1, t2, weapon, use_partner, log_prob).

    Candidates are restricted to slots where the toggle can actually change
    the rollout: a slot that currently FIRES (toggling cancels it) or a slot
    that is LEGAL (toggling can add a shot). Toggling an illegal, non-firing
    slot is a guaranteed no-op and would waste an edit.
    """
    logits = score_all_slots(policy.slot_actor, states, nw)  # [B,P,T,M], with grad
    B, P, T, M = logits.shape
    n_slots = T * M
    effective = fire_state | legal_state  # [B,P,T,M]

    # --- Stage 1: any effective slot ---
    idx1, lp1 = _sample_masked(logits.reshape(B, P, n_slots),
                                effective.reshape(B, P, n_slots), greedy)
    t1 = idx1 // M
    weapon = idx1 % M

    # --- Stage 2: NONE, or another effective round of the same weapon ---
    w_idx = weapon.view(B, P, 1, 1).expand(B, P, T, 1)
    same_weapon_scores = logits.gather(-1, w_idx).squeeze(-1)          # [B,P,T]
    same_weapon_ok = effective.gather(-1, w_idx).squeeze(-1)            # [B,P,T]
    round_idx = torch.arange(T, device=logits.device).view(1, 1, T).expand(B, P, T)
    same_weapon_ok = same_weapon_ok & (round_idx != t1.unsqueeze(-1))

    none_col = policy.none_logit.view(1, 1, 1).expand(B, P, 1)
    stage2 = torch.cat([same_weapon_scores, none_col], dim=-1)          # index T = NONE
    stage2_mask = torch.cat([same_weapon_ok,
                              torch.ones(B, P, 1, dtype=torch.bool, device=logits.device)], dim=-1)
    idx2, lp2 = _sample_masked(stage2, stage2_mask, greedy)

    use_partner = idx2 < T
    t2 = idx2.clamp_max(T - 1)

    return t1, t2, weapon, use_partner, lp1 + lp2


def apply_combined(flip_mask, t1, t2, weapon, use_partner):
    """Toggle slot A, and the partner slot too when one was chosen."""
    B, P, T, M = flip_mask.shape
    new_mask = flip_mask.clone()
    b_idx = torch.arange(B, device=flip_mask.device).view(B, 1).expand(B, P)
    p_idx = torch.arange(P, device=flip_mask.device).view(1, P).expand(B, P)

    new_mask[b_idx.reshape(-1), p_idx.reshape(-1), t1.reshape(-1), weapon.reshape(-1)] ^= True

    if use_partner.any():
        u = use_partner
        new_mask[b_idx[u], p_idx[u], t2[u], weapon[u]] ^= True
    return new_mask


def self_play_gnn_combined(base_actor, policy, episode, epoch,
                            n_edits=3, batch_size=4, para_size=8, logger=None):
    base_actor.eval()
    policy.train()
    losses = Average_Meter()
    tot_base = tot_final = 0.0
    tot_partner_rate = 0.0
    n_edit_total = 0

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

            log_probs, rewards = [], []
            for _ in range(n_edits):
                legal_state = legal_slots_from_states(states, nw, nt)
                t1, t2, weapon, use_partner, log_prob = propose_combined(
                    policy, states, nw, mt, fire_state, legal_state)
                flip_mask = apply_combined(flip_mask, t1, t2, weapon, use_partner)

                obj_new, states, fire_state = simulate_with_flips(
                    base_actor, ae, wtp, flip_mask, nw, nt, mt, return_fire_state=True)
                log_probs.append(log_prob)
                rewards.append(obj_prev - obj_new)
                obj_prev = obj_new

                tot_partner_rate += use_partner.float().mean().item()
                n_edit_total += 1

            loss = 0.0
            for log_prob, reward in zip(log_probs, rewards):
                adv = reward - reward.mean(dim=1, keepdim=True)
                adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-6)
                loss = loss - (log_prob * adv).mean()
            loss = loss / max(len(log_probs), 1)

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy.optimizer.step()

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
        'move_rate': tot_partner_rate / max(n_edit_total, 1),
    }
