"""
JOINT (single-process) training of the two-agent pipeline:

  Agent 1 -- the PROPOSER: the N+1-way policy (Sinkhorn/comm actor) that
      decides, per weapon per round, whether to fire (target choice is then
      handled by the auction). Trained by REINFORCE on its own rollout,
      exactly the recipe already validated in
      Dynamic_Sampling_GNN_moderate.py::self_play_gnn_moderate (reward-to-go,
      POMO-style baseline across the para dimension, ammo-utilization bonus,
      entropy bonus, NO critic).

  Agent 2 -- the AUDITOR: the improvement policy that inspects the
      proposer's COMPLETED schedule and flips individual (round, weapon)
      fire/hold decisions to improve the true objective. Trained by
      REINFORCE on per-edit objective deltas (dense, directly attributable),
      as in Dynamic_Sampling_GNN_improve.py.

Both agents are updated in the SAME loop on the SAME instance each step, so
the auditor continuously co-adapts to a proposer that is still changing --
unlike the earlier two-stage setup (train proposer, freeze it, then train
auditor on top), which produced the current project best of 0.1291 (see
brerla_improvement_local_search memory).

Note on coupling: the proposer is rewarded on ITS OWN rollout quality, not
on the post-audit objective. This is deliberate -- it keeps the proposer's
training signal identical to the recipe already known to work here, and
avoids handing it a reward that depends on a second, simultaneously-moving
policy. A fully-coupled variant (proposer rewarded on the post-audit
result) is a real alternative but is a moving-target/credit-assignment risk
this project has repeatedly been burned by; not attempted here.

New file -- does not modify any existing training code.
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

from Dynamic_Sampling_GNN_moderate import get_random_moderate_problem_size
from Dynamic_Sampling_GNN_improve import simulate_with_flips, score_all_slots
from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters


def get_random_wide_problem_size(m_lo=5, m_hi=30, t_lo=5, t_hi=10):
    """Wide-scale curriculum, structured like the held-out eval configs.

    The standing curriculum (`get_random_moderate_problem_size`) draws M, N
    and T independently from U[5,7], while evaluation runs up to
    70M_100N_15T -- a 10-14x extrapolation in weapon count. That asymmetry
    hurts the PARALLEL proposer specifically: sequential decoding carries
    its coordination mechanism to any M for free (each weapon conditions on
    the ones already committed this round), whereas a parallel decoder has
    to have LEARNED coordination, and coordination patterns learned at M=5-7
    have no reason to transfer to M=70. This is the leading suspect for the
    Large/Battlefield-tier deficit, and training scale is the one lever
    never tried against it.

    Mirrors the eval configs' structure rather than sampling independently:
    N >= M always (more targets than weapons), and T grows with scale, as in
    (5,5,5) ... (30,40,10) ... (70,100,15).
    """
    num_weapons = random.randint(m_lo, m_hi)
    num_targets = random.randint(num_weapons, max(num_weapons, int(round(1.4 * num_weapons))))
    max_time = random.randint(t_lo, t_hi)
    # Same ammo/prep/cost formulas as the standing curriculum, so the only
    # thing that changes is problem SCALE (ammo already scales with T).
    amm_list = [random.randint(1, max(2, max_time // 2 + 1)) for _ in range(num_weapons)]
    prep_list = [random.randint(0, 1) for _ in range(num_weapons)]
    cost_list = [random.randint(2, 10) * 10 for _ in range(num_weapons)]
    return (num_weapons, num_targets, max_time, amm_list, prep_list, cost_list)


def _train_proposer(actor, ae, wtp, nw, nt, mt, amm_list, entropy_coef=1e-3):
    """One REINFORCE update for the proposer on its own sampled rollout.
    Mirrors self_play_gnn_moderate's math exactly (reward-to-go, per-step
    para-baselined advantage, ammo bonus, entropy bonus, no critic)."""
    env = Environment(assignment_encoding=ae.clone(),
                       weapon_to_target_prob=wtp.clone(), max_time=mt)

    log_probs, entropies, rewards = [], [], []
    total_fires = None
    original_value = env.original_target_value[:, :, 0:nt].sum(2)
    prev_value = original_value.clone()

    for _ in range(mt):
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)

        safe_policy = policy.clamp_min(1e-8)
        entropies.append(-(safe_policy * safe_policy.log()).sum(dim=-1))

        b_sz, p_sz = env.assignment_encoding.shape[:2]
        flat_policy = policy.reshape(-1, nt + 1)
        action = torch.multinomial(flat_policy, 1).view(b_sz, p_sz, nw)

        per_weapon_log_prob = torch.log(
            policy.gather(-1, action.unsqueeze(-1)).clamp_min(1e-8)
        ).squeeze(-1)
        log_probs.append(per_weapon_log_prob.sum(dim=-1, keepdim=True))

        env.update_internal_variables_parallel(selected_actions=action)

        fires = (action < nt).float().sum(dim=-1, keepdim=True)
        total_fires = fires if total_fires is None else total_fires + fires

        curr_value = env.current_target_value[:, :, 0:nt].sum(2)
        rewards.append((prev_value - curr_value) / (original_value + 1e-8))
        prev_value = curr_value

        env.time_update()

    total_ammo = float(sum(amm_list))
    rewards[-1] = rewards[-1] + 0.1 * (total_fires.squeeze(-1) / max(total_ammo, 1.0))

    T_steps = len(rewards)
    running = torch.zeros_like(rewards[0])
    returns = [None] * T_steps
    for t in reversed(range(T_steps)):
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


def _train_auditor(base_actor, improve_actor, ae, wtp, nw, nt, mt, n_edits, auction_fn=None):
    """One REINFORCE update for the auditor over a sequence of n_edits
    flips applied to the proposer's completed (argmax + auction) schedule.
    Reward for each edit is the objective delta it caused."""
    B, P = ae.shape[:2]
    flip_mask = torch.zeros(B, P, mt, nw, dtype=torch.bool, device=DEVICE)

    obj_prev, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt,
                                            auction_fn=auction_fn)
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

        obj_new, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt,
                                               auction_fn=auction_fn)
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


def self_play_joint(base_actor, improve_actor, episode, epoch,
                     n_edits=3, batch_size=8, para_size=8, entropy_coef=1e-3,
                     train_proposer_this_step=True, problem_size_fn=None,
                     auction_fn=None, logger=None):
    """One joint step, ALTERNATING: exactly one of the two agents is updated
    per call (`train_proposer_this_step` selects which), rather than both.

    Rationale: updating both every step makes each agent chase a target that
    moved within the same step, and costs a full proposer episode PLUS
    (n_edits+1) re-simulations every time. Alternating gives each agent a
    briefly-stationary counterpart to learn against -- the standard
    alternating-update pattern for two-network training -- and is cheaper on
    average, since proposer-only steps skip the edit re-simulations entirely.

    Returns (proposer_loss, auditor_loss, info); the loss belonging to the
    agent NOT updated this step is 0 and its info fields are absent.
    """
    base_actor.train()
    improve_actor.train()

    prop_losses, aud_losses = Average_Meter(), Average_Meter()
    tot_destr = tot_base = tot_audited = 0.0

    if problem_size_fn is None:
        problem_size_fn = get_random_moderate_problem_size

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
                p_loss, destr = _train_proposer(base_actor, ae, wtp, nw, nt, mt, amm, entropy_coef)
                prop_losses.push(torch.tensor(p_loss), 1)
                tot_destr += destr
            else:
                base_actor.eval()  # auditor sees the proposer's deterministic (argmax) behavior
                a_loss, base_obj, audited_obj = _train_auditor(
                    base_actor, improve_actor, ae, wtp, nw, nt, mt, n_edits, auction_fn=auction_fn)
                base_actor.train()
                aud_losses.push(torch.tensor(a_loss), 1)
                tot_base += base_obj
                tot_audited += audited_obj

            if logger is not None:
                who = 'proposer' if train_proposer_this_step else 'auditor'
                logger.info(f"Episode {ep+1}/{episode} | updated {who}")
        finally:
            restore_hyperparameters(original_hyperparams)

    # Average_Meter.result() divides by a zero count for whichever agent was
    # not updated this step (returning nan), so report 0.0 for it instead.
    p_out = prop_losses.result() if train_proposer_this_step else 0.0
    a_out = 0.0 if train_proposer_this_step else aud_losses.result()

    return p_out, a_out, {
        'destruction_ratio': tot_destr / episode,
        'base_objective': tot_base / episode,
        'audited_objective': tot_audited / episode,
    }
