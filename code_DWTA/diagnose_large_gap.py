"""
WHERE does the Large/Battlefield-tier gap to Sequential come from?

Aggregate objectives say our pipeline loses every Large/Battlefield config
narrowly but consistently (e.g. 30M_40N_10T: proposer 0.0642, +auditor
0.0415, Sequential 0.0317) while winning big at Small/Medium. Since the
proposer alone is ~2x worse than Sequential there, the deficit is in the
PROPOSER, not in the auditor's budget -- but "the proposer is worse" is not
actionable. This decomposes the difference into behaviours we can act on:

  fire_rate            fraction of weapon-decisions that actually fire
                       -- are we firing too little (hoarding ammo) or too
                       much (wasting it early)?
  ammo_used            fraction of total ammunition expended by episode end
                       -- leftover ammo is unrealized destruction.
  dispersion           distinct targets / weapons firing, per round
                       -- low means many weapons piled onto few targets,
                       the mean-field failure mode Proposition 2 predicts
                       should worsen as M grows.
  wasteful_redundancy  share of redundant same-round hits landing on
                       already-nearly-dead targets (survival < threshold)
                       -- redundancy that was NOT worth it.
  destruction_by_round when in the episode value is actually destroyed
                       -- reveals mis-timing (all early / all late).

Compares three arms on identical instances: Sequential, our proposer alone,
and proposer+auditor. Run on Large and Battlefield configs.
"""
import sys
from collections import Counter

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_improve import create_gnn_actor_improve
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals, _round_dispersion, _round_wasteful_redundancy
from Dynamic_Sampling_GNN_improve import simulate_with_flips, score_all_slots

CONFIGS = [(30, 40, 10), (70, 100, 15)]
N_EVAL = 10
SEED = 123
EVAL_EDITS = 10

BASE_CKPT = "result/Joint_base_seed5_simlr_best.pt"
IMPROVE_CKPT = "result/Joint_improve_seed5_simlr_best.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"


def _blank_stats():
    return {'fire_rate': [], 'ammo_used': [], 'dispersion': [], 'waste': [],
            'obj': [], 'destr_by_round': []}


def _collect(stats, init_value, remaining, fires, decisions, ammo_total,
             disps, wastes, destr_by_round):
    stats['obj'].append(remaining / max(init_value, 1e-8))
    stats['fire_rate'].append(fires / max(decisions, 1))
    stats['ammo_used'].append(fires / max(ammo_total, 1))
    d = [x for x in disps if x is not None]
    w = [x for x in wastes if x is not None]
    if d:
        stats['dispersion'].append(float(np.mean(d)))
    if w:
        stats['waste'].append(float(np.mean(w)))
    stats['destr_by_round'].append(destr_by_round)


@torch.no_grad()
def run_parallel(base_actor, improve_actor, ae, wtp, nw, nt, mt, use_auditor):
    """One episode of proposer(+auditor) with per-round behavioural stats.
    When use_auditor, the flip mask is the one the auditor greedily settles
    on (best-so-far), then the episode is replayed under it to collect
    stats for the schedule actually reported."""
    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)

    if use_auditor:
        obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
        best_obj, best_mask = obj.mean().item(), flip_mask.clone()
        n_slots = mt * nw
        for _ in range(min(EVAL_EDITS, n_slots)):
            logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
            logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
            idx = logits.argmax(dim=-1)
            nf = flip_mask.reshape(1, 1, n_slots).clone()
            nf.scatter_(-1, idx.unsqueeze(-1), True)
            flip_mask = nf.reshape(1, 1, mt, nw)
            obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
            if obj.mean().item() < best_obj:
                best_obj, best_mask = obj.mean().item(), flip_mask.clone()
        flip_mask = best_mask

    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()
    prev = init_value
    fires = decisions = 0
    disps, wastes, destr = [], [], []

    for t in range(mt):
        orig = env.original_target_value[0, 0, 0:nt]
        survival_before = (env.current_target_value[0, 0, 0:nt] / orig.clamp_min(1e-8)).tolist()

        policy, _ = base_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = (policy[:, :, :nw, :].argmax(dim=-1) < nt) ^ flip_mask[:, :, t, :]
        action = auction_round_action(env.current_target_value[:, :, 0:nt],
                                       env.weapon_to_target_prob[:, :, :nw, :nt],
                                       env.mask_per_weapon[:, :, :nw, :nt] > 0,
                                       must_fire=must_fire)
        fired = [int(a) for a in action[0, 0].tolist() if a < nt]
        fires += len(fired)
        decisions += nw
        disps.append(_round_dispersion(fired))
        wastes.append(_round_wasteful_redundancy(fired, survival_before))

        env.update_internal_variables_parallel(selected_actions=action)
        cur = env.current_target_value[:, :, 0:nt].sum().item()
        destr.append((prev - cur) / max(init_value, 1e-8))
        prev = cur
        env.time_update()

    return init_value, prev, fires, decisions, disps, wastes, destr


@torch.no_grad()
def run_sequential(seq_actor, ae, wtp, nw, nt, mt):
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()
    prev = init_value
    fires = decisions = 0
    disps, wastes, destr = [], [], []
    n_edges = nw * nt

    for t in range(mt):
        orig = env.original_target_value[0, 0, 0:nt]
        survival_before = (env.current_target_value[0, 0, 0:nt] / orig.clamp_min(1e-8)).tolist()
        fired = []
        for _ in range(nw):
            policy, _ = seq_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask.clone())
            idx = int(policy.view(-1, n_edges + 1).argmax(dim=1).item())
            if idx < n_edges:
                fired.append(idx % nt)
            env.update_internal_variables(selected_action=torch.tensor([[idx]], device=DEVICE))
        fires += len(fired)
        decisions += nw
        disps.append(_round_dispersion(fired))
        wastes.append(_round_wasteful_redundancy(fired, survival_before))

        cur = env.current_target_value[:, :, 0:nt].sum().item()
        destr.append((prev - cur) / max(init_value, 1e-8))
        prev = cur
        env.time_update()

    return init_value, prev, fires, decisions, disps, wastes, destr


if __name__ == "__main__":
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()
    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.load_state_dict(torch.load(IMPROVE_CKPT, map_location=DEVICE, weights_only=False))
    improve_actor.eval()
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    seq_actor.eval()

    for (M, N, T) in CONFIGS:
        print(f"\n{'='*72}\n{M}M_{N}N_{T}T\n{'='*72}", flush=True)
        rng = np.random.default_rng(SEED)
        arms = {'Sequential': _blank_stats(), 'proposer': _blank_stats(), 'proposer+auditor': _blank_stats()}

        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            patch_globals(M, N, T, AMM, PREP, COST)
            ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=np.asarray(P), TW=TW,
                                        max_time=T, batch_size=1, alpha=1.0, amm=AMM)
            ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
            ammo_total = float(sum(AMM))

            runs = {
                'Sequential': run_sequential(seq_actor, ae, wtp, M, N, T),
                'proposer': run_parallel(base_actor, improve_actor, ae, wtp, M, N, T, False),
                'proposer+auditor': run_parallel(base_actor, improve_actor, ae, wtp, M, N, T, True),
            }
            for name, r in runs.items():
                init_v, remaining, fires, decisions, disps, wastes, destr = r
                _collect(arms[name], init_v, remaining, fires, decisions, ammo_total, disps, wastes, destr)
            print(f"  instance {i+1}/{N_EVAL}", flush=True)

        print(f"\n{'arm':<20}{'obj':>8}{'fire_rate':>11}{'ammo_used':>11}{'dispersion':>12}{'waste':>8}")
        for name, s in arms.items():
            print(f"{name:<20}{np.mean(s['obj']):>8.4f}{np.mean(s['fire_rate']):>11.3f}"
                  f"{np.mean(s['ammo_used']):>11.3f}{np.mean(s['dispersion']):>12.3f}"
                  f"{np.mean(s['waste']):>8.3f}")

        print(f"\ndestruction by round (fraction of initial value):")
        for name, s in arms.items():
            per_round = np.mean(np.array(s['destr_by_round']), axis=0)
            print(f"  {name:<18}" + " ".join(f"{v:.3f}" for v in per_round))
