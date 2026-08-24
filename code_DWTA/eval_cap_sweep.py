"""
Is the multifire auction's hand-set `max_per_target` cap still needed once
the learned auditor sits on top of it?

The cap (default 2) was introduced because uncapped greedy marginal-value
assignment inherits Greedy's myopia -- it can keep stacking weapons onto
whichever target currently has the highest marginal value, since the
built-in diminishing-returns term (survival[n] shrinks per assignment)
does not always overcome a high base value*probability. It was tuned by
inspecting SCIP solutions at small scale, and empirically beat the uncapped
version back when there was no auditor.

That situation has changed: the auditor's whole job is to cancel shots that
turn out not to pay, which is exactly the failure the cap was patching. If
the auditor absorbs it, a hand-tuned constant is replaced by a learned
mechanism -- a strictly better story for "what does the learned component
contribute", and one fewer magic number to defend.

Sweeps cap in {2, 3, unlimited} x {proposer alone, proposer+auditor} across
all 12 configs, against Sequential.
"""
import sys
from functools import partial

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_improve import create_gnn_actor_improve
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals
from Dynamic_Sampling_GNN_improve import score_all_slots

ALL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
SEQUENTIAL = {
    (5, 5, 5): 0.1694, (5, 7, 5): 0.3071, (10, 15, 5): 0.2996,
    (15, 15, 5): 0.0958, (15, 20, 5): 0.2073, (20, 30, 5): 0.2953,
    (30, 30, 10): 0.0162, (30, 40, 10): 0.0317, (40, 50, 10): 0.0283,
    (50, 50, 15): 0.0878, (50, 70, 15): 0.0798, (70, 100, 15): 0.1003,
}
N_EVAL = 10
SEED = 123
EVAL_EDITS = 10
BASE_CKPT = "result/Joint_base_seed5_altlr_best.pt"      # the ACTIVE-auditor checkpoint
IMPROVE_CKPT = "result/Joint_improve_seed5_altlr_best.pt"

CAPS = [("cap2", 2), ("cap3", 3), ("uncap", 10 ** 9)]


@torch.no_grad()
def simulate(base_actor, ae, wtp, flip_mask, nw, nt, mt, auction_fn):
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    original_value = env.original_target_value[:, :, 0:nt].sum(2)
    states = []
    for t in range(mt):
        states.append((env.assignment_encoding.clone(),
                        env.weapon_to_target_prob.clone(),
                        env.mask_per_weapon.clone()))
        policy, _ = base_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = (policy[:, :, :nw, :].argmax(dim=-1) < nt) ^ flip_mask[:, :, t, :]
        action = auction_fn(env.current_target_value[:, :, 0:nt],
                            env.weapon_to_target_prob[:, :, :nw, :nt],
                            env.mask_per_weapon[:, :, :nw, :nt] > 0,
                            must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    return (env.current_target_value[:, :, 0:nt].sum(2) / (original_value + 1e-8)), states


@torch.no_grad()
def eval_instance(base_actor, improve_actor, V, P, TW, nw, nt, mt, amm, prep, cost, auction_fn):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate(base_actor, ae, wtp, flip_mask, nw, nt, mt, auction_fn)
    base_obj = obj.mean().item()
    best = base_obj

    n_slots = mt * nw
    for _ in range(min(EVAL_EDITS, n_slots)):
        logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
        logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
        idx = logits.argmax(dim=-1)
        nf = flip_mask.reshape(1, 1, n_slots).clone()
        nf.scatter_(-1, idx.unsqueeze(-1), True)
        flip_mask = nf.reshape(1, 1, mt, nw)
        obj, states = simulate(base_actor, ae, wtp, flip_mask, nw, nt, mt, auction_fn)
        best = min(best, obj.mean().item())
    return base_obj, best


if __name__ == "__main__":
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()
    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.load_state_dict(torch.load(IMPROVE_CKPT, map_location=DEVICE, weights_only=False))
    improve_actor.eval()

    hdr = f"{'config':<15}{'Seq':>9}"
    for name, _ in CAPS:
        hdr += f"{name+' prop':>12}{name+' aud':>11}"
    print(hdr, flush=True)

    cols = {'seq': []}
    for name, _ in CAPS:
        cols[name + '_prop'] = []
        cols[name + '_aud'] = []

    for (M, N, T) in ALL_CONFIGS:
        rng = np.random.default_rng(SEED)
        acc = {name + s: [] for name, _ in CAPS for s in ('_prop', '_aud')}
        for _ in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            P = np.asarray(P)
            for name, cap in CAPS:
                fn = partial(auction_round_action_multifire, max_per_target=cap)
                b, a = eval_instance(base_actor, improve_actor, V, P, TW, M, N, T, AMM, PREP, COST, fn)
                acc[name + '_prop'].append(b)
                acc[name + '_aud'].append(a)

        seq = SEQUENTIAL[(M, N, T)]
        cols['seq'].append(seq)
        row = f"{M}M_{N}N_{T}T{'':<6}{seq:>9.4f}"
        for name, _ in CAPS:
            p = float(np.mean(acc[name + '_prop']))
            a = float(np.mean(acc[name + '_aud']))
            cols[name + '_prop'].append(p)
            cols[name + '_aud'].append(a)
            row += f"{p:>12.4f}{a:>11.4f}"
        print(row, flush=True)

    row = f"{'MEAN':<15}{np.mean(cols['seq']):>9.4f}"
    for name, _ in CAPS:
        row += f"{np.mean(cols[name+'_prop']):>12.4f}{np.mean(cols[name+'_aud']):>11.4f}"
    print("\n" + row)

    print()
    for name, _ in CAPS:
        w = sum(1 for i in range(len(ALL_CONFIGS)) if cols[name + '_aud'][i] < cols['seq'][i])
        label = name + ' + auditor'
        print(f"{label:<22} beats Sequential on {w}/{len(ALL_CONFIGS)} configs")
