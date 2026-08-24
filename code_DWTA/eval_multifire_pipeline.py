"""
Does swapping the 1:1 eviction auction for the capacitated MULTIFIRE
auction close the Large/Battlefield gap to Sequential?

Motivation (diagnose_large_gap.py, 2026-08-23): at Large/Battlefield tier
our pipeline's per-round dispersion is EXACTLY 1.000 -- two weapons can
never engage the same target in the same round -- because the 1:1 eviction
auction structurally forbids it. Sequential runs at 0.891/0.918 dispersion
with a wasteful-redundancy rate of 0.001, i.e. it concentrates fire and
that concentration is almost never wasted. Meanwhile we FIRE MORE (0.275
vs 0.260) and BURN MORE AMMO (0.901 vs 0.853) yet destroy LESS -- so the
deficit is shot effectiveness, not shot volume, exactly what an inability
to double-team a high-value target would produce.

`auction_round_action_multifire` (Bertsekas & Castanon 1989 capacitated
transportation auction, max_per_target=2) lifts precisely that constraint,
and is inference-only -- no retraining. A previous session measured it
winning decisively at 30M_30N_10T (0.0251 -> 0.0137, beating Sequential's
0.0162, 9/10 instances flipping) but rejected it because the 12-config
MEAN got worse; that was before the diagnosis explained why it helps where
it helps, and before the auditor layer existed.

Reports 1:1 vs multifire for both the proposer alone and proposer+auditor,
per config, against Sequential.
"""
import sys

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_improve import create_gnn_actor_improve
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action, auction_round_action_multifire
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
BASE_CKPT = "result/Joint_base_seed5_simlr_best.pt"
IMPROVE_CKPT = "result/Joint_improve_seed5_simlr_best.pt"


@torch.no_grad()
def simulate(base_actor, ae, wtp, flip_mask, nw, nt, mt, auction_fn):
    """Same contract as Dynamic_Sampling_GNN_improve.simulate_with_flips but
    with the auction function injectable, so 1:1 and multifire can be run
    through an otherwise identical pipeline."""
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
    final_value = env.current_target_value[:, :, 0:nt].sum(2)
    return final_value / (original_value + 1e-8), states


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

    print(f"{'config':<15}{'Seq':>9}{'1:1 prop':>10}{'1:1 aud':>10}"
          f"{'mf prop':>10}{'mf aud':>10}{'best beats Seq?':>17}", flush=True)

    cols = {k: [] for k in ['seq', 'p11', 'a11', 'pmf', 'amf']}
    wins = 0
    for (M, N, T) in ALL_CONFIGS:
        rng = np.random.default_rng(SEED)
        acc = {k: [] for k in ['p11', 'a11', 'pmf', 'amf']}
        for _ in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            P = np.asarray(P)
            b, a = eval_instance(base_actor, improve_actor, V, P, TW, M, N, T, AMM, PREP, COST,
                                  auction_round_action)
            acc['p11'].append(b); acc['a11'].append(a)
            b, a = eval_instance(base_actor, improve_actor, V, P, TW, M, N, T, AMM, PREP, COST,
                                  auction_round_action_multifire)
            acc['pmf'].append(b); acc['amf'].append(a)

        seq = SEQUENTIAL[(M, N, T)]
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        best = min(m['a11'], m['amf'])
        if best < seq:
            wins += 1
        cols['seq'].append(seq)
        for k in acc:
            cols[k].append(m[k])
        print(f"{M}M_{N}N_{T}T{'':<6}{seq:>9.4f}{m['p11']:>10.4f}{m['a11']:>10.4f}"
              f"{m['pmf']:>10.4f}{m['amf']:>10.4f}"
              f"{('YES' if best < seq else 'no'):>17}", flush=True)

    print(f"\n{'MEAN':<15}{np.mean(cols['seq']):>9.4f}{np.mean(cols['p11']):>10.4f}"
          f"{np.mean(cols['a11']):>10.4f}{np.mean(cols['pmf']):>10.4f}{np.mean(cols['amf']):>10.4f}")
    print(f"\nBest-of-both-auctions beats Sequential on {wins}/{len(ALL_CONFIGS)} configs")
