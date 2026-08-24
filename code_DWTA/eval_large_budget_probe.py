"""
Decisive cheap probe: on ONE representative Large-tier config where the
improvement pipeline loses to Sequential, does a much larger edit budget
ever close the gap?

Rationale: the K=3/10/30 sweep showed the improvement curve flattening hard
at Large tier (30M_40N_10T: 0.0642 -> 0.0505 -> 0.0424 -> 0.0415, versus
Sequential's 0.0317), so budget alone may not be enough. But the budget is
also wildly unequal across scales -- K=30 covers 120% of 5M_5N_5T's 25
slots but only 2.9% of 70M_100N_15T's 1050 -- so the Large-tier deficit
might be a budget artifact after all. Running one config to a large budget
answers this for ~1/6 the cost of sweeping the whole tier.

Interpretation:
  - if the curve crosses Sequential, scale the budget with problem size and
    re-sweep the tier;
  - if it plateaus well above, the Large-tier gap is structural (consistent
    with the manuscript's own Proposition 2: the parallel decoder's
    within-round coordination disadvantage grows with M) and no amount of
    single-flip editing will fix it -- stop spending compute here.
"""
import sys
import time

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_improve import create_gnn_actor_improve
from common.Dynamic_Instance_generation import input_generation
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals
from Dynamic_Sampling_GNN_improve import simulate_with_flips, score_all_slots

CONFIG = (30, 40, 10)          # Large tier, clear loss margin (0.0415 vs Seq 0.0317)
SEQUENTIAL = 0.0317
N_EVAL = 10
SEED = 123
MAX_BUDGET = 250
REPORT_AT = [3, 10, 30, 60, 120, 250]

BASE_CKPT = "result/CommSinkhornSCoPE_multiscale_seed5_best_actor.pt"
IMPROVE_CKPT = "result/Improve_multiscale_seed5_k3_best_actor.pt"


@torch.no_grad()
def probe(base_actor, improve_actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
    best = obj.mean().item()

    n_slots = mt * nw
    out = {0: best}
    for k in range(1, MAX_BUDGET + 1):
        if k > n_slots:
            break
        logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
        logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
        idx = logits.argmax(dim=-1)
        new_flip = flip_mask.reshape(1, 1, n_slots).clone()
        new_flip.scatter_(-1, idx.unsqueeze(-1), True)
        flip_mask = new_flip.reshape(1, 1, mt, nw)
        obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
        best = min(best, obj.mean().item())
        if k in REPORT_AT:
            out[k] = best
    return out


if __name__ == "__main__":
    M, N, T = CONFIG
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()
    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.load_state_dict(torch.load(IMPROVE_CKPT, map_location=DEVICE, weights_only=False))
    improve_actor.eval()

    print(f"probing {M}M_{N}N_{T}T ({M*T} slots), Sequential reference = {SEQUENTIAL:.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    per_k = {k: [] for k in [0] + REPORT_AT}
    t0 = time.time()
    for i in range(N_EVAL):
        V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
        res = probe(base_actor, improve_actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST)
        for k, v in res.items():
            per_k[k].append(v)
        print(f"  instance {i+1}/{N_EVAL} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'K':>6}{'objective':>12}{'vs Seq':>10}")
    for k in [0] + REPORT_AT:
        if per_k[k]:
            m = float(np.mean(per_k[k]))
            print(f"{k:>6}{m:>12.4f}{('BEATS' if m < SEQUENTIAL else f'+{m-SEQUENTIAL:.4f}'):>10}")
