"""
Can a LARGER edit budget push the flip-only improvement policy past
Sequential on the configs where it currently loses?

Context: at K=30 the improvement policy beats Sequential on the MEAN
(0.1291 vs 0.1432) but loses 8/12 configs individually -- it wins big on
Small/Medium and loses narrowly on Large/Battlefield. The budget sweep at
K=3/10/30 showed the small configs had already SATURATED (K exceeds their
total slot count: 5M_5N_5T has only 25 slots) while the large ones were
still descending monotonically (70M_100N_15T has 1050 slots -- K=30 edits
touches 2.9% of them). So the per-config losses may be a budget artifact
rather than a limitation of the method.

Edit budget is a pure inference-time parameter (greedy edits, keep-best),
so this needs no retraining. Uses the K=3-TRAINED checkpoint, which is the
better one (see brerla_improvement_local_search memory: the K=10-trained
retrain scored worse and overwrote the default path).
"""
import sys

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
BASE_CKPT = "result/CommSinkhornSCoPE_multiscale_seed5_best_actor.pt"
IMPROVE_CKPT = "result/Improve_multiscale_seed5_k3_best_actor.pt"
BUDGETS = [30, 60, 120, 250]


@torch.no_grad()
def eval_instance(base_actor, improve_actor, V, P, TW, nw, nt, mt, amm, prep, cost, budgets):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
    best = obj.mean().item()
    base_obj = best

    n_slots = mt * nw
    out = {}
    for k in range(1, max(budgets) + 1):
        if k <= n_slots:
            logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
            logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
            idx = logits.argmax(dim=-1)
            new_flip = flip_mask.reshape(1, 1, n_slots).clone()
            new_flip.scatter_(-1, idx.unsqueeze(-1), True)
            flip_mask = new_flip.reshape(1, 1, mt, nw)
            obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
            best = min(best, obj.mean().item())
        if k in budgets:
            out[k] = best
    return base_obj, out


if __name__ == "__main__":
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()

    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.load_state_dict(torch.load(IMPROVE_CKPT, map_location=DEVICE, weights_only=False))
    improve_actor.eval()

    print(f"{'config':<15}{'slots':>7}{'Seq':>9}{'base':>9}" +
          "".join(f"{'K='+str(b):>9}" for b in BUDGETS) + f"{'beats Seq?':>12}", flush=True)

    all_base, all_seq = [], []
    all_budget = {b: [] for b in BUDGETS}
    for M, N, T in ALL_CONFIGS:
        rng = np.random.default_rng(SEED)
        base_objs = []
        budget_objs = {b: [] for b in BUDGETS}
        for _ in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            b_obj, res = eval_instance(base_actor, improve_actor, V, np.asarray(P), TW,
                                        M, N, T, AMM, PREP, COST, BUDGETS)
            base_objs.append(b_obj)
            for b in BUDGETS:
                budget_objs[b].append(res[b])

        bm = float(np.mean(base_objs))
        seq = SEQUENTIAL[(M, N, T)]
        all_base.append(bm)
        all_seq.append(seq)
        row = f"{M}M_{N}N_{T}T{'':<6}{M*T:>7}{seq:>9.4f}{bm:>9.4f}"
        for b in BUDGETS:
            v = float(np.mean(budget_objs[b]))
            all_budget[b].append(v)
            row += f"{v:>9.4f}"
        final = float(np.mean(budget_objs[max(BUDGETS)]))
        row += f"{('YES' if final < seq else 'no'):>12}"
        print(row, flush=True)

    row = f"{'MEAN':<15}{'':>7}{np.mean(all_seq):>9.4f}{np.mean(all_base):>9.4f}"
    for b in BUDGETS:
        row += f"{np.mean(all_budget[b]):>9.4f}"
    print("\n" + row)
    wins = sum(1 for i in range(len(ALL_CONFIGS))
               if all_budget[max(BUDGETS)][i] < all_seq[i])
    print(f"Beats Sequential on {wins}/{len(ALL_CONFIGS)} configs at K={max(BUDGETS)}")
