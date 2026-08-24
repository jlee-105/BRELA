"""
Does the improvement policy's gain scale with the EDIT BUDGET?

The trained improvement policy (rl/DWTA_GNN_TRAIN_improve_multiscale.py)
was trained with n_edits=3 and gave only 0.1357 -> 0.1327 (2.2%). Since
edits are chosen greedily and best-so-far is kept at inference, the edit
budget is a pure inference-time search parameter -- it can be raised
without retraining. This script sweeps it on the ALREADY-TRAINED
checkpoint, to decide whether a longer/retrained run at higher n_edits is
worth the cost before committing to it.
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
N_EVAL = 10
SEED = 123
BASE_CKPT = "result/CommSinkhornSCoPE_multiscale_seed5_best_actor.pt"
IMPROVE_CKPT = "result/Improve_multiscale_seed5_best_actor.pt"
BUDGETS = [3, 10, 30]


@torch.no_grad()
def eval_instance(base_actor, improve_actor, V, P, TW, nw, nt, mt, amm, prep, cost, budgets):
    """Returns {budget: best_objective_within_that_budget} -- one pass,
    recording the running best at each budget checkpoint (a budget of K is
    a prefix of a budget of K' > K, since edits are greedy)."""
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
    best = obj.mean().item()
    base_obj = best

    max_budget = max(budgets)
    n_slots = mt * nw
    out = {}
    for k in range(1, max_budget + 1):
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

    header = f"{'config':<16}{'base':>9}" + "".join(f"{'K='+str(b):>9}" for b in BUDGETS)
    print(header, flush=True)

    all_base = []
    all_budget = {b: [] for b in BUDGETS}
    for M, N, T in ALL_CONFIGS:
        rng = np.random.default_rng(SEED)
        base_objs = []
        budget_objs = {b: [] for b in BUDGETS}
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            b_obj, res = eval_instance(base_actor, improve_actor, V, np.asarray(P), TW,
                                        M, N, T, AMM, PREP, COST, BUDGETS)
            base_objs.append(b_obj)
            for b in BUDGETS:
                budget_objs[b].append(res[b])
        bm = float(np.mean(base_objs))
        all_base.append(bm)
        row = f"{M}M_{N}N_{T}T{'':<7}{bm:>9.4f}"
        for b in BUDGETS:
            v = float(np.mean(budget_objs[b]))
            all_budget[b].append(v)
            row += f"{v:>9.4f}"
        print(row, flush=True)

    row = f"{'MEAN':<16}{np.mean(all_base):>9.4f}"
    for b in BUDGETS:
        row += f"{np.mean(all_budget[b]):>9.4f}"
    print("\n" + row)
