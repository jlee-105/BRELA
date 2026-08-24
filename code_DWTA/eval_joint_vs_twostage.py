"""
Head-to-head at a MATCHED edit budget: jointly-trained (single-process,
alternating proposer/auditor) vs the two-stage setup (pretrained proposer
frozen, auditor trained on top).

The joint run was evaluated during training at K=10 only (0.1310), while
the two-stage best number (0.1291) was measured at K=30. Edit budget is a
pure inference-time parameter, so this re-scores BOTH pipelines at the same
budgets and reports them side by side -- otherwise the comparison is
confounded by budget, not by training scheme.

Also reports the proposer-alone objective for each, so the auditor's
marginal contribution is separable from the proposer's own quality.
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
SEQUENTIAL_MEAN = 0.1432
N_EVAL = 10
SEED = 123
BUDGETS = [3, 10, 30]

PIPELINES = {
    "two-stage": ("result/CommSinkhornSCoPE_multiscale_seed5_best_actor.pt",
                   "result/Improve_multiscale_seed5_k3_best_actor.pt"),
    "joint-alt": ("result/Joint_base_seed5_altlr_best.pt",
                   "result/Joint_improve_seed5_altlr_best.pt"),
    "joint-sim": ("result/Joint_base_seed5_simlr_best.pt",
                   "result/Joint_improve_seed5_simlr_best.pt"),
}


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


def run_pipeline(name, base_ckpt, imp_ckpt):
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(base_ckpt, map_location=DEVICE, weights_only=False))
    base_actor.eval()

    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.load_state_dict(torch.load(imp_ckpt, map_location=DEVICE, weights_only=False))
    improve_actor.eval()

    print(f"\n=== {name} ===", flush=True)
    print(f"    base={base_ckpt}\n    improve={imp_ckpt}", flush=True)
    print(f"{'config':<15}{'proposer':>10}" + "".join(f"{'K='+str(b):>9}" for b in BUDGETS), flush=True)

    all_base = []
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
        all_base.append(bm)
        row = f"{M}M_{N}N_{T}T{'':<6}{bm:>10.4f}"
        for b in BUDGETS:
            v = float(np.mean(budget_objs[b]))
            all_budget[b].append(v)
            row += f"{v:>9.4f}"
        print(row, flush=True)

    means = {b: float(np.mean(all_budget[b])) for b in BUDGETS}
    row = f"{'MEAN':<15}{np.mean(all_base):>10.4f}" + "".join(f"{means[b]:>9.4f}" for b in BUDGETS)
    print(row, flush=True)
    return float(np.mean(all_base)), means


if __name__ == "__main__":
    results = {}
    for name, (b_ck, i_ck) in PIPELINES.items():
        try:
            results[name] = run_pipeline(name, b_ck, i_ck)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {name}: {e}", flush=True)

    print("\n\n=== SUMMARY (mean over 12 configs, lower is better) ===")
    print(f"{'pipeline':<15}{'proposer':>10}" + "".join(f"{'K='+str(b):>9}" for b in BUDGETS))
    for name, (bm, means) in results.items():
        print(f"{name:<15}{bm:>10.4f}" + "".join(f"{means[b]:>9.4f}" for b in BUDGETS))
    print(f"{'Sequential':<15}{'-':>10}{SEQUENTIAL_MEAN:>9.4f}" + " " * 18 + "  (reference)")
