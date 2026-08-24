"""
Fill Section~\\ref{sec:repair-results} of the manuscript.

The earlier budget-sweep script (eval_improve_budget_sweep.py) is NOT usable
for this: it runs the 1:1 eviction auction, which Lemma 1 of the paper shows
is a structurally different mechanism from the capacitated one the pipeline
now uses. Its numbers are therefore not comparable to Table 6 and must not be
reported. This script re-runs the same three questions against the current
pipeline instead:

  (a) marginal contribution of the repair layer on top of SCoPE-Comm,
  (b) the shape of the K-sweep -- the guarantee is that it cannot rise, so
      the empirical claim is WHERE IT FLATTENS,
  (c) which auction the construction pass used, held fixed at multifire.

Checkpoints: the "mfloop" pair, i.e. jointly trained with the capacitated
(multifire) auction inside the training loop, matching inference.
"""
import sys

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_improve import create_gnn_actor_improve
from common.Dynamic_Instance_generation import input_generation
from common.auction_refinement import auction_round_action_multifire
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
BASE_CKPT = "result/Joint_base_seed5_mfloop_best.pt"
IMPROVE_CKPT = "result/Joint_improve_seed5_mfloop_best.pt"
BUDGETS = [3, 10, 30, 100]


def auction(*args, **kwargs):
    """Uncapped capacitated auction -- the mechanism described in Sec. 5.4."""
    return auction_round_action_multifire(*args, **kwargs)


@torch.no_grad()
def eval_instance(base_actor, improve_actor, V, P, TW, nw, nt, mt, amm, prep, cost, budgets):
    """One greedy edit sequence, recording the running best at each budget.
    Valid because edits are greedy and never undone, so the schedules seen
    under budget K are a prefix of those seen under any larger budget."""
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                               max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt,
                                      auction_fn=auction)
    best = obj.mean().item()
    base_obj = best

    n_slots = mt * nw
    out = {}
    first_flat = None          # budget at which the running best stops moving
    for k in range(1, max(budgets) + 1):
        if k <= n_slots:
            logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
            logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
            idx = logits.argmax(dim=-1)
            new_flip = flip_mask.reshape(1, 1, n_slots).clone()
            new_flip.scatter_(-1, idx.unsqueeze(-1), True)
            flip_mask = new_flip.reshape(1, 1, mt, nw)
            obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt,
                                              auction_fn=auction)
            cur = obj.mean().item()
            if cur < best - 1e-9:
                best = cur
                first_flat = None
            elif first_flat is None:
                first_flat = k
        if k in budgets:
            out[k] = best
    return base_obj, out, (first_flat if first_flat is not None else max(budgets))


if __name__ == "__main__":
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()

    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.load_state_dict(torch.load(IMPROVE_CKPT, map_location=DEVICE, weights_only=False))
    improve_actor.eval()

    # The base class carries a learned Sinkhorn scale. The manuscript describes
    # SCoPE-Comm as communication-only, so report the trained value: if it is
    # not ~0, the architecture section is understating what the model contains.
    scale = getattr(base_actor, 'sinkhorn_scale', None)
    if scale is not None:
        print(f"[check] trained sinkhorn_scale = {float(scale):.6f}"
              f"   (near 0 means the Sinkhorn term is inactive)\n", flush=True)

    header = f"{'config':<16}{'K=0':>9}" + "".join(f"{'K='+str(b):>9}" for b in BUDGETS) + f"{'flat@':>8}"
    print(header, flush=True)

    all_base, all_flat = [], []
    all_budget = {b: [] for b in BUDGETS}
    for M, N, T in ALL_CONFIGS:
        rng = np.random.default_rng(SEED)
        base_objs, flats = [], []
        budget_objs = {b: [] for b in BUDGETS}
        for _ in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            b_obj, res, flat = eval_instance(base_actor, improve_actor, V, np.asarray(P), TW,
                                             M, N, T, AMM, PREP, COST, BUDGETS)
            base_objs.append(b_obj)
            flats.append(flat)
            for b in BUDGETS:
                budget_objs[b].append(res[b])
        bm = float(np.mean(base_objs))
        fm = float(np.mean(flats))
        all_base.append(bm)
        all_flat.append(fm)
        row = f"{M}M_{N}N_{T}T{'':<7}{bm:>9.4f}"
        for b in BUDGETS:
            v = float(np.mean(budget_objs[b]))
            all_budget[b].append(v)
            row += f"{v:>9.4f}"
        print(row + f"{fm:>8.1f}", flush=True)

    row = f"{'MEAN':<16}{np.mean(all_base):>9.4f}"
    for b in BUDGETS:
        row += f"{np.mean(all_budget[b]):>9.4f}"
    print("\n" + row + f"{np.mean(all_flat):>8.1f}")
