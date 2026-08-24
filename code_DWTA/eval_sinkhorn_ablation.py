"""
Resolve the manuscript's UNREPORTED-COMPONENT TODO: the trained checkpoint
carries a learned Sinkhorn bonus scalar (sinkhorn_scale=0.0213). Is it doing
anything? Decisive test: same checkpoint, same instances, construction pass
with the scalar as trained vs forced to exactly 0.

Protocol matches eval_repair_paper_table.py (seed 123, 10 instances/config,
many-to-one auction, zero flip mask = construction only) so the "as trained"
column should reproduce that script's K=0 column.
"""
import sys

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.Dynamic_Instance_generation import input_generation
from common.auction_refinement import auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals
from Dynamic_Sampling_GNN_improve import simulate_with_flips

ALL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123
BASE_CKPT = "result/Joint_base_seed5_mfloop_best.pt"


@torch.no_grad()
def construction_objective(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                               max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, _ = simulate_with_flips(actor, ae, wtp, flip_mask, nw, nt, mt,
                                 auction_fn=auction_round_action_multifire)
    return obj.mean().item()


if __name__ == "__main__":
    actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    actor.eval()

    trained_scale = float(actor.sinkhorn_scale.detach())
    print(f"trained sinkhorn_scale = {trained_scale:.6f}", flush=True)

    print(f"\n{'config':<16}{'as trained':>12}{'scale=0':>12}{'delta':>12}", flush=True)
    on_all, off_all = [], []
    for M, N, T in ALL_CONFIGS:
        # Same instance stream for both settings.
        rng = np.random.default_rng(SEED)
        instances = [generate_moderate_temporal_instance(M, N, T, rng=rng) for _ in range(N_EVAL)]

        actor.sinkhorn_scale.data.fill_(trained_scale)
        on = [construction_objective(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST)
              for (V, P, TW, AMM, PREP, COST) in instances]

        actor.sinkhorn_scale.data.fill_(0.0)
        off = [construction_objective(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST)
               for (V, P, TW, AMM, PREP, COST) in instances]

        on_m, off_m = float(np.mean(on)), float(np.mean(off))
        on_all.append(on_m)
        off_all.append(off_m)
        print(f"{M}M_{N}N_{T}T{'':<7}{on_m:>12.4f}{off_m:>12.4f}{off_m-on_m:>+12.4f}", flush=True)

    print(f"\n{'MEAN':<16}{np.mean(on_all):>12.4f}{np.mean(off_all):>12.4f}"
          f"{np.mean(off_all)-np.mean(on_all):>+12.4f}")
    print("\n(delta ~ 0 means the Sinkhorn term is inactive and can be reported as such;")
    print(" a nonzero delta means the term must be described in the architecture section.)")
