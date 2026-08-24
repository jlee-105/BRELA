"""Pure auction (must_fire=None -- auction decides fire-vs-hold itself, no
RL involved) on the MODERATE temporal-dilemma curriculum, across all 12
tiered-benchmark scales. Mirrors eval_greedy_moderate_all_configs.py: same
seed=123 generator, same fixed instances per config, so Greedy/SCIP/Auction/
AM/POMO are all compared on identical data.
"""
import time

import numpy as np
import torch

from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

N_EVAL = 10
SEED = 123

ALL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]


@torch.no_grad()
def eval_instance_auction(V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        action = auction_round_action(remaining_value, prob, legal_mask)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


def run_config(M, N, T):
    rng = np.random.default_rng(SEED)  # same seed/order as common.rl4co_eval's moderate loader
    objs = []
    t0 = time.time()
    for i in range(N_EVAL):
        V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
        objs.append(eval_instance_auction(V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
    elapsed = time.time() - t0
    mean_obj = float(np.mean(objs))
    print(f"{M}M_{N}N_{T}T{'':<8}{mean_obj:>10.4f}  ({elapsed:.1f}s, n={N_EVAL})", flush=True)
    return mean_obj


if __name__ == "__main__":
    print(f"{'config':<18}{'auction':>10}")
    for M, N, T in ALL_CONFIGS:
        run_config(M, N, T)
