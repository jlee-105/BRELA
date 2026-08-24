"""Measures ammo-utilization rate (fraction of total available ammo actually
fired) for SCoPE vs SCoPE-Comm (both with auction refinement) across all 12
moderate-curriculum configs -- quantitative support for the paper's
coordination-analysis section (participation-level coordination gap)."""
import argparse

import numpy as np
import torch

from common.DWTA_GNN import create_gnn_actor
from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123


@torch.no_grad()
def ammo_utilization_one_instance(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)

    total_fires = 0
    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :nw, :].argmax(dim=-1)
        must_fire = policy_choice < nt
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        total_fires += int((action < nt).sum().item())
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    total_ammo = int(sum(amm))
    return total_fires / max(total_ammo, 1)


def eval_all(actor):
    actor.eval()
    per_config = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        utils = []
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            utils.append(ammo_utilization_one_instance(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        per_config.append((M, N, T, float(np.mean(utils))))
    return per_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope_ckpt', type=str, default='result/SCoPE_multiscale_seed5_best_actor.pt')
    parser.add_argument('--comm_ckpt', type=str, default='result/CommSCoPE_multiscale_seed5_best_actor.pt')
    args = parser.parse_args()

    scope = create_gnn_actor().to(DEVICE)
    scope.load_state_dict(torch.load(args.scope_ckpt, map_location=DEVICE, weights_only=False))
    comm = create_gnn_actor_comm().to(DEVICE)
    comm.load_state_dict(torch.load(args.comm_ckpt, map_location=DEVICE, weights_only=False))

    scope_util = eval_all(scope)
    comm_util = eval_all(comm)

    print(f"{'config':<18}{'SCoPE util%':>14}{'Comm util%':>14}")
    for (M, N, T, su), (_, _, _, cu) in zip(scope_util, comm_util):
        print(f"{M}M_{N}N_{T}T{'':<8}{su*100:>13.1f}%{cu*100:>13.1f}%")
    mean_s = sum(u for *_, u in scope_util) / len(scope_util)
    mean_c = sum(u for *_, u in comm_util) / len(comm_util)
    print(f"{'MEAN':<18}{mean_s*100:>13.1f}%{mean_c*100:>13.1f}%")
