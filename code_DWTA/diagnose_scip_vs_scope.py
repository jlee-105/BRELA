"""Diagnostic: for ONE 5M_5N_5T moderate instance (seed=123, index given),
solve with SCIP (exact optimal schedule) and run SCoPE (best multiscale
checkpoint) on the SAME instance, printing both round-by-round firing
schedules side by side so the actual divergence is visible, not just the
aggregate objective gap.
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + pyscipopt both link OpenMP;
                                                          # this is the first script to import both

import numpy as np
import torch

from common.DWTA_GNN import create_gnn_actor
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals
from opt.SCIP import solve_wta_scip

M, N, T = 5, 5, 5
SEED = 123


def get_instance(index):
    rng = np.random.default_rng(SEED)
    inst = None
    for i in range(index + 1):
        inst = generate_moderate_temporal_instance(M, N, T, rng=rng)
    return inst


def print_instance(V, P, TW, AMM, PREP):
    print(f"Target values (V):      {[round(v, 2) for v in V]}")
    print(f"Target windows (TW):    {TW}")
    print(f"Weapon ammo (AMM):      {AMM}")
    print(f"Weapon prep/cooldown:   {PREP}")
    print(f"Kill prob P (weapon x target):")
    for m in range(M):
        print(f"  W{m}: {[round(p, 2) for p in P[m]]}")


def run_scip(V, P, TW, AMM, PREP):
    obj, solution_3d, _, status, gap = solve_wta_scip(M, N, T, V, P, A=AMM, W=PREP, tw=TW, Time_Limit=600)
    print(f"\nSCIP status={status} gap={gap:.4f} objective(remaining)={obj:.4f} norm={obj/sum(V):.4f}")
    print("SCIP optimal firing schedule (round: weapon -> target):")
    for t in range(T):
        fires = [(m, n) for m in range(M) for n in range(N) if round(solution_3d[m, n, t]) == 1]
        if fires:
            print(f"  round {t}: " + ", ".join(f"W{m}->T{n}" for m, n in fires))
        else:
            print(f"  round {t}: (no fires)")
    return obj / sum(V)


@torch.no_grad()
def run_scope(V, P, TW, AMM, PREP, ckpt_path):
    actor = create_gnn_actor().to(DEVICE)
    actor.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    actor.eval()

    patch_globals(M, N, T, AMM, PREP, [1] * M)
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=P, TW=TW,
                                max_time=T, batch_size=1, alpha=1.0, amm=AMM)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=T)
    init_value = env.current_target_value[:, :, 0:N].sum().item()

    print("\nSCoPE firing schedule (round: weapon -> target):")
    for t in range(T):
        remaining_value = env.current_target_value[:, :, 0:N]
        prob = env.weapon_to_target_prob[:, :, :M, :N]
        legal_mask = env.mask_per_weapon[:, :, :M, :N] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :M, :].argmax(dim=-1)
        must_fire = policy_choice < N
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        flat = action.view(-1).tolist()
        fires = [(m, n) for m, n in enumerate(flat) if n < N]
        if fires:
            print(f"  round {t}: " + ", ".join(f"W{m}->T{n}" for m, n in fires))
        else:
            print(f"  round {t}: (no fires)")
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:N].sum().item()
    norm = remaining / max(init_value, 1e-8)
    print(f"\nSCoPE objective(remaining)={remaining:.4f} norm={norm:.4f}")
    return norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='which of the 10 fixed 5M_5N_5T instances (0-9)')
    parser.add_argument('--ckpt', type=str, default='result/SCoPE_multiscale_seed5_best_actor.pt')
    args = parser.parse_args()

    V, P, TW, AMM, PREP, COST = get_instance(args.index)
    P = np.asarray(P)

    print(f"=== 5M_5N_5T moderate instance #{args.index} ===")
    print_instance(V, P, TW, AMM, PREP)

    scip_norm = run_scip(V, P, TW, AMM, PREP)
    scope_norm = run_scope(V, P, TW, AMM, PREP, args.ckpt)

    print(f"\n=== SUMMARY: SCIP={scip_norm:.4f}  SCoPE={scope_norm:.4f}  gap={scope_norm - scip_norm:+.4f} ===")
