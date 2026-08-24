"""3-way diagnostic: SCIP-optimal vs Sequential vs SCoPE-Comm (Parallel+auction),
on the SAME small moderate-curriculum instance(s), printing round-by-round
firing schedules so the actual structural divergence is visible -- not just
the aggregate objective gap. Extends diagnose_scip_vs_scope.py (SCIP vs SCoPE
only) with Sequential and the current best CommSCoPE checkpoint, since the
paper's live question is specifically "what does Sequential do differently
from Parallel that Parallel could imitate."
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals
try:
    from opt.SCIP import solve_wta_scip
except ModuleNotFoundError:
    solve_wta_scip = None  # pyscipopt not installed in this env; skip SCIP, still compare Seq vs Comm

M, N, T = 5, 5, 5
SEED = 123
COMM_CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"


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
    print("Kill prob P (weapon x target):")
    for m in range(M):
        print(f"  W{m}: {[round(p, 2) for p in P[m]]}")


def run_scip(V, P, TW, AMM, PREP):
    obj, solution_3d, _, status, gap = solve_wta_scip(M, N, T, V, P, A=AMM, W=PREP, tw=TW, Time_Limit=600)
    print(f"\nSCIP status={status} gap={gap:.4f} objective(remaining)={obj:.4f} norm={obj/sum(V):.4f}")
    print("SCIP optimal firing schedule (round: weapon -> target):")
    for t in range(T):
        fires = [(m, n) for m in range(M) for n in range(N) if round(solution_3d[m, n, t]) == 1]
        print(f"  round {t}: " + (", ".join(f"W{m}->T{n}" for m, n in fires) if fires else "(no fires)"))
    return obj / sum(V)


@torch.no_grad()
def run_comm(V, P, TW, AMM, PREP):
    actor = create_gnn_actor_comm().to(DEVICE)
    actor.load_state_dict(torch.load(COMM_CKPT, map_location=DEVICE, weights_only=False))
    actor.eval()

    patch_globals(M, N, T, AMM, PREP, [1] * M)
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=P, TW=TW,
                                max_time=T, batch_size=1, alpha=1.0, amm=AMM)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=T)
    init_value = env.current_target_value[:, :, 0:N].sum().item()

    print("\nSCoPE-Comm firing schedule (round: weapon -> target):")
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
        print(f"  round {t}: " + (", ".join(f"W{m}->T{n}" for m, n in fires) if fires else "(no fires)"))
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:N].sum().item()
    norm = remaining / max(init_value, 1e-8)
    print(f"\nSCoPE-Comm objective(remaining)={remaining:.4f} norm={norm:.4f}")
    return norm


@torch.no_grad()
def run_sequential(V, P, TW, AMM, PREP):
    actor = create_gnn_actor_sequential().to(DEVICE)
    actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    actor.eval()

    patch_globals(M, N, T, AMM, PREP, [1] * M)
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=P, TW=TW,
                                max_time=T, batch_size=1, alpha=1.0, amm=AMM)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=T)
    init_value = env.current_target_value[:, :, 0:N].sum().item()

    num_edges = M * N
    print("\nSequential firing schedule (round: weapon -> target):")
    for t in range(T):
        fires = []
        for _ in range(M):
            mask = env.mask.clone()
            policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
            flat = policy.view(-1, num_edges + 1)
            action_idx = int(flat.argmax(dim=1).item())
            if action_idx < num_edges:
                fires.append((action_idx // N, action_idx % N))
            action = torch.tensor([[action_idx]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        print(f"  round {t}: " + (", ".join(f"W{m}->T{n}" for m, n in fires) if fires else "(no fires)"))
        env.time_update()

    remaining = env.current_target_value[:, :, 0:N].sum().item()
    norm = remaining / max(init_value, 1e-8)
    print(f"\nSequential objective(remaining)={remaining:.4f} norm={norm:.4f}")
    return norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    V, P, TW, AMM, PREP, COST = get_instance(args.index)
    P = np.asarray(P)

    print(f"=== 5M_5N_5T moderate instance #{args.index} ===")
    print_instance(V, P, TW, AMM, PREP)

    scip_norm = run_scip(V, P, TW, AMM, PREP) if solve_wta_scip is not None else None
    seq_norm = run_sequential(V, P, TW, AMM, PREP)
    comm_norm = run_comm(V, P, TW, AMM, PREP)

    if scip_norm is not None:
        print(f"\n=== SUMMARY: SCIP={scip_norm:.4f}  Sequential={seq_norm:.4f}  SCoPE-Comm={comm_norm:.4f} "
              f"(Seq-SCIP={seq_norm-scip_norm:+.4f}, Comm-SCIP={comm_norm-scip_norm:+.4f}, Comm-Seq={comm_norm-seq_norm:+.4f}) ===")
    else:
        print(f"\n=== SUMMARY (SCIP unavailable, pyscipopt not installed): Sequential={seq_norm:.4f}  "
              f"SCoPE-Comm={comm_norm:.4f}  (Comm-Seq={comm_norm-seq_norm:+.4f}) ===")
