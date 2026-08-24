"""
Diagnostic: how much does the Sinkhorn-trained actor's fire/hold (must_fire)
decision actually differ from the plain comm-layer actor's, HOLDING STATE
FIXED (both policies evaluated on the exact same round of a single shared
reference trajectory, driven by CommSCoPE's own auction decisions).

Motivation: the auction discards the trained policy's target CHOICE
entirely -- only the binary must_fire gate survives into the final
auction-refined action (see common/auction_refinement.py). So if
CommSinkhornSCoPE (comm + Sinkhorn target-coordination bonus) genuinely
improves on plain CommSCoPE at all, the improvement MUST flow through
must_fire, not through target choice, since target choice never reaches the
environment in the reported "SCoPE" pipeline. This script measures directly:
(1) what fraction of (weapon, round) must_fire decisions actually differ
between the two checkpoints on identical state, and (2) whether that
disagreement concentrates in any particular scale tier.

If disagreement is near-zero everywhere, the earlier 0.1465 -> 0.1357
improvement is likely noise/checkpoint variance, not a real Sinkhorn effect
-- worth knowing before investing further in target-choice-coordination
architecture (comm layer, Sinkhorn) versus redirecting effort toward the
fire/hold decision specifically.
"""
import numpy as np
import torch

from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

N_EVAL = 10
SEED = 123
SCOPE_CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"
SINKHORN_CKPT = "result/CommSinkhornSCoPE_multiscale_seed5_best_actor.pt"

ALL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]


@torch.no_grad()
def diagnose_instance(scope_actor, sinkhorn_actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    # Reference trajectory is driven by CommSCoPE's OWN auction decisions --
    # Sinkhorn actor's policy is evaluated on this same state every round
    # but never used to advance the environment, so both actors always see
    # IDENTICAL state at comparison time.
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)

    total_decisions = 0
    disagreements = 0
    scope_fires = 0
    sinkhorn_fires = 0

    for _ in range(mt):
        policy_scope, _ = scope_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_sinkhorn, _ = sinkhorn_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)

        choice_scope = policy_scope[:, :, :nw, :].argmax(dim=-1)
        choice_sinkhorn = policy_sinkhorn[:, :, :nw, :].argmax(dim=-1)

        must_fire_scope = choice_scope < nt
        must_fire_sinkhorn = choice_sinkhorn < nt

        total_decisions += nw
        disagreements += (must_fire_scope != must_fire_sinkhorn).sum().item()
        scope_fires += must_fire_scope.sum().item()
        sinkhorn_fires += must_fire_sinkhorn.sum().item()

        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire_scope)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    return total_decisions, disagreements, scope_fires, sinkhorn_fires


if __name__ == "__main__":
    scope_actor = create_gnn_actor_comm().to(DEVICE)
    scope_actor.load_state_dict(torch.load(SCOPE_CKPT, map_location=DEVICE, weights_only=False))
    scope_actor.eval()

    sinkhorn_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    sinkhorn_actor.load_state_dict(torch.load(SINKHORN_CKPT, map_location=DEVICE, weights_only=False))
    sinkhorn_actor.eval()

    print(f"{'config':<16}{'decisions':>10}{'disagree%':>11}{'scope_fire%':>13}{'sinkhorn_fire%':>15}")
    grand_total, grand_disagree = 0, 0
    for M, N, T in ALL_CONFIGS:
        rng = np.random.default_rng(SEED)
        instances = [generate_moderate_temporal_instance(M, N, T, rng=rng) for _ in range(N_EVAL)]
        tot_d, tot_dis, tot_sf, tot_kf = 0, 0, 0, 0
        for V, P, TW, AMM, PREP, COST in instances:
            P = np.asarray(P)
            d, dis, sf, kf = diagnose_instance(scope_actor, sinkhorn_actor, V, P, TW, M, N, T, AMM, PREP, COST)
            tot_d += d; tot_dis += dis; tot_sf += sf; tot_kf += kf
        grand_total += tot_d
        grand_disagree += tot_dis
        print(f"{M}M_{N}N_{T}T{'':<7}{tot_d:>10}{100*tot_dis/tot_d:>10.2f}%{100*tot_sf/tot_d:>12.2f}%{100*tot_kf/tot_d:>14.2f}%", flush=True)

    print(f"\nOVERALL: {grand_total} decisions, {100*grand_disagree/grand_total:.2f}% disagreement rate")
