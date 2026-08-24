"""Full 12-config comparison: SCoPE-Comm (current best checkpoint) with 1:1
eviction auction vs the capped many-to-one auction (auction_round_action_multifire),
vs Sequential (current best checkpoint) -- same-checkpoint, inference-only
swap (no retraining). Follow-up to diagnose_dispersion_large.py's finding
that the 1:1 auction structurally forbids same-round target concentration
(dispersion locked at 1.0 every round), which multifire lifts and which
decisively fixed the Large-tier (30M_30N_10T) gap (Comm 0.0251->0.0137,
beating Sequential's 0.0162). This checks whether the fix holds/helps
across the full scale range, not just Large.
"""
import time

import numpy as np
import torch

from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action, auction_round_action_multifire, auction_round_action_multifire_guided
GUIDE_WEIGHT = 0.5
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

N_EVAL = 10
SEED = 123
COMM_CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"

ALL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]


@torch.no_grad()
def eval_comm(actor, V, P, TW, nw, nt, mt, amm, prep, cost, mode):
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
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :nw, :].argmax(dim=-1)
        must_fire = policy_choice < nt
        if mode == "multi":
            action = auction_round_action_multifire(remaining_value, prob, legal_mask, must_fire=must_fire)
        elif mode == "guided":
            target_scores = policy[:, :, :nw, :nt]
            legal_f = legal_mask.float()
            denom = (target_scores * legal_f).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            policy_target_prob = (target_scores * legal_f) / denom
            action = auction_round_action_multifire_guided(
                remaining_value, prob, legal_mask, policy_target_prob,
                must_fire=must_fire, guide_weight=GUIDE_WEIGHT)
        else:
            action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


@torch.no_grad()
def eval_seq(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    num_edges = nw * nt
    for _ in range(mt):
        for _ in range(nw):
            mask = env.mask.clone()
            policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
            flat = policy.view(-1, num_edges + 1)
            action_idx = int(flat.argmax(dim=1).item())
            action = torch.tensor([[action_idx]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


if __name__ == "__main__":
    comm_actor = create_gnn_actor_comm().to(DEVICE)
    comm_actor.load_state_dict(torch.load(COMM_CKPT, map_location=DEVICE, weights_only=False))
    comm_actor.eval()
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    seq_actor.eval()

    print(f"{'config':<16}{'Seq':>8}{'1:1':>8}{'multi':>8}{'guided':>8}{'guided_wins?':>13}")
    seq_all, c11_all, cmf_all, cg_all = [], [], [], []
    for M, N, T in ALL_CONFIGS:
        rng = np.random.default_rng(SEED)
        instances = [generate_moderate_temporal_instance(M, N, T, rng=rng) for _ in range(N_EVAL)]
        seq_objs, c11_objs, cmf_objs, cg_objs = [], [], [], []
        for V, P, TW, AMM, PREP, COST in instances:
            P = np.asarray(P)
            seq_objs.append(eval_seq(seq_actor, V, P, TW, M, N, T, AMM, PREP, COST))
            c11_objs.append(eval_comm(comm_actor, V, P, TW, M, N, T, AMM, PREP, COST, mode="1:1"))
            cmf_objs.append(eval_comm(comm_actor, V, P, TW, M, N, T, AMM, PREP, COST, mode="multi"))
            cg_objs.append(eval_comm(comm_actor, V, P, TW, M, N, T, AMM, PREP, COST, mode="guided"))
        seq_m, c11_m, cmf_m, cg_m = np.mean(seq_objs), np.mean(c11_objs), np.mean(cmf_objs), np.mean(cg_objs)
        seq_all.append(seq_m); c11_all.append(c11_m); cmf_all.append(cmf_m); cg_all.append(cg_m)
        beats_seq = "YES" if cg_m < seq_m else "no"
        print(f"{M}M_{N}N_{T}T{'':<7}{seq_m:>8.4f}{c11_m:>8.4f}{cmf_m:>8.4f}{cg_m:>8.4f}{beats_seq:>13}", flush=True)

    print(f"\n{'MEAN':<16}{np.mean(seq_all):>8.4f}{np.mean(c11_all):>8.4f}{np.mean(cmf_all):>8.4f}{np.mean(cg_all):>8.4f}")
