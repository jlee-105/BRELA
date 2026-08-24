"""Inference-time scaling: Sequential vs SCoPE-Comm+auction ("ours"), across
all 12 moderate-curriculum held-out configs. Pure decode time only (forward
passes + auction where applicable) -- no SCIP, no instance-generation
overhead counted. Uses each decoder's own current BEST checkpoint (same ones
used in the main results table).
"""
import time

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

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
N_WARMUP = 2  # untimed warm-up instances per config, not counted (JIT/cuDNN warm-up noise)
SEED = 123

SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"
COMM_CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_sequential(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)

    _sync()
    t0 = time.perf_counter()
    for _ in range(mt):
        for _ in range(nw):
            mask = env.mask.clone()
            if (mask > 0).any():
                policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                flat = policy.view(-1, nw * nt + 1)
                action = flat.argmax(dim=1).view(1, 1)
            else:
                action = torch.tensor([[nw * nt]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        env.time_update()
    _sync()
    return time.perf_counter() - t0


@torch.no_grad()
def time_ours(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    """SCoPE-Comm actor forward + 1:1 auction refinement, our full pipeline."""
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)

    _sync()
    t0 = time.perf_counter()
    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :nw, :].argmax(dim=-1)
        must_fire = policy_choice < nt
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    _sync()
    return time.perf_counter() - t0


def eval_all(actor, time_fn):
    actor.eval()
    per_config = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        instances = [generate_moderate_temporal_instance(M, N, T, rng=rng) for _ in range(N_WARMUP + N_EVAL)]
        for V, P, TW, AMM, PREP, COST in instances[:N_WARMUP]:
            time_fn(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST)
        times = []
        for V, P, TW, AMM, PREP, COST in instances[N_WARMUP:]:
            times.append(time_fn(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        per_config.append((M, N, T, float(np.mean(times)), float(np.std(times))))
    return per_config


if __name__ == "__main__":
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    ours_actor = create_gnn_actor_comm().to(DEVICE)
    ours_actor.load_state_dict(torch.load(COMM_CKPT, map_location=DEVICE, weights_only=False))

    seq_times = eval_all(seq_actor, time_sequential)
    ours_times = eval_all(ours_actor, time_ours)

    print(f"{'config':<16}{'Sequential(s)':>16}{'Ours(s)':>12}{'speedup':>10}")
    for (M, N, T, sm, ss), (_, _, _, om, os_) in zip(seq_times, ours_times):
        speedup = sm / max(om, 1e-9)
        print(f"{M}M_{N}N_{T}T{'':<7}{sm:>16.4f}{om:>12.4f}{speedup:>9.1f}x")

    mean_seq = sum(t for *_, t, _ in seq_times) / len(seq_times)
    mean_ours = sum(t for *_, t, _ in ours_times) / len(ours_times)
    print(f"{'MEAN':<16}{mean_seq:>16.4f}{mean_ours:>12.4f}{mean_seq/mean_ours:>9.1f}x")
