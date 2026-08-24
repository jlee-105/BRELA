"""Rollout-style improvement for the Sequential decoder: same idea as
eval_parallel_rollout_sampling.py (replicate one instance across P copies,
sample instead of argmax, simulate all P to completion in one batched-per-
decision forward pass, keep the best of P final trajectories) -- no critic,
no beam search, no retraining. P=1 with sample=False reproduces today's
plain argmax decode exactly (the number already reported in the main
results table).

Sequential has no auction step (its own decoder resolves within-round
coordination one edge at a time), so there is no Python-loop bottleneck like
the parallel-rollout prototype hit -- P rollouts cost P x (M forward calls
per round), same asymptotic shape as a single decode, just P times more of
it, fully batched.
"""
import time

import numpy as np
import torch

from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"
CONFIGS = [(5, 5, 5), (30, 30, 10), (70, 100, 15)]
N_EVAL = 1
SEED = 123
P_VALUES = [1, 8, 32]


@torch.no_grad()
def run(actor, V, P, TW, nw, nt, mt, amm, prep, cost, num_par, sample):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae = ae.unsqueeze(1).repeat(1, num_par, 1, 1).contiguous()
    wtp = wtp.unsqueeze(1).repeat(1, num_par, 1, 1).contiguous()
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum(dim=2)[0, 0].item()

    num_edges = nw * nt
    num_actions = num_edges + 1
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(mt):
        for _ in range(nw):
            mask = env.mask.clone()
            policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
            flat = policy.view(-1, num_actions)
            if sample:
                action = torch.multinomial(flat.clamp_min(1e-8), 1).view(1, num_par)
            else:
                action = flat.argmax(dim=1).view(1, num_par)
            env.update_internal_variables(selected_action=action)
        env.time_update()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    final_value_per_par = env.current_target_value[:, :, 0:nt].sum(dim=2)[0]  # [num_par]
    best_remaining = final_value_per_par.min().item()
    return init_value, best_remaining, elapsed


def eval_config(actor, M, N, T):
    rng = np.random.default_rng(SEED)
    instances = [generate_moderate_temporal_instance(M, N, T, rng=rng) for _ in range(N_EVAL)]
    results = {p: {"obj": [], "time": []} for p in P_VALUES}
    for V, Pp, TW, AMM, PREP, COST in instances:
        for p in P_VALUES:
            sample = p > 1
            init_v, best_rem, elapsed = run(actor, V, np.asarray(Pp), TW, M, N, T, AMM, PREP, COST,
                                             num_par=p, sample=sample)
            results[p]["obj"].append(best_rem / max(init_v, 1e-8))
            results[p]["time"].append(elapsed)
    return {p: (float(np.mean(r["obj"])), float(np.mean(r["time"]))) for p, r in results.items()}


if __name__ == "__main__":
    actor = create_gnn_actor_sequential().to(DEVICE)
    actor.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
    actor.eval()

    header = f"{'config':<16}" + "".join(f"{'P='+str(p)+' obj':>12}{'P='+str(p)+' time':>12}" for p in P_VALUES)
    print(header)
    for M, N, T in CONFIGS:
        res = eval_config(actor, M, N, T)
        row = f"{M}M_{N}N_{T}T{'':<7}"
        for p in P_VALUES:
            obj, t = res[p]
            row += f"{obj:>12.4f}{t:>12.4f}"
        print(row, flush=True)
