"""Prototype test: can the parallel decoder (SCoPE-Comm) get a rollout-style
improvement analogous to Sequential's existing rollout/beam machinery
(rl_rollout/multi_policy_rollout.py, BEAM_WITH_SIMULATION_TRUNC_CRITIC.py)?

Idea: the parallel decoder's tensors already carry a P (para) dimension used
for the POMO-style training baseline (see rl/Dynamic_Sampling_GNN_moderate.py,
lines ~78-121: instance replicated across P, ONE batched forward pass per
round, action sampled via multinomial per round). At INFERENCE time we can
reuse exactly that pattern: replicate one instance across P copies, sample
(not argmax) each round, auction-refine, step all P forward together in ONE
batched forward pass per round, and at the end keep the best of the P final
trajectories. This is structurally cheaper than an equivalent Sequential
rollout, which needs P x W sequential forward passes per round (P independent
rollouts, each still W serial per-weapon calls).

No training involved -- reuses the already-trained CommSCoPE best checkpoint,
inference-only, same as auction_refinement.py.
"""
import time

import numpy as np
import torch

from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"
CONFIGS = [(5, 5, 5), (30, 30, 10), (70, 100, 15)]
N_EVAL = 1
SEED = 123
P_VALUES = [1, 8, 32]  # P=1 == plain argmax decode (today's reported "Ours")


@torch.no_grad()
def run(actor, V, P, TW, nw, nt, mt, amm, prep, cost, num_par, sample):
    """sample=False, num_par=1 reproduces today's argmax decode exactly."""
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae = ae.unsqueeze(1).repeat(1, num_par, 1, 1).contiguous()
    wtp = wtp.unsqueeze(1).repeat(1, num_par, 1, 1).contiguous()
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum(dim=2)[0, 0].item()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        if sample:
            flat = policy[:, :, :nw, :].reshape(-1, nt + 1)
            choice = torch.multinomial(flat.clamp_min(1e-8), 1).view(1, num_par, nw)
        else:
            choice = policy[:, :, :nw, :].argmax(dim=-1)
        must_fire = choice < nt
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
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
    actor = create_gnn_actor_comm().to(DEVICE)
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
