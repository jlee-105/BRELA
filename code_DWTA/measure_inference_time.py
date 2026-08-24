"""
MEASURED inference time (not estimated) for every pipeline under
consideration, so the parallel-vs-sequential decision rests on real numbers.

Everything so far has leaned on a stale figure from an earlier session
(Parallel 879ms vs Sequential 5447ms at 70x100x15) whose measurement
conditions -- in particular whether the auction was included -- are not
recorded. That matters, because the auction is a Python loop over weapons,
so "parallel decoding is cheap" may not survive contact with it, and every
audit edit multiplies whatever the base cost is by (K+1).

Structural expectation to check against:
  parallel   : T   network forwards per simulation + T auction passes
  sequential : M*T network forwards per simulation, no auction
so at 70M_100N_15T sequential does ~1050 forwards per simulation versus
parallel's 15 -- but parallel pays for the auction's inner loop over M
weapons, which is exactly what the stale number may have omitted.

Reports base decode time and the K-edit audited time for each arm.
"""
import sys
import time

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_improve import create_gnn_actor_improve
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action, auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals
from Dynamic_Sampling_GNN_improve import simulate_with_flips, score_all_slots
from Dynamic_Sampling_GNN_seq_joint import simulate_sequential_with_flips

CONFIGS = [(5, 5, 5), (20, 30, 5), (30, 40, 10), (70, 100, 15)]
N_TIMED = 3          # instances per measurement (wall-clock, so keep small)
EDIT_BUDGETS = [3, 10]
SEED = 123

BASE_CKPT = "result/Joint_base_seed5_simlr_best.pt"
IMPROVE_CKPT = "result/Joint_improve_seed5_simlr_best.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def time_parallel(base_actor, improve_actor, ae, wtp, nw, nt, mt, n_edits, auction_fn):
    """Times the full reported pipeline: base decode + n_edits audit edits,
    each edit costing one full re-simulation plus one auditor scoring pass."""
    _sync()
    t0 = time.perf_counter()

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    env_sim = lambda fm: _simulate(base_actor, ae, wtp, fm, nw, nt, mt, auction_fn)
    obj, states = env_sim(flip_mask)
    base_done = time.perf_counter()

    n_slots = mt * nw
    for _ in range(min(n_edits, n_slots)):
        logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
        logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
        idx = logits.argmax(dim=-1)
        nf = flip_mask.reshape(1, 1, n_slots).clone()
        nf.scatter_(-1, idx.unsqueeze(-1), True)
        flip_mask = nf.reshape(1, 1, mt, nw)
        obj, states = env_sim(flip_mask)

    _sync()
    total = time.perf_counter() - t0
    return base_done - t0, total


@torch.no_grad()
def _simulate(base_actor, ae, wtp, flip_mask, nw, nt, mt, auction_fn):
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    original_value = env.original_target_value[:, :, 0:nt].sum(2)
    states = []
    for t in range(mt):
        states.append((env.assignment_encoding.clone(),
                        env.weapon_to_target_prob.clone(),
                        env.mask_per_weapon.clone()))
        policy, _ = base_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = (policy[:, :, :nw, :].argmax(dim=-1) < nt) ^ flip_mask[:, :, t, :]
        action = auction_fn(env.current_target_value[:, :, 0:nt],
                            env.weapon_to_target_prob[:, :, :nw, :nt],
                            env.mask_per_weapon[:, :, :nw, :nt] > 0,
                            must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    return (env.current_target_value[:, :, 0:nt].sum(2) / (original_value + 1e-8)), states


@torch.no_grad()
def time_sequential(seq_actor, improve_actor, ae, wtp, nw, nt, mt, n_edits):
    _sync()
    t0 = time.perf_counter()

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate_sequential_with_flips(seq_actor, ae, wtp, flip_mask, nw, nt, mt)
    base_done = time.perf_counter()

    n_slots = mt * nw
    for _ in range(min(n_edits, n_slots)):
        logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
        logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
        idx = logits.argmax(dim=-1)
        nf = flip_mask.reshape(1, 1, n_slots).clone()
        nf.scatter_(-1, idx.unsqueeze(-1), True)
        flip_mask = nf.reshape(1, 1, mt, nw)
        obj, states = simulate_sequential_with_flips(seq_actor, ae, wtp, flip_mask, nw, nt, mt)

    _sync()
    return base_done - t0, time.perf_counter() - t0


if __name__ == "__main__":
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()
    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.load_state_dict(torch.load(IMPROVE_CKPT, map_location=DEVICE, weights_only=False))
    improve_actor.eval()
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    seq_actor.eval()

    print(f"{'config':<14}{'arm':<22}{'base(s)':>10}" +
          "".join(f"{'K='+str(k)+'(s)':>11}" for k in EDIT_BUDGETS), flush=True)

    for (M, N, T) in CONFIGS:
        rng = np.random.default_rng(SEED)
        inst = []
        for _ in range(N_TIMED):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            patch_globals(M, N, T, AMM, PREP, COST)
            ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=np.asarray(P), TW=TW,
                                        max_time=T, batch_size=1, alpha=1.0, amm=AMM)
            inst.append((ae.unsqueeze(1), wtp.unsqueeze(1), AMM, PREP, COST))

        arms = {
            'parallel+1:1': lambda a, w, k: time_parallel(base_actor, improve_actor, a, w, M, N, T, k,
                                                            auction_round_action),
            'parallel+multifire': lambda a, w, k: time_parallel(base_actor, improve_actor, a, w, M, N, T, k,
                                                                  auction_round_action_multifire),
            'sequential': lambda a, w, k: time_sequential(seq_actor, improve_actor, a, w, M, N, T, k),
        }

        for name, fn in arms.items():
            # warm up once so CUDA init / autotune is not charged to the first timing
            patch_globals(M, N, T, inst[0][2], inst[0][3], inst[0][4])
            fn(inst[0][0], inst[0][1], 1)

            base_times, budget_times = [], {k: [] for k in EDIT_BUDGETS}
            for (ae, wtp, AMM, PREP, COST) in inst:
                patch_globals(M, N, T, AMM, PREP, COST)
                for k in EDIT_BUDGETS:
                    b, tot = fn(ae, wtp, k)
                    budget_times[k].append(tot)
                    if k == EDIT_BUDGETS[0]:
                        base_times.append(b)

            row = f"{f'{M}M_{N}N_{T}T':<14}{name:<22}{np.mean(base_times):>10.3f}"
            row += "".join(f"{np.mean(budget_times[k]):>11.3f}" for k in EDIT_BUDGETS)
            print(row, flush=True)
