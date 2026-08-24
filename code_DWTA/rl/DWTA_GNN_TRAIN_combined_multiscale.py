"""
Train the COMBINED edit-operator improvement policy
(rl/Dynamic_Sampling_GNN_combined.py) on top of the frozen current-best base
pipeline, on the MODERATE curriculum, multi-scale (M,N,T ~ U[5,7]).

Directly comparable to:
  - rl/DWTA_GNN_TRAIN_improve_multiscale.py  (flip-only, project best 0.1291)
  - rl/DWTA_GNN_TRAIN_move_multiscale.py     (move-only, 0.1330, worse)
same base, harness and eval; only the edit operator differs.

Also logs `move_rate` -- the fraction of edits where the policy chose to
pair a partner (a real move) rather than a plain flip. That number is the
direct empirical answer to whether temporal rescheduling is actually useful
here, since the policy is free to ignore it entirely.

Usage: python rl/DWTA_GNN_TRAIN_combined_multiscale.py --total_steps 200 --eval_every 50 --eval_edits 30
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_improve import create_gnn_actor_improve
from common.Dynamic_Instance_generation import input_generation
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

from Dynamic_Sampling_GNN_improve import simulate_with_flips
from Dynamic_Sampling_GNN_combined import (CombinedEditPolicy, self_play_gnn_combined,
                                             propose_combined, apply_combined,
                                             legal_slots_from_states)

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123
BASE_CKPT = "result/CommSinkhornSCoPE_multiscale_seed5_best_actor.pt"


@torch.no_grad()
def eval_instance_combined(base_actor, policy, V, P, TW, nw, nt, mt, amm, prep, cost, n_edits):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states, fire_state = simulate_with_flips(
        base_actor, ae, wtp, flip_mask, nw, nt, mt, return_fire_state=True)
    base_obj = obj.mean().item()
    best = base_obj
    n_moves = 0

    for _ in range(n_edits):
        legal_state = legal_slots_from_states(states, nw, nt)
        t1, t2, weapon, use_partner, _ = propose_combined(
            policy, states, nw, mt, fire_state, legal_state, greedy=True)
        flip_mask = apply_combined(flip_mask, t1, t2, weapon, use_partner)
        n_moves += int(use_partner.sum().item())
        obj, states, fire_state = simulate_with_flips(
            base_actor, ae, wtp, flip_mask, nw, nt, mt, return_fire_state=True)
        best = min(best, obj.mean().item())

    return base_obj, best, n_moves


def eval_all_configs(base_actor, policy, n_edits):
    policy.eval()
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        b_objs, c_objs, moves = [], [], 0
        for _ in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            b, c, nm = eval_instance_combined(base_actor, policy, V, np.asarray(P), TW,
                                               M, N, T, AMM, PREP, COST, n_edits)
            b_objs.append(b)
            c_objs.append(c)
            moves += nm
        results.append((M, N, T, float(np.mean(b_objs)), float(np.mean(c_objs)), moves))
    policy.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--n_edits', type=int, default=3)
    parser.add_argument('--eval_edits', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--para_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()
    for p in base_actor.parameters():
        p.requires_grad_(False)

    policy = CombinedEditPolicy(create_gnn_actor_improve()).to(DEVICE)
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy.optimizer, T_max=args.total_steps, eta_min=args.lr * 0.01)

    print(f"[COMBINED] base={BASE_CKPT} (frozen), n_edits={args.n_edits} (train) / "
          f"{args.eval_edits} (eval), batch={args.batch_size}x{args.para_size}", flush=True)

    best_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/Combined_multiscale_seed{args.seed}_best_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        loss, info = self_play_gnn_combined(
            base_actor, policy, episode=1, epoch=step,
            n_edits=args.n_edits, batch_size=args.batch_size, para_size=args.para_size)
        scheduler.step()

        if step % 10 == 0:
            print(f"[COMBINED] step {step}/{args.total_steps} loss={loss:.4f} "
                  f"base={info['base_objective']:.4f} -> edited={info['improved_objective']:.4f} "
                  f"move_rate={info['move_rate']:.2f} none_logit={policy.none_logit.item():.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [COMBINED] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(base_actor, policy, args.eval_edits)
            b_scores, c_scores, tot_moves = [], [], 0
            for M, N, T, b, c, nm in results:
                b_scores.append(b)
                c_scores.append(c)
                tot_moves += nm
                print(f"    {M}M_{N}N_{T}T: base={b:.4f}  combined={c:.4f}  (moves={nm})", flush=True)
            mean_b = sum(b_scores) / len(b_scores)
            mean_c = sum(c_scores) / len(c_scores)
            if mean_c < best_score:
                best_score = mean_c
                best_step = step
                torch.save(policy.state_dict(), best_path)
            print(f"    mean_base={mean_b:.4f}  mean_combined={mean_c:.4f}  "
                  f"total_moves={tot_moves}  (best so far: {best_score:.4f} at step {best_step})", flush=True)

    print(f"Best combined mean={best_score:.4f} at step {best_step} -> {best_path}")


if __name__ == "__main__":
    main()
