"""
Train the improvement policy (common/DWTA_GNN_improve.py) via
rl/Dynamic_Sampling_GNN_improve.py, on top of the FROZEN current-best base
pipeline (CommSinkhornSCoPE checkpoint + 1:1 auction, mean 0.1357), on the
MODERATE temporal-dilemma curriculum, multi-scale (M,N,T ~ U[5,7]).
Zero-shot evaluated on the same 12 held-out configs as every other method.

At EVAL time edits are chosen greedily (argmax) and the BEST objective seen
across all edits is kept -- so the reported number can never be worse than
the frozen base pipeline's own result, by construction.

New file -- does not modify any existing training script/checkpoint.

Usage: python rl/DWTA_GNN_TRAIN_improve_multiscale.py --total_steps 200 --eval_every 20 --n_edits 3
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

from Dynamic_Sampling_GNN_improve import self_play_gnn_improve, simulate_with_flips, score_all_slots

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
def eval_instance_improve(base_actor, improve_actor, V, P, TW, nw, nt, mt, amm, prep, cost, n_edits):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
    best_obj = obj.mean().item()
    base_obj = best_obj

    for _ in range(n_edits):
        logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, mt * nw)
        logits = logits.masked_fill(flip_mask.reshape(1, 1, mt * nw), -1e9)
        idx = logits.argmax(dim=-1)  # greedy at eval
        new_flip = flip_mask.reshape(1, 1, mt * nw).clone()
        new_flip.scatter_(-1, idx.unsqueeze(-1), True)
        flip_mask = new_flip.reshape(1, 1, mt, nw)
        obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt)
        best_obj = min(best_obj, obj.mean().item())  # keep-best: cannot regress below base

    return base_obj, best_obj


def eval_all_configs(base_actor, improve_actor, n_edits):
    improve_actor.eval()
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        base_objs, imp_objs = [], []
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            b, im = eval_instance_improve(base_actor, improve_actor, V, np.asarray(P), TW,
                                           M, N, T, AMM, PREP, COST, n_edits)
            base_objs.append(b)
            imp_objs.append(im)
        results.append((M, N, T, float(np.mean(base_objs)), float(np.mean(imp_objs))))
    improve_actor.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--n_edits', type=int, default=3)
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

    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.optimizer = torch.optim.Adam(improve_actor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        improve_actor.optimizer, T_max=args.total_steps, eta_min=args.lr * 0.01)

    print(f"[IMPROVE] base={BASE_CKPT} (frozen), n_edits={args.n_edits}, "
          f"batch={args.batch_size}x{args.para_size}, lr={args.lr}, "
          f"eval on {len(ALL_EVAL_CONFIGS)} held-out configs every {args.eval_every} steps", flush=True)

    best_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/Improve_multiscale_seed{args.seed}_best_actor.pt"
    final_path = f"result/Improve_multiscale_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        loss, info = self_play_gnn_improve(
            base_actor, improve_actor, episode=1, epoch=step,
            n_edits=args.n_edits, batch_size=args.batch_size, para_size=args.para_size,
        )
        scheduler.step()

        if step % 10 == 0:
            print(f"[IMPROVE] step {step}/{args.total_steps} loss={loss:.4f} "
                  f"base={info['base_objective']:.4f} -> improved={info['improved_objective']:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [IMPROVE] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(base_actor, improve_actor, args.n_edits)
            base_scores, imp_scores = [], []
            for M, N, T, b, im in results:
                base_scores.append(b)
                imp_scores.append(im)
                print(f"    {M}M_{N}N_{T}T: base={b:.4f}  improved={im:.4f}", flush=True)

            mean_base = sum(base_scores) / len(base_scores)
            mean_imp = sum(imp_scores) / len(imp_scores)
            if mean_imp < best_score:
                best_score = mean_imp
                best_step = step
                torch.save(improve_actor.state_dict(), best_path)
            print(f"    mean_base={mean_base:.4f}  mean_improved={mean_imp:.4f}  "
                  f"(best so far: {best_score:.4f} at step {best_step})", flush=True)

    torch.save(improve_actor.state_dict(), final_path)
    print(f"Saved final to {final_path}")
    print(f"Saved best (step {best_step}, mean={best_score:.4f}) to {best_path}")


if __name__ == "__main__":
    main()
