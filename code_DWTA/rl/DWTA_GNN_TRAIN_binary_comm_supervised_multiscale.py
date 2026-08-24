"""
Train EdgeAwareGNN_ACTOR_BINARY_COMM (common/DWTA_GNN_binary_comm.py) via
PURE SUPERVISED rollout-labeled BCE (rl/Dynamic_Sampling_GNN_binary_supervised.py)
-- no REINFORCE -- on the MODERATE temporal-dilemma curriculum, multi-scale
(M,N,T ~ U[5,7]). Zero-shot evaluated on the same 12 held-out configs as
Sequential/CommSCoPE/CommSinkhornSCoPE/BinaryComm1to1(REINFORCE), for a
directly comparable number.

See Dynamic_Sampling_GNN_binary_supervised.py's module docstring for the
motivation: every REINFORCE-based binary-actor attempt this session hit
instability (collapse to always-fire or always-hold). This replaces the
sparse RL reward with a dense, directly-computed (rollout-based, not a
learned critic) supervised target every decision.

New file -- does not modify any existing training script/checkpoint.

Usage: python rl/DWTA_GNN_TRAIN_binary_comm_supervised_multiscale.py --total_steps 300 --eval_every 20
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
from common.DWTA_GNN_binary_comm import create_gnn_actor_binary_comm
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

from Dynamic_Sampling_GNN_binary_supervised import self_play_gnn_binary_supervised

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123


@torch.no_grad()
def eval_instance_binary_1to1(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
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
        fire_prob, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = fire_prob > 0.5
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


def eval_all_configs(actor):
    actor.eval()
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        objs = []
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            objs.append(eval_instance_binary_1to1(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        results.append((M, N, T, float(np.mean(objs))))
    actor.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=300, help='one step = one episode')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--rollout_k', type=int, default=2)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    actor = create_gnn_actor_binary_comm().to(DEVICE)
    lr = args.lr if args.lr is not None else ACTOR_LEARNING_RATE
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor.optimizer, T_max=args.total_steps, eta_min=lr * 0.01)

    print(f"[BINARY-COMM-SUPERVISED] training on M,N,T ~ U[5,7], rollout_k={args.rollout_k}, "
          f"evaluating zero-shot on {len(ALL_EVAL_CONFIGS)} held-out configs every {args.eval_every} steps")

    best_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/BinaryCommSupervised_multiscale_seed{args.seed}_best_actor.pt"
    final_path = f"result/BinaryCommSupervised_multiscale_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        bce_loss, epoch_info = self_play_gnn_binary_supervised(
            actor, episode=1, epoch=step,
            generator_fn=generate_moderate_temporal_instance,
            rollout_k=args.rollout_k,
        )
        scheduler.step()

        if step % 5 == 0:
            print(f"[BINARY-COMM-SUPERVISED] step {step}/{args.total_steps} "
                  f"bce_loss={bce_loss:.4f} destruction={epoch_info['destruction_ratio']:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [BINARY-COMM-SUPERVISED] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(actor)
            scores = []
            for M, N, T, score in results:
                scores.append(score)
                print(f"    {M}M_{N}N_{T}T: BinaryCommSupervised={score:.4f}", flush=True)

            mean_score = sum(scores) / len(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_step = step
                torch.save(actor.state_dict(), best_path)
            print(f"    mean_across_12_configs={mean_score:.4f}  "
                  f"(best so far: {best_score:.4f} at step {best_step})", flush=True)

    torch.save(actor.state_dict(), final_path)
    print(f"Saved final actor to {final_path}")
    print(f"Saved best actor (step {best_step}, mean={best_score:.4f}) to {best_path}")


if __name__ == "__main__":
    main()
