"""
Train EdgeAwareGNN_ACTOR_BINARY_COMM (common/DWTA_GNN_binary_comm.py --
binary fire/hold actor + weapon-to-weapon communication layer) with the
STANDARD 1:1 eviction auction (auction_round_action) in the training loop --
NOT the multifire variant -- on the MODERATE temporal-dilemma curriculum,
multi-scale (M,N,T ~ U[5,7]). Zero-shot evaluated on the same 12 held-out
configs as Sequential/CommSCoPE/CommSinkhornSCoPE, for a directly comparable
number.

Why this specific combination, not yet tried: rl/DWTA_GNN_TRAIN_binary_comm_multiscale.py
already trains this same binary+comm architecture, but with
auction_round_action_multifire in the loop, and got WORSE results (0.1532
mean, losing decisively at Large tier specifically -- see Finding 5 in
brerla_auction_train_inference_mismatch memory). That confounds two
questions: (a) is the binary-only architecture (target choice not learned
at all, since the auction discards it anyway) better than the full N+1-way
actor's indirect approach, and (b) is multifire a good auction choice to
train WITH. This script isolates (a) by pairing the binary+comm actor with
the SAME 1:1 auction that Sequential/CommSCoPE/Sinkhorn are all reported
against, giving a clean apples-to-apples comparison for the first time.

Motivation for trying this at all (2026-08-20 session): a direct diagnostic
(diagnose_sinkhorn_must_fire.py) showed CommSinkhornSCoPE's ~0.11 mean
improvement over CommSCoPE is NOT coming from better target coordination
(the auction discards target choice entirely) but from an indirect,
opaque shift in the fire/hold gate (Sinkhorn-adjusted edge scores
occasionally lose to the no-op score). Since only fire/hold ever reaches
the environment in the reported pipeline, training a policy whose ONLY
output IS fire/hold (this file) should be a more direct, better-targeted
way to improve exactly the one channel that matters, instead of hoping a
target-choice-shaping mechanism leaks through indirectly.

New file -- does not modify rl/DWTA_GNN_TRAIN_binary_comm_multiscale.py or
any existing checkpoint. Mirrors its harness/eval/best-checkpoint pattern.

Usage: python rl/DWTA_GNN_TRAIN_binary_comm_1to1_multiscale.py --total_steps 400 --eval_every 20
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

from Dynamic_Sampling_GNN_binary import self_play_gnn_binary

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
    parser.add_argument('--total_steps', type=int, default=400, help='one step = one episode')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    actor = create_gnn_actor_binary_comm().to(DEVICE)
    lr = args.lr if args.lr is not None else ACTOR_LEARNING_RATE
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor.optimizer, T_max=args.total_steps, eta_min=lr * 0.01)

    ENTROPY_START, ENTROPY_END = 0.3, 0.01
    decay_rate = (ENTROPY_END / ENTROPY_START) ** (1.0 / args.total_steps)

    print(f"[BINARY-COMM-1TO1] training on M,N,T ~ U[5,7], evaluating zero-shot "
          f"on {len(ALL_EVAL_CONFIGS)} held-out configs every {args.eval_every} steps")

    best_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/BinaryComm1to1_multiscale_seed{args.seed}_best_actor.pt"
    final_path = f"result/BinaryComm1to1_multiscale_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        entropy_coef = ENTROPY_START * (decay_rate ** step)
        actor_loss, epoch_info = self_play_gnn_binary(
            actor, episode=1, epoch=step,
            generator_fn=generate_moderate_temporal_instance,
            entropy_coef=entropy_coef,
            auction_fn=auction_round_action,
        )
        scheduler.step()

        if step % 20 == 0:
            print(f"[BINARY-COMM-1TO1] step {step}/{args.total_steps} "
                  f"actor_loss={actor_loss:.4f} destruction={epoch_info['destruction_ratio']:.4f} "
                  f"entropy_coef={entropy_coef:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [BINARY-COMM-1TO1] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(actor)
            scores = []
            for M, N, T, score in results:
                scores.append(score)
                print(f"    {M}M_{N}N_{T}T: BinaryComm1to1={score:.4f}", flush=True)

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
