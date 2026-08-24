"""
Train EdgeAwareGNN_ACTOR_BINARY_COMM (common/DWTA_GNN_binary_comm.py --
binary fire/hold actor + weapon-to-weapon communication layer) with the
CAPACITATED many-to-one auction (auction_round_action_multifire) INSIDE the
training loop, on the MODERATE temporal-dilemma curriculum, multi-scale
(M,N,T ~ U[5,7]). Zero-shot evaluated on the same 12 held-out configs as
every other method tonight.

Combines three findings (see memory brerla_auction_train_inference_mismatch.md):
  1. Training the full N+1-way actor never uses the auction at all -- the
     REINFORCE gradient credits a target choice the auction later discards
     at inference, a train/inference mismatch. The binary actor sidesteps
     this entirely: its only learned decision (fire/hold, Bernoulli) is
     exactly what determines must_fire, which auction consumes -- so
     training the binary actor WITH the auction in the loop (as
     rl/Dynamic_Sampling_GNN_binary.py already does) has a clean,
     unambiguous log-prob with no mismatch.
  2. The 1:1 eviction auction structurally forbids same-round target
     concentration (dispersion locked at 1.0every round) -- decisively
     wrong at Large/Battlefield scale. Using auction_round_action_multifire
     instead, even just as an inference-time swap on an already-trained
     checkpoint, flipped 30M_30N_10T from losing to Sequential to winning.
     Training WITH multifire in the loop (this script) lets the policy's
     fire/hold judgment itself adapt to how multifire actually behaves,
     which it never got to do when multifire was only swapped in post-hoc.
  3. A naive mean-pool global context is a weaker fire/hold signal than
     genuine weapon-to-weapon communication (common/DWTA_GNN_comm.py).

New file -- does not modify rl/Dynamic_Sampling_GNN_moderate.py,
common/DWTA_GNN.py, or any existing checkpoint. Mirrors
rl/DWTA_GNN_TRAIN_comm_multiscale.py's harness/eval/best-checkpoint pattern.

Usage: python rl/DWTA_GNN_TRAIN_binary_comm_multiscale.py --total_steps 1500
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
from common.auction_refinement import auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

from Dynamic_Sampling_GNN_binary import self_play_gnn_binary, get_random_temporal_problem_size

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123


@torch.no_grad()
def eval_instance_binary_multifire(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
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
        action = auction_round_action_multifire(remaining_value, prob, legal_mask, must_fire=must_fire)
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
            objs.append(eval_instance_binary_multifire(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        results.append((M, N, T, float(np.mean(objs))))
    actor.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=1500, help='one step = one episode')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    actor = create_gnn_actor_binary_comm().to(DEVICE)
    lr = args.lr if args.lr is not None else ACTOR_LEARNING_RATE
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor.optimizer, T_max=args.total_steps, eta_min=lr * 0.01)

    ENTROPY_START, ENTROPY_END = 0.3, 0.01
    decay_rate = (ENTROPY_END / ENTROPY_START) ** (1.0 / args.total_steps)

    print(f"[BINARY-COMM-MULTIFIRE] training on M,N,T ~ U[5,7], evaluating zero-shot "
          f"on {len(ALL_EVAL_CONFIGS)} held-out configs every {args.eval_every} steps")

    best_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/BinaryCommMultifire_multiscale_seed{args.seed}_best_actor.pt"
    final_path = f"result/BinaryCommMultifire_multiscale_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        entropy_coef = ENTROPY_START * (decay_rate ** step)
        actor_loss, epoch_info = self_play_gnn_binary(
            actor, episode=1, epoch=step,
            generator_fn=generate_moderate_temporal_instance,
            entropy_coef=entropy_coef,
            auction_fn=auction_round_action_multifire,
        )
        scheduler.step()

        if step % 20 == 0:
            print(f"[BINARY-COMM-MULTIFIRE] step {step}/{args.total_steps} "
                  f"actor_loss={actor_loss:.4f} destruction={epoch_info['destruction_ratio']:.4f} "
                  f"entropy_coef={entropy_coef:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [BINARY-COMM-MULTIFIRE] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(actor)
            scores = []
            for M, N, T, score in results:
                scores.append(score)
                print(f"    {M}M_{N}N_{T}T: BinaryCommMultifire={score:.4f}", flush=True)

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
