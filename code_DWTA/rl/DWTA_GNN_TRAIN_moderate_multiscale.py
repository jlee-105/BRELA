"""
Train the FULL N+1-way Parallel GNN actor (SCoPE = this actor + auction
refinement) on the MODERATE temporal-dilemma curriculum, multi-scale
(M,N,T ~ U[5,7], resampled every episode), with the SAME rigor now applied
to the RL4CO AM/POMO baselines: cosine LR decay, grad-norm clipping (already
built into self_play_gnn_moderate), and explicit best-checkpoint selection
via periodic zero-shot evaluation on all 12 held-out tiered-benchmark
configs (5x5x5 up to 70x100x15) -- so SCoPE/Sequential are compared to
Greedy/SCIP/Auction/AM/POMO on identical footing.

Reports BOTH the raw Parallel policy (argmax, no auction) and the
auction-refined SCoPE score (must_fire = policy's own argmax != no-op,
auction decides which target -- common/auction_refinement.py) every eval
round; best-checkpoint selection is based on SCoPE's mean score across the
12 configs, since SCoPE (not raw Parallel) is this project's actual
proposed model.

New file -- does not modify rl/Dynamic_Sampling_GNN.py or any existing
training script/checkpoint.

Usage: python rl/DWTA_GNN_TRAIN_moderate_multiscale.py --total_steps 1500
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
from common.DWTA_GNN import create_gnn_actor
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

from Dynamic_Sampling_GNN_moderate import self_play_gnn_moderate

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123


@torch.no_grad()
def eval_instance_parallel_and_scope(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    """Returns (raw_parallel_objective, scope_objective) for one instance --
    raw = policy's own argmax target choice; scope = same policy's fire/hold
    decision (argmax != no-op) with auction deciding WHICH target."""
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env_raw = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    env_scope = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    init_value = env_raw.current_target_value[:, :, 0:nt].sum().item()

    for _ in range(mt):
        policy, _ = actor(env_raw.assignment_encoding, env_raw.weapon_to_target_prob, env_raw.mask_per_weapon)
        action = policy.argmax(dim=-1)
        env_raw.update_internal_variables_parallel(selected_actions=action)
        env_raw.time_update()

        remaining_value = env_scope.current_target_value[:, :, 0:nt]
        prob = env_scope.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env_scope.mask_per_weapon[:, :, :nw, :nt] > 0
        policy_s, _ = actor(env_scope.assignment_encoding, env_scope.weapon_to_target_prob, env_scope.mask_per_weapon)
        policy_choice = policy_s[:, :, :nw, :].argmax(dim=-1)
        must_fire = policy_choice < nt
        action_s = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        env_scope.update_internal_variables_parallel(selected_actions=action_s)
        env_scope.time_update()

    remaining_raw = env_raw.current_target_value[:, :, 0:nt].sum().item()
    remaining_scope = env_scope.current_target_value[:, :, 0:nt].sum().item()
    return remaining_raw / max(init_value, 1e-8), remaining_scope / max(init_value, 1e-8)


def eval_all_configs(actor):
    """Zero-shot eval on all 12 configs' fixed MODERATE test sets (seed=123,
    same generator/order as common/rl4co_eval.py's moderate loader)."""
    actor.eval()
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        raw_objs, scope_objs = [], []
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            r_raw, r_scope = eval_instance_parallel_and_scope(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST)
            raw_objs.append(r_raw)
            scope_objs.append(r_scope)
        results.append((M, N, T, float(np.mean(raw_objs)), float(np.mean(scope_objs))))
    actor.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=1500, help='one step = one episode')
    parser.add_argument('--lr', type=float, default=None, help='defaults to ACTOR_LEARNING_RATE hyperparam')
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    actor = create_gnn_actor().to(DEVICE)
    lr = args.lr if args.lr is not None else ACTOR_LEARNING_RATE
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor.optimizer, T_max=args.total_steps, eta_min=lr * 0.01)

    print(f"[SCOPE-MULTISCALE] training on M,N,T ~ U[5,7] each, evaluating zero-shot "
          f"on {len(ALL_EVAL_CONFIGS)} held-out configs (moderate curriculum) every {args.eval_every} steps")

    best_scope_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/SCoPE_multiscale_seed{args.seed}_best_actor.pt"
    final_path = f"result/SCoPE_multiscale_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        actor_loss, epoch_info = self_play_gnn_moderate(actor, episode=1, epoch=step)
        scheduler.step()

        if step % 20 == 0:
            print(f"[SCOPE-MULTISCALE] step {step}/{args.total_steps} "
                  f"actor_loss={actor_loss:.4f} destruction={epoch_info['destruction_ratio']:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [SCOPE-MULTISCALE] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(actor)
            scope_scores = []
            for M, N, T, raw_score, scope_score in results:
                scope_scores.append(scope_score)
                print(f"    {M}M_{N}N_{T}T: Parallel(raw)={raw_score:.4f}  SCoPE(auction)={scope_score:.4f}", flush=True)

            mean_scope = sum(scope_scores) / len(scope_scores)
            if mean_scope < best_scope_score:
                best_scope_score = mean_scope
                best_step = step
                torch.save(actor.state_dict(), best_path)
            print(f"    mean_SCoPE_across_12_configs={mean_scope:.4f}  "
                  f"(best so far: {best_scope_score:.4f} at step {best_step})", flush=True)

    torch.save(actor.state_dict(), final_path)
    print(f"Saved final actor to {final_path}")
    print(f"Saved best actor (step {best_step}, mean_SCoPE={best_scope_score:.4f}) to {best_path}")


if __name__ == "__main__":
    main()
