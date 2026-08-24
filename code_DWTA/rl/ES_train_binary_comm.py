"""
Evolution Strategies (ES, Salimans et al. 2017 -- antithetic sampling,
Monte Carlo estimate of the fitness gradient w.r.t. a Gaussian parameter
perturbation) training for EdgeAwareGNN_ACTOR_BINARY_COMM
(common/DWTA_GNN_binary_comm.py) -- binary fire/hold + comm layer -- with
the standard 1:1 auction downstream, on the MODERATE temporal-dilemma
curriculum, multi-scale (M,N,T ~ U[5,7]).

Motivation (2026-08-20 session, end of a long investigation -- see
brerla_sinkhorn_coordination_experiment memory for the full trail): every
REINFORCE-based binary-actor attempt this session collapsed (always-fire or
always-hold). A supervised rollout-labeling alternative was also tried and
also failed (destruction=0.0 recurring even under pure BCE, no REINFORCE at
all) -- and was separately criticized because its per-weapon fire-vs-hold
ROLLOUT LABELS are computed by asking the (known-myopic) AUCTION which
target to fire at within each branch, so the labels inherit the auction's
own weakness.

ES sidesteps BOTH problems simultaneously:
  1. It never computes a policy gradient / log-prob / backprop through a
     sampled action at all -- it perturbs the actor's PARAMETERS with
     Gaussian noise and compares the resulting policy's TRUE end-to-end
     fitness (destruction ratio, after the full network->auction->
     environment pipeline). Fitness as a function of PARAMETERS is smooth
     regardless of whether the output distribution is binary/bang-bang,
     unlike a policy-gradient estimator's variance, which explodes
     precisely for that kind of output.
  2. Unlike the supervised-rollout approach, there is no intermediate
     "which target would the auction pick" label to get contaminated by
     the auction's myopia -- ES optimizes the REAL downstream result of
     the actual deployed (network + auction) pipeline directly, so the
     auction's own weakness is just a fixed part of the environment being
     optimized against, not a corrupting influence on the training signal
     itself.

Trains the BINARY (not N+1-way) architecture specifically because the
auction discards target choice regardless of training algorithm -- training
richer target-choice information via ES would be exactly as wasted an
effort as it was under REINFORCE (see the must_fire diagnostic in the same
memory file). Binary is the only architecture whose entire learned output
is the one channel (fire/hold) that actually reaches the environment.

New file -- does not modify common/DWTA_GNN_binary_comm.py or any existing
training script/checkpoint.

Usage: python rl/ES_train_binary_comm.py --total_steps 100 --population 15
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

from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters
from Dynamic_Sampling_GNN_binary import get_random_temporal_problem_size

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_params[idx:idx + n].view_as(p))
        idx += n


@torch.no_grad()
def evaluate_fitness(actor, assignment_encoding, weapon_to_target_prob, num_weapons, num_targets, max_time):
    """Run one full deterministic (fire_prob>0.5) episode + 1:1 auction,
    return mean destruction ratio (higher is better -- ES maximizes this)."""
    env = Environment(assignment_encoding=assignment_encoding.clone(),
                       weapon_to_target_prob=weapon_to_target_prob.clone(), max_time=max_time)
    original_value = env.original_target_value[:, :, 0:num_targets].sum(2)

    for _ in range(max_time):
        remaining_value = env.current_target_value[:, :, 0:num_targets]
        prob = env.weapon_to_target_prob[:, :, :num_weapons, :num_targets]
        legal_mask = env.mask_per_weapon[:, :, :num_weapons, :num_targets] > 0
        fire_prob, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = fire_prob > 0.5
        action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    final_value = env.current_target_value[:, :, 0:num_targets].sum(2)
    destruction_ratio = 1 - (final_value / (original_value + 1e-8))
    return destruction_ratio.mean().item()


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
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        objs = []
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            objs.append(eval_instance_binary_1to1(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        results.append((M, N, T, float(np.mean(objs))))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=100, help='ES iterations')
    parser.add_argument('--population', type=int, default=15, help='antithetic pairs (total evals/iter = 2x this)')
    parser.add_argument('--sigma', type=float, default=0.02, help='parameter noise std')
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    actor = create_gnn_actor_binary_comm().to(DEVICE)
    actor.eval()  # deterministic forward (dropout off) for stable fitness comparisons across population members
    theta = get_flat_params(actor)
    n_params = theta.numel()
    print(f"[ES-BINARY-COMM] {n_params} parameters, population={2*args.population} evals/iter, "
          f"sigma={args.sigma}, lr={args.lr}, on M,N,T ~ U[5,7]")

    best_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/ES_BinaryComm_multiscale_seed{args.seed}_best_actor.pt"
    final_path = f"result/ES_BinaryComm_multiscale_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        num_weapons, num_targets, max_time = get_random_temporal_problem_size()
        V, P, TW, amm_list, prep_list, cost_list = generate_moderate_temporal_instance(
            num_weapons, num_targets, max_time
        )
        original_hyperparams = patch_hyperparameters_for_epoch(
            num_weapons, num_targets, max_time, amm_list,
            prep_list=prep_list, cost_list=cost_list,
        )
        try:
            ae, wtp = input_generation(
                NUM_WEAPON=num_weapons, NUM_TARGET=num_targets,
                value=V, prob=np.array(P), TW=TW, max_time=max_time,
                batch_size=TRAIN_BATCH, amm=amm_list,
            )
            ae = ae.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
            wtp = wtp.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()

            epsilons = torch.randn(args.population, n_params, device=DEVICE)
            diffs = torch.zeros(args.population, device=DEVICE)

            for i in range(args.population):
                eps = epsilons[i]
                set_flat_params(actor, theta + args.sigma * eps)
                f_pos = evaluate_fitness(actor, ae, wtp, num_weapons, num_targets, max_time)
                set_flat_params(actor, theta - args.sigma * eps)
                f_neg = evaluate_fitness(actor, ae, wtp, num_weapons, num_targets, max_time)
                diffs[i] = f_pos - f_neg

            diffs = diffs / (diffs.std() + 1e-8)
            grad_estimate = (epsilons * diffs.unsqueeze(1)).sum(dim=0) / (args.population * args.sigma)
            theta = theta + args.lr * grad_estimate
            set_flat_params(actor, theta)

        finally:
            restore_hyperparameters(original_hyperparams)

        if step % 5 == 0:
            fitness_mean = evaluate_fitness(actor, ae, wtp, num_weapons, num_targets, max_time)
            print(f"[ES-BINARY-COMM] step {step}/{args.total_steps} "
                  f"({num_weapons}Mx{num_targets}Nx{max_time}T) fitness(destruction)={fitness_mean:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [ES-BINARY-COMM] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(actor)
            scores = []
            for M, N, T, score in results:
                scores.append(score)
                print(f"    {M}M_{N}N_{T}T: ESBinaryComm={score:.4f}", flush=True)

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
