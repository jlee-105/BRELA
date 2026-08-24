"""
Evaluation harness comparing the parallel multi-pointer decoder
(common/DWTA_GNN.py::EdgeAwareGNN_ACTOR) against the reconstructed
sequential/original decoder (common/DWTA_GNN_sequential.py::
EdgeAwareGNN_ACTOR_Sequential) on identical held-out instances.

Greedy (argmax) rollout only -- no beam search, since the beam/rollout code
in rl_rollout/ is hard-coded for the old flat single-edge decoding and would
need a separate rewrite to support the parallel actor (out of scope here;
see BReRLA_revision_plan.md's soft-coupling validation plan).

For each shot, "pre-shot value fraction" is the target's remaining value
fraction (relative to its own original value) AT THE TIME THE ACTING
POLICY OBSERVED THE STATE it decided from -- i.e. the start-of-round value
for the parallel decoder (all weapons in a round decide from the same
snapshot, so this is what they could see) and the start-of-decision value
for the sequential decoder (state updates between each single decision).
This directly measures whether the policy chose to fire on an
already-nearly-dead target given what it could observe -- the exact
question the soft-coupling hypothesis is about.

Usage:
    python eval_coordination_comparison.py \
        --parallel-checkpoint result/GNN_TRAIN_<date>/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt \
        --sequential-checkpoint result/GNN_TRAIN_SEQ_<date>/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt \
        --out eval_coordination_comparison.csv
"""

import argparse
import csv
import statistics
import time

import torch

from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_HYPER_PARAMETER import *
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.DWTA_GNN import create_gnn_actor
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.utilities import compute_overkill_rate, compute_fire_dispersion

EVAL_N = 50
EVAL_NW, EVAL_NT, EVAL_MT = 5, 5, 5
EVAL_SEED = 42
EVAL_ALPHA = 1.0
OVERKILL_THRESHOLD = 0.1


def generate_fixed_eval_instances():
    """Same seed-42/50-instance/5x5x5 convention as result/GNN_TRAIN_20260308/
    checkpoint_evaluation.txt, so both architectures are scored on identical
    held-out instances."""
    torch.manual_seed(EVAL_SEED)
    instances = []
    for _ in range(EVAL_N):
        assignment_encoding, weapon_to_target_prob = input_generation(
            NUM_WEAPON=EVAL_NW, NUM_TARGET=EVAL_NT,
            value=None, prob=None, TW=None,
            max_time=EVAL_MT, batch_size=1, alpha=EVAL_ALPHA,
        )
        instances.append((assignment_encoding.unsqueeze(1).clone(), weapon_to_target_prob.unsqueeze(1).clone()))
    return instances


def _instance_summary(shot_fracs, shot_counts, original_value_per_target, env, elapsed):
    final_value = env.current_target_value[0, 0, :EVAL_NT].sum().item()
    init_value = original_value_per_target.sum().item()
    destruction = 1.0 - final_value / max(init_value, 1e-8)
    return {
        'destruction_ratio': destruction,
        'overkill_rate': compute_overkill_rate(shot_fracs, threshold=OVERKILL_THRESHOLD),
        'fire_dispersion': compute_fire_dispersion(shot_counts),
        'inference_time_s': elapsed,
        'num_shots': len(shot_fracs),
    }


def eval_parallel(actor, instances):
    actor.eval()
    results = []
    with torch.no_grad():
        for assignment_encoding, weapon_to_target_prob in instances:
            env = Environment(assignment_encoding=assignment_encoding.clone(),
                               weapon_to_target_prob=weapon_to_target_prob.clone(),
                               max_time=EVAL_MT)
            original_value_per_target = env.original_target_value[0, 0, :EVAL_NT].clone()
            shot_fracs = []
            shot_counts = [0] * EVAL_NT

            t0 = time.time()
            for _ in range(EVAL_MT):
                pre_value = env.current_target_value[0, 0, :EVAL_NT].clone()
                policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
                action = policy.argmax(dim=-1)  # [1, 1, W] greedy per-weapon pick

                for w in range(EVAL_NW):
                    tgt = action[0, 0, w].item()
                    if tgt < EVAL_NT:
                        frac = (pre_value[tgt] / (original_value_per_target[tgt] + 1e-8)).item()
                        shot_fracs.append(frac)
                        shot_counts[tgt] += 1

                env.update_internal_variables_parallel(selected_actions=action)
                env.time_update()
            elapsed = time.time() - t0

            results.append(_instance_summary(shot_fracs, shot_counts, original_value_per_target, env, elapsed))
    return results


def eval_sequential(actor, instances):
    actor.eval()
    results = []
    with torch.no_grad():
        for assignment_encoding, weapon_to_target_prob in instances:
            env = Environment(assignment_encoding=assignment_encoding.clone(),
                               weapon_to_target_prob=weapon_to_target_prob.clone(),
                               max_time=EVAL_MT)
            original_value_per_target = env.original_target_value[0, 0, :EVAL_NT].clone()
            shot_fracs = []
            shot_counts = [0] * EVAL_NT
            num_actions = EVAL_NW * EVAL_NT

            t0 = time.time()
            for _ in range(EVAL_MT):
                for _ in range(EVAL_NW):
                    pre_value = env.current_target_value[0, 0, :EVAL_NT].clone()
                    policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask)
                    flat = policy.view(-1, num_actions + 1)
                    action_idx = flat.argmax(dim=1).item()

                    if action_idx < num_actions:
                        tgt = action_idx % EVAL_NT
                        frac = (pre_value[tgt] / (original_value_per_target[tgt] + 1e-8)).item()
                        shot_fracs.append(frac)
                        shot_counts[tgt] += 1

                    action = torch.tensor([[action_idx]], device=DEVICE)
                    env.update_internal_variables(selected_action=action)
                env.time_update()
            elapsed = time.time() - t0

            results.append(_instance_summary(shot_fracs, shot_counts, original_value_per_target, env, elapsed))
    return results


def summarize(results, label):
    keys = ['destruction_ratio', 'overkill_rate', 'fire_dispersion', 'inference_time_s', 'num_shots']
    summary = {'architecture': label, 'n_instances': len(results)}
    for k in keys:
        values = [r[k] for r in results]
        summary[f'{k}_mean'] = statistics.mean(values)
        summary[f'{k}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def print_comparison(summaries):
    print(f"\n{'Metric':<24}", end="")
    for s in summaries:
        print(f"{s['architecture']:>22}", end="")
    print()
    print("-" * (24 + 22 * len(summaries)))
    for k, label in [
        ('destruction_ratio', 'Destruction ratio'),
        ('overkill_rate', f'Overkill rate (<{OVERKILL_THRESHOLD:.0%})'),
        ('fire_dispersion', 'Fire dispersion'),
        ('inference_time_s', 'Inference time (s)'),
        ('num_shots', 'Shots/instance'),
    ]:
        print(f"{label:<24}", end="")
        for s in summaries:
            print(f"{s[k + '_mean']:>12.4f} +/-{s[k + '_std']:>6.4f}", end="")
        print()
    print(f"\n(n={summaries[0]['n_instances']} held-out instances, seed={EVAL_SEED}, "
          f"{EVAL_NW}x{EVAL_NT}x{EVAL_MT}, greedy rollout, 1 training seed per architecture "
          f"-- directional signal only, not multi-seed statistical evidence)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parallel-checkpoint', required=True)
    parser.add_argument('--sequential-checkpoint', required=True)
    parser.add_argument('--out', default='eval_coordination_comparison.csv')
    args = parser.parse_args()

    instances = generate_fixed_eval_instances()

    parallel_actor = create_gnn_actor().to(DEVICE)
    parallel_actor.load_state_dict(torch.load(args.parallel_checkpoint, map_location=DEVICE))
    parallel_results = eval_parallel(parallel_actor, instances)

    sequential_actor = create_gnn_actor_sequential().to(DEVICE)
    sequential_actor.load_state_dict(torch.load(args.sequential_checkpoint, map_location=DEVICE))
    sequential_results = eval_sequential(sequential_actor, instances)

    summaries = [
        summarize(parallel_results, 'Parallel (new)'),
        summarize(sequential_results, 'Sequential (original)'),
    ]
    print_comparison(summaries)

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['architecture', 'instance_idx', 'destruction_ratio', 'overkill_rate',
                          'fire_dispersion', 'inference_time_s', 'num_shots'])
        for label, results in [('parallel', parallel_results), ('sequential', sequential_results)]:
            for i, r in enumerate(results):
                writer.writerow([label, i, r['destruction_ratio'], r['overkill_rate'],
                                  r['fire_dispersion'], r['inference_time_s'], r['num_shots']])
    print(f"\nPer-instance results written to {args.out}")


if __name__ == "__main__":
    main()
