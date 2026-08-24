"""
Supervised warm-start (behavior cloning / imitation learning) of the FULL
N+1-way Parallel GNN actor from SCIP-(near-)optimal small-scale (M,N,T ~
U[5,7]) MODERATE-curriculum solutions -- see generate_scip_teacher_dataset.py
for how the dataset is built, and brerla_scip_imitation_warmstart memory for
the rationale: REINFORCE alone (AM, POMO, SCoPE self-play, all tried this
session) repeatedly peaks early then degrades on this curriculum; a
supervised warm-start should give REINFORCE a much better starting point to
fine-tune from, instead of starting from random weights.

Teacher forcing: at each round, the environment's state comes from applying
SCIP's OWN optimal actions so far (not the network's own predictions) --
standard behavior cloning, avoids compounding distributional shift WITHIN
a single training trajectory (the classic BC failure mode still applies
ACROSS full self-play rollouts at eval time, which is exactly why this is
a warm-start for REINFORCE fine-tuning, not a final answer on its own).

New file -- does not modify any existing training script. Does not use a
critic (plain supervised cross-entropy loss per round per weapon).

Usage: python train_scip_warmstart.py --dataset result/scip_teacher_dataset.pt --epochs 30
"""
import argparse
import os
import random
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN import create_gnn_actor
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
EVAL_SEED = 123

# Measured on scip_teacher_dataset.pt: 69.7% of (weapon, round) labels are
# no-op, 30.3% fire (split across up to N target classes each) -- unweighted
# CE collapsed entirely to predicting no-op (see module docstring / commit
# history). Down-weight no-op by fire_count/noop_count so it can't win on
# numerical majority alone.
NOOP_CLASS_WEIGHT = 3268 / 7517


@torch.no_grad()
def eval_instance_parallel_and_scope(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
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
    actor.eval()
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(EVAL_SEED)
        raw_objs, scope_objs = [], []
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            r_raw, r_scope = eval_instance_parallel_and_scope(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST)
            raw_objs.append(r_raw)
            scope_objs.append(r_scope)
        results.append((M, N, T, float(np.mean(raw_objs)), float(np.mean(scope_objs))))
    actor.train()
    return results


def teacher_force_one_instance(actor, item):
    """Runs the environment forced along SCIP's own optimal action sequence,
    computing cross-entropy loss between the actor's prediction and SCIP's
    action at every round for every weapon. Returns the scalar loss
    (mean over rounds*weapons) for this instance."""
    M, N, T = item['M'], item['N'], item['T']
    V, P, TW, AMM, PREP = item['V'], np.asarray(item['P']), item['TW'], item['AMM'], item['PREP']
    actions = item['actions']  # [T][M], target idx or N=no-op

    patch_globals(M, N, T, AMM, PREP, [1] * M)
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=P, TW=TW,
                                max_time=T, batch_size=1, alpha=1.0, amm=AMM)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=T)

    total_loss = 0.0
    for t in range(T):
        # Clone before feeding to the actor -- env.update_internal_variables_parallel /
        # time_update() mutate these buffers in place next round, which would
        # otherwise corrupt the autograd graph for this round's forward pass
        # (matches the existing self_play_gnn(_moderate) pattern).
        current_state = env.assignment_encoding.clone()
        current_prob = env.weapon_to_target_prob.clone()
        current_mask = env.mask_per_weapon.clone()

        policy, _ = actor(current_state, current_prob, current_mask)
        label = torch.tensor(actions[t], device=DEVICE, dtype=torch.long).view(1, 1, M)  # [1,1,M]
        # Class-weighted CE: SCIP-optimal labels are ~70% no-op / 30% fire
        # (measured on the teacher dataset), and that 30% is further split
        # across up to N target classes -- unweighted CE collapsed to
        # "always predict no-op" (verified: CE_loss and eval score were both
        # perfectly frozen across 30 epochs of otherwise-working training).
        # Down-weight the no-op class by the empirical fire/no-op ratio so
        # it can't win purely on numerical majority.
        weight = torch.ones(N + 1, device=DEVICE)
        weight[N] = NOOP_CLASS_WEIGHT
        loss_t = F.cross_entropy(policy.view(M, N + 1), label.view(M), weight=weight)
        total_loss = total_loss + loss_t

        # Teacher-force: step the environment using SCIP's OWN action, not the network's.
        env.update_internal_variables_parallel(selected_actions=label)
        env.time_update()

    return total_loss / T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='result/scip_teacher_dataset.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=2, help='epochs between 12-config zero-shot evals')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = torch.load(args.dataset, weights_only=False)
    print(f"[WARMSTART] loaded {len(dataset)} SCIP-teacher instances from {args.dataset}")

    actor = create_gnn_actor().to(DEVICE)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    best_scope_score = float('inf')
    best_epoch = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/SCIP_warmstart_seed{args.seed}_best_actor.pt"
    final_path = f"result/SCIP_warmstart_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(dataset)
        actor.train()
        epoch_loss = 0.0
        for item in dataset:
            loss = teacher_force_one_instance(actor, item)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)

        print(f"[WARMSTART] epoch {epoch}/{args.epochs} CE_loss={epoch_loss:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"--- [WARMSTART] zero-shot eval at epoch {epoch} ---", flush=True)
            results = eval_all_configs(actor)
            scope_scores = []
            for M, N, T, raw_score, scope_score in results:
                scope_scores.append(scope_score)
                print(f"    {M}M_{N}N_{T}T: Parallel(raw)={raw_score:.4f}  SCoPE(auction)={scope_score:.4f}", flush=True)
            mean_scope = sum(scope_scores) / len(scope_scores)
            if mean_scope < best_scope_score:
                best_scope_score = mean_scope
                best_epoch = epoch
                torch.save(actor.state_dict(), best_path)
            print(f"    mean_SCoPE_across_12_configs={mean_scope:.4f}  "
                  f"(best so far: {best_scope_score:.4f} at epoch {best_epoch})", flush=True)

    torch.save(actor.state_dict(), final_path)
    print(f"Saved final actor to {final_path}")
    print(f"Saved best actor (epoch {best_epoch}, mean_SCoPE={best_scope_score:.4f}) to {best_path}")


if __name__ == "__main__":
    main()
