"""
Train EdgeAwareGNN_ACTOR_COMM_QVALUE (common/DWTA_GNN_comm_qvalue.py) via
Monte-Carlo Q-value regression (rl/Dynamic_Sampling_GNN_qvalue.py) through
the Q-value-driven capacitated auction
(common/auction_refinement_qvalue.py::auction_round_action_multifire_qvalue)
-- NO REINFORCE anywhere -- on the MODERATE temporal-dilemma curriculum,
multi-scale (M,N,T ~ U[5,7]). Zero-shot evaluated on the same 12 held-out
configs as every other method this session, for a directly comparable
number.

REDA-inspired (see brerla_sinkhorn_coordination_experiment memory):
Q-values ARE the auction's benefit matrix (not discarded target-choice
info, unlike CommSCoPE/Sinkhorn's must_fire-only survival to the auction),
trained via regression to each weapon's own reward-to-go, not
policy-gradient sampling -- sidesteps every instability this whole session's
REINFORCE/rollout-supervised binary attempts hit.

New file -- does not modify any existing training script/checkpoint.

Usage: python rl/DWTA_GNN_TRAIN_qvalue_multiscale.py --total_steps 300 --eval_every 20
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
from common.DWTA_GNN_comm_qvalue import create_gnn_actor_comm_qvalue
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement_qvalue import auction_round_action_multifire_qvalue
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

from Dynamic_Sampling_GNN_qvalue import self_play_gnn_qvalue

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123


@torch.no_grad()
def eval_instance_qvalue(actor, V, P, TW, nw, nt, mt, amm, prep, cost, max_per_target=2):
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
        q_values, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        action, _ = auction_round_action_multifire_qvalue(
            q_values, remaining_value, prob, legal_mask, max_per_target=max_per_target,
        )
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
            objs.append(eval_instance_qvalue(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        results.append((M, N, T, float(np.mean(objs))))
    actor.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=300, help='one step = one episode')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--max_per_target', type=int, default=2)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    actor = create_gnn_actor_comm_qvalue().to(DEVICE)
    lr = args.lr if args.lr is not None else ACTOR_LEARNING_RATE
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor.optimizer, T_max=args.total_steps, eta_min=lr * 0.01)

    print(f"[QVALUE-MULTISCALE] training on M,N,T ~ U[5,7], Monte-Carlo Q regression (no REINFORCE), "
          f"evaluating zero-shot on {len(ALL_EVAL_CONFIGS)} held-out configs every {args.eval_every} steps")

    best_score = float('inf')
    best_step = -1
    os.makedirs("result", exist_ok=True)
    best_path = f"result/QValue_multiscale_seed{args.seed}_best_actor.pt"
    final_path = f"result/QValue_multiscale_seed{args.seed}_final_actor.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        loss, epoch_info = self_play_gnn_qvalue(actor, episode=1, epoch=step, max_per_target=args.max_per_target)
        scheduler.step()

        if step % 10 == 0:
            print(f"[QVALUE-MULTISCALE] step {step}/{args.total_steps} "
                  f"mse_loss={loss:.4f} destruction={epoch_info['destruction_ratio']:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [QVALUE-MULTISCALE] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(actor)
            scores = []
            for M, N, T, score in results:
                scores.append(score)
                print(f"    {M}M_{N}N_{T}T: QValue={score:.4f}", flush=True)

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
