"""
Single-process JOINT training of the two-agent pipeline (proposer +
auditor) -- see rl/Dynamic_Sampling_GNN_joint.py for the design.

Structurally this alternates two networks in one loop the way GAN training
does, but the two agents are COOPERATIVE, not adversarial: both are
optimizing the same objective (minimize remaining target value). There is
no minimax, so GAN's characteristic adversarial-oscillation failure mode
does not apply; the shared risk is only that each agent is chasing a
moving target while the other keeps updating.

Evaluated zero-shot on the same 12 held-out configs as every other method,
reporting BOTH the proposer-alone objective and the post-audit objective,
so the auditor's marginal contribution is visible directly.

Usage:
  # warm-start proposer from the current best checkpoint (default)
  python rl/DWTA_GNN_TRAIN_joint_multiscale.py --total_steps 200 --eval_every 50
  # or train both from scratch
  python rl/DWTA_GNN_TRAIN_joint_multiscale.py --base_ckpt none
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
from common.auction_refinement import auction_round_action, auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

from Dynamic_Sampling_GNN_joint import self_play_joint, get_random_wide_problem_size
from Dynamic_Sampling_GNN_improve import simulate_with_flips, score_all_slots

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123
DEFAULT_BASE_CKPT = "result/CommSinkhornSCoPE_multiscale_seed5_best_actor.pt"


@torch.no_grad()
def eval_instance(base_actor, improve_actor, V, P, TW, nw, nt, mt, amm, prep, cost, n_edits,
                   auction_fn=None):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)

    flip_mask = torch.zeros(1, 1, mt, nw, dtype=torch.bool, device=DEVICE)
    obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt,
                                       auction_fn=auction_fn)
    base_obj = obj.mean().item()
    best = base_obj

    n_slots = mt * nw
    for _ in range(min(n_edits, n_slots)):
        logits = score_all_slots(improve_actor, states, nw).reshape(1, 1, n_slots)
        logits = logits.masked_fill(flip_mask.reshape(1, 1, n_slots), -1e9)
        idx = logits.argmax(dim=-1)
        new_flip = flip_mask.reshape(1, 1, n_slots).clone()
        new_flip.scatter_(-1, idx.unsqueeze(-1), True)
        flip_mask = new_flip.reshape(1, 1, mt, nw)
        obj, states = simulate_with_flips(base_actor, ae, wtp, flip_mask, nw, nt, mt,
                                           auction_fn=auction_fn)
        best = min(best, obj.mean().item())  # keep-best: cannot regress below proposer

    return base_obj, best


def eval_all_configs(base_actor, improve_actor, n_edits, auction_fn=None):
    base_actor.eval()
    improve_actor.eval()
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        b_objs, a_objs = [], []
        for _ in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            b, a = eval_instance(base_actor, improve_actor, V, np.asarray(P), TW,
                                  M, N, T, AMM, PREP, COST, n_edits, auction_fn=auction_fn)
            b_objs.append(b)
            a_objs.append(a)
        results.append((M, N, T, float(np.mean(b_objs)), float(np.mean(a_objs))))
    base_actor.train()
    improve_actor.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=200)
    # The proposer must use the SAME lr the standalone recipe uses
    # (ACTOR_LEARNING_RATE = 5e-4, see rl/DWTA_GNN_TRAIN_comm_sinkhorn_multiscale.py).
    # An earlier default of 1e-4 here -- copied from the auditor's schedule --
    # left the proposer 5x undertrained and was the entire reason joint
    # training looked worse than the two-stage setup (proposer stuck at
    # ~0.22 after 400 updates instead of reaching ~0.1357).
    parser.add_argument('--base_lr', type=float, default=ACTOR_LEARNING_RATE)
    parser.add_argument('--improve_lr', type=float, default=1e-4)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--eval_edits', type=int, default=30, help='inference edit budget at eval')
    parser.add_argument('--n_edits', type=int, default=3, help='edits per training step')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--para_size', type=int, default=8)
    parser.add_argument('--base_ckpt', type=str, default=DEFAULT_BASE_CKPT,
                        help='warm-start checkpoint for the proposer, or "none" to train from scratch')
    parser.add_argument('--curriculum', type=str, default='moderate',
                        choices=['moderate', 'wide'],
                        help="'moderate': the standing M,N,T ~ U[5,7] curriculum. "
                             "'wide': M ~ U[wide_m_lo, wide_m_hi], N >= M, T ~ U[5,10] -- "
                             "targets the Large/Battlefield deficit, since the parallel "
                             "proposer must LEARN coordination that sequential decoding "
                             "gets structurally for free at any M.")
    parser.add_argument('--wide_m_lo', type=int, default=5)
    parser.add_argument('--wide_m_hi', type=int, default=30)
    parser.add_argument('--wide_t_lo', type=int, default=5)
    parser.add_argument('--wide_t_hi', type=int, default=10)
    parser.add_argument('--auction', type=str, default='1to1',
                        choices=['1to1', 'multifire'],
                        help="target-assignment rule. '1to1' is the eviction auction; "
                             "'multifire' is the capacitated many-to-one variant, which "
                             "measurement showed is both better at Large tier (it lifts the "
                             "1:1 auction's hard dispersion=1.0 constraint) and faster.")
    parser.add_argument('--mode', type=str, default='alternating',
                        choices=['alternating', 'simultaneous'],
                        help="'alternating': one agent per step. 'simultaneous': both every step "
                             "(one step therefore costs twice as much, so use half the total_steps "
                             "for a matched number of per-agent updates and matched simulation cost).")
    parser.add_argument('--tag', type=str, default='', help='suffix for output paths (avoids clobbering)')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    if args.base_ckpt.lower() != 'none':
        base_actor.load_state_dict(torch.load(args.base_ckpt, map_location=DEVICE, weights_only=False))
        print(f"[JOINT] proposer warm-started from {args.base_ckpt}")
    else:
        print("[JOINT] proposer trained from scratch")
    base_actor.optimizer = torch.optim.Adam(base_actor.parameters(), lr=args.base_lr)

    improve_actor = create_gnn_actor_improve().to(DEVICE)
    improve_actor.optimizer = torch.optim.Adam(improve_actor.parameters(), lr=args.improve_lr)

    auction_fn = (auction_round_action_multifire if args.auction == 'multifire'
                  else auction_round_action)
    print(f"[JOINT] auction={args.auction}")

    if args.curriculum == 'wide':
        problem_size_fn = lambda: get_random_wide_problem_size(
            args.wide_m_lo, args.wide_m_hi, args.wide_t_lo, args.wide_t_hi)
        print(f"[JOINT] curriculum=wide  M~U[{args.wide_m_lo},{args.wide_m_hi}], "
              f"N>=M, T~U[{args.wide_t_lo},{args.wide_t_hi}]")
    else:
        problem_size_fn = None  # falls back to the standing U[5,7] curriculum
        print("[JOINT] curriculum=moderate  M,N,T ~ U[5,7]")

    # T_max must match how many times each scheduler actually steps:
    # alternating -> once every other step; simultaneous -> every step.
    # Getting this wrong leaves the cosine schedule only half-completed.
    per_agent_steps = max(1, args.total_steps // 2) if args.mode == 'alternating' else args.total_steps
    base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_actor.optimizer, T_max=per_agent_steps, eta_min=args.base_lr * 0.01)
    imp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        improve_actor.optimizer, T_max=per_agent_steps, eta_min=args.improve_lr * 0.01)

    print(f"[JOINT] proposer+auditor co-trained in one loop, n_edits={args.n_edits} (train) / "
          f"{args.eval_edits} (eval), batch={args.batch_size}x{args.para_size}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    os.makedirs("result", exist_ok=True)
    best_base_path = f"result/Joint_base_seed{args.seed}{suffix}_best.pt"
    best_imp_path = f"result/Joint_improve_seed{args.seed}{suffix}_best.pt"

    best_score = float('inf')
    best_step = -1
    t0 = time.time()

    last_p_loss = last_a_loss = 0.0
    last_destr = last_base = last_audited = 0.0

    for step in range(1, args.total_steps + 1):
        if args.mode == 'alternating':
            # Exactly one agent is updated per step (see
            # Dynamic_Sampling_GNN_joint.self_play_joint's docstring).
            train_proposer = (step % 2 == 1)
            p_loss, a_loss, info = self_play_joint(
                base_actor, improve_actor, episode=1, epoch=step,
                n_edits=args.n_edits, batch_size=args.batch_size, para_size=args.para_size,
                train_proposer_this_step=train_proposer, problem_size_fn=problem_size_fn, auction_fn=auction_fn,
            )
            if train_proposer:
                last_p_loss, last_destr = p_loss, info['destruction_ratio']
                base_sched.step()
            else:
                last_a_loss = a_loss
                last_base, last_audited = info['base_objective'], info['audited_objective']
                imp_sched.step()
        else:  # simultaneous -- both agents updated every step
            p_loss, _, p_info = self_play_joint(
                base_actor, improve_actor, episode=1, epoch=step,
                n_edits=args.n_edits, batch_size=args.batch_size, para_size=args.para_size,
                train_proposer_this_step=True, problem_size_fn=problem_size_fn, auction_fn=auction_fn,
            )
            _, a_loss, a_info = self_play_joint(
                base_actor, improve_actor, episode=1, epoch=step,
                n_edits=args.n_edits, batch_size=args.batch_size, para_size=args.para_size,
                train_proposer_this_step=False, problem_size_fn=problem_size_fn, auction_fn=auction_fn,
            )
            last_p_loss, last_destr = p_loss, p_info['destruction_ratio']
            last_a_loss = a_loss
            last_base, last_audited = a_info['base_objective'], a_info['audited_objective']
            base_sched.step()
            imp_sched.step()

        if step % 10 == 0:
            print(f"[JOINT] step {step}/{args.total_steps} "
                  f"proposer_loss={last_p_loss:.4f} auditor_loss={last_a_loss:.4f} "
                  f"destr={last_destr:.4f} "
                  f"base={last_base:.4f}->audited={last_audited:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            print(f"--- [JOINT] zero-shot eval at step {step} ---", flush=True)
            results = eval_all_configs(base_actor, improve_actor, args.eval_edits,
                                        auction_fn=auction_fn)
            b_scores, a_scores = [], []
            for M, N, T, b, a in results:
                b_scores.append(b)
                a_scores.append(a)
                print(f"    {M}M_{N}N_{T}T: proposer={b:.4f}  audited={a:.4f}", flush=True)

            mean_b = sum(b_scores) / len(b_scores)
            mean_a = sum(a_scores) / len(a_scores)
            if mean_a < best_score:
                best_score = mean_a
                best_step = step
                torch.save(base_actor.state_dict(), best_base_path)
                torch.save(improve_actor.state_dict(), best_imp_path)
            print(f"    mean_proposer={mean_b:.4f}  mean_audited={mean_a:.4f}  "
                  f"(best so far: {best_score:.4f} at step {best_step})", flush=True)

    print(f"Best audited mean={best_score:.4f} at step {best_step}")
    print(f"Saved: {best_base_path}, {best_imp_path}")


if __name__ == "__main__":
    main()
