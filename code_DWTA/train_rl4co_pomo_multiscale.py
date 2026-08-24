"""
Train RL4CO's POMO policy with genuine multi-scale training -- same
rationale/design as train_rl4co_am_multiscale.py (mirrors our own GNN's
get_random_problem_size curriculum: M,N,T ~ U[5,7] each, resampled every
step, evaluated zero-shot on all 12 held-out configs). POMO's multistart
mechanism (DWTAEnv.get_num_starts/select_start_nodes,
common/rl4co_dwta_env.py) already reads the per-batch env's own N (a fresh
DWTAEnv is constructed every step from that step's generator, so there is no
staleness -- unlike DWTAContext, which had to be fixed to never cache M/N/T
at all since ONE policy instance is reused across every batch).

Baseline: POMO's own "shared" baseline (mean reward across the N multistart
rollouts of the SAME instance) -- not a critic, consistent with this
project's standing "never use critic" rule.

Usage: python train_rl4co_pomo_multiscale.py --total_steps 2000
"""
import argparse
import random
import time

import torch

from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.ops import unbatchify

from common.rl4co_dwta_env import DWTAEnv, DWTAModerateGenerator
from common.rl4co_dwta_embeddings import DWTAContext, DWTADynamicEmbedding, DWTAInitEmbedding
from common.rl4co_eval import eval_rl4co_policy, load_moderate_fixed_instances_as_td

SCALE_LOW, SCALE_HIGH = 5, 7

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]


def sample_scale():
    return (random.randint(SCALE_LOW, SCALE_HIGH),
            random.randint(SCALE_LOW, SCALE_HIGH),
            random.randint(SCALE_LOW, SCALE_HIGH))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--total_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = AttentionModelPolicy(
        embed_dim=args.embed_dim,
        num_encoder_layers=6,
        num_heads=8,
        env_name='dwta',
        normalization='batch',  # was 'instance' -- InstanceNorm1d computes stats over only N+1
                                 # nodes per instance, so train-time (N~5-7) and eval-time (N up to
                                 # 100) normalization statistics come from wildly different sample
                                 # counts, which broke generalization at the largest scales. BatchNorm
                                 # (AM's default, which generalized fine everywhere) pools stats across
                                 # the whole batch*nodes dimension instead, independent of N.
        init_embedding=DWTAInitEmbedding(args.embed_dim),
        dynamic_embedding=DWTADynamicEmbedding(args.embed_dim),
        context_embedding=DWTAContext(args.embed_dim),
        use_graph_context=False,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.lr * 0.01)

    print(f"[MULTISCALE-POMO] training on M,N,T ~ U[{SCALE_LOW},{SCALE_HIGH}] each, "
          f"evaluating zero-shot on {len(ALL_EVAL_CONFIGS)} held-out configs "
          f"(moderate curriculum) every {args.eval_every} steps")

    best_mean_score = float('inf')
    best_step = -1
    best_path = f"result/RL4CO_POMO_multiscale_seed{args.seed}_best_policy.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        M, N, T = sample_scale()
        gen = DWTAModerateGenerator(num_weapon=M, num_target=N, max_time=T)
        env = DWTAEnv(generator=gen)
        n_start = N  # one forced start per target (no-op excluded), matches DWTAEnv.get_num_starts

        batch = gen(batch_size=[args.batch_size])
        td = env.reset(batch.to(device))

        policy.train()
        out = policy(td, env, phase="train", decode_type="multistart_sampling", num_starts=n_start)
        reward = unbatchify(out["reward"], n_start)  # [batch, n_start]
        log_likelihood = unbatchify(out["log_likelihood"], n_start)  # [batch, n_start]

        # POMO's own shared baseline: mean reward across the n_start rollouts
        # of the SAME instance -- not a critic.
        advantage = reward - reward.mean(dim=1, keepdim=True)
        loss = -(advantage.detach() * log_likelihood).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % 20 == 0:
            print(f"[MULTISCALE-POMO] step {step}/{args.total_steps} ({M}Mx{N}Nx{T}T) "
                  f"loss={loss.item():.4f} mean_reward={reward.mean().item():.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            policy.eval()
            print(f"--- [MULTISCALE-POMO] zero-shot eval at step {step} ---", flush=True)
            scores = []
            for eM, eN, eT in ALL_EVAL_CONFIGS:
                eval_gen = DWTAModerateGenerator(num_weapon=eM, num_target=eN, max_time=eT)
                eval_env = DWTAEnv(generator=eval_gen)
                eval_td, scip_mean, greedy_mean, n = load_moderate_fixed_instances_as_td(
                    device, M=eM, N=eN, T=eT
                )
                # Single-path greedy decode for the reported number -- NOT
                # best-of-multistart (see brerla_pomo_fair_eval memory: POMO
                # must be compared apples-to-apples against single-path AM/
                # Greedy/SCoPE, not its own best-of-N).
                pomo_mean = eval_rl4co_policy(policy, eval_env, eval_td)
                scores.append(pomo_mean)
                ref = f"Greedy={greedy_mean:.4f}" if greedy_mean is not None else ""
                if scip_mean is not None:
                    ref += f" SCIP={scip_mean:.4f}"
                print(f"    {eM}M_{eN}N_{eT}T: POMO={pomo_mean:.4f}  {ref}", flush=True)

            mean_score = sum(scores) / len(scores)
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                best_step = step
                torch.save(policy.state_dict(), best_path)
            print(f"    mean_across_12_configs={mean_score:.4f}  "
                  f"(best so far: {best_mean_score:.4f} at step {best_step})", flush=True)
            policy.train()

    save_path = f"result/RL4CO_POMO_multiscale_seed{args.seed}_final_policy.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Saved final policy to {save_path}")
    print(f"Saved best policy (step {best_step}, mean_across_12_configs={best_mean_score:.4f}) to {best_path}")


if __name__ == "__main__":
    main()
