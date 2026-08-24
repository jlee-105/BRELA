"""
Train RL4CO's AM policy with genuine multi-scale training, mirroring our
own GNN's own curriculum (rl/Dynamic_Sampling_GNN.py::get_random_problem_size:
num_weapon/num_target/max_time each ~ randint(5, 7), resampled every
episode/training step) -- for a fair comparison against SCoPE/Sequential,
which are BOTH trained this way and then evaluated zero-shot on much larger
held-out configs (5x5x5 up to 70x100x15).

This needs a custom loop instead of RL4COTrainer.fit(): RL4CO's Trainer
assumes one fixed-size env/dataset for the whole run. Here a fresh
(M, N, T) is drawn every training step, a throwaway DWTAEnv is built for it
(cheap -- no persistent state), and one manual REINFORCE update (plain
mean-baseline, no critic -- consistent with this project's standing "never
use critic" rule) is applied. This only works because DWTAContext/
DWTAInitEmbedding/DWTADynamicEmbedding (common/rl4co_dwta_embeddings.py) are
now genuinely size-agnostic (a earlier version was not -- see
brerla_rl4co_baseline_curriculum / the DWTAContext fix earlier this
session); the SAME policy weights are reused across every batch regardless
of (M, N, T).

Usage: python train_rl4co_am_multiscale.py --total_steps 2000
"""
import argparse
import random
import time

import torch

from rl4co.models.zoo.am.policy import AttentionModelPolicy

from common.rl4co_dwta_env import DWTAEnv, DWTAModerateGenerator
from common.rl4co_dwta_embeddings import DWTAContext, DWTADynamicEmbedding, DWTAInitEmbedding
from common.rl4co_eval import eval_rl4co_policy, load_moderate_fixed_instances_as_td

# Mirrors rl/Dynamic_Sampling_GNN.py::get_random_problem_size exactly
SCALE_LOW, SCALE_HIGH = 5, 7

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]


def sample_scale():
    M = random.randint(SCALE_LOW, SCALE_HIGH)
    N = random.randint(SCALE_LOW, SCALE_HIGH)
    T = random.randint(SCALE_LOW, SCALE_HIGH)
    return M, N, T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
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
        num_encoder_layers=3,
        num_heads=8,
        env_name='dwta',
        init_embedding=DWTAInitEmbedding(args.embed_dim),
        dynamic_embedding=DWTADynamicEmbedding(args.embed_dim),
        context_embedding=DWTAContext(args.embed_dim),
        use_graph_context=False,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    # Cosine decay to 1% of peak lr -- the un-annealed run oscillated instead
    # of converging (best checkpoint was step ~1800/3000, not the last one).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.lr * 0.01)

    print(f"[MULTISCALE] training on M,N,T ~ U[{SCALE_LOW},{SCALE_HIGH}] each, "
          f"evaluating zero-shot on {len(ALL_EVAL_CONFIGS)} held-out configs "
          f"(moderate curriculum) every {args.eval_every} steps")

    best_mean_score = float('inf')
    best_step = -1
    best_path = f"result/RL4CO_AM_multiscale_seed{args.seed}_best_policy.pt"

    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        M, N, T = sample_scale()
        gen = DWTAModerateGenerator(num_weapon=M, num_target=N, max_time=T)
        env = DWTAEnv(generator=gen)  # env.M/N/T set from gen in __init__; embeddings read shape from td, not self

        batch = gen(batch_size=[args.batch_size])
        td = env.reset(batch.to(device))

        policy.train()
        out = policy(td, env, decode_type="sampling")
        reward = out["reward"]  # [batch], = -remaining_value.sum(-1)
        log_likelihood = out["log_likelihood"]  # [batch]

        baseline = reward.mean().detach()  # plain mean-baseline REINFORCE -- NOT a critic
        advantage = reward - baseline
        loss = -(advantage.detach() * log_likelihood).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % 20 == 0:
            print(f"[MULTISCALE] step {step}/{args.total_steps} ({M}Mx{N}Nx{T}T) "
                  f"loss={loss.item():.4f} mean_reward={reward.mean().item():.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.eval_every == 0 or step == args.total_steps:
            policy.eval()
            print(f"--- [MULTISCALE] zero-shot eval at step {step} ---", flush=True)
            scores = []
            for eM, eN, eT in ALL_EVAL_CONFIGS:
                eval_gen = DWTAModerateGenerator(num_weapon=eM, num_target=eN, max_time=eT)
                eval_env = DWTAEnv(generator=eval_gen)
                eval_env.M, eval_env.N, eval_env.T = eM, eN, eT
                eval_td, scip_mean, greedy_mean, n = load_moderate_fixed_instances_as_td(
                    device, M=eM, N=eN, T=eT
                )
                am_mean = eval_rl4co_policy(policy, eval_env, eval_td)
                scores.append(am_mean)
                ref = f"Greedy={greedy_mean:.4f}" if greedy_mean is not None else ""
                if scip_mean is not None:
                    ref += f" SCIP={scip_mean:.4f}"
                print(f"    {eM}M_{eN}N_{eT}T: AM={am_mean:.4f}  {ref}", flush=True)

            mean_score = sum(scores) / len(scores)
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                best_step = step
                torch.save(policy.state_dict(), best_path)
            print(f"    mean_across_12_configs={mean_score:.4f}  "
                  f"(best so far: {best_mean_score:.4f} at step {best_step})", flush=True)
            policy.train()

    save_path = f"result/RL4CO_AM_multiscale_seed{args.seed}_final_policy.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Saved final policy to {save_path}")
    print(f"Saved best policy (step {best_step}, mean_across_12_configs={best_mean_score:.4f}) to {best_path}")


if __name__ == "__main__":
    main()
