"""
Train RL4CO's POMO on DWTAEnv, as an external neural-CO baseline for
comparison against SCoPE. New file, does not modify any existing training
code. Shares DWTAEnv/embeddings/progress-eval with train_rl4co_am.py.

POMO's multistart mechanism is adapted for DWTAP's legality constraints --
see DWTAEnv.select_start_nodes's docstring (common/rl4co_dwta_env.py):
forced first actions fall back to no-op for instances where the forced
target hasn't emerged yet at round 0, instead of firing on a target that
doesn't exist yet. State augmentation (POMO's dihedral8, meant for 2D
routing coordinates) is disabled (num_augment=1) since DWTAP has no spatial
"locs" feature for it to act on.

Usage: python train_rl4co_pomo.py --num_weapon 5 --num_target 5 --max_time 5 --curriculum moderate
"""
import argparse

import torch
from rl4co.models.zoo.pomo import POMO
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer

from common.rl4co_dwta_env import DWTAEnv, DWTAGenerator, DWTAModerateGenerator
from common.rl4co_dwta_embeddings import DWTAContext, DWTADynamicEmbedding, DWTAInitEmbedding
from common.rl4co_eval import SCIPProgressCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_weapon', type=int, default=5)
    parser.add_argument('--num_target', type=int, default=5)
    parser.add_argument('--max_time', type=int, default=5)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_data_size', type=int, default=10_000)
    parser.add_argument('--val_data_size', type=int, default=1_000)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=5,
                         help='epochs between progress evals against the fixed SCIP-validated test set')
    parser.add_argument('--curriculum', type=str, default='standard', choices=['standard', 'moderate'],
                         help='standard = main tiered benchmark distribution (little hold-vs-fire '
                              'structure); moderate = temporal-dilemma curriculum used for the '
                              'RL+Auction hybrid (has real hold-vs-fire structure)')
    parser.add_argument('--num_starts', type=int, default=None,
                         help='cap on POMO multistart count; default (None) uses one start per '
                              'target (N), which can be memory-prohibitive for large N (e.g. N=100) '
                              '-- pass an explicit cap for large configs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    M, N, T = args.num_weapon, args.num_target, args.max_time
    if args.curriculum == 'moderate':
        gen = DWTAModerateGenerator(num_weapon=M, num_target=N, max_time=T)
    else:
        gen = DWTAGenerator(num_weapon=M, num_target=N, max_time=T)
    env = DWTAEnv(generator=gen)

    policy = AttentionModelPolicy(
        embed_dim=args.embed_dim,
        num_encoder_layers=6,
        num_heads=8,
        env_name='dwta',
        normalization='instance',
        init_embedding=DWTAInitEmbedding(args.embed_dim),
        dynamic_embedding=DWTADynamicEmbedding(args.embed_dim),
        context_embedding=DWTAContext(args.embed_dim),
        use_graph_context=False,
    )

    model = POMO(
        env,
        policy=policy,
        baseline='shared',
        num_augment=1,  # no spatial "locs" feature for dihedral8 augmentation to act on
        num_starts=args.num_starts,  # None -> auto: env.get_num_starts -> N; capped explicitly for large N
        batch_size=args.batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        optimizer_kwargs={'lr': 1e-4},
    )

    trainer = RL4COTrainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_checkpointing=True,
        default_root_dir=f"result/RL4CO_POMO_{args.curriculum}_{M}M_{N}N_{T}T_seed{args.seed}",
        callbacks=[SCIPProgressCallback(M, N, T, eval_every=args.eval_every, curriculum=args.curriculum, tag='POMO')],
    )
    trainer.fit(model)

    save_path = f"result/RL4CO_POMO_{args.curriculum}_{M}M_{N}N_{T}T_seed{args.seed}/final_policy.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Saved policy to {save_path}")


if __name__ == "__main__":
    main()
