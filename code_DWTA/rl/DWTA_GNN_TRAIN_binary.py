"""
Training script for the RL+Auction hybrid (binary fire/hold actor). New
file -- does not modify or overwrite any existing training script/results.
Run: python rl/DWTA_GNN_TRAIN_binary.py --seed 5
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.utilities import Get_Logger
from common.DWTA_GNN_binary import create_gnn_actor_binary
from common.temporal_dilemma_generator import generate_temporal_dilemma_instance
from eval_greedy_benchmark import eval_instance_greedy

sys.path.append(os.path.dirname(__file__))
from Dynamic_Sampling_GNN_binary import self_play_gnn_binary

_PROGRESS_NW, _PROGRESS_NT, _PROGRESS_MT = 10, 10, 5
_PROGRESS_N = 10


def _load_fixed_progress_instances(seed=123):
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(_PROGRESS_N):
        instances.append(generate_temporal_dilemma_instance(_PROGRESS_NW, _PROGRESS_NT, _PROGRESS_MT, rng=rng))
    return instances


class GNN_REINFORCETrainer_Binary:
    def __init__(self, output_dir=None):
        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_dir = f"GNN_TRAIN_BINARY_{date_str}"

        logger_result = Get_Logger(output_dir)
        if isinstance(logger_result, tuple):
            self.logger, self.output_dir = logger_result
        else:
            self.logger = logger_result
            self.output_dir = os.path.join("TRAIN", output_dir)

        self.actor = create_gnn_actor_binary().to(DEVICE)
        self.logger.info("Binary fire/hold actor initialized (RL+Auction hybrid, new architecture)")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Training: {TOTAL_EPOCH} epochs, {TOTAL_EPISODE} episodes/epoch, temporal-dilemma curriculum")

    def _save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(self.output_dir, f"CheckPoint_epoch{epoch:05d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, "GNN_ACTOR_BINARY_state_dic.pt"))

    def train(self):
        self.logger.info("Starting binary fire/hold REINFORCE training...")
        start_time = time.time()

        progress_instances = _load_fixed_progress_instances()
        greedy_objs = [
            eval_instance_greedy(V, P, TW, _PROGRESS_NW, _PROGRESS_NT, _PROGRESS_MT, amm, prep, cost)["objective"]
            for (V, P, TW, amm, prep, cost) in progress_instances
        ]
        greedy_mean = float(np.mean(greedy_objs))
        self.logger.info(f"[PROGRESS] Greedy baseline on temporal-dilemma instances "
                          f"({_PROGRESS_NW}x{_PROGRESS_NT}x{_PROGRESS_MT}, {len(progress_instances)} instances): "
                          f"objective={greedy_mean:.4f} (fixed reference)")

        for epoch in range(1, TOTAL_EPOCH + 1):
            epoch_start = time.time()
            actor_loss, epoch_cfg = self_play_gnn_binary(
                actor=self.actor, episode=TOTAL_EPISODE, epoch=epoch, logger=self.logger,
            )
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            self.logger.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch:03d} | "
                f"Objective: {epoch_cfg['objective']:.6f} | Destruction: {epoch_cfg['destruction_ratio']:.2%} | "
                f"Actor Loss: {actor_loss:.6f} | Epoch time: {epoch_time:.1f}s"
            )

            if epoch % 5 == 0:
                self._save_checkpoint(epoch)
                self.actor.eval()
                with torch.no_grad():
                    import json
                    import pandas as pd
                    from common.Dynamic_Instance_generation import input_generation
                    from common.DWTA_Simulator import Environment
                    from common.auction_refinement import auction_round_action
                    from eval_tiered_benchmark import patch_globals

                    rl_objs = []
                    for (V, P, TW, amm, prep, cost) in progress_instances:
                        patch_globals(_PROGRESS_NW, _PROGRESS_NT, _PROGRESS_MT, amm, prep, cost)
                        ae, wtp = input_generation(
                            NUM_WEAPON=_PROGRESS_NW, NUM_TARGET=_PROGRESS_NT, value=V, prob=P, TW=TW,
                            max_time=_PROGRESS_MT, batch_size=1, alpha=1.0, amm=amm,
                        )
                        ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
                        env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=_PROGRESS_MT)
                        init_value = env.current_target_value[:, :, 0:_PROGRESS_NT].sum().item()
                        for _ in range(_PROGRESS_MT):
                            remaining_value = env.current_target_value[:, :, 0:_PROGRESS_NT]
                            prob = env.weapon_to_target_prob[:, :, :_PROGRESS_NW, :_PROGRESS_NT]
                            legal_mask = env.mask_per_weapon[:, :, :_PROGRESS_NW, :_PROGRESS_NT] > 0
                            fire_prob, _ = self.actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
                            must_fire = fire_prob > 0.5
                            action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
                            env.update_internal_variables_parallel(selected_actions=action)
                            env.time_update()
                        remaining = env.current_target_value[:, :, 0:_PROGRESS_NT].sum().item()
                        rl_objs.append(remaining / max(init_value, 1e-8))
                self.actor.train()
                rl_mean = float(np.mean(rl_objs))
                gap = rl_mean - greedy_mean
                self.logger.info(
                    f"[PROGRESS] epoch {epoch}: RL+Auction={rl_mean:.4f} vs Greedy={greedy_mean:.4f} "
                    f"(gap={gap:+.4f}, {'RL+Auction AHEAD' if gap < 0 else 'Greedy ahead'})"
                )

        self.logger.info(f"Training completed in {(time.time()-start_time)/3600:.2f} hours")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        import random as _random
        _random.seed(args.seed)

    output_dir = f"GNN_TRAIN_BINARY_seed{args.seed}" if args.seed is not None else None
    trainer = GNN_REINFORCETrainer_Binary(output_dir=output_dir)
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
