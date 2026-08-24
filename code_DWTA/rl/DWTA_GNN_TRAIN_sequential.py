"""
GNN-based REINFORCE Training for the SEQUENTIAL (originally-published
BReRLA) decoder -- comparison point for DWTA_GNN_TRAIN.py's parallel
multi-pointer decoder. Structurally a thin copy of DWTA_GNN_TRAIN.py; see
common/DWTA_GNN_sequential.py and rl/Dynamic_Sampling_GNN_sequential.py for
what actually differs (decoding architecture / training loop).
"""

import os
import time
import torch
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.utilities import Get_Logger, Average_Meter
from common.DWTA_GNN import create_gnn_critic
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from torch import no_grad

sys.path.append(os.path.dirname(__file__))
from Dynamic_Sampling_GNN_sequential import self_play_gnn, _flat_mask_to_critic_mask

# 2026-08-06: periodic in-loop comparison against Greedy on a fixed instance set,
# added so progress can be tracked without a separate eval process competing for
# the GPU (see UPDATE.md "GPU contention" note).
import json
import numpy as np
import pandas as pd
from eval_tiered_benchmark import eval_instance_sequential
from eval_greedy_benchmark import eval_instance_greedy

_PROGRESS_CONFIG = ("TEST_INSTANCE/20M_30N_5T.xlsx", 20, 30, 5)
_PROGRESS_N = 10


def _load_progress_instances():
    fname, nw, nt, mt = _PROGRESS_CONFIG
    df = pd.read_excel(fname)
    instances = []
    for i in range(min(_PROGRESS_N, len(df))):
        row = df.iloc[i]
        instances.append((
            json.loads(row["V"]), np.array(json.loads(row["P"])), json.loads(row["TW"]),
            json.loads(row["AMM"]), json.loads(row["PREP"]), json.loads(row["COST"]),
        ))
    return instances, nw, nt, mt


class GNN_REINFORCETrainer_Sequential:
    """GNN-based REINFORCE Trainer for the sequential (original) DWTA decoder."""

    def __init__(self, output_dir=None, ablation=None):
        self.ablation = ablation
        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_dir = f"GNN_TRAIN_SEQ_{date_str}"

        logger_result = Get_Logger(output_dir)
        if isinstance(logger_result, tuple):
            self.logger, self.output_dir = logger_result
        else:
            self.logger = logger_result
            self.output_dir = os.path.join("TRAIN", output_dir)

        self.actor = create_gnn_actor_sequential().to(DEVICE)
        self.critic = create_gnn_critic().to(DEVICE)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARNING_RATE)

        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer

        self.epoch_objectives = []

        self.eval_objectives = []
        self.eval_initial_objectives = []
        self.eval_destruction_ratios = []
        self.eval_no_action_ratios = []

        self.logger.info(f"GNN REINFORCE Trainer (SEQUENTIAL decoder) initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Multi-scale training: Random 5-7W x 5-7T x 5-7T per epoch")
        self.logger.info(f"Model: EMBEDDING_DIM={EMBEDDING_DIM}, HEAD_NUM={HEAD_NUM}")
        self.logger.info(f"Training: {TOTAL_EPOCH} epochs, {TOTAL_EPISODE} episodes/epoch")

    def train_epoch(self, epoch):
        actor_loss, critic_loss, epoch_cfg = self_play_gnn(
            old_actor=None,
            actor=self.actor,
            critic=self.critic,
            episode=TOTAL_EPISODE,
            temp=None,
            epoch=epoch,
            logger=self.logger,
            ablation=self.ablation
        )
        self.logger.info(
            f"Epoch {epoch}: {epoch_cfg['num_weapons']}Wx{epoch_cfg['num_targets']}Tx{epoch_cfg['max_time']}T, "
            f"AMM={epoch_cfg['amm']}"
        )
        self.last_epoch_cfg = epoch_cfg
        self.last_epoch_objective = epoch_cfg.get('objective', None)
        self.last_epoch_destruction = epoch_cfg.get('destruction_ratio', None)

        return actor_loss, critic_loss

    def _log_epoch_results(self, epoch, actor_loss, critic_loss, epoch_objective, total_time, epoch_time=None):
        total_time_str = f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}"
        epoch_time_str = f"{epoch_time:.1f}s" if epoch_time is not None else "N/A"

        self.epoch_objectives.append(epoch_objective)

        actor_grad_norm = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
        critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)

        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']

        log_msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Epoch: {epoch:03d} | Objective: {epoch_objective:.6f} | "
                  f"Actor Loss: {actor_loss:.6f} | Critic Loss: {critic_loss:.6f} | "
                  f"Actor LR: {actor_lr:.2e} | Critic LR: {critic_lr:.2e} | "
                  f"Actor Grad: {actor_grad_norm:.4f} | Critic Grad: {critic_grad_norm:.4f} | "
                  f"Time: {total_time_str} | Epoch time: {epoch_time_str}")

        self.logger.info(log_msg)

    def _evaluate_policy(self) -> float:
        """Evaluate with fixed alpha=1.0, seed=42, 50 instances. Uses the
        current (non-stale) common.DWTA_Simulator.Environment -- NOT
        rl_rollout.DWTA_Simulator_rollout, which the revision plan flags as
        a less-refactored duplicate with a possibly-inconsistent damage
        formula."""
        EVAL_ALPHA = 1.0
        EVAL_N = 50
        EVAL_NW, EVAL_NT, EVAL_MT = 5, 5, 5
        EVAL_SEED = 42

        self.actor.eval()
        with no_grad():
            torch.manual_seed(EVAL_SEED)
            total_obj = 0.0
            total_init = 0.0
            total_fires = 0
            total_decisions = 0

            for _ in range(EVAL_N):
                assignment_encoding, weapon_to_target_prob = input_generation(
                    NUM_WEAPON=EVAL_NW, NUM_TARGET=EVAL_NT,
                    value=None, prob=None, TW=None,
                    max_time=EVAL_MT, batch_size=1, alpha=EVAL_ALPHA,
                )
                assignment_encoding = assignment_encoding.unsqueeze(1)
                weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1)

                env = Environment(assignment_encoding=assignment_encoding,
                                   weapon_to_target_prob=weapon_to_target_prob, max_time=EVAL_MT)

                init_obj = env.current_target_value[:, :, 0:EVAL_NT].sum().item()
                total_init += init_obj

                for _ in range(EVAL_MT):
                    for _ in range(EVAL_NW):
                        mask = env.mask.clone()
                        if (mask > 0).any():
                            policy, _ = self.actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                            flat = policy.view(-1, EVAL_NW * EVAL_NT + 1)
                            selected_action = flat.argmax(dim=1).view(1, 1)
                            if selected_action.item() < EVAL_NW * EVAL_NT:
                                total_fires += 1
                        else:
                            selected_action = torch.tensor([[EVAL_NW * EVAL_NT]], device=DEVICE)
                        total_decisions += 1
                        env.update_internal_variables(selected_action=selected_action)
                    env.time_update()

                total_obj += env.current_target_value[:, :, 0:EVAL_NT].sum().item()

            avg_init = total_init / EVAL_N
            avg_final = total_obj / EVAL_N
            destruction_ratio = 1.0 - avg_final / max(avg_init, 1e-8)
            fire_ratio = total_fires / max(total_decisions, 1)

            self.last_eval_init = avg_init
            self.last_eval_final = avg_final
            self.last_eval_destruction = destruction_ratio
            self.last_eval_fire_ratio = fire_ratio
            self.last_eval_no_action_ratio = 1.0 - fire_ratio

            return avg_final

    def _save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(self.output_dir, f"CheckPoint_epoch{epoch:05d}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, "GNN_ACTOR_state_dic.pt"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, "GNN_CRITIC_state_dic.pt"))

    def _save_final_results(self):
        with open(os.path.join(self.output_dir, "epoch_objectives.txt"), "w") as f:
            for i, obj in enumerate(self.epoch_objectives):
                f.write(f"Epoch {i+1}: {obj:.6f}\n")

        self._save_checkpoint(TOTAL_EPOCH)

        self.logger.info(f"Final results saved to {self.output_dir}")

    def train(self):
        self.logger.info("Starting GNN-based DWTA REINFORCE Training (SEQUENTIAL decoder)...")
        start_time = time.time()

        # Precompute Greedy's result once (deterministic, does not depend on training).
        progress_instances, p_nw, p_nt, p_mt = _load_progress_instances()
        greedy_objs = [
            eval_instance_greedy(V, P, TW, p_nw, p_nt, p_mt, amm, prep, cost)["objective"]
            for (V, P, TW, amm, prep, cost) in progress_instances
        ]
        greedy_mean = float(np.mean(greedy_objs))
        self.logger.info(
            f"[PROGRESS] Greedy baseline on {_PROGRESS_CONFIG[0]} ({p_nw}x{p_nt}x{p_mt}, "
            f"{len(progress_instances)} instances): objective={greedy_mean:.4f} (fixed reference)"
        )

        for epoch in range(1, TOTAL_EPOCH + 1):
            epoch_start_time = time.time()

            actor_loss, critic_loss = self.train_epoch(epoch)

            epoch_objective = (
                self.last_epoch_objective if getattr(self, 'last_epoch_objective', None) is not None
                else 25.0 - (epoch / TOTAL_EPOCH) * 10.0
            )

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time

            self._log_epoch_results(epoch, actor_loss, critic_loss, epoch_objective, total_time, epoch_time)

            if EVALUATION_PERIOD and (epoch % EVALUATION_PERIOD == 0):
                eval_avg = self._evaluate_policy()

                self.eval_objectives.append(eval_avg)
                self.eval_initial_objectives.append(getattr(self, 'last_eval_init', 0.0))
                self.eval_destruction_ratios.append(getattr(self, 'last_eval_destruction', 0.0))
                self.eval_no_action_ratios.append(getattr(self, 'last_eval_no_action_ratio', 0.0))

                self.logger.info(
                    f"[EVAL] alpha=1.0 | RemVal={eval_avg:.2f} | "
                    f"Init={getattr(self, 'last_eval_init', 0.0):.2f} | "
                    f"Destr={getattr(self, 'last_eval_destruction', 0.0):.2%} | "
                    f"FireRate={getattr(self, 'last_eval_fire_ratio', 0.0):.2%}"
                )

            # 2026-08-06: save every 5 epochs from the start (was: nothing before epoch
            # 100) so an early, if less-trained, checkpoint is available quickly on demand.
            should_save = (epoch % 5 == 0)

            if should_save:
                self._save_checkpoint(epoch)
                self.logger.info(f"Checkpoint saved at epoch {epoch}")

                # In-loop progress check against Greedy on the same fixed instances --
                # avoids running a separate eval process that would compete for the GPU.
                self.actor.eval()
                with torch.no_grad():
                    scope_objs = [
                        eval_instance_sequential(self.actor, V, P, TW, p_nw, p_nt, p_mt, amm, prep, cost)["objective"]
                        for (V, P, TW, amm, prep, cost) in progress_instances
                    ]
                self.actor.train()
                scope_mean = float(np.mean(scope_objs))
                gap = scope_mean - greedy_mean
                self.logger.info(
                    f"[PROGRESS] epoch {epoch}: Sequential={scope_mean:.4f} vs "
                    f"Greedy={greedy_mean:.4f} (gap={gap:+.4f}, {'Sequential AHEAD' if gap < 0 else 'Greedy ahead'})"
                )

        self._save_final_results()

        total_training_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_training_time/3600:.2f} hours")


def main():
    import argparse
    import random as _random

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                         help='Random seed for reproducible multi-seed comparison runs.')
    parser.add_argument('--ablation', type=str, default=None,
                         choices=['no_reward_to_go'],
                         help='Ablation variant, mirrors the parallel trainer (only '
                              'no_reward_to_go implemented for sequential so far).')
    args = parser.parse_args()

    print("Starting GNN-based DWTA REINFORCE Training (SEQUENTIAL decoder)...")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        _random.seed(args.seed)
        print(f"Seed set to {args.seed}")

    if args.ablation is not None:
        suffix = f"seed{args.seed}" if args.seed is not None else "seed0"
        output_dir = f"GNN_TRAIN_SEQ_ABL_{args.ablation}_{suffix}"
    elif args.seed is not None:
        output_dir = f"GNN_TRAIN_SEQ_seed{args.seed}"
    else:
        output_dir = None
    trainer = GNN_REINFORCETrainer_Sequential(output_dir=output_dir, ablation=args.ablation)
    trainer.train()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
