"""
GNN-based REINFORCE Training for Dynamic Weapon-Target Assignment (DWTA)
Author: AI Assistant  
Date: 2025-01-21

Clean training script with simple folder imports.
"""

import os
import time
import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import ast
import json
import numpy as np

# Simple folder imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.utilities import Get_Logger, Average_Meter
from common.DWTA_GNN import create_gnn_actor, create_gnn_critic
from common.Dynamic_Instance_generation import input_generation
from rl_rollout.DWTA_Simulator_rollout import Environment as RolloutEnv
from torch import no_grad

# Import from same directory (rl/)
sys.path.append(os.path.dirname(__file__))
from Dynamic_Sampling_GNN import self_play_gnn


class GNN_REINFORCETrainer:
    """GNN-based REINFORCE Trainer for DWTA problem."""
    
    def __init__(self, output_dir=None):
        """Initialize GNN REINFORCE trainer."""
        # Create output directory
        if output_dir is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_dir = f"GNN_TRAIN_{date_str}"
        
        # Initialize logger
        logger_result = Get_Logger(output_dir)
        if isinstance(logger_result, tuple):
            self.logger, self.output_dir = logger_result
        else:
            self.logger = logger_result
            self.output_dir = os.path.join("TRAIN", output_dir)
        
        # Create models
        self.actor = create_gnn_actor().to(DEVICE)
        self.critic = create_gnn_critic().to(DEVICE)
        
        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LEARNING_RATE)
        
        # Set optimizers for models (required by self_play_gnn)
        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer
        
        # Training metrics
        self.epoch_objectives = []
        
        # Evaluation metrics storage
        self.eval_objectives = []
        self.eval_initial_objectives = []
        self.eval_destruction_ratios = []
        self.eval_no_action_ratios = []
        
        self.logger.info(f"GNN REINFORCE Trainer initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Multi-scale training: Random 5-7W × 5-7T × 5-7T per epoch")
        self.logger.info(f"Model: EMBEDDING_DIM={EMBEDDING_DIM}, HEAD_NUM={HEAD_NUM}")
        self.logger.info(f"Training: {TOTAL_EPOCH} epochs, {TOTAL_EPISODE} episodes/epoch")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        actor_loss, critic_loss, epoch_cfg = self_play_gnn(
            old_actor=None,
            actor=self.actor,
            critic=self.critic,
            episode=TOTAL_EPISODE,
            temp=None,
            epoch=epoch,
            logger=self.logger
        )
        # Log epoch configuration
        self.logger.info(
            f"Epoch {epoch}: {epoch_cfg['num_weapons']}W×{epoch_cfg['num_targets']}T×{epoch_cfg['max_time']}T, "
            f"AMM={epoch_cfg['amm']}"
        )
        # Store measured objective/metrics for this epoch
        self.last_epoch_cfg = epoch_cfg
        self.last_epoch_objective = epoch_cfg.get('objective', None)
        self.last_epoch_destruction = epoch_cfg.get('destruction_ratio', None)

        return actor_loss, critic_loss

    def _log_epoch_results(self, epoch, actor_loss, critic_loss, epoch_objective, total_time, epoch_time=None):
        """Log epoch results."""
        total_time_str = f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}"
        epoch_time_str = f"{epoch_time:.1f}s" if epoch_time is not None else "N/A"
        
        # Store objective for plotting
        self.epoch_objectives.append(epoch_objective)
        
        # Calculate gradient norms
        actor_grad_norm = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
        critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
        
        # Get current learning rates
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        
        # Plot training progress (separate file)
        import matplotlib.pyplot as plt
        
        # 1. Training plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_objectives) + 1), self.epoch_objectives, 'b-', label='Training Objective', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training Objective')
        plt.title(f'Training Progress (Epoch {epoch})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Ensure directory exists before saving
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, "training_progress.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Evaluation plot (separate file)
        if self.eval_objectives:
            plt.figure(figsize=(10, 6))
            eval_epochs = [i * UPDATE_PERIOD for i in range(1, len(self.eval_objectives) + 1)]
            
            # Main evaluation metrics
            plt.plot(eval_epochs, self.eval_objectives, 'r-', label='Eval Objective', linewidth=2, marker='o')
            plt.plot(eval_epochs, self.eval_initial_objectives, 'g--', label='Eval Initial', linewidth=1.5, alpha=0.7)
            
            # Add destruction ratio on secondary y-axis
            ax_twin = plt.gca().twinx()
            ax_twin.plot(eval_epochs, [ratio * 100 for ratio in self.eval_destruction_ratios], 'orange', 
                        label='Destruction %', linewidth=1.5, linestyle=':', marker='s', markersize=4)
            ax_twin.set_ylabel('Destruction Ratio (%)', color='orange')
            ax_twin.tick_params(axis='y', labelcolor='orange')
            
            plt.xlabel('Epoch')
            plt.ylabel('Evaluation Objective')
            plt.title(f'Evaluation Performance (Epoch {epoch})')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            
            plt.savefig(os.path.join(self.output_dir, "evaluation_progress.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Expanded logging
        log_msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Epoch: {epoch:03d} | Objective: {epoch_objective:.6f} | "
                  f"Actor Loss: {actor_loss:.6f} | Critic Loss: {critic_loss:.6f} | "
                  f"Actor LR: {actor_lr:.2e} | Critic LR: {critic_lr:.2e} | "
                  f"Actor Grad: {actor_grad_norm:.4f} | Critic Grad: {critic_grad_norm:.4f} | "
                  f"Time: {total_time_str} | Epoch time: {epoch_time_str}")
        
        self.logger.info(log_msg)

    def _evaluate_policy(self) -> float:
        """Evaluate with fixed alpha=1.0, seed=42, 50 instances."""
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

                env = RolloutEnv(assignment_encoding=assignment_encoding,
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
        """Save model checkpoints."""
        checkpoint_dir = os.path.join(self.output_dir, f"CheckPoint_epoch{epoch:05d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, "GNN_ACTOR_state_dic.pt"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, "GNN_CRITIC_state_dic.pt"))

    def _save_final_results(self):
        """Save final training results."""
        # Save epoch objectives
        with open(os.path.join(self.output_dir, "epoch_objectives.txt"), "w") as f:
            for i, obj in enumerate(self.epoch_objectives):
                f.write(f"Epoch {i+1}: {obj:.6f}\n")
        
        # Save final plots as separate files
        import matplotlib.pyplot as plt
        
        # 1. Final Training plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.epoch_objectives) + 1), self.epoch_objectives, 'b-', label='Training Objective', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training Objective')
        plt.title('Final Training Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "training_progress.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Final Evaluation plot
        if self.eval_objectives:
            plt.figure(figsize=(10, 6))
            eval_epochs = [i * UPDATE_PERIOD for i in range(1, len(self.eval_objectives) + 1)]
            
            # Main evaluation metrics
            plt.plot(eval_epochs, self.eval_objectives, 'r-', label='Eval Objective', linewidth=2, marker='o')
            plt.plot(eval_epochs, self.eval_initial_objectives, 'g--', label='Eval Initial', linewidth=1.5, alpha=0.7)
            
            # Add destruction ratio on secondary y-axis
            ax_twin = plt.gca().twinx()
            ax_twin.plot(eval_epochs, [ratio * 100 for ratio in self.eval_destruction_ratios], 'orange', 
                        label='Destruction %', linewidth=1.5, linestyle=':', marker='s', markersize=4)
            ax_twin.set_ylabel('Destruction Ratio (%)', color='orange')
            ax_twin.tick_params(axis='y', labelcolor='orange')
            
            plt.xlabel('Epoch')
            plt.ylabel('Evaluation Objective')
            plt.title('Final Evaluation Performance')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            
            plt.savefig(os.path.join(self.output_dir, "evaluation_progress.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save final models
        self._save_checkpoint(TOTAL_EPOCH)
        
        self.logger.info(f"Final results and training plot saved to {self.output_dir}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting GNN-based DWTA REINFORCE Training...")
        start_time = time.time()
        
        for epoch in range(1, TOTAL_EPOCH + 1):
            epoch_start_time = time.time()
            
            # Train one epoch
            actor_loss, critic_loss = self.train_epoch(epoch)
            
            # Use measured objective (mean leftover target value across all train instances)
            epoch_objective = (
                self.last_epoch_objective if getattr(self, 'last_epoch_objective', None) is not None
                else 25.0 - (epoch / TOTAL_EPOCH) * 10.0  # fallback
            )
            
            # Calculate timing
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            # Log results
            self._log_epoch_results(epoch, actor_loss, critic_loss, epoch_objective, total_time, epoch_time)

            # 주기적 평가 수행 및 로그
            if EVALUATION_PERIOD and (epoch % EVALUATION_PERIOD == 0):
                
                eval_avg = self._evaluate_policy()
                
                # Store evaluation metrics for plotting
                self.eval_objectives.append(eval_avg)
                self.eval_initial_objectives.append(getattr(self, 'last_eval_init', 0.0))
                self.eval_destruction_ratios.append(getattr(self, 'last_eval_destruction', 0.0))
                self.eval_no_action_ratios.append(getattr(self, 'last_eval_no_action_ratio', 0.0))
                
                self.logger.info(
                    f"[EVAL] alpha=0.5 | RemVal={eval_avg:.2f} | "
                    f"Init={getattr(self, 'last_eval_init', 0.0):.2f} | "
                    f"Destr={getattr(self, 'last_eval_destruction', 0.0):.2%} | "
                    f"FireRate={getattr(self, 'last_eval_fire_ratio', 0.0):.2%}"
                )

            
            # Save checkpoint strategy:
            # 1~99: No saves
            # 100~150: Every 10 epochs
            # 151~200: Every epoch
            should_save = False
            if epoch < 100:
                should_save = False
            elif epoch <= 150:
                should_save = (epoch % 10 == 0)
            else:
                should_save = True
                
            if should_save:
                self._save_checkpoint(epoch)
                self.logger.info(f"Checkpoint saved at epoch {epoch}")
        
        # Save final results
        self._save_final_results()
        
        total_training_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_training_time/3600:.2f} hours")


def main():
    """Main training function."""
    print("Starting GNN-based DWTA REINFORCE Training...")
    
    # Create and run trainer
    trainer = GNN_REINFORCETrainer()
    trainer.train()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()