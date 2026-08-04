"""
Expert Iteration (ExIt) Training for DWTA
Bertsekas-style iterative policy improvement via multi-policy rollout.

Loop:
  1. Collect expert trajectories using multi-policy beam search
  2. Train actor with hybrid loss (imitation + REINFORCE)
  3. Train critic on rollout-improved value estimates
  4. Repeat — each iteration the base policy improves,
     so the search also improves (policy improvement theorem).
"""
import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.utilities import Get_Logger, Average_Meter
from common.DWTA_GNN import create_gnn_actor, create_gnn_critic
from common.Dynamic_Instance_generation import input_generation
from rl_rollout.DWTA_Simulator_rollout import Environment
from rl.Dynamic_Sampling_GNN import (
    self_play_gnn,
    get_random_problem_size,
    patch_hyperparameters_for_epoch,
    restore_hyperparameters,
)
from rl.expert_trajectory import ExpertBuffer, collect_expert_trajectories


class ExItTrainer:
    """Expert Iteration trainer combining REINFORCE pre-training with search-based imitation."""

    def __init__(self, output_dir=None):
        date_str = datetime.now().strftime("%Y%m%d")
        if output_dir is None:
            output_dir = f"EXIT_TRAIN_{date_str}"

        self.logger, self.output_dir = Get_Logger(output_dir)

        self.actor = create_gnn_actor().to(DEVICE)
        self.critic = create_gnn_critic().to(DEVICE)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY,
        )
        self.actor.optimizer = self.actor_optimizer
        self.critic.optimizer = self.critic_optimizer

        self.expert_buffer = ExpertBuffer(capacity=EXIT_EXPERT_BUFFER_SIZE)

        # Base RL-style evaluation: fixed seed, same structure
        self.EVAL_SEED = 42
        self.EVAL_N = 50
        self.EVAL_NW, self.EVAL_NT, self.EVAL_MT = 5, 5, 5
        self.EVAL_ALPHA = 1.0
        self.epoch_objectives = []
        self.checkpoint_eval_results = []

        self.logger.info(f"ExIt Trainer initialised  |  device={DEVICE}")
        self.logger.info(f"  N_EXIT_ITERS={N_EXIT_ITERS}  REINFORCE_EPOCHS={TOTAL_EPOCH}")
        self.logger.info(f"  Eval: seed={self.EVAL_SEED}, n={self.EVAL_N}, size={self.EVAL_NW}x{self.EVAL_NT}x{self.EVAL_MT} (same as base RL)")

    # ------------------------------------------------------------------
    # Phase A: REINFORCE warm-up (reuses existing self_play_gnn)
    # ------------------------------------------------------------------
    def _reinforce_epoch(self, epoch):
        actor_loss, critic_loss, epoch_cfg = self_play_gnn(
            old_actor=None,
            actor=self.actor,
            critic=self.critic,
            episode=TOTAL_EPISODE,
            temp=None,
            epoch=epoch,
            logger=self.logger,
        )
        return actor_loss, critic_loss, epoch_cfg

    # ------------------------------------------------------------------
    # Phase B: Expert trajectory collection + imitation update
    # ------------------------------------------------------------------
    def _collect_and_store_expert(self, n_instances=10, beta=2, exit_iter=1):
        self.expert_buffer.clear()
        self.logger.info(f"  Collecting expert trajectories ({n_instances} instances, multi-scale, beta={beta}) ...")
        trajs = collect_expert_trajectories(
            actor=self.actor,
            critic=self.critic,
            num_instances=n_instances,
            beta=beta,
            to_go_weight=1.0,
            exit_iter=exit_iter,
        )
        for (s, wtp, m, a, v, nw, nt) in trajs:
            self.expert_buffer.push(s, wtp, m, a, v, nw, nt)
        self.logger.info(f"  Buffer size: {len(self.expert_buffer)}")

    def _patch_nw_nt(self, nw, nt):
        """Set NUM_WEAPONS, NUM_TARGETS for actor/critic forward (multi-scale buffer)."""
        import common.Dynamic_HYPER_PARAMETER as HP
        globals()['NUM_WEAPONS'] = nw
        globals()['NUM_TARGETS'] = nt
        HP.NUM_WEAPONS = nw
        HP.NUM_TARGETS = nt
        for name in ('common.DWTA_Simulator', 'rl_rollout.DWTA_Simulator_rollout'):
            try:
                Sim = __import__(name, fromlist=['Environment'])
                Sim.NUM_WEAPONS = nw
                Sim.NUM_TARGETS = nt
            except ImportError:
                pass

    def _imitation_update(self, n_steps=20, batch_size=64, beta_weight=0.8):
        """Train actor to imitate expert actions from the buffer (multi-scale, beta_weight scales loss)."""
        if len(self.expert_buffer) < batch_size:
            return 0.0

        self.actor.train()
        total_loss = 0.0

        for _ in range(n_steps):
            states, wtps, masks, actions, values, nw, nt = self.expert_buffer.sample(batch_size)
            self._patch_nw_nt(nw, nt)

            if states.dim() == 2:
                states = states.unsqueeze(0)
            if states.dim() == 3:
                states = states.unsqueeze(1)
            if wtps.dim() == 2:
                wtps = wtps.unsqueeze(0).unsqueeze(1)
            elif wtps.dim() == 3:
                wtps = wtps.unsqueeze(1)
            if masks.dim() == 1:
                masks = masks.unsqueeze(0)
            if masks.dim() == 2:
                masks = masks.unsqueeze(1)

            policy, _ = self.actor(states, prob=wtps, mask=masks)
            policy_flat = policy.view(-1, policy.size(-1))
            actions_flat = actions.view(-1).long()

            ce_loss = F.cross_entropy(
                torch.log(policy_flat.clamp_min(1e-8)),
                actions_flat,
            )
            loss = beta_weight * ce_loss

            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            total_loss += loss.item()

        return total_loss / max(n_steps, 1)

    def _critic_update_from_buffer(self, n_steps=10, batch_size=64):
        """Update critic to better predict final values from expert rollouts (multi-scale)."""
        if len(self.expert_buffer) < batch_size:
            return 0.0

        self.critic.train()
        total_loss = 0.0

        for _ in range(n_steps):
            states, _, masks, _, values, nw, nt = self.expert_buffer.sample(batch_size)
            self._patch_nw_nt(nw, nt)

            if states.dim() == 2:
                states = states.unsqueeze(0)
            if states.dim() == 3:
                states = states.unsqueeze(1)
            if masks.dim() == 1:
                masks = masks.unsqueeze(0)
            if masks.dim() == 2:
                masks = masks.unsqueeze(1)

            pred_value = self.critic(states, masks).squeeze(-1).view(-1)
            target = values.view(-1)

            loss = F.mse_loss(pred_value, target)

            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
            total_loss += loss.item()

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Evaluation: fixed seed (same as base RL) for reproducibility
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_fixed_seed(self):
        """Evaluate on fixed seed=42, 50 instances, 5x5x5, same as base RL."""
        self.actor.eval()
        nw, nt, mt = self.EVAL_NW, self.EVAL_NT, self.EVAL_MT
        amm = [2] * nw
        prep = [1] * nw
        cost = [50] * nw
        original = patch_hyperparameters_for_epoch(nw, nt, mt, amm, prep_list=prep, cost_list=cost)
        try:
            torch.manual_seed(self.EVAL_SEED)
            total_remaining_norm = 0.0
            for _ in range(self.EVAL_N):
                ae, wtp = input_generation(
                    NUM_WEAPON=nw, NUM_TARGET=nt,
                    value=None, prob=None, TW=None,
                    max_time=mt, batch_size=1, alpha=self.EVAL_ALPHA,
                )
                ae = ae.unsqueeze(1)
                wtp = wtp.unsqueeze(1)
                env = Environment(ae, wtp, max_time=mt)
                init_val = env.current_target_value[:, :, 0:nt].sum().item()
                for t in range(mt):
                    for w in range(nw):
                        mask = env.mask.clone()
                        if (mask > 0).any():
                            policy, _ = self.actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                            flat = policy.view(-1, nw * nt + 1)
                            action = flat.argmax(dim=1).view(1, 1)
                        else:
                            action = torch.tensor([[nw * nt]], device=DEVICE)
                        env.update_internal_variables(selected_action=action)
                    env.time_update()
                remaining = env.current_target_value[:, :, 0:nt].sum().item()
                remaining_norm = remaining / max(init_val, 1e-8)
                total_remaining_norm += remaining_norm
            avg_remaining_norm = total_remaining_norm / self.EVAL_N
            destruction = 1.0 - avg_remaining_norm
            objective = self.EVAL_ALPHA * avg_remaining_norm
            return objective, destruction, avg_remaining_norm
        finally:
            restore_hyperparameters(original)

    # ------------------------------------------------------------------
    # Evaluation (multi-scale: same get_random_problem_size as base RL)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, n_instances=5, exit_iter=1, alpha=1.0):
        self.actor.eval()
        total_obj = 0.0

        for i in range(n_instances):
            nw, nt, mt, ep_amm_list, ep_prep_list, ep_cost_list = get_random_problem_size(exit_iter, i)
            original_hyperparams = patch_hyperparameters_for_epoch(
                nw, nt, mt, ep_amm_list,
                prep_list=ep_prep_list, cost_list=ep_cost_list,
            )
            try:
                ae, wtp = input_generation(
                    NUM_WEAPON=nw, NUM_TARGET=nt,
                    value=None, prob=None, TW=None,
                    max_time=mt, batch_size=1, alpha=alpha,
                )
                ae = ae.unsqueeze(1)
                wtp = wtp.unsqueeze(1)
                env = Environment(ae, wtp, max_time=mt)

                for t in range(mt):
                    for w in range(nw):
                        mask = env.mask.clone()
                        if (mask > 0).any():
                            policy, _ = self.actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                            flat = policy.view(-1, nw * nt + 1)
                            action = flat.argmax(dim=1).view(1, 1)
                        else:
                            action = torch.tensor([[nw * nt]], device=DEVICE)
                        env.update_internal_variables(selected_action=action)
                    env.time_update()

                total_obj += env.current_target_value[:, :, 0:nt].sum().item()
            finally:
                restore_hyperparameters(original_hyperparams)

        avg_obj = total_obj / max(n_instances, 1)
        return avg_obj

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _should_save_base_rl_style(self, global_ep):
        """Same schedule as base RL: 1-99 no save, 100-150 every 10, 151-200 every epoch."""
        if global_ep < 100:
            return False
        if global_ep <= 150:
            return (global_ep % 10 == 0)
        return True

    def _save_checkpoint(self, exit_iter, epoch):
        ckpt_dir = os.path.join(self.output_dir, f"ExIt_{exit_iter:02d}_epoch{epoch:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(ckpt_dir, "GNN_ACTOR_state_dic.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_dir, "GNN_CRITIC_state_dic.pt"))

    def _save_checkpoint_base_rl_style(self, global_ep):
        """Save to CheckPoint_epoch{ep:05d}/ (same folder name as base RL)."""
        ckpt_dir = os.path.join(self.output_dir, f"CheckPoint_epoch{global_ep:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(ckpt_dir, "GNN_ACTOR_state_dic.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_dir, "GNN_CRITIC_state_dic.pt"))

    def _save_checkpoint_to_dir(self, ckpt_dir, epoch):
        """Save current model to a specific directory (e.g. ExIt_best)."""
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(ckpt_dir, "GNN_ACTOR_state_dic.pt"))
        torch.save(self.critic.state_dict(), os.path.join(ckpt_dir, "GNN_CRITIC_state_dic.pt"))
        with open(os.path.join(ckpt_dir, "epoch.txt"), "w") as f:
            f.write(f"{epoch}\n")

    def _save_final_results(self, best_obj, best_exit_iter, best_epoch):
        """Save epoch_objectives.txt and checkpoint_evaluation.txt (same structure as base RL)."""
        with open(os.path.join(self.output_dir, "epoch_objectives.txt"), "w") as f:
            for i, obj in enumerate(self.epoch_objectives):
                f.write(f"Epoch {i+1}: {obj:.6f}\n")
        with open(os.path.join(self.output_dir, "checkpoint_evaluation.txt"), "w") as f:
            f.write(f"seed={self.EVAL_SEED}, alpha={self.EVAL_ALPHA}, n_instances={self.EVAL_N}, size={self.EVAL_NW}x{self.EVAL_NT}x{self.EVAL_MT}\n")
            f.write(f"{'Epoch':>5} {'Objective':>10} {'Destruction':>12} {'RemVal':>10}\n")
            for ep, obj, destr, rem in self.checkpoint_eval_results:
                f.write(f"{ep:5d} {obj:10.4f} {destr:12.2%} {rem:10.4f}\n")
            f.write(f"\nBEST: Epoch {best_epoch}, Objective = {best_obj:.4f}\n")
        self.logger.info(f"Results saved: epoch_objectives.txt, checkpoint_evaluation.txt")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Starting Expert Iteration Training")
        self.logger.info("=" * 60)
        global_start = time.time()
        best_avg_obj = float('inf')
        best_exit_iter = None

        reinforce_epochs_per_iter = max(TOTAL_EPOCH // N_EXIT_ITERS, 20)

        for exit_iter in range(1, N_EXIT_ITERS + 1):
            iter_start = time.time()
            beta_weight = EXIT_IMITATION_WEIGHT - (
                (EXIT_IMITATION_WEIGHT - EXIT_IMITATION_WEIGHT_MIN)
                * (exit_iter - 1) / max(N_EXIT_ITERS - 1, 1)
            )

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ExIt Iteration {exit_iter}/{N_EXIT_ITERS}  |  beta={beta_weight:.3f}")
            self.logger.info(f"{'='*60}")

            # Step 1: REINFORCE training epochs (with per-epoch eval, same as base RL)
            t1 = time.time()
            self.logger.info(f"  [Step 1] REINFORCE training ({reinforce_epochs_per_iter} epochs) ...")
            for ep in range(1, reinforce_epochs_per_iter + 1):
                a_loss, c_loss, cfg = self._reinforce_epoch(
                    epoch=(exit_iter - 1) * reinforce_epochs_per_iter + ep,
                )
                if ep % 10 == 0 or ep == reinforce_epochs_per_iter:
                    self.logger.info(
                        f"    REINFORCE ep {ep}/{reinforce_epochs_per_iter} | "
                        f"actor_loss={a_loss:.4f}  critic_loss={c_loss:.4f}"
                    )
                if EVALUATION_PERIOD and (ep % EVALUATION_PERIOD == 0):
                    global_ep = (exit_iter - 1) * reinforce_epochs_per_iter + ep
                    obj, destr, rem = self._evaluate_fixed_seed()
                    self.epoch_objectives.append(rem)
                    if self._should_save_base_rl_style(global_ep):
                        self.checkpoint_eval_results.append((global_ep, obj, destr, rem))
                        self._save_checkpoint_base_rl_style(global_ep)
                        self.logger.info(f"    CheckPoint_epoch{global_ep:05d} saved")
                    self.logger.info(
                        f"    [EVAL] ep {global_ep} | Obj={obj:.4f} | Destr={destr:.2%} | RemVal={rem:.4f}"
                    )
            self.logger.info(f"  [Step 1] time: {time.time() - t1:.1f}s")

            # Step 2: Collect expert trajectories (multi-scale, same as base RL)
            t2 = time.time()
            self.logger.info(f"  [Step 2] Collecting expert trajectories ...")
            self._collect_and_store_expert(n_instances=TRAIN_BATCH, beta=2, exit_iter=exit_iter)
            self.logger.info(f"  [Step 2] time: {time.time() - t2:.1f}s")

            # Step 3: Imitation learning from expert buffer
            t3 = time.time()
            self.logger.info(f"  [Step 3] Imitation update (beta_weight={beta_weight:.3f}) ...")
            imit_loss = self._imitation_update(n_steps=30, batch_size=64, beta_weight=beta_weight)
            self.logger.info(f"    Imitation loss: {imit_loss:.6f}")
            self.logger.info(f"  [Step 3] time: {time.time() - t3:.1f}s")

            # Step 4: Critic update from expert buffer
            t4 = time.time()
            critic_buf_loss = self._critic_update_from_buffer(n_steps=20, batch_size=64)
            self.logger.info(f"    Critic buffer loss: {critic_buf_loss:.6f}")
            self.logger.info(f"  [Step 4] time: {time.time() - t4:.1f}s")

            # Step 5: Evaluate (fixed seed, same as base RL)
            t5 = time.time()
            obj, destr, rem = self._evaluate_fixed_seed()
            iter_time = time.time() - iter_start
            self.logger.info(
                f"  [Eval] Obj={obj:.4f} | Destr={destr:.2%} | RemVal={rem:.4f} | Iter time: {iter_time:.0f}s"
            )
            self.logger.info(f"  [Step 5 eval] time: {time.time() - t5:.1f}s")

            # Step 6: Save checkpoint
            self._save_checkpoint(exit_iter, reinforce_epochs_per_iter * exit_iter)

            # Track best by eval remaining value (lower is better)
            if rem < best_avg_obj:
                best_avg_obj = rem
                best_exit_iter = exit_iter
                best_epoch = reinforce_epochs_per_iter * exit_iter
                self._save_checkpoint_to_dir(os.path.join(self.output_dir, "ExIt_best"), best_epoch)
                self.logger.info(f"  [Best] Updated best model -> ExIt_best (remaining value: {best_avg_obj:.4f})")

        self._save_final_results(best_avg_obj, best_exit_iter, best_epoch)

        total_time = time.time() - global_start
        self.logger.info(f"\nExIt Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best model: iteration {best_exit_iter} (eval remaining value: {best_avg_obj:.4f}) -> {self.output_dir}/ExIt_best")
        self.logger.info(f"Final checkpoint saved to {self.output_dir}")


def main():
    print("Starting ExIt (Expert Iteration) Training for DWTA ...")
    trainer = ExItTrainer()
    trainer.train()
    print("ExIt Training completed successfully!")


if __name__ == "__main__":
    main()
