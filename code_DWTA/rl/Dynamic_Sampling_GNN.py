"""
GNN-based Dynamic Sampling for DWTA Problem
Multi-episodic REINFORCE with batch*para graph processing
Now supports random multi-scale training for better generalization!
"""

import torch
import torch.nn.functional as F
import random
import os
import sys

# Add path for common modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import *
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.utilities import Average_Meter


def get_random_problem_size(epoch, episode):
    """
    Get random problem size for multi-scale training per episode.
    Dimensions: [5, 7], Ammo: [2, 4], Prep: [1, 2], Cost: [20, 100].
    """
    num_weapons = random.randint(5, 7)
    num_targets = random.randint(5, 7)
    max_time = random.randint(5, 7)
    amm_list = [random.randint(1, 3) for _ in range(num_weapons)]
    prep_list = [random.randint(1, 2) for _ in range(num_weapons)]
    cost_list = [random.randint(2, 10) * 10 for _ in range(num_weapons)]
    print(f"Epoch {epoch}, Episode {episode}: {num_weapons}W x {num_targets}T x {max_time}T, AMM={amm_list}")
    return (num_weapons, num_targets, max_time, amm_list, prep_list, cost_list)


def patch_hyperparameters_for_epoch(num_weapons, num_targets, max_time, amm_list, prep_list=None, cost_list=None):
    """
    Temporarily patch global hyperparameters for multi-scale training.
    """
    original_values = {
        'NUM_WEAPONS': NUM_WEAPONS,
        'NUM_TARGETS': NUM_TARGETS, 
        'MAX_TIME': MAX_TIME,
        'AMM': AMM.copy(),
        'PREPARATION_TIME': PREPARATION_TIME.copy() if isinstance(PREPARATION_TIME, list) else list(PREPARATION_TIME),
        'WEAPON_COST': WEAPON_COST.copy() if isinstance(WEAPON_COST, list) else list(WEAPON_COST),
    }
    
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import common.Dynamic_HYPER_PARAMETER as HP
    
    globals()['NUM_WEAPONS'] = num_weapons
    globals()['NUM_TARGETS'] = num_targets
    globals()['MAX_TIME'] = max_time
    globals()['AMM'] = amm_list
    if prep_list is not None:
        globals()['PREPARATION_TIME'] = prep_list
    if cost_list is not None:
        globals()['WEAPON_COST'] = cost_list
    
    HP.NUM_WEAPONS = num_weapons
    HP.NUM_TARGETS = num_targets
    HP.MAX_TIME = max_time
    HP.AMM = amm_list
    if prep_list is not None:
        HP.PREPARATION_TIME = prep_list
    if cost_list is not None:
        HP.WEAPON_COST = cost_list

    for sim_module_name in ('common.DWTA_Simulator', 'rl_rollout.DWTA_Simulator_rollout'):
        try:
            Sim = __import__(sim_module_name, fromlist=['Environment'])
            Sim.NUM_WEAPONS = num_weapons
            Sim.NUM_TARGETS = num_targets
            Sim.MAX_TIME = max_time
            Sim.AMM = amm_list
            if prep_list is not None:
                Sim.PREPARATION_TIME = prep_list
            if cost_list is not None:
                Sim.WEAPON_COST = cost_list
        except ImportError:
            pass

    # ExIt beam search uses this module; must match env (nw, nt) for multi-scale
    try:
        BeamMod = __import__('unused.BEAM_WITH_SIMULATION_NOT_TRUNC', fromlist=['Beam_Search'])
        BeamMod.NUM_WEAPONS = num_weapons
        BeamMod.NUM_TARGETS = num_targets
        BeamMod.MAX_TIME = max_time
        BeamMod.AMM = amm_list
        if prep_list is not None:
            BeamMod.PREPARATION_TIME = prep_list
        if cost_list is not None:
            BeamMod.WEAPON_COST = cost_list
    except ImportError:
        pass

    return original_values


def restore_hyperparameters(original_values):
    """
    Restore original hyperparameters after epoch.
    
    Args:
        original_values: Dictionary of original hyperparameter values
    """
    import common.Dynamic_HYPER_PARAMETER as HP
    
    for key in ('NUM_WEAPONS', 'NUM_TARGETS', 'MAX_TIME', 'AMM', 'PREPARATION_TIME', 'WEAPON_COST'):
        globals()[key] = original_values[key]
    
    HP.NUM_WEAPONS = original_values['NUM_WEAPONS']
    HP.NUM_TARGETS = original_values['NUM_TARGETS']
    HP.MAX_TIME = original_values['MAX_TIME']
    HP.AMM = original_values['AMM']
    HP.PREPARATION_TIME = original_values['PREPARATION_TIME']
    HP.WEAPON_COST = original_values['WEAPON_COST']

    for sim_module_name in ('common.DWTA_Simulator', 'rl_rollout.DWTA_Simulator_rollout'):
        try:
            Sim = __import__(sim_module_name, fromlist=['Environment'])
            Sim.NUM_WEAPONS = original_values['NUM_WEAPONS']
            Sim.NUM_TARGETS = original_values['NUM_TARGETS']
            Sim.MAX_TIME = original_values['MAX_TIME']
            Sim.AMM = original_values['AMM']
            Sim.PREPARATION_TIME = original_values['PREPARATION_TIME']
            Sim.WEAPON_COST = original_values['WEAPON_COST']
        except ImportError:
            pass


def self_play_gnn(old_actor, actor, critic, episode, temp, epoch, logger=None):
    """
    Multi-episodic REINFORCE training for GNN with random multi-scale training.
    Uses only final returns (REINFORCE) with variance reduction.
    Each episode has random configuration.
    """
    
    try:
        # Set models to training mode
        actor.train()
        critic.train()
        
        # Initialize metrics
        actor_losses = Average_Meter()
        critic_losses = Average_Meter()
        
        # Entropy regularization coefficient (can be moved to hyperparams)
        entropy_coef = 1e-3 if 'ENTROPY_COEF' not in globals() else ENTROPY_COEF
        
        # Run training episodes
        total_entropy = 0.0
        total_steps = 0
        total_objective = 0.0
        total_destruction_ratio = 0.0
        
        for ep in range(episode):
            # Get random problem size for this episode
            ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list, ep_prep_list, ep_cost_list = get_random_problem_size(epoch, ep)
            
            # Fixed alpha=1.0: maximize destruction only, ignore cost
            ep_alpha = 1.0
            
            # Patch hyperparameters for this episode
            original_hyperparams = patch_hyperparameters_for_epoch(
                ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list,
                prep_list=ep_prep_list, cost_list=ep_cost_list,
            )
            
            try:
                # Generate training instances with episode-specific alpha
                assignment_encoding, weapon_to_target_prob = input_generation(
                    NUM_WEAPON=ep_num_weapons,
                    NUM_TARGET=ep_num_targets,
                    value=None,
                    prob=None,
                    TW=None,
                    max_time=ep_max_time,
                    batch_size=TRAIN_BATCH,
                    alpha=ep_alpha,
                )
                
                # Expand to [batch, para, assignment, feature] for multi-episodic (avoid shared memory)
                assignment_encoding = assignment_encoding.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
                weapon_to_target_prob = weapon_to_target_prob.unsqueeze(1).repeat(1, NUM_PAR, 1, 1).contiguous()
                
                # Create environment for batch*para instances
                env = Environment(
                    assignment_encoding=assignment_encoding,
                    weapon_to_target_prob=weapon_to_target_prob,
                    max_time=ep_max_time  # Use episode-specific values
                )

                # Log target emerging time windows (start/end) for visibility
                if logger is not None:
                    try:
                        ts = env.target_start_time[0, 0, :ep_num_targets].detach().cpu().tolist()
                        te = env.target_end_time[0, 0, :ep_num_targets].detach().cpu().tolist()
                        logger.info(f"Target time windows (batch0): start={ts}, end={te}")
                    except Exception:
                        pass
                
                # Storage
                log_probs = []
                values = []
                entropies = []
                rewards = []  # per-step immediate reward, normalized value reduction (Eq. 17 form)

                # original_value must be captured before any actions are taken
                original_value = env.original_target_value[:, :, 0:ep_num_targets].sum(2)  # [batch, para]
                prev_value = original_value.clone()

                # Execute episode: ONE decision round per time step. All live weapons
                # choose simultaneously (parallel multi-pointer decoding) instead of the
                # previous per-weapon sequential loop (~M actor calls per time step).
                for time_step in range(ep_max_time):
                    # Current state
                    current_state = env.assignment_encoding.clone()
                    current_prob = env.weapon_to_target_prob.clone()

                    # Policy: [batch, para, W, T+1] -- independent softmax per weapon,
                    # computed in a single forward pass (shared GNN embeddings give each
                    # weapon global context; no explicit conflict handler)
                    policy, _ = actor(
                        assignment_embedding=current_state,
                        prob=current_prob,
                        mask=env.mask_per_weapon.clone()
                    )

                    # Entropy (per weapon, this step)
                    safe_policy = policy.clamp_min(1e-8)
                    step_entropy = -(safe_policy * safe_policy.log()).sum(dim=-1)  # [batch, para, W]
                    entropies.append(step_entropy)
                    # step_entropy.mean() already averages over (batch, para, W), so the
                    # running counter must increment by 1 per call, not by ep_num_weapons,
                    # or avg_entropy would be spuriously divided by an extra factor of W.
                    total_entropy += step_entropy.mean().item()
                    total_steps += 1

                    # Sample each weapon's action independently from its own [T+1]
                    # distribution (flatten batch*para*W for multinomial, then restore shape).
                    # Derive batch/para sizes from the actual tensor rather than the global
                    # TRAIN_BATCH/NUM_PAR constants, to stay correct under any future change
                    # to how multi-scale episodes are batched.
                    b_sz, p_sz = current_state.shape[:2]
                    flat_policy = policy.reshape(-1, ep_num_targets + 1)
                    action = torch.multinomial(flat_policy, 1).view(b_sz, p_sz, ep_num_weapons)

                    # Value: single state-value estimate this step; doubles as Phi(s_t)
                    # for potential-based shaping below. Uses the per-weapon mask so the
                    # critic can mask out illegal (weapon,target) pairs when pooling
                    # (see EdgeAwareGNN_CRITIC.forward) and recover exact (W,T) without guessing.
                    value = critic(current_state, env.mask_per_weapon.clone())

                    # Store: joint log-prob of the simultaneous action = sum over weapons
                    # of each weapon's own log-prob, log pi(a_t|s_t) = sum_m log pi_m(a_m,t|s_t)
                    per_weapon_log_prob = torch.log(
                        policy.gather(-1, action.unsqueeze(-1)).clamp_min(1e-8)
                    ).squeeze(-1)  # [batch, para, W]
                    joint_log_prob = per_weapon_log_prob.sum(dim=-1, keepdim=True)  # [batch, para, 1]

                    values.append(value.clone())
                    log_probs.append(joint_log_prob)

                    # Env step: apply ALL weapons' decisions simultaneously
                    env.update_internal_variables_parallel(selected_actions=action)

                    # Immediate reward for this time step's simultaneous decisions
                    curr_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)
                    step_reward = (prev_value - curr_value) / (original_value + 1e-8)
                    rewards.append(step_reward)
                    prev_value = curr_value

                    env.time_update()

                # Final returns (for critic target / logging): maximize destruction
                final_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)  # [batch, para]

                destruction_ratio = 1 - (final_value / (original_value + 1e-8))
                returns = destruction_ratio

                # --- Potential-based reward shaping (Ng, Harada & Russell, ICML 1999) ---
                # Phi(s_t) := critic(s_t), detached. Adding gamma*Phi(s_{t+1}) - Phi(s_t) to
                # each step's reward provably leaves the optimal policy unchanged (it
                # telescopes to a constant over a full trajectory), so this stays a valid
                # REINFORCE objective -- it is NOT actor-critic bootstrapping, and does not
                # introduce bias if the critic is inaccurate. Phi(s_T) (terminal) := 0.
                gamma = 1.0  # undiscounted, finite horizon -- matches Eq. (17)'s reward form
                T_steps = len(rewards)
                phi = [v.squeeze(-1).detach() for v in values]
                phi_next = phi[1:] + [torch.zeros_like(phi[0])]
                shaped_rewards = [rewards[t] + gamma * phi_next[t] - phi[t] for t in range(T_steps)]

                # Shaped reward-to-go per decision step: a step's credit should not depend
                # on reward realized before that step was taken (standard REINFORCE
                # variance reduction, unbiased; see Sutton & Barto).
                shaped_returns = [None] * T_steps
                running = torch.zeros_like(shaped_rewards[0])
                for t in reversed(range(T_steps)):
                    running = shaped_rewards[t] + gamma * running
                    shaped_returns[t] = running

                # Per-instance, per-step advantage normalization across para (same POMO-style
                # shared baseline as before, now applied at every decision step instead of
                # once per episode)
                advantages = []
                for G_t in shaped_returns:
                    baseline_t = G_t.mean(dim=1, keepdim=True)
                    adv_t = G_t - baseline_t
                    adv_std_t = adv_t.std(dim=1, keepdim=True).clamp_min(1e-6)
                    advantages.append(adv_t / adv_std_t)

                # Update Actor (REINFORCE) with entropy bonus
                actor_loss = 0
                for log_prob, adv_t in zip(log_probs, advantages):
                    actor_loss = actor_loss + (-(log_prob.squeeze(-1) * adv_t).mean())
                if entropies:
                    entropy_mean = torch.stack(entropies).mean()
                    actor_loss = actor_loss - entropy_coef * entropy_mean
                
                actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor.optimizer.step()
                
                # Update Critic to predict 0-1 destruction ratio
                critic_loss = 0
                for value in values:
                    critic_loss = critic_loss + F.mse_loss(value.squeeze(-1), returns)
                
                critic.optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic.optimizer.step()
                
                # Metrics
                actor_losses.push(torch.tensor(actor_loss.item()), 1)
                critic_losses.push(torch.tensor(critic_loss.item()), 1)
                
                # Accumulate episode metrics
                total_objective += final_value.mean().item()
                total_destruction_ratio += returns.mean().item()
                
                # Per-episode progress logging
                if logger is not None:
                    logger.info(
                        f"Episode {ep+1}/{episode} | Actor Loss: {actor_loss.item():.6f} | "
                        f"Critic Loss: {critic_loss.item():.6f}"
                    )
                    
            finally:
                # Always restore hyperparameters for this episode
                restore_hyperparameters(original_hyperparams)
        
        # Store average entropy for logging
        avg_entropy = total_entropy / max(total_steps, 1)
        
        # Aggregate episode metrics for logging at trainer level
        epoch_objective = total_objective / episode
        avg_destruction = total_destruction_ratio / episode

        return actor_losses.result(), critic_losses.result(), {
            'num_weapons': 'mixed',  # Mixed across episodes
            'num_targets': 'mixed',  # Mixed across episodes
            'max_time': 'mixed',     # Mixed across episodes
            'amm': 'mixed',          # Mixed across episodes
            'objective': epoch_objective,
            'destruction_ratio': avg_destruction,
        }
    
    except Exception as e:
        print(f"Error in self_play_gnn: {e}")
        raise 