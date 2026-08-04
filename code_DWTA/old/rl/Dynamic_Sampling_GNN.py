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
    All dimensions (weapons, targets, time) are randomly sampled from [5, 7].
    Ammunition is randomly sampled from [1, 3] for each weapon.
    """
    num_weapons = random.randint(5, 7)
    num_targets = random.randint(5, 7)
    max_time = random.randint(5, 7)
    amm_list = [random.randint(1, 3) for _ in range(num_weapons)]
    print(f"Epoch {epoch}, Episode {episode}: {num_weapons}W×{num_targets}T×{max_time}T, AMM={amm_list}")
    return (num_weapons, num_targets, max_time, amm_list)


def patch_hyperparameters_for_epoch(num_weapons, num_targets, max_time, amm_list):
    """
    Temporarily patch global hyperparameters for multi-scale training.
    
    Args:
        num_weapons: Number of weapons for this epoch
        num_targets: Number of targets for this epoch  
        max_time: Maximum time steps for this epoch
        amm_list: List of ammunition values for each weapon
    """
    # Store original values
    original_values = {
        'NUM_WEAPONS': NUM_WEAPONS,
        'NUM_TARGETS': NUM_TARGETS, 
        'MAX_TIME': MAX_TIME,
        'AMM': AMM.copy(),  # Store original AMM array
        'PREPARATION_TIME': PREPARATION_TIME.copy() if isinstance(PREPARATION_TIME, list) else list(PREPARATION_TIME)
    }
    
    # Patch global variables in both current module and hyperparameter module
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import common.Dynamic_HYPER_PARAMETER as HP
    
    globals()['NUM_WEAPONS'] = num_weapons
    globals()['NUM_TARGETS'] = num_targets
    globals()['MAX_TIME'] = max_time
    globals()['AMM'] = amm_list
    # Use original PREPARATION_TIME (length 50) - no dynamic generation
    # globals()['PREPARATION_TIME'] remains unchanged
    
    # Also patch the hyperparameter module
    HP.NUM_WEAPONS = num_weapons
    HP.NUM_TARGETS = num_targets
    HP.MAX_TIME = max_time
    HP.AMM = amm_list
    # HP.PREPARATION_TIME remains unchanged (use original length 50)

    # Also patch simulator module globals (used inside env methods)
    import common.DWTA_Simulator as Sim
    Sim.NUM_WEAPONS = num_weapons
    Sim.NUM_TARGETS = num_targets
    Sim.MAX_TIME = max_time
    Sim.AMM = amm_list
    # Sim.PREPARATION_TIME remains unchanged (use original length 50)
    
    return original_values


def restore_hyperparameters(original_values):
    """
    Restore original hyperparameters after epoch.
    
    Args:
        original_values: Dictionary of original hyperparameter values
    """
    import common.Dynamic_HYPER_PARAMETER as HP
    
    globals()['NUM_WEAPONS'] = original_values['NUM_WEAPONS']
    globals()['NUM_TARGETS'] = original_values['NUM_TARGETS']
    globals()['MAX_TIME'] = original_values['MAX_TIME']
    globals()['AMM'] = original_values['AMM']
    globals()['PREPARATION_TIME'] = original_values['PREPARATION_TIME']
    
    # Also restore the hyperparameter module
    HP.NUM_WEAPONS = original_values['NUM_WEAPONS']
    HP.NUM_TARGETS = original_values['NUM_TARGETS']
    HP.MAX_TIME = original_values['MAX_TIME']
    HP.AMM = original_values['AMM']
    HP.PREPARATION_TIME = original_values['PREPARATION_TIME']

    import common.DWTA_Simulator as Sim
    Sim.NUM_WEAPONS = original_values['NUM_WEAPONS']
    Sim.NUM_TARGETS = original_values['NUM_TARGETS']
    Sim.MAX_TIME = original_values['MAX_TIME']
    Sim.AMM = original_values['AMM']
    Sim.PREPARATION_TIME = original_values['PREPARATION_TIME']


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
            ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list = get_random_problem_size(epoch, ep)
            
            # Patch hyperparameters for this episode
            original_hyperparams = patch_hyperparameters_for_epoch(
                ep_num_weapons, ep_num_targets, ep_max_time, ep_amm_list
            )
            
            try:
                # Generate training instances: [batch, assignment, feature] with explicit size
                assignment_encoding, weapon_to_target_prob = input_generation(
                    NUM_WEAPON=ep_num_weapons,  # Use episode-specific values
                    NUM_TARGET=ep_num_targets,  # Use episode-specific values
                    value=None,
                    prob=None,
                    TW=None,
                    max_time=ep_max_time,       # Use episode-specific values
                    batch_size=TRAIN_BATCH
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
                
                # Execute episode for all batch*para instances using episode-specific dimensions
                for time_step in range(ep_max_time):
                    for weapon_idx in range(ep_num_weapons):
                        # Current state
                        current_state = env.assignment_encoding.clone()
                        current_prob = env.weapon_to_target_prob.clone()
                        
                        # Policy
                        policy, _ = actor(
                            assignment_embedding=current_state,
                            prob=current_prob,
                            mask=env.mask.clone()
                        )
                        
                        # Entropy (per step)
                        safe_policy = policy.clamp_min(1e-8)
                        step_entropy = -(safe_policy * safe_policy.log()).sum(dim=-1)  # [batch, para]
                        entropies.append(step_entropy)
                        total_entropy += step_entropy.mean().item()
                        total_steps += 1
                        
                        # Sample action using episode-specific dimensions
                        action = torch.multinomial(
                            policy.view(-1, ep_num_weapons * ep_num_targets + 1), 1
                        ).view(TRAIN_BATCH, NUM_PAR)
                        
                        # Value
                        value = critic(current_state, env.mask.clone())
                        
                        # Store
                        values.append(value.clone())
                        log_probs.append(torch.log(policy.gather(-1, action.unsqueeze(-1)).clamp_min(1e-8)))
                        
                        # Env step
                        env.update_internal_variables(selected_action=action)
                    
                    env.time_update()
                
                # Final returns (REINFORCE): lower final_value is better using episode-specific dimensions
                final_value = env.current_target_value[:, :, 0:ep_num_targets].sum(2)  # [batch, para]
                original_value = env.original_target_value[:, :, 0:ep_num_targets].sum(2)  # [batch, para]
                
                # Convert to 0-1 scale: destruction ratio
                destruction_ratio = 1 - (final_value / (original_value + 1e-8))  # [0, 1] range
                returns = destruction_ratio  # Higher destruction ratio is better
                
                # For actor loss, we still use the advantage with original values
                returns_for_actor = -final_value
                
                # For actor: use original value-based advantage
                baseline_actor = returns_for_actor.mean(dim=1, keepdim=True)  # [batch, 1]
                advantage_actor = returns_for_actor - baseline_actor  # [batch, para]
                
                # Per-instance advantage normalization across para
                adv_mean = advantage_actor.mean(dim=1, keepdim=True)
                adv_std = advantage_actor.std(dim=1, keepdim=True).clamp_min(1e-6)
                advantage_normalized = (advantage_actor - adv_mean) / adv_std
                
                # Update Actor (REINFORCE) with entropy bonus
                actor_loss = 0
                for log_prob in log_probs:
                    actor_loss = actor_loss + (-(log_prob.squeeze(-1) * advantage_normalized).mean())
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