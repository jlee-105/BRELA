"""
Dynamic Instance Generation for DWTA Problem
Optimized version with vectorized operations for maximum efficiency.
"""

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
import random
import torch
import numpy as np


def validate_hyperparameters():
    """
    Validate hyperparameters consistency and constraints.
    
    Raises:
        AssertionError: If hyperparameters are inconsistent or invalid
    """
    assert NUM_WEAPONS > 0, f"NUM_WEAPONS must be positive, got {NUM_WEAPONS}"
    assert NUM_TARGETS > 0, f"NUM_TARGETS must be positive, got {NUM_TARGETS}"
    assert len(AMM) >= NUM_WEAPONS, f"AMM list too short: need {NUM_WEAPONS}, got {len(AMM)}"
    assert MAX_TIME > 0, f"MAX_TIME must be positive, got {MAX_TIME}"
    assert MIN_TARGET_VALUE <= MAX_TARGET_VALUE, "MIN_TARGET_VALUE must be <= MAX_TARGET_VALUE"
    assert 0 < LOW_PROB < HIGH_PROB <= 1, f"Invalid probability range: {LOW_PROB} to {HIGH_PROB}"


def _create_no_action_encoding(batch_size, max_time):
    """
    Create vectorized no-action encoding for all batches.
    
    Args:
        batch_size: Number of batches
        max_time: Maximum simulation time
        
    Returns:
        torch.Tensor: No-action encoding [batch_size, 9]
    """
    no_action_features = torch.tensor([
        0,                      # ammunition (no weapon)
        0.0,                    # weapon availability
        max_time / MAX_TIME,    # remaining time
        0.0,                    # weapon waiting time
        0,                      # number of fired on target
        0,                      # target start time
        0.0,                    # target active time
        0.0,                    # target value
        0.0                     # probability
    ], device=DEVICE, dtype=torch.float32)
    
    return no_action_features.unsqueeze(0).expand(batch_size, -1)


def _generate_training_instances(batch_size, max_time, num_weapons, num_targets, TW=None):
    """
    Generate training instances using current global parameters (dynamic).
    
    Args:
        batch_size: Number of training instances to generate
        max_time: Maximum simulation time
        
    Returns:
        tuple: (assignment_encoding, weapon_to_target_prob)
    """
    # Import current global values dynamically
    from .Dynamic_HYPER_PARAMETER import NUM_WEAPONS, NUM_TARGETS, MIN_TARGET_VALUE, MAX_TARGET_VALUE, AMM, MAX_TIME as GLOBAL_MAX_TIME
    
    # Pre-allocate tensors using current global values
    assignment_encoding = torch.zeros(
        (batch_size, num_weapons * num_targets + 1, 9), 
        device=DEVICE, dtype=torch.float32
    )
    
    # Generate target values for all batches (vectorized)
    target_values = torch.randint(
        MIN_TARGET_VALUE, MAX_TARGET_VALUE + 1, 
        (batch_size, num_targets), device=DEVICE, dtype=torch.float32
    ) / MAX_TARGET_VALUE
    
    # Generate target time windows
    if TW is not None:
        # Fixed evaluation windows supplied: TW = [(start, end), ...] in absolute time units
        starts = torch.tensor([tw[0] for tw in TW[:num_targets]], device=DEVICE, dtype=torch.float32) / MAX_TIME
        ends = torch.tensor([tw[1] for tw in TW[:num_targets]], device=DEVICE, dtype=torch.float32) / MAX_TIME
        target_emerge_times = starts.unsqueeze(0).expand(batch_size, -1).contiguous()
        target_end_times = ends.unsqueeze(0).expand(batch_size, -1).contiguous()
    else:
        # Random for training: start time 0 to 50% of max_time (as integer), end time = max_time
        max_start_time = max(0, max_time // 2)  # 50% of max_time, minimum 0
        target_emerge_times = torch.randint(
            0, max_start_time + 1, (batch_size, num_targets), device=DEVICE, dtype=torch.float32
        ) / max_time
        target_end_times = torch.full(
            (batch_size, num_targets), max_time, device=DEVICE, dtype=torch.float32
        ) / max_time
    
    # Generate weapon-target probabilities (vectorized)
    weapon_target_probs = torch.rand(
        (batch_size, num_weapons, num_targets), device=DEVICE, dtype=torch.float32
    ) * (HIGH_PROB - LOW_PROB) + LOW_PROB
    
    # Pre-compute ammunition ratios using current AMM
    amm_ratios = torch.tensor(AMM[:num_weapons], device=DEVICE, dtype=torch.float32) / max(AMM[:num_weapons])
    
    # Build assignment encoding for all batches
    for batch_idx in range(batch_size):
        assignment_idx = 0
        for weapon_idx in range(num_weapons):
            for target_idx in range(num_targets):
                features = torch.tensor([
                    amm_ratios[weapon_idx],                           # ammunition
                    1.0,                                              # weapon availability
                    max_time / max_time,                              # remaining time (normalized)
                    0.0,                                              # weapon waiting time
                    0.0,                                              # number of fired on target
                    target_emerge_times[batch_idx, target_idx],       # target start time
                    target_end_times[batch_idx, target_idx],          # target end time
                    target_values[batch_idx, target_idx],             # target value
                    weapon_target_probs[batch_idx, weapon_idx, target_idx]  # probability
                ], device=DEVICE, dtype=torch.float32)
                
                assignment_encoding[batch_idx, assignment_idx] = features
                assignment_idx += 1
        
        # Add no-action encoding
        assignment_encoding[batch_idx, -1] = _create_no_action_encoding(1, max_time).squeeze(0)
    
    return assignment_encoding, weapon_target_probs


def _generate_evaluation_instances(value, prob, TW, max_time):
    """
    Generate evaluation instances using provided parameters.
    
    Args:
        value: Target values
        prob: Weapon-target probabilities
        TW: Target time windows
        max_time: Maximum simulation time
        
    Returns:
        tuple: (assignment_encoding, weapon_to_target_prob)
    """
    batch_size = 1  # Evaluation uses single batch
    
    # Get actual problem size from input data
    num_weapons, num_targets = prob.shape
    
    assignment_encoding = torch.zeros(
        (batch_size, num_weapons * num_targets + 1, 9), 
        device=DEVICE, dtype=torch.float32
    )
    
    # Convert inputs to tensors
    target_values = torch.tensor(value, device=DEVICE, dtype=torch.float32) / MAX_TARGET_VALUE
    target_emerge_times = torch.tensor([tw[0] for tw in TW], device=DEVICE, dtype=torch.float32) / MAX_TIME
    target_end_times = torch.tensor([tw[1] for tw in TW], device=DEVICE, dtype=torch.float32) / MAX_TIME
    weapon_target_probs = torch.tensor(prob, device=DEVICE, dtype=torch.float32)
    
    # Pre-compute ammunition ratios (scale to actual number of weapons)
    amm_ratios = torch.tensor([4] * num_weapons, device=DEVICE, dtype=torch.float32) / 4  # Normalized to 1.0
    
    # Build assignment encoding
    assignment_idx = 0
    for weapon_idx in range(num_weapons):
        for target_idx in range(num_targets):
            features = torch.tensor([
                amm_ratios[weapon_idx],                           # ammunition
                1.0,                                              # weapon availability
                max_time / MAX_TIME,                              # remaining time
                0.0,                                              # weapon waiting time
                0.0,                                              # number of fired on target
                target_emerge_times[target_idx],                  # target start time
                target_end_times[target_idx],                     # target end time
                target_values[target_idx],                        # target value
                weapon_target_probs[weapon_idx, target_idx]       # probability
            ], device=DEVICE, dtype=torch.float32)
            
            assignment_encoding[0, assignment_idx] = features
            assignment_idx += 1
    
    # Add no-action encoding
    assignment_encoding[0, -1] = _create_no_action_encoding(1, max_time).squeeze(0)
    
    # Reshape probability matrix
    weapon_target_probs = weapon_target_probs.unsqueeze(0)  # Add batch dimension
    
    return assignment_encoding, weapon_target_probs


def input_generation(NUM_WEAPON, NUM_TARGET, value, prob, TW, max_time, batch_size):
    """
    Generate input instances for DWTA problem with optimized vectorized operations.
    
    Args:
        NUM_WEAPON: Number of weapons (for compatibility, uses global NUM_WEAPONS)
        NUM_TARGET: Number of targets (for compatibility, uses global NUM_TARGETS)
        value: Target values (None for training, list for evaluation)
        prob: Weapon-target probabilities (None for training, array for evaluation)
        TW: Target time windows (None for training, list for evaluation)
        max_time: Maximum simulation time
        batch_size: Batch size for training
        
    Returns:
        tuple: (assignment_encoding, weapon_to_target_prob)
            - assignment_encoding: [batch_size, num_assignments+1, 9]
            - weapon_to_target_prob: [batch_size, num_weapons, num_targets]
    """
    # Validate hyperparameters
    validate_hyperparameters()
    
    if value is None:
        # Training/evaluation (size-agnostic) with optional fixed TW
        return _generate_training_instances(batch_size, max_time, NUM_WEAPON, NUM_TARGET, TW=TW)
    else:
        # Evaluation mode: use provided parameters (size-agnostic)
        return _generate_evaluation_instances(value, prob, TW, max_time)
