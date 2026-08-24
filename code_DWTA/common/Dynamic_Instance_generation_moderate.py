"""
Batched, vectorized MODERATE temporal-dilemma training-instance generator,
producing the exact same (assignment_encoding, weapon_to_target_prob) tensor
format as common/Dynamic_Instance_generation.py's _generate_training_instances
-- so it's a drop-in swap inside self-play training loops (see
rl/Dynamic_Sampling_GNN_moderate.py), just replacing the STANDARD value/
time-window distribution with the MODERATE curriculum's continuous
value-vs-emergence-time correlation (common/temporal_dilemma_generator_moderate.py).

Ammo/prep are kept SHARED across the whole batch (one value per weapon,
patched in globally per episode, exactly like the standard training path's
get_random_problem_size/patch_hyperparameters_for_epoch) rather than
per-instance-random like the standalone eval-side generator -- the
Environment class's ammo/reload bookkeeping assumes one global AMM/PREP
config per batch, so genuinely per-instance-varying ammo within a single
batched training step isn't supported without a larger Environment rewrite.
Value/probability/time-window ARE fully per-instance random, which is the
part of the moderate curriculum that actually creates the hold-vs-fire
dilemma; the fixed eval-side test set (common/rl4co_eval.py) still uses the
true per-instance-varying-ammo generator unchanged.
"""
import torch

from .Dynamic_HYPER_PARAMETER import MAX_TARGET_VALUE, NUM_FEATURES
from .TORCH_OBJECTS import DEVICE
from .Dynamic_Instance_generation import _create_no_action_encoding


def generate_moderate_training_instances(batch_size, num_weapons, num_targets, max_time, amm_list):
    """Vectorized batch generator matching the MODERATE curriculum's value
    distribution (base_value U(2,6) + time-correlated bonus U(2,5), noisy --
    see temporal_dilemma_generator_moderate.py's docstring for the
    rationale), for use in self-play training. amm_list: per-weapon ammo
    count, shared across the batch (see module docstring)."""
    assignment_encoding = torch.zeros(
        (batch_size, num_weapons * num_targets + 1, NUM_FEATURES),
        device=DEVICE, dtype=torch.float32,
    )

    start = torch.randint(0, max_time, (batch_size, num_targets), device=DEVICE, dtype=torch.float32)
    base_value = torch.rand(batch_size, num_targets, device=DEVICE) * 4 + 2  # U(2, 6)
    time_bonus = (start / max(max_time - 1, 1)) * (torch.rand(batch_size, num_targets, device=DEVICE) * 3 + 2)  # U(2, 5)
    value = base_value + time_bonus  # ~[2, 11], smoothly increasing with start time + noise

    target_values = value / MAX_TARGET_VALUE  # same normalization convention as the standard path
    target_emerge_times = start / max_time
    target_end_times = torch.full((batch_size, num_targets), float(max_time - 1), device=DEVICE) / max_time

    weapon_target_probs = torch.rand(batch_size, num_weapons, num_targets, device=DEVICE) * 0.6 + 0.3  # U(0.3, 0.9)

    amm_ratios = torch.tensor(amm_list[:num_weapons], device=DEVICE, dtype=torch.float32) / max(amm_list[:num_weapons])

    for batch_idx in range(batch_size):
        assignment_idx = 0
        for weapon_idx in range(num_weapons):
            for target_idx in range(num_targets):
                features = torch.tensor([
                    amm_ratios[weapon_idx],
                    1.0,
                    max_time / max_time,
                    0.0,
                    0.0,
                    target_emerge_times[batch_idx, target_idx],
                    target_end_times[batch_idx, target_idx],
                    target_values[batch_idx, target_idx],
                    weapon_target_probs[batch_idx, weapon_idx, target_idx],
                ], device=DEVICE, dtype=torch.float32)
                assignment_encoding[batch_idx, assignment_idx] = features
                assignment_idx += 1
        assignment_encoding[batch_idx, -1] = _create_no_action_encoding(1, max_time).squeeze(0)

    return assignment_encoding, weapon_target_probs
