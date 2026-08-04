import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_HYPER_PARAMETER import NUM_WEAPONS, MAX_TIME
from unused.BEAM_WITH_SIMULATION_NOT_TRUNC import Beam_Search, batch_dimension_resize, alpha


class Beam_Search_Trunc(Beam_Search):
    """
    Beam search variant with beta-truncated rollout and critic cost-to-go.
    - Simulates only `beta` steps ahead using the actor (greedy per group)
    - Then adds critic(state) as a to-go value to estimate final objective
    - Selects global top-k (alpha) groups as usual
    """
    def __init__(self, env, actor, value, available_actions, beta: int = 2, to_go_weight: float = 1.0):
        super().__init__(env=env, actor=actor, value=value, available_actions=available_actions)
        self.beta = beta
        self.to_go_weight = to_go_weight

    @torch.no_grad()
    def do_beam_simulation(self, possible_node_index, time, w_index):
        # initial flag
        initial = 'yes' if (time + w_index == 0) else 'no'

        # derive starting point from fires made so far
        start_time = int(self.n_fires.item()) // NUM_WEAPONS
        start_weapon = int(self.n_fires.item()) % NUM_WEAPONS

        # If we start at weapon 0, advance the clock once (to match original logic)
        if start_weapon == 0:
            self.time_update()

        # Remaining horizon in (time, weapon) pairs
        remaining_weapons = NUM_WEAPONS - start_weapon
        remaining_rounds = MAX_TIME - start_time
        remaining_steps = remaining_weapons + (remaining_rounds - 1) * NUM_WEAPONS if remaining_rounds > 0 else 0

        # Limit simulation to beta steps
        steps_to_sim = min(self.beta, max(0, remaining_steps))

        # Simulate up to beta steps using actor greedily for each parallel group
        cur_time = start_time
        cur_weapon = start_weapon
        steps_done = 0
        while steps_done < steps_to_sim and cur_time < MAX_TIME:
            # For current weapon index within this time
            if cur_weapon < NUM_WEAPONS:
                policy, _ = self.policy(
                    assignment_embedding=self.assignment_encoding,
                    prob=self.weapon_to_target_prob,
                    mask=self.mask
                )
                action_index = policy.view(-1, NUM_WEAPONS * self.target_arange.size(0) + 1).argmax(dim=1).view(
                    self.assignment_encoding.size(0), self.assignment_encoding.size(1)
                )
                self.update_internal_variables(selected_action=action_index)
                cur_weapon += 1
                steps_done += 1
            else:
                # Start of next time slot
                self.time_update()
                cur_time += 1
                cur_weapon = 0

        # After truncated rollout, estimate final objective = current remainder + critic to-go
        # Current remainder per group
        remainder = self.current_target_value[:, :, 0:self.target_arange.size(0)].sum(2)

        if self.to_go_value is not None:
            try:
                # Critic expects (state, mask); returns [batch, para, 1] in 0-1 range
                to_go_ratio = self.to_go_value(state=self.assignment_encoding.clone(), mask=self.mask.clone())
                to_go_ratio = to_go_ratio.squeeze(-1)  # [batch, para]
                
                # Scale by current remainder: critic predicts destruction ratio (0-1)
                # to_go_value = expected additional destruction in absolute terms
                to_go = to_go_ratio * remainder  # Scale by current target value
            except Exception:
                to_go = torch.zeros_like(remainder)
        else:
            to_go = torch.zeros_like(remainder)

        # Combine with weight (lower is better). Subtract expected additional destruction.
        self.beam_result = remainder - self.to_go_weight * to_go

        # Select best groups globally
        batch_indices, group_indices = self.select_best_actions(selected_actions=possible_node_index, initial=initial)
        return batch_indices, group_indices


def batch_dimension_resize_trunc(env, batch_index, group_index):
    """Alias passthrough to the existing resize to keep API symmetrical."""
    return batch_dimension_resize(env=env, batch_index=batch_index, group_index=group_index) 