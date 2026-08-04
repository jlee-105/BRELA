import os
import sys
import copy
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_HYPER_PARAMETER import (
    NUM_WEAPONS, NUM_TARGETS, MAX_TIME,
    NUM_ROLLOUT_POLICIES, ROLLOUT_SAMPLES_PER_POLICY,
)
from unused.BEAM_WITH_SIMULATION_NOT_TRUNC import Beam_Search, batch_dimension_resize, alpha
from rl_rollout.multi_policy_rollout import ROLLOUT_POLICIES


def _snapshot_beam_state(beam):
    """Save a lightweight snapshot of mutable beam tensors for multi-policy rollout."""
    keys = [
        'assignment_encoding', 'weapon_to_target_prob', 'weapon_to_target_assign',
        'current_target_value', 'original_target_value', 'possible_weapons',
        'available_actions', 'mask', 'weapon_availability', 'amm_availability',
        'weapon_wait_time', 'time_left', 'target_availability',
        'target_start_time', 'target_end_time', 'n_target_hit',
        'clock', 'n_fires', 'total_cost',
    ]
    snap = {}
    for k in keys:
        v = getattr(beam, k, None)
        if v is not None and isinstance(v, torch.Tensor):
            snap[k] = v.clone()
    return snap


def _restore_beam_state(beam, snap):
    """Restore beam tensors from a snapshot."""
    for k, v in snap.items():
        setattr(beam, k, v.clone())


class Beam_Search_Trunc(Beam_Search):
    """
    Multi-policy truncated rollout with cost-aware scoring.

    At each decision point the search:
      1. Expands candidate actions (inherited from Beam_Search)
      2. For each candidate, runs *multiple* truncated rollouts using
         different base policies (greedy / stochastic / urgency-aware)
      3. Evaluates each rollout with: remainder - to_go_weight * critic_to_go (no cost term)
      4. Aggregates across policies (mean) and selects the best candidate
    """

    def __init__(self, env, actor, value, available_actions,
                 beta: int = 2, to_go_weight: float = 1.0,
                 use_greedy_only: bool = False):
        super().__init__(env=env, actor=actor, value=value, available_actions=available_actions)
        self.beta = beta  # None or <0 => rollout to end of episode (Bertsekas full rollout)
        self.to_go_weight = to_go_weight
        self._source_env = env  # stash for deferred cost init in reset()
        self._use_greedy_only = use_greedy_only  # True => argmax only (single policy)

    def reset(self):
        super().reset()
        env = self._source_env
        batch_size = self.assignment_encoding.size(0)
        para_size = self.assignment_encoding.size(1)

        from common.Dynamic_HYPER_PARAMETER import WEAPON_COST
        cost_list = WEAPON_COST[:NUM_WEAPONS]
        if len(cost_list) < NUM_WEAPONS:
            pad_val = WEAPON_COST[-1] if len(WEAPON_COST) > 0 else 50
            cost_list = cost_list + [pad_val] * (NUM_WEAPONS - len(cost_list))
        self.weapon_cost_per_fire = torch.tensor(cost_list, device=DEVICE, dtype=torch.float32)
        self.max_possible_cost = float(self.weapon_cost_per_fire.sum() * MAX_TIME)

        if hasattr(env, 'total_cost'):
            self.total_cost = env.total_cost.expand(batch_size, para_size).clone()
        else:
            self.total_cost = torch.zeros(batch_size, para_size, device=DEVICE)

    # ------------------------------------------------------------------
    # Internal: single-policy truncated rollout (returns score tensor)
    # ------------------------------------------------------------------
    def _run_single_rollout(self, policy_fn, start_time, start_weapon, steps_to_sim):
        """Execute a truncated rollout with the given policy_fn and return score per group."""
        cur_time, cur_weapon, steps_done = start_time, start_weapon, 0

        while steps_done < steps_to_sim and cur_time < MAX_TIME:
            if cur_weapon < NUM_WEAPONS:
                action_index = policy_fn(self)
                self.update_internal_variables(selected_action=action_index)
                cur_weapon += 1
                steps_done += 1
            else:
                self.time_update()
                cur_time += 1
                cur_weapon = 0

        # Remaining target value per group
        remainder = self.current_target_value[:, :, 0:self.target_arange.size(0)].sum(2)

        # Critic to-go (to_go_weight=0 in train/full rollout; inference uses weight > 0)
        if self.to_go_value is not None:
            try:
                to_go_ratio = self.to_go_value(
                    state=self.assignment_encoding.clone(), mask=self.mask.clone()
                ).squeeze(-1)
                to_go = to_go_ratio * remainder
            except Exception:
                to_go = torch.zeros_like(remainder)
        else:
            to_go = torch.zeros_like(remainder)
        score = remainder - self.to_go_weight * to_go
        return score

    # ------------------------------------------------------------------
    # Policy wrappers that match _run_single_rollout's expected signature
    # ------------------------------------------------------------------
    @staticmethod
    def _greedy_fn(beam_obj):
        policy, _ = beam_obj.policy(
            assignment_embedding=beam_obj.assignment_encoding,
            prob=beam_obj.weapon_to_target_prob,
            mask=beam_obj.mask,
        )
        return policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).argmax(dim=1).view(
            beam_obj.assignment_encoding.size(0), beam_obj.assignment_encoding.size(1)
        )

    @staticmethod
    def _stochastic_fn(beam_obj):
        policy, _ = beam_obj.policy(
            assignment_embedding=beam_obj.assignment_encoding,
            prob=beam_obj.weapon_to_target_prob,
            mask=beam_obj.mask,
        )
        flat = policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).clamp_min(1e-8)
        return torch.multinomial(flat, 1).squeeze(-1).view(
            beam_obj.assignment_encoding.size(0), beam_obj.assignment_encoding.size(1)
        )

    @staticmethod
    def _urgency_fn(beam_obj):
        policy, _ = beam_obj.policy(
            assignment_embedding=beam_obj.assignment_encoding,
            prob=beam_obj.weapon_to_target_prob,
            mask=beam_obj.mask,
        )
        batch_size = beam_obj.assignment_encoding.size(0)
        para_size = beam_obj.assignment_encoding.size(1)

        time_remaining = (beam_obj.target_end_time - beam_obj.clock).float()
        safety = (time_remaining / max(float(MAX_TIME), 1.0)).clamp(0.0, 1.0)
        urgency = 1.0 - safety
        urgency_wt = urgency.unsqueeze(2).expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
        urgency_flat = urgency_wt.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS)
        no_act = torch.zeros(batch_size, para_size, 1, device=DEVICE)
        bonus = torch.cat([urgency_flat, no_act], dim=-1)

        adjusted = policy * (1.0 + bonus) * beam_obj.mask
        return adjusted.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).argmax(dim=1).view(
            batch_size, para_size
        )

    POLICY_FNS = None  # populated on first call

    def _get_policy_fns(self):
        if Beam_Search_Trunc.POLICY_FNS is None:
            Beam_Search_Trunc.POLICY_FNS = [
                self._greedy_fn,
                self._stochastic_fn,
                self._urgency_fn,
            ]
        return Beam_Search_Trunc.POLICY_FNS

    # ------------------------------------------------------------------
    # Main entry: multi-policy truncated beam simulation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def do_beam_simulation(self, possible_node_index, time, w_index):
        initial = 'yes' if (time + w_index == 0) else 'no'

        start_time = int(self.n_fires.item()) // NUM_WEAPONS
        start_weapon = int(self.n_fires.item()) % NUM_WEAPONS

        if start_weapon == 0:
            self.time_update()

        remaining_weapons = NUM_WEAPONS - start_weapon
        remaining_rounds = MAX_TIME - start_time
        remaining_steps = remaining_weapons + (remaining_rounds - 1) * NUM_WEAPONS if remaining_rounds > 0 else 0
        # beta None or <0 => full rollout to end of episode (Bertsekas); else truncate at beta
        if self.beta is None or self.beta < 0:
            steps_to_sim = max(0, remaining_steps)
        else:
            steps_to_sim = min(self.beta, max(0, remaining_steps))

        # Save state before rollouts
        snapshot = _snapshot_beam_state(self)

        # use_greedy_only => argmax only (single policy); else multi-policy (greedy, stochastic, urgency)
        policy_fns = [self._greedy_fn] if self._use_greedy_only else self._get_policy_fns()
        all_scores = []

        for pfn in policy_fns:
            _restore_beam_state(self, snapshot)
            score = self._run_single_rollout(pfn, start_time, start_weapon, steps_to_sim)
            all_scores.append(score)

        # Aggregate: mean across policies (Bertsekas multi-heuristic: best-of is also valid)
        stacked = torch.stack(all_scores, dim=0)  # [K, batch, para]
        self.beam_result = stacked.mean(dim=0)     # [batch, para]

        # Restore original state for action selection
        _restore_beam_state(self, snapshot)

        batch_indices, group_indices = self.select_best_actions(
            selected_actions=possible_node_index, initial=initial
        )
        return batch_indices, group_indices


def batch_dimension_resize_trunc(env, batch_index, group_index):
    """Alias passthrough to the existing resize to keep API symmetrical."""
    return batch_dimension_resize(env=env, batch_index=batch_index, group_index=group_index)