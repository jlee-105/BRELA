"""
Custom RL4CO environment for DWTAP, so the library's standard AM (Attention
Model) and POMO baselines can be run unmodified against SCoPE.

Decision structure: exactly like our own Sequential decoder
(rl/Dynamic_Sampling_GNN_sequential.py) -- one (weapon, target-or-no-op)
choice at a time, in a fixed round-major/weapon-minor order (round 0: weapon
0..M-1 each decide once, then round 1: weapon 0..M-1, etc.), for
M*T total construction steps per episode. This lets AM/POMO's native
single-pointer autoregressive decoder apply unmodified -- the "route" being
constructed is just this fixed sequence of decision slots, each with its own
legality mask (ammo, reload cooldown, target time window), exactly matching
common/DWTA_Simulator.py's physics (ported here independently, not
imported, since RL4CO's TensorDict step/reset interface is a different
programming model from our Environment class).
"""
from typing import Optional

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase


class DWTAGenerator:
    """Generates DWTAP instances in RL4CO's batched-tensor convention.
    Mirrors common/Dynamic_Instance_generation.py's random-instance
    parameters (LOW_PROB/HIGH_PROB=0.2/0.9, target value range, etc.) so
    training distributions match SCoPE's own multi-scale curriculum as
    closely as this fixed-size-per-batch generator allows."""

    def __init__(self, num_weapon=5, num_target=5, max_time=5,
                 min_value=1, max_value=10, low_prob=0.2, high_prob=0.9,
                 min_amm=1, max_amm=3, min_prep=1, max_prep=2, **kwargs):
        self.num_weapon = num_weapon
        self.num_target = num_target
        self.max_time = max_time
        self.min_value = min_value
        self.max_value = max_value
        self.low_prob = low_prob
        self.high_prob = high_prob
        self.min_amm = min_amm
        self.max_amm = max_amm
        self.min_prep = min_prep
        self.max_prep = max_prep

    def __call__(self, batch_size) -> TensorDict:
        bs = batch_size[0] if isinstance(batch_size, (list, tuple)) else batch_size
        M, N, T = self.num_weapon, self.num_target, self.max_time

        value = torch.randint(self.min_value, self.max_value + 1, (bs, N)).float()
        prob = torch.rand(bs, M, N) * (self.high_prob - self.low_prob) + self.low_prob
        # Target windows: start in [0, T//2], end = T-1 (matches
        # common/Dynamic_Instance_generation.py's training-mode default)
        start = torch.randint(0, max(1, T // 2) + 1, (bs, N)).float()
        end = torch.full((bs, N), float(T - 1))
        amm = torch.randint(self.min_amm, self.max_amm + 1, (bs, M)).float()
        prep = torch.randint(self.min_prep, self.max_prep + 1, (bs, M)).float()

        return TensorDict(
            {
                "value": value,
                "prob": prob,
                "tw_start": start,
                "tw_end": end,
                "amm": amm,
                "prep": prep,
            },
            batch_size=[bs],
        )


class DWTAModerateGenerator:
    """Batched torch reimplementation of
    common/temporal_dilemma_generator_moderate.py's
    generate_moderate_temporal_instance, so RL4CO's AM/POMO train on the SAME
    moderate temporal-dilemma curriculum used for the RL+Auction hybrid.
    The main tiered benchmark (DWTAGenerator above) was found to have almost
    no hold-vs-fire structure -- training/evaluating AM/POMO only there would
    bias the comparison toward myopic strategies (Greedy/Auction), which have
    no reason to lose on instances with no such structure to exploit."""

    def __init__(self, num_weapon=5, num_target=5, max_time=5, **kwargs):
        self.num_weapon = num_weapon
        self.num_target = num_target
        self.max_time = max_time

    def __call__(self, batch_size) -> TensorDict:
        bs = batch_size[0] if isinstance(batch_size, (list, tuple)) else batch_size
        M, N, T = self.num_weapon, self.num_target, self.max_time

        start = torch.randint(0, T, (bs, N)).float()
        base_value = torch.rand(bs, N) * 4 + 2  # U(2, 6)
        time_bonus = (start / max(T - 1, 1)) * (torch.rand(bs, N) * 3 + 2)  # U(2, 5)
        value = base_value + time_bonus  # roughly [2, 11], smoothly increasing with start time
        tw_end = torch.full((bs, N), float(T - 1))

        amm = torch.randint(1, max(2, T // 2 + 1), (bs, M)).float()
        prep = torch.randint(0, 2, (bs, M)).float()
        prob = torch.rand(bs, M, N) * 0.6 + 0.3  # U(0.3, 0.9)

        return TensorDict(
            {
                "value": value,
                "prob": prob,
                "tw_start": start,
                "tw_end": tw_end,
                "amm": amm,
                "prep": prep,
            },
            batch_size=[bs],
        )


class DWTAEnv(RL4COEnvBase):
    """DWTAP as an M*T-step sequential construction problem for RL4CO's
    native single-pointer AM/POMO decoders (see module docstring)."""

    name = "dwta"

    def __init__(self, generator: DWTAGenerator = None, generator_params: dict = {}, **kwargs):
        super().__init__(**kwargs)
        if generator is None:
            generator = DWTAGenerator(**generator_params)
        self.generator = generator
        self.M = generator.num_weapon
        self.N = generator.num_target
        self.T = generator.max_time
        self._make_spec(generator)

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        bs = batch_size[0] if isinstance(batch_size, (list, tuple)) else batch_size
        M, N = self.M, self.N

        td_reset = TensorDict(
            {
                "value": td["value"].to(device),
                "prob": td["prob"].to(device),
                "tw_start": td["tw_start"].to(device),
                "tw_end": td["tw_end"].to(device),
                "amm": td["amm"].to(device),
                "prep": td["prep"].to(device),
                "remaining_value": td["value"].to(device).clone(),
                "ammo_left": td["amm"].to(device).clone(),
                "reload_until": torch.full((bs, M), -1.0, device=device),  # last round still on cooldown
                "step_idx": torch.zeros(bs, dtype=torch.int64, device=device),  # 0..M*T-1
            },
            batch_size=[bs],
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def _current_weapon_round(td: TensorDict, M: int):
        weapon = td["step_idx"] % M
        rnd = td["step_idx"] // M
        return weapon, rnd

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """[batch, N+1] mask for the CURRENT step's weapon; last column = no-op, always legal."""
        M, N = self.M, self.N
        weapon, rnd = self._current_weapon_round(td, M)
        bs = td.batch_size[0]
        idx = torch.arange(bs, device=td.device)

        ammo_ok = td["ammo_left"][idx, weapon] > 0  # [batch]
        reload_ok = rnd.float() > td["reload_until"][idx, weapon]  # [batch]
        weapon_ok = (ammo_ok & reload_ok).unsqueeze(-1)  # [batch, 1]

        tw_ok = (td["tw_start"] <= rnd.unsqueeze(-1).float()) & (td["tw_end"] >= rnd.unsqueeze(-1).float())  # [batch, N]
        target_ok = td["remaining_value"] > 1e-9  # [batch, N] -- no point re-targeting a dead target

        legal_targets = weapon_ok & tw_ok & target_ok  # [batch, N]
        noop = torch.ones(bs, 1, dtype=torch.bool, device=td.device)
        return torch.cat([legal_targets, noop], dim=-1)

    def _step(self, td: TensorDict) -> TensorDict:
        M, N = self.M, self.N
        bs = td.batch_size[0]
        idx = torch.arange(bs, device=td.device)
        weapon, rnd = self._current_weapon_round(td, M)
        action = td["action"]  # [batch], in 0..N (N = no-op)

        is_fire = action < N
        target = torch.where(is_fire, action, torch.zeros_like(action))

        p = td["prob"][idx, weapon, target]  # [batch]
        rv = td["remaining_value"].clone()
        new_val = rv[idx, target] * (1 - p)
        rv[idx, target] = torch.where(is_fire, new_val, rv[idx, target])
        td["remaining_value"] = rv

        ammo = td["ammo_left"].clone()
        ammo[idx, weapon] = torch.where(is_fire, ammo[idx, weapon] - 1, ammo[idx, weapon])
        td["ammo_left"] = ammo

        reload_until = td["reload_until"].clone()
        reload_until[idx, weapon] = torch.where(
            is_fire, rnd.float() + td["prep"][idx, weapon], reload_until[idx, weapon]
        )
        td["reload_until"] = reload_until

        td["step_idx"] = td["step_idx"] + 1
        done = td["step_idx"] >= (M * self.T)
        td["done"] = done
        td["reward"] = torch.zeros(bs, device=td.device)  # computed at episode end via _get_reward

        # Avoid indexing past the end on the terminal step
        safe_td = td.clone()
        if done.any():
            safe_td["step_idx"] = torch.where(done, torch.zeros_like(td["step_idx"]), td["step_idx"])
        td.set("action_mask", self.get_action_mask(safe_td if done.any() else td))
        return td

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Negative remaining value (RL4CO maximizes reward; we want to
        minimize remaining value, matching SCoPE/Greedy's own objective)."""
        return -td["remaining_value"].sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """No additional check needed beyond what the action_mask already
        enforces during rollout (ammo/reload/time-window legality) -- unlike
        routing problems, there is no separate global constraint (e.g. tour
        length) to re-verify post-hoc."""
        pass

    def get_num_starts(self, td: TensorDict) -> int:
        """For POMO multistart: one forced start per TARGET (no-op excluded,
        analogous to how CVRP/OP exclude the depot from start-node choices --
        forcing the first decision to no-op wouldn't diversify anything)."""
        return self.N

    def select_start_nodes(self, td: TensorDict, num_starts: int) -> torch.Tensor:
        """POMO's standard multistart trick (force weapon 0/round 0's action
        to each of `num_starts` different values, one per rollout replica)
        assumes every start is legal -- true for TSP/CVRP-style problems
        where all nodes are always reachable, but NOT true here: a target
        may not have emerged yet (tw_start > 0), so forcing round 0 to fire
        at it would fabricate a shot at a target that doesn't exist yet.
        Instead: force target `s` only where legal for that specific
        instance (per the just-computed reset action_mask); fall back to
        no-op for instances where target `s` isn't legal at round 0. This
        keeps every forced-start trajectory physically valid, at the cost of
        some starts degenerating to the same no-op action for a given
        instance (acceptable -- still diversifies across the batch)."""
        bs = td.batch_size[0]
        legal = td["action_mask"][:, :num_starts]  # [bs, num_starts], round-0 weapon-0 legality
        starts = torch.arange(num_starts, device=td.device)
        action = starts.repeat_interleave(bs)  # [num_starts*bs], block s = start s repeated bs times
        legal_flat = legal.t().reshape(-1)  # [num_starts, bs] -> flat, matches block ordering
        noop = torch.full_like(action, self.N)
        return torch.where(legal_flat, action, noop)

    def _make_spec(self, generator: DWTAGenerator):
        M, N, T = self.M, self.N, self.T
        self.observation_spec = Composite(
            value=Unbounded(shape=(N,), dtype=torch.float32),
            prob=Unbounded(shape=(M, N), dtype=torch.float32),
            remaining_value=Unbounded(shape=(N,), dtype=torch.float32),
            action_mask=Unbounded(shape=(N + 1,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Bounded(shape=(1,), dtype=torch.int64, low=0, high=N + 1)
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)
