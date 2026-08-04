"""
Multi-Policy Rollout Module for DWTA
Implements three base policies for Bertsekas-style multi-heuristic rollout:
  1. Greedy: argmax of actor output
  2. Stochastic: sample from actor distribution
  3. Urgency-aware: boost targets whose time window is about to close
"""
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import DEVICE


def greedy_action(actor, env):
    """Select action with highest probability (argmax)."""
    policy, _ = actor(
        env.assignment_encoding, env.weapon_to_target_prob, env.mask
    )
    action = policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).argmax(dim=1)
    return action.view(env.assignment_encoding.size(0), env.assignment_encoding.size(1))


def stochastic_action(actor, env):
    """Sample action from the actor's probability distribution."""
    policy, _ = actor(
        env.assignment_encoding, env.weapon_to_target_prob, env.mask
    )
    flat = policy.view(-1, NUM_WEAPONS * NUM_TARGETS + 1)
    action = torch.multinomial(flat.clamp_min(1e-8), 1).squeeze(-1)
    return action.view(env.assignment_encoding.size(0), env.assignment_encoding.size(1))


def _compute_urgency_bonus(env):
    """
    Compute per-action urgency bonus based on how close each target's
    time window is to closing.  Targets about to disappear get a large bonus
    so the urgency policy prioritises them.

    Returns tensor of shape [batch, para, NUM_WEAPONS * NUM_TARGETS + 1].
    """
    batch_size = env.assignment_encoding.size(0)
    para_size = env.assignment_encoding.size(1)

    time_remaining = (env.target_end_time - env.clock).float()
    max_remaining = float(MAX_TIME)
    # 0 = target about to vanish (high urgency), 1 = plenty of time
    safety = (time_remaining / max(max_remaining, 1.0)).clamp(0.0, 1.0)
    urgency = 1.0 - safety  # high when time window almost closed

    # Expand to [batch, para, W, T] then flatten to action dim
    urgency_wt = urgency.unsqueeze(2).expand(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
    urgency_flat = urgency_wt.reshape(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS)

    # Append zero bonus for no-action
    no_action_bonus = torch.zeros(batch_size, para_size, 1, device=DEVICE)
    return torch.cat([urgency_flat, no_action_bonus], dim=-1)


def urgency_action(actor, env):
    """Greedy action biased toward targets whose time window is closing."""
    policy, _ = actor(
        env.assignment_encoding, env.weapon_to_target_prob, env.mask
    )
    bonus = _compute_urgency_bonus(env)
    adjusted = policy * (1.0 + bonus)
    # Re-mask invalid actions
    adjusted = adjusted * env.mask
    action = adjusted.view(-1, NUM_WEAPONS * NUM_TARGETS + 1).argmax(dim=1)
    return action.view(env.assignment_encoding.size(0), env.assignment_encoding.size(1))


# Registry for easy iteration
ROLLOUT_POLICIES = {
    'greedy': greedy_action,
    'stochastic': stochastic_action,
    'urgency': urgency_action,
}


def get_action_by_policy(policy_name, actor, env):
    """Dispatch to the named rollout policy."""
    return ROLLOUT_POLICIES[policy_name](actor, env)
