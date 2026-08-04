"""
Systematic Experiment Runner for DWTA
Evaluates all methods across constraint levels and cost settings.

Usage:
    python run_experiments.py --methods all --size 5x5x5
    python run_experiments.py --methods greedy,rl_greedy --constraints strong --cost balanced
"""
import os
import sys
import time
import json
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from common.TORCH_OBJECTS import DEVICE
import common.Dynamic_HYPER_PARAMETER as HP
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_GNN import create_gnn_actor, create_gnn_critic
from rl_rollout.DWTA_Simulator_rollout import Environment
from rl_rollout.BEAM_WITH_SIMULATION_TRUNC_CRITIC import (
    Beam_Search_Trunc, batch_dimension_resize_trunc,
)


# ======================================================================
# Constraint presets
# ======================================================================
CONSTRAINT_PRESETS = {
    'none': {
        'prep_time': [0] * 50,
        'amm': [999] * 50,
        'tw_max_start': 0,       # all targets active from t=0
    },
    'weak': {
        'prep_time': [1] * 50,
        'amm': [3] * 50,
        'tw_max_start': 0.25,    # start in [0, horizon/4]
    },
    'strong': {
        'prep_time': [1, 2] * 25,
        'amm': [2] * 50,
        'tw_max_start': 0.5,     # start in [0, horizon/2]
    },
}

COST_PRESETS = {
    'none':     1.0,   # COST_WEIGHT=1.0 means ignore cost
    'balanced': 0.5,
    'focused':  0.2,
}


# ======================================================================
# Patch hyperparameters for a given experiment config
# ======================================================================
def _apply_config(num_weapons, num_targets, max_time, constraint, cost_setting):
    preset = CONSTRAINT_PRESETS[constraint]
    HP.NUM_WEAPONS = num_weapons
    HP.NUM_TARGETS = num_targets
    HP.MAX_TIME = max_time
    HP.PREPARATION_TIME = preset['prep_time']
    HP.AMM = preset['amm']
    HP.COST_WEIGHT = COST_PRESETS[cost_setting]

    # Also patch simulator module globals
    import common.DWTA_Simulator as Sim
    for attr in ('NUM_WEAPONS', 'NUM_TARGETS', 'MAX_TIME', 'PREPARATION_TIME', 'AMM', 'COST_WEIGHT'):
        setattr(Sim, attr, getattr(HP, attr))
    import rl_rollout.DWTA_Simulator_rollout as RSim
    for attr in ('NUM_WEAPONS', 'NUM_TARGETS', 'MAX_TIME', 'PREPARATION_TIME', 'AMM', 'COST_WEIGHT'):
        if hasattr(RSim, attr):
            setattr(RSim, attr, getattr(HP, attr))


# ======================================================================
# Method runners — each returns (remaining_value, total_cost, solve_time)
# ======================================================================
@torch.no_grad()
def run_greedy(instance_ae, instance_wtp, nw, nt, mt):
    """Myopic greedy: always fire at target with highest value*prob."""
    ae = instance_ae.unsqueeze(1).clone()
    wtp = instance_wtp.unsqueeze(1).clone()
    env = Environment(ae, wtp, max_time=mt)

    t0 = time.time()
    for t in range(mt):
        for w in range(nw):
            mask = env.mask.clone()
            # Simple heuristic: highest expected damage
            policy_vals = torch.zeros(nw * nt + 1, device=DEVICE)
            cv = env.current_target_value.reshape(1, 1, nw, nt)
            for wi in range(nw):
                for ti in range(nt):
                    idx = wi * nt + ti
                    policy_vals[idx] = cv[0, 0, wi, ti] * wtp[0, 0, wi, ti]
            policy_vals[-1] = -1e9  # discourage no-action unless forced
            policy_vals = policy_vals * mask.view(-1)
            action_idx = policy_vals.argmax().item()
            if mask.view(-1).sum() <= 1:
                action_idx = nw * nt  # no-action
            action = torch.tensor([[action_idx]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        env.time_update()
    elapsed = time.time() - t0

    rem = env.current_target_value[:, :, 0:nt].sum().item()
    cost = env.total_cost.sum().item()
    return rem, cost, elapsed


@torch.no_grad()
def run_rl_greedy(actor, instance_ae, instance_wtp, nw, nt, mt):
    """RL greedy: argmax of actor policy at each step."""
    ae = instance_ae.unsqueeze(1).clone()
    wtp = instance_wtp.unsqueeze(1).clone()
    env = Environment(ae, wtp, max_time=mt)

    t0 = time.time()
    for t in range(mt):
        for w in range(nw):
            mask = env.mask.clone()
            if (mask > 0).any():
                policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                action = policy.view(-1, nw * nt + 1).argmax(dim=1).view(1, 1)
            else:
                action = torch.tensor([[nw * nt]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        env.time_update()
    elapsed = time.time() - t0

    rem = env.current_target_value[:, :, 0:nt].sum().item()
    cost = env.total_cost.sum().item()
    return rem, cost, elapsed


@torch.no_grad()
def run_rl_search(actor, critic, instance_ae, instance_wtp, nw, nt, mt, beta=2, to_go_weight=1.0, use_greedy_only=False):
    """RL + beam search rollout. beta=None or <0 => full rollout; use_greedy_only => argmax only."""
    ae = instance_ae.unsqueeze(1).expand(1, HP.VAL_PARA, nt * nw + 1, HP.NUM_FEATURES).contiguous()
    wtp = instance_wtp.unsqueeze(1).expand(1, HP.VAL_PARA, nw, nt).contiguous()
    env = Environment(ae, wtp, max_time=mt)

    t0 = time.time()
    for t in range(mt):
        for w in range(nw):
            mask = env.mask.clone()
            if (mask > 0).any():
                beam = Beam_Search_Trunc(
                    env=env, actor=actor, value=critic,
                    available_actions=mask, beta=beta, to_go_weight=to_go_weight,
                    use_greedy_only=use_greedy_only,
                )
                beam.reset()
                expanded = beam.expand_actions()
                b_idx, g_idx = beam.do_beam_simulation(
                    possible_node_index=expanded, time=t, w_index=w,
                )
                selected_action = g_idx.unsqueeze(1)
                env = batch_dimension_resize_trunc(env=env, batch_index=b_idx, group_index=g_idx)
            else:
                selected_action = torch.tensor(
                    [nw * nt], device=DEVICE
                )[None, :].expand(1, HP.VAL_PARA)
            env.update_internal_variables(selected_action=selected_action)
        env.time_update()
    elapsed = time.time() - t0

    obj = env.current_target_value[:, :, 0:nt].sum(2)
    rem = obj.min().item()
    cost = env.total_cost.min().item()
    return rem, cost, elapsed


# ======================================================================
# Main experiment loop
# ======================================================================
def run_experiment_matrix(
    methods, sizes, constraints, cost_settings,
    n_instances=5, actor_path=None, critic_path=None,
    rollout_beta=2, rollout_to_go_weight=1.0, rollout_use_greedy_only=False,
):
    actor, critic = None, None
    if any(m in methods for m in ('rl_greedy', 'rl_search')):
        actor = create_gnn_actor().to(DEVICE)
        critic = create_gnn_critic().to(DEVICE)
        if actor_path and os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location=DEVICE, weights_only=False))
            print(f"Loaded actor from {actor_path}")
        if critic_path and os.path.exists(critic_path):
            critic.load_state_dict(torch.load(critic_path, map_location=DEVICE, weights_only=False))
            print(f"Loaded critic from {critic_path}")
        actor.eval()
        critic.eval()

    results = []

    for size_str in sizes:
        nw, nt, mt = [int(x) for x in size_str.split('x')]

        for constraint in constraints:
            for cost_setting in cost_settings:
                _apply_config(nw, nt, mt, constraint, cost_setting)

                alpha = COST_PRESETS[cost_setting]
                for inst_idx in range(n_instances):
                    ae, wtp = input_generation(
                        NUM_WEAPON=nw, NUM_TARGET=nt,
                        value=None, prob=None, TW=None,
                        max_time=mt, batch_size=1,
                        alpha=alpha,
                    )
                    init_value = (ae[:, :-1, HP.TARGET_VALUE_INDEX] * HP.MAX_TARGET_VALUE).sum().item()

                    for method in methods:
                        if method == 'greedy':
                            rem, cost, elapsed = run_greedy(ae, wtp, nw, nt, mt)
                        elif method == 'rl_greedy':
                            rem, cost, elapsed = run_rl_greedy(actor, ae, wtp, nw, nt, mt)
                        elif method == 'rl_search':
                            rem, cost, elapsed = run_rl_search(
                                actor, critic, ae, wtp, nw, nt, mt,
                                beta=rollout_beta, to_go_weight=rollout_to_go_weight,
                                use_greedy_only=rollout_use_greedy_only,
                            )
                        else:
                            continue

                        destr = 1.0 - rem / max(init_value, 1e-8)
                        results.append({
                            'size': size_str,
                            'constraint': constraint,
                            'cost_setting': cost_setting,
                            'cost_weight': COST_PRESETS[cost_setting],
                            'instance': inst_idx,
                            'method': method,
                            'init_value': round(init_value, 4),
                            'remaining_value': round(rem, 4),
                            'destruction_ratio': round(destr, 4),
                            'total_cost': round(cost, 2),
                            'solve_time_s': round(elapsed, 4),
                        })
                        print(
                            f"  {size_str} | {constraint:6s} | cost={cost_setting:8s} | "
                            f"inst={inst_idx} | {method:12s} | "
                            f"rem={rem:8.2f} cost={cost:8.1f} destr={destr:.2%} t={elapsed:.2f}s"
                        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="DWTA experiment runner")
    parser.add_argument('--methods', type=str, default='greedy,rl_greedy,rl_search',
                        help='Comma-separated methods (greedy,rl_greedy,rl_search,all)')
    parser.add_argument('--sizes', type=str, default='5x5x5',
                        help='Comma-separated sizes like 5x5x5,10x10x5')
    parser.add_argument('--constraints', type=str, default='none,weak,strong',
                        help='Comma-separated constraint levels')
    parser.add_argument('--costs', type=str, default='none,balanced,focused',
                        help='Comma-separated cost settings')
    parser.add_argument('--n_instances', type=int, default=5)
    parser.add_argument('--actor_path', type=str, default=None)
    parser.add_argument('--critic_path', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--rollout_beta', type=int, default=2,
                        help='Rollout depth for rl_search; -1 = full rollout to end')
    parser.add_argument('--rollout_to_go_weight', type=float, default=1.0)
    parser.add_argument('--rollout_greedy_only', action='store_true',
                        help='Use only greedy policy in rollout (argmax only)')
    args = parser.parse_args()

    methods = args.methods.split(',')
    if 'all' in methods:
        methods = ['greedy', 'rl_greedy', 'rl_search']

    sizes = args.sizes.split(',')
    constraints = args.constraints.split(',')
    cost_settings = args.costs.split(',')

    print(f"Device: {DEVICE}")
    print(f"Methods: {methods}")
    print(f"Sizes: {sizes}")
    print(f"Constraints: {constraints}")
    print(f"Cost settings: {cost_settings}")
    print(f"Instances per config: {args.n_instances}")
    print()

    rollout_beta = None if args.rollout_beta < 0 else args.rollout_beta
    df = run_experiment_matrix(
        methods=methods,
        sizes=sizes,
        constraints=constraints,
        cost_settings=cost_settings,
        n_instances=args.n_instances,
        actor_path=args.actor_path,
        critic_path=args.critic_path,
        rollout_beta=rollout_beta,
        rollout_to_go_weight=args.rollout_to_go_weight,
        rollout_use_greedy_only=args.rollout_greedy_only,
    )

    out_path = args.output or f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n=== SUMMARY ===")
    summary = df.groupby(['size', 'constraint', 'cost_setting', 'method']).agg({
        'remaining_value': 'mean',
        'destruction_ratio': 'mean',
        'total_cost': 'mean',
        'solve_time_s': 'mean',
    }).round(4)
    print(summary.to_string())


if __name__ == '__main__':
    main()
