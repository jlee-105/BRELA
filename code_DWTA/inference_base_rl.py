"""
Base RL Inference: actor argmax only, no search.
Evaluates on all test instances from TEST_INSTANCE/ Excel files.
Supports ExIt_05 checkpoint via --actor_path or auto-discovery.
"""
import argparse
import glob
import sys
import os
import time
import json
import ast
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from common.TORCH_OBJECTS import *
import common.Dynamic_HYPER_PARAMETER as HP
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_GNN import create_gnn_actor
from rl_rollout.DWTA_Simulator_rollout import Environment

EVAL_ALPHA = 1.0
EXIT_05_PATTERN = "result/EXIT_TRAIN_*/ExIt_05_epoch00200/GNN_ACTOR_state_dic.pt"
TEST_DIR = "TEST_INSTANCE"

TEST_FILES = [
    "5M_5N_5T.xlsx",
    "5M_7N_5T.xlsx",
    "7M_5N_5T.xlsx",
    "15M_15N_5T.xlsx",
    "15M_20N_5T.xlsx",
    "20M_20N_5T.xlsx",
    "30M_30N_5T.xlsx",
    "30M_30N_10T.xlsx",
    "50M_50N_10T.xlsx",
]

SCALE_MAP = {
    "5M_5N_5T": "Small", "5M_7N_5T": "Small", "7M_5N_5T": "Small",
    "15M_15N_5T": "Medium", "15M_20N_5T": "Medium", "20M_20N_5T": "Medium",
    "30M_30N_5T": "Large", "30M_30N_10T": "Large", "50M_50N_10T": "Large",
}


def resolve_actor_path(actor_path):
    """Resolve actor checkpoint path. If None, glob for latest ExIt_05."""
    if actor_path is not None:
        if not os.path.isfile(actor_path):
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
        return actor_path
    candidates = glob.glob(EXIT_05_PATTERN)
    if not candidates:
        raise FileNotFoundError(
            f"No ExIt_05 checkpoint found (searched {EXIT_05_PATTERN}). "
            "Run ExIt training or pass --actor_path path/to/GNN_ACTOR_state_dic.pt"
        )
    # Use most recently modified
    return max(candidates, key=lambda p: os.path.getmtime(p))


def patch_globals(nw, nt, mt, amm, prep, cost):
    """Patch global hyperparameters for this instance."""
    HP.NUM_WEAPONS = nw
    HP.NUM_TARGETS = nt
    HP.MAX_TIME = mt
    HP.AMM = amm
    HP.PREPARATION_TIME = prep
    HP.WEAPON_COST = cost

    import rl_rollout.DWTA_Simulator_rollout as RSim
    RSim.NUM_WEAPONS = nw
    RSim.NUM_TARGETS = nt
    RSim.MAX_TIME = mt
    RSim.AMM = amm
    RSim.PREPARATION_TIME = prep
    RSim.WEAPON_COST = cost


@torch.no_grad()
def evaluate_instance(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    """Run actor greedy on a single instance."""
    patch_globals(nw, nt, mt, amm, prep, cost)

    ae, wtp = input_generation(
        NUM_WEAPON=nw, NUM_TARGET=nt,
        value=V, prob=P, TW=TW,
        max_time=mt, batch_size=1, alpha=EVAL_ALPHA,
    )
    ae = ae.unsqueeze(1)
    wtp = wtp.unsqueeze(1)

    env = Environment(ae, wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    t0 = time.time()
    total_fires = 0

    for _ in range(mt):
        for _ in range(nw):
            mask = env.mask.clone()
            if (mask > 0).any():
                policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                flat = policy.view(-1, nw * nt + 1)
                action = flat.argmax(dim=1).view(1, 1)
                if action.item() < nw * nt:
                    total_fires += 1
            else:
                action = torch.tensor([[nw * nt]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        env.time_update()

    elapsed = time.time() - t0
    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    total_cost = env.total_cost.sum().item()
    max_cost = env.max_possible_cost

    remaining_norm = remaining / max(init_value, 1e-8)
    cost_norm = total_cost / max(max_cost, 1e-8)
    objective = EVAL_ALPHA * remaining_norm + (1 - EVAL_ALPHA) * cost_norm
    destruction = 1.0 - remaining_norm

    return {
        'init_value': round(init_value, 4),
        'remaining_value': round(remaining, 4),
        'remaining_norm': round(remaining_norm, 4),
        'total_cost': round(total_cost, 2),
        'cost_norm': round(cost_norm, 4),
        'objective': round(objective, 4),
        'destruction': round(destruction, 4),
        'fires': total_fires,
        'time_s': round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Base RL inference (argmax only) on test instances.")
    parser.add_argument(
        "--actor_path",
        type=str,
        default=None,
        help="Path to GNN_ACTOR_state_dic.pt. If omitted, uses latest ExIt_05 checkpoint under result/EXIT_TRAIN_*/ExIt_05_epoch00200/.",
    )
    args = parser.parse_args()

    actor_path = resolve_actor_path(args.actor_path)
    print(f"Device: {DEVICE}")
    print(f"Model: {actor_path}")
    print(f"Alpha: {EVAL_ALPHA}")
    print()

    actor = create_gnn_actor().to(DEVICE)
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE, weights_only=False))
    actor.eval()
    print("Model loaded.\n")

    all_results = []

    for fname in TEST_FILES:
        fpath = os.path.join(TEST_DIR, fname)
        if not os.path.exists(fpath):
            print(f"SKIP: {fpath} not found")
            continue

        df = pd.read_excel(fpath)
        label = fname.replace(".xlsx", "")
        scale = SCALE_MAP.get(label, "?")
        nw = int(df.iloc[0]['M'])
        nt = int(df.iloc[0]['N'])
        mt = int(df.iloc[0]['T'])

        print(f"--- {label} ({scale}) | {nw}W x {nt}T x {mt}T | {len(df)} instances ---")

        file_results = []
        for i in range(len(df)):
            row = df.iloc[i]
            V = json.loads(row['V'])
            P = np.array(json.loads(row['P']))
            TW = json.loads(row['TW'])
            amm = json.loads(row['AMM'])
            prep = json.loads(row['PREP'])
            cost_list = json.loads(row['COST'])

            result = evaluate_instance(actor, V, P, TW, nw, nt, mt, amm, prep, cost_list)
            result['size'] = label
            result['scale'] = scale
            result['instance'] = i
            file_results.append(result)

        all_results.extend(file_results)

        objs = [r['objective'] for r in file_results]
        destrs = [r['destruction'] for r in file_results]
        costs = [r['cost_norm'] for r in file_results]
        times = [r['time_s'] for r in file_results]

        print(f"  Objective:    {np.mean(objs):.4f} (+/- {np.std(objs):.4f})")
        print(f"  Destruction:  {np.mean(destrs):.2%} (+/- {np.std(destrs):.2%})")
        print(f"  Cost (norm):  {np.mean(costs):.4f} (+/- {np.std(costs):.4f})")
        print(f"  Time:         {np.mean(times):.4f}s")
        print()

    results_df = pd.DataFrame(all_results)
    out_path = "result/rollout_rl_trained_base_inference_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"All results saved to {out_path}")

    print("\n=== SUMMARY ===")
    summary = results_df.groupby(['scale', 'size']).agg({
        'objective': ['mean', 'std'],
        'destruction': ['mean', 'std'],
        'cost_norm': ['mean', 'std'],
        'time_s': 'mean',
    }).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
