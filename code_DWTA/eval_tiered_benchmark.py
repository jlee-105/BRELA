"""
Unified evaluation of parallel vs sequential decoder checkpoints (5 seeds each)
across the 4-tier x 3-config held-out benchmark.

Uses common.DWTA_Simulator.Environment for both decoder types (the current,
non-stale simulator -- NOT rl_rollout.DWTA_Simulator_rollout).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import common.Dynamic_HYPER_PARAMETER as HP
import common.DWTA_Simulator as Sim
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_GNN import create_gnn_actor
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.DWTA_Simulator import Environment

N_EVAL = 10
TEST_DIR = "TEST_INSTANCE"

TIERS = {
    "Small": ["5M_5N_5T.xlsx", "5M_7N_5T.xlsx", "10M_15N_5T.xlsx"],
    "Medium": ["15M_15N_5T.xlsx", "15M_20N_5T.xlsx", "20M_30N_5T.xlsx"],
    "Large": ["30M_30N_10T.xlsx", "30M_40N_10T.xlsx", "40M_50N_10T.xlsx"],
    "Battlefield": ["50M_50N_15T.xlsx", "50M_70N_15T.xlsx", "70M_100N_15T.xlsx"],
}

SEED_DIRS = {
    "parallel": {0: "GNN_TRAIN_20260803", 1: "GNN_TRAIN_seed1", 2: "GNN_TRAIN_seed2",
                 3: "GNN_TRAIN_seed3", 4: "GNN_TRAIN_seed4"},
    "sequential": {0: "GNN_TRAIN_SEQ_20260803", 1: "GNN_TRAIN_SEQ_seed1", 2: "GNN_TRAIN_SEQ_seed2",
                   3: "GNN_TRAIN_SEQ_seed3", 4: "GNN_TRAIN_SEQ_seed4"},
}


def patch_globals(nw, nt, mt, amm, prep, cost):
    HP.NUM_WEAPONS = nw
    HP.NUM_TARGETS = nt
    HP.MAX_TIME = mt
    HP.AMM = amm
    HP.PREPARATION_TIME = prep
    HP.WEAPON_COST = cost
    Sim.AMM = amm
    Sim.PREPARATION_TIME = prep
    Sim.WEAPON_COST = cost
    # common/DWTA_Simulator.py does `from .Dynamic_HYPER_PARAMETER import *`, which
    # freezes MAX_TIME as a local name at first import -- patching HP.MAX_TIME alone
    # does not propagate. Without this, target time windows de-normalize using the
    # stale default (5) instead of this config's mt, breaking T!=5 tiers (Large/
    # Battlefield). Training's patch_hyperparameters_for_epoch already does this
    # correctly (rl/Dynamic_Sampling_GNN.py); this eval-side helper had missed it.
    Sim.MAX_TIME = mt


WASTE_SURVIVAL_THRESHOLD = 0.2  # a target below 20% of its original value counts as "nearly dead"


def _round_dispersion(target_indices):
    """target_indices: target index for every weapon that fired THIS timestep
    (no-ops excluded). Returns None if nobody fired this round (undefined)."""
    if not target_indices:
        return None
    return len(set(target_indices)) / len(target_indices)


def _round_wasteful_redundancy(target_indices, survival_before):
    """Not all redundant (same-target, same-round) fire is wasteful: concentrating
    on a still-healthy, high-value target can be the correct call. Only count a
    redundant hit as wasteful if the target was ALREADY nearly dead (low pre-round
    survival ratio) before this round's simultaneous hits were applied -- i.e. the
    extra hit(s) had little marginal value left to capture regardless of where else
    they could have gone. `survival_before[n]` = current_target_value[n] /
    original_target_value[n] at the START of this round (before this round's fires).
    Returns None if nobody fired this round."""
    if not target_indices:
        return None
    from collections import Counter
    counts = Counter(target_indices)
    wasteful = 0
    for n, k in counts.items():
        if k >= 2 and survival_before[n] < WASTE_SURVIVAL_THRESHOLD:
            wasteful += (k - 1)  # first hit on a target is never "redundant"
    return wasteful / len(target_indices)


@torch.no_grad()
def eval_instance_parallel(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    total_fires, total_decisions = 0, 0
    round_dispersions = []
    round_waste = []
    t0 = time.time()
    for _ in range(mt):
        survival_before = (env.current_target_value[0, 0, :nt] / env.original_target_value[0, 0, :nt]).tolist()
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        action = policy.argmax(dim=-1)  # [1, 1, W]
        flat_action = action.view(-1)
        fired_targets = flat_action[flat_action < nt].tolist()  # this round's targets, one per firing weapon
        round_dispersions.append(_round_dispersion(fired_targets))
        round_waste.append(_round_wasteful_redundancy(fired_targets, survival_before))
        total_fires += len(fired_targets)
        total_decisions += nw
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    elapsed = time.time() - t0

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return _pack_result(init_value, remaining, total_fires, total_decisions, elapsed, round_dispersions, round_waste)


@torch.no_grad()
def eval_instance_sequential(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    total_fires, total_decisions = 0, 0
    round_dispersions = []
    round_waste = []
    t0 = time.time()
    for _ in range(mt):
        survival_before = (env.current_target_value[0, 0, :nt] / env.original_target_value[0, 0, :nt]).tolist()
        fired_targets = []  # this round's targets, one per firing weapon (order of decision, not weapon id)
        for _ in range(nw):
            mask = env.mask.clone()
            if (mask > 0).any():
                policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
                flat = policy.view(-1, nw * nt + 1)
                action = flat.argmax(dim=1).view(1, 1)
                action_idx = action.item()
                if action_idx < nw * nt:
                    total_fires += 1
                    fired_targets.append(action_idx % nt)  # target_index = action_idx % num_targets
            else:
                action = torch.tensor([[nw * nt]], device=DEVICE)
            total_decisions += 1
            env.update_internal_variables(selected_action=action)
        round_dispersions.append(_round_dispersion(fired_targets))
        round_waste.append(_round_wasteful_redundancy(fired_targets, survival_before))
        env.time_update()
    elapsed = time.time() - t0

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return _pack_result(init_value, remaining, total_fires, total_decisions, elapsed, round_dispersions, round_waste)


def _pack_result(init_value, remaining, total_fires, total_decisions, elapsed, round_dispersions, round_waste):
    remaining_norm = remaining / max(init_value, 1e-8)
    valid_dispersions = [d for d in round_dispersions if d is not None]
    valid_waste = [w for w in round_waste if w is not None]
    return {
        "init_value": init_value,
        "remaining_value": remaining,
        "objective": remaining_norm,
        "destruction": 1.0 - remaining_norm,
        "fire_rate": total_fires / max(total_decisions, 1),
        "time_s": elapsed,
        # Per-timestep target-dispersion ratio (distinct targets hit / weapons that
        # fired, that round), averaged over rounds with >=1 fire. 1.0 = every shot
        # this round hit a different target; low = piling onto the same target
        # within a single simultaneous-decision round (see brerla_open_todos memory).
        "dispersion": (sum(valid_dispersions) / len(valid_dispersions)) if valid_dispersions else None,
        "redundant_fire_rate": (1.0 - sum(valid_dispersions) / len(valid_dispersions)) if valid_dispersions else None,
        # Fraction of this round's fires that were a 2nd+ hit on a target whose
        # PRE-round survival ratio was already below WASTE_SURVIVAL_THRESHOLD --
        # i.e. redundant AND the target was already nearly dead. Distinguishes
        # wasteful piling-on from justified concentration on a still-valuable target.
        "wasteful_redundancy": (sum(valid_waste) / len(valid_waste)) if valid_waste else None,
    }


def load_actor(decoder_type, seed):
    ckpt_dir = SEED_DIRS[decoder_type][seed]
    path = f"result/{ckpt_dir}/CheckPoint_epoch00200/GNN_ACTOR_state_dic.pt"
    actor = (create_gnn_actor() if decoder_type == "parallel" else create_gnn_actor_sequential()).to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    actor.eval()
    return actor


RESULTS_CSV = "result/tiered_benchmark_results.csv"
PROGRESS_LOG = "result/tiered_benchmark_progress.log"


def _log(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(msg + "\n")


def _append_row(row_out, wrote_header):
    df = pd.DataFrame([row_out])
    df.to_csv(RESULTS_CSV, mode="a", header=not wrote_header, index=False)


def run():
    eval_fn = {"parallel": eval_instance_parallel, "sequential": eval_instance_sequential}
    open(PROGRESS_LOG, "w").close()
    if __import__("os").path.exists(RESULTS_CSV):
        __import__("os").remove(RESULTS_CSV)
    wrote_header = False

    total_jobs = 2 * 5 * sum(len(files) for files in TIERS.values())
    job_i = 0

    for decoder_type in ["parallel", "sequential"]:
        for seed in range(5):
            _log(f"=== {decoder_type} seed {seed} ===")
            actor = load_actor(decoder_type, seed)

            for tier, files in TIERS.items():
                for fname in files:
                    job_i += 1
                    df = pd.read_excel(f"{TEST_DIR}/{fname}")
                    nw, nt, mt = int(df.iloc[0]["M"]), int(df.iloc[0]["N"]), int(df.iloc[0]["T"])

                    metrics = []
                    config_t0 = time.time()
                    for i in range(min(N_EVAL, len(df))):
                        row = df.iloc[i]
                        V = json.loads(row["V"])
                        P = np.array(json.loads(row["P"]))
                        TW = json.loads(row["TW"])
                        amm = json.loads(row["AMM"])
                        prep = json.loads(row["PREP"])
                        cost = json.loads(row["COST"])
                        result = eval_fn[decoder_type](actor, V, P, TW, nw, nt, mt, amm, prep, cost)
                        metrics.append(result)
                        _log(f"    [{job_i}/{total_jobs}] {fname} instance {i+1}/{min(N_EVAL, len(df))} "
                             f"done in {result['time_s']:.2f}s")

                    obj = np.array([m["objective"] for m in metrics])
                    destr = np.array([m["destruction"] for m in metrics])
                    fire = np.array([m["fire_rate"] for m in metrics])
                    tsec = np.array([m["time_s"] for m in metrics])
                    disp = np.array([m["dispersion"] for m in metrics if m["dispersion"] is not None])
                    waste = np.array([m["wasteful_redundancy"] for m in metrics if m["wasteful_redundancy"] is not None])

                    row_out = {
                        "decoder": decoder_type, "seed": seed, "tier": tier,
                        "config": fname.replace(".xlsx", ""), "M": nw, "N": nt, "T": mt,
                        "n_instances": len(metrics),
                        "objective_mean": obj.mean(), "objective_std": obj.std(),
                        "destruction_mean": destr.mean(), "destruction_std": destr.std(),
                        "fire_rate_mean": fire.mean(),
                        "time_s_mean": tsec.mean(), "time_s_std": tsec.std(),
                        "dispersion_mean": disp.mean() if len(disp) else None,
                        "dispersion_std": disp.std() if len(disp) else None,
                        "wasteful_redundancy_mean": waste.mean() if len(waste) else None,
                        "wasteful_redundancy_std": waste.std() if len(waste) else None,
                    }
                    _append_row(row_out, wrote_header)
                    wrote_header = True
                    disp_str = f"{disp.mean():.3f}" if len(disp) else "n/a"
                    waste_str = f"{waste.mean():.3f}" if len(waste) else "n/a"
                    _log(f"  [{job_i}/{total_jobs}] {fname}: obj={obj.mean():.4f}+-{obj.std():.4f} "
                         f"destr={destr.mean():.2%} disp={disp_str} waste={waste_str} time={tsec.mean():.3f}s "
                         f"(config took {time.time()-config_t0:.1f}s)")

    _log(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    run()
