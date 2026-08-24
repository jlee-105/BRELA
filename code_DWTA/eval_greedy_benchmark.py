"""
Classical maximum-marginal-return greedy baseline (Kolitz & Eckler 1988-style),
evaluated on the same 4-tier x 12-config x 10-instance held-out benchmark as
eval_tiered_benchmark.py, using the same common.DWTA_Simulator.Environment so
ammo/reload/time-window constraints are handled identically to SCoPE/Sequential
(not reimplemented). No trained parameters -> no seed loop, same as SCIP.

Per round, repeatedly assigns the single (weapon, target) pair with the largest
expected marginal value destroyed among not-yet-assigned weapons and legal
targets -- this is exactly the classical greedy algorithm for monotone
submodular maximization under a partition matroid analyzed in the paper's
Proposition 2 (Section V, "Sequential decoding as matroid greedy").
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.TORCH_OBJECTS import DEVICE
from eval_tiered_benchmark import (
    N_EVAL, TEST_DIR, TIERS, patch_globals,
    _pack_result, _round_dispersion, _round_wasteful_redundancy,
)

RESULTS_CSV = "result/greedy_benchmark_results.csv"
PROGRESS_LOG = "result/greedy_benchmark_progress.log"


def _log(msg):
    print(msg, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(msg + "\n")


def _greedy_round_action(env, nw, nt):
    """One round of max-marginal-return greedy. Returns an action tensor [1,1,W]
    in the same format update_internal_variables_parallel expects (target index,
    or nt for no-op)."""
    mask = env.mask_per_weapon[0, 0, :, :nt].cpu().numpy() > 0  # [W, N] legality
    prob = env.weapon_to_target_prob[0, 0, :nw, :nt].cpu().numpy()  # [W, N] = P[m,n]
    remaining = env.current_target_value[0, 0, :nt].cpu().numpy().copy()  # mutated within this round

    action = np.full(nw, nt, dtype=np.int64)  # default: no-op
    assigned = np.zeros(nw, dtype=bool)

    while True:
        gain = remaining[None, :] * prob  # expected value destroyed per (weapon,target)
        gain = np.where(mask & ~assigned[:, None], gain, -np.inf)
        if not np.isfinite(gain).any():
            break
        m, n = np.unravel_index(np.argmax(gain), gain.shape)
        action[m] = n
        assigned[m] = True
        remaining[n] *= (1 - prob[m, n])  # diminishing returns for further hits on n this round

    return torch.tensor(action, device=DEVICE).view(1, 1, nw)


@torch.no_grad()
def eval_instance_greedy(V, P, TW, nw, nt, mt, amm, prep, cost):
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
        action = _greedy_round_action(env, nw, nt)
        flat_action = action.view(-1)
        fired_targets = flat_action[flat_action < nt].tolist()
        round_dispersions.append(_round_dispersion(fired_targets))
        round_waste.append(_round_wasteful_redundancy(fired_targets, survival_before))
        total_fires += len(fired_targets)
        total_decisions += nw
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    elapsed = time.time() - t0

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return _pack_result(init_value, remaining, total_fires, total_decisions, elapsed, round_dispersions, round_waste)


def _append_row(row_out, wrote_header):
    df = pd.DataFrame([row_out])
    df.to_csv(RESULTS_CSV, mode="a", header=not wrote_header, index=False)


def run():
    open(PROGRESS_LOG, "w").close()
    import os
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)
    wrote_header = False

    total_jobs = sum(len(files) for files in TIERS.values())
    job_i = 0

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
                result = eval_instance_greedy(V, P, TW, nw, nt, mt, amm, prep, cost)
                metrics.append(result)
                _log(f"    [{job_i}/{total_jobs}] {fname} instance {i+1}/{min(N_EVAL, len(df))} "
                     f"done in {result['time_s']:.3f}s")

            obj = np.array([m["objective"] for m in metrics])
            destr = np.array([m["destruction"] for m in metrics])
            fire = np.array([m["fire_rate"] for m in metrics])
            tsec = np.array([m["time_s"] for m in metrics])
            disp = np.array([m["dispersion"] for m in metrics if m["dispersion"] is not None])
            waste = np.array([m["wasteful_redundancy"] for m in metrics if m["wasteful_redundancy"] is not None])

            row_out = {
                "tier": tier, "config": fname.replace(".xlsx", ""), "M": nw, "N": nt, "T": mt,
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
            _log(f"  [{job_i}/{total_jobs}] {fname}: obj={obj.mean():.4f}+-{obj.std():.4f} "
                 f"destr={destr.mean():.2%} disp={disp_str} time={tsec.mean():.3f}s "
                 f"(config took {time.time()-config_t0:.1f}s)")

    _log(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    run()
