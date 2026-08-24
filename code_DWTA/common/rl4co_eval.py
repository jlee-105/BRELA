"""Progress-eval helper for RL4CO baselines (AM/POMO): loads the SAME fixed
TEST_INSTANCE files (and their pre-computed SCIP-optimal reference) used by
Greedy/Sequential/Parallel evaluation (see eval_tiered_benchmark.py,
eval_scip_benchmark.py), so AM/POMO training progress is tracked against
ground truth on an apples-to-apples basis rather than only its own reward.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from tensordict.tensordict import TensorDict

_HERE = os.path.dirname(__file__)
TEST_DIR = os.path.join(_HERE, "..", "TEST_INSTANCE")
RESULT_DIR = os.path.join(_HERE, "..", "result")
N_EVAL = 10  # same instances (rows 0..9) used everywhere else in this codebase

# (M, N, T) -> (test_instance_xlsx, scip_result_csv)
_SCIP_LOOKUP = {
    (5, 5, 5): ("5M_5N_5T.xlsx", "scip_5M_5N_5T_120s.csv"),
    (5, 7, 5): ("5M_7N_5T.xlsx", "scip_5M_7N_5T_600s.csv"),
    (10, 15, 5): ("10M_15N_5T.xlsx", "scip_10M_15N_5T_600s.csv"),
    (15, 15, 5): ("15M_15N_5T.xlsx", "scip_15M_15N_5T_600s.csv"),
    (15, 20, 5): ("15M_20N_5T.xlsx", "scip_15M_20N_5T_600s.csv"),
    (20, 30, 5): ("20M_30N_5T.xlsx", "scip_20M_30N_5T_600s.csv"),
    (30, 30, 10): ("30M_30N_10T.xlsx", "scip_30M_30N_10T_600s.csv"),
    (30, 40, 10): ("30M_40N_10T.xlsx", "scip_30M_40N_10T_600s.csv"),
    (40, 50, 10): ("40M_50N_10T.xlsx", "scip_40M_50N_10T_600s.csv"),
    (50, 50, 15): ("50M_50N_15T.xlsx", "scip_50M_50N_15T_600s.csv"),
    (50, 70, 15): ("50M_70N_15T.xlsx", "scip_50M_70N_15T_600s.csv"),
    (70, 100, 15): ("70M_100N_15T.xlsx", "scip_70M_100N_15T_600s.csv"),
}

# Precomputed Greedy reference (result/full_12config_scip_parallel_greedy.csv),
# for the same instances, so progress logs can show SCIP / Greedy / AM side by side.
_GREEDY_LOOKUP = {
    (5, 5, 5): 0.1591184913767322,
    (5, 7, 5): 0.309966218494326,
}


def load_fixed_instances_as_td(M, N, T, device):
    """Returns (td, scip_mean, greedy_mean, n) for the fixed TEST_INSTANCE
    file matching (M, N, T), or (None, None, None, None) if no such fixed
    file / SCIP reference exists for this config."""
    key = (M, N, T)
    if key not in _SCIP_LOOKUP:
        return None, None, None, None
    fname, scip_csv = _SCIP_LOOKUP[key]
    fpath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(fpath):
        return None, None, None, None

    df = pd.read_excel(fpath)
    n = min(N_EVAL, len(df))
    value = torch.zeros(n, N)
    prob = torch.zeros(n, M, N)
    tw_start = torch.zeros(n, N)
    tw_end = torch.zeros(n, N)
    amm = torch.zeros(n, M)
    prep = torch.zeros(n, M)
    for i in range(n):
        row = df.iloc[i]
        V = json.loads(row["V"])
        P = np.array(json.loads(row["P"]))
        TW = json.loads(row["TW"])
        A = json.loads(row["AMM"])
        W = json.loads(row["PREP"])
        value[i] = torch.tensor(V, dtype=torch.float32)
        prob[i] = torch.tensor(P, dtype=torch.float32)
        tw_start[i] = torch.tensor([tw[0] for tw in TW], dtype=torch.float32)
        tw_end[i] = torch.tensor([tw[1] for tw in TW], dtype=torch.float32)
        amm[i] = torch.tensor(A, dtype=torch.float32)
        prep[i] = torch.tensor(W, dtype=torch.float32)

    td = TensorDict(
        {
            "value": value.to(device),
            "prob": prob.to(device),
            "tw_start": tw_start.to(device),
            "tw_end": tw_end.to(device),
            "amm": amm.to(device),
            "prep": prep.to(device),
        },
        batch_size=[n],
        device=device,
    )

    scip_path = os.path.join(RESULT_DIR, scip_csv)
    scip_mean = None
    if os.path.exists(scip_path):
        scip_df = pd.read_csv(scip_path)
        scip_mean = float(scip_df["objective_norm"].head(n).mean())

    greedy_mean = _GREEDY_LOOKUP.get(key)
    return td, scip_mean, greedy_mean, n


# Only 5x5x5 has a proven SCIP reference so far (9/10 optimal) -- SCIP is
# slow, so the other 11 configs are deferred; Greedy/AM/POMO all still run
# and are compared against each other without a SCIP anchor until it's done.
_MODERATE_SCIP_REFERENCE = {
    (5, 5, 5): 0.14789626288423424,  # seed=123, 9/10 proven optimal
}
_MODERATE_N = 10
_MODERATE_SEED = 123


def load_moderate_fixed_instances_as_td(device, M=5, N=5, T=5, seed=_MODERATE_SEED, n=_MODERATE_N):
    """Fixed MODERATE temporal-dilemma test set (same seed=123 generator
    used to validate the RL+Auction hybrid) for any (M, N, T) -- for
    Greedy/AM/POMO progress-eval on the curriculum that actually has
    hold-vs-fire structure, instead of the standard benchmark (which does
    not, and biases any comparison toward Greedy/Auction). Returns
    (td, scip_mean_or_None, greedy_mean, n); scip_mean is only populated for
    configs with a precomputed reference (see _MODERATE_SCIP_REFERENCE)."""
    from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
    from eval_greedy_benchmark import eval_instance_greedy

    rng = np.random.default_rng(seed)
    value = torch.zeros(n, N)
    prob = torch.zeros(n, M, N)
    tw_start = torch.zeros(n, N)
    tw_end = torch.zeros(n, N)
    amm = torch.zeros(n, M)
    prep = torch.zeros(n, M)
    greedy_objs = []
    for i in range(n):
        V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
        value[i] = torch.tensor(V, dtype=torch.float32)
        prob[i] = torch.tensor(np.asarray(P), dtype=torch.float32)
        tw_start[i] = torch.tensor([tw[0] for tw in TW], dtype=torch.float32)
        tw_end[i] = torch.tensor([tw[1] for tw in TW], dtype=torch.float32)
        amm[i] = torch.tensor(AMM, dtype=torch.float32)
        prep[i] = torch.tensor(PREP, dtype=torch.float32)
        greedy_objs.append(eval_instance_greedy(V, P, TW, M, N, T, AMM, PREP, COST)["objective"])

    td = TensorDict(
        {
            "value": value.to(device),
            "prob": prob.to(device),
            "tw_start": tw_start.to(device),
            "tw_end": tw_end.to(device),
            "amm": amm.to(device),
            "prep": prep.to(device),
        },
        batch_size=[n],
        device=device,
    )
    scip_mean = _MODERATE_SCIP_REFERENCE.get((M, N, T))
    greedy_mean = float(np.mean(greedy_objs))
    return td, scip_mean, greedy_mean, n


@torch.no_grad()
def eval_rl4co_policy(policy, env, td, decode_type="greedy"):
    """Greedy rollout of `policy` on fixed instance batch `td`; returns mean
    normalized remaining-value objective (lower is better -- directly
    comparable to Greedy/Auction/SCIP's own normalized objective)."""
    was_training = policy.training
    policy.eval()
    init_value = td["value"].sum(-1).clone()
    td_reset = env.reset(td.clone())
    out = policy(td_reset, env=env, decode_type=decode_type)
    remaining = -out["reward"]  # env reward = -remaining_value.sum(-1)
    norm = (remaining / init_value.clamp_min(1e-8)).mean().item()
    if was_training:
        policy.train()
    return norm


class SCIPProgressCallback(Callback):
    """Every `eval_every` epochs, greedy-decodes the current policy on a
    fixed SCIP-validated test set and logs SCIP-optimal / Greedy / model
    side by side -- same fixed instances and same normalized objective used
    to evaluate Greedy/Sequential/Parallel elsewhere in this codebase, so
    numbers are directly comparable. Shared by train_rl4co_am.py and
    train_rl4co_pomo.py. `curriculum='moderate'` uses the temporal-dilemma
    test set (the one with actual hold-vs-fire structure) instead of the
    standard benchmark -- see brerla_rl4co_baseline_curriculum memory for
    why that's the one that should end up in the paper's baseline table."""

    def __init__(self, M, N, T, eval_every=5, curriculum='standard', tag='model'):
        self.M, self.N, self.T = M, N, T
        self.eval_every = eval_every
        self.curriculum = curriculum
        self.tag = tag
        self.td = None
        self.scip_mean = None
        self.greedy_mean = None
        self.n = None

    def on_fit_start(self, trainer, pl_module):
        device = pl_module.device
        if self.curriculum == 'moderate':
            self.td, self.scip_mean, self.greedy_mean, self.n = load_moderate_fixed_instances_as_td(
                device, M=self.M, N=self.N, T=self.T
            )
        else:
            self.td, self.scip_mean, self.greedy_mean, self.n = load_fixed_instances_as_td(
                self.M, self.N, self.T, device
            )
        if self.td is None:
            print(f"[PROGRESS] no fixed SCIP-validated test set for "
                  f"{self.M}M_{self.N}N_{self.T}T ({self.curriculum}) -- skipping progress eval")
            return
        ref = f"SCIP={self.scip_mean:.4f}"
        if self.greedy_mean is not None:
            ref += f" Greedy={self.greedy_mean:.4f}"
        print(f"[PROGRESS] fixed test set: {self.n} instances of "
              f"{self.M}M_{self.N}N_{self.T}T ({self.curriculum}) ({ref})")

    def on_train_epoch_end(self, trainer, pl_module):
        if self.td is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_every != 0 and epoch != trainer.max_epochs - 1:
            return
        model_mean = eval_rl4co_policy(pl_module.policy, pl_module.env, self.td)
        msg = f"[PROGRESS] epoch {epoch}: {self.tag}={model_mean:.4f}"
        if self.scip_mean is not None:
            msg += f"  SCIP={self.scip_mean:.4f}  gap={model_mean - self.scip_mean:+.4f}"
        if self.greedy_mean is not None:
            msg += f"  Greedy={self.greedy_mean:.4f}"
        print(msg, flush=True)
        pl_module.log(f"progress/{self.tag}_vs_scip", model_mean, on_epoch=True)
