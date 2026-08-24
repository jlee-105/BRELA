"""
Generate an offline imitation-learning dataset: MODERATE-curriculum
instances at small scale (M,N,T ~ U[5,7], same range as our own multi-scale
self-play curriculum), each solved to optimality (or near-optimality) by
SCIP, with the full per-round per-weapon optimal action extracted as a
supervision label. Used by train_scip_warmstart.py for teacher-forced
behavior cloning, as a warm-start before REINFORCE fine-tuning (see
brerla_scip_imitation_warmstart memory for the rationale: REINFORCE alone
has repeatedly (AM, POMO, SCoPE) peaked early then degraded on this
curriculum, so a supervised warm-start from SCIP-optimal small-scale
solutions should give REINFORCE a much better starting point).

SCIP is fast at this scale (seconds), so this is generated once, offline,
and reused for many supervised training epochs (no more SCIP calls needed
after this).

Usage: python generate_scip_teacher_dataset.py --n_instances 400 --out result/scip_teacher_dataset.pt
"""
import argparse
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + pyscipopt OpenMP conflict

import numpy as np
import torch

from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from opt.SCIP import solve_wta_scip

SCALE_LOW, SCALE_HIGH = 5, 7
SCIP_TIME_LIMIT = 15  # 5x5x5 solves in ~4s, but 7x7x7 can take much longer -- imitation targets
                       # don't need PROVEN optimality, just a good incumbent, so cap tightly and
                       # accept SCIP's best-found-so-far under timeout rather than waiting it out


def sample_scale(rng):
    M = int(rng.integers(SCALE_LOW, SCALE_HIGH + 1))
    N = int(rng.integers(SCALE_LOW, SCALE_HIGH + 1))
    T = int(rng.integers(SCALE_LOW, SCALE_HIGH + 1))
    return M, N, T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_instances', type=int, default=400)
    parser.add_argument('--seed', type=int, default=777)  # distinct from the eval seed (123)
    parser.add_argument('--out', type=str, default='result/scip_teacher_dataset.pt')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = []
    n_optimal = 0
    t0 = time.time()

    i = 0
    while i < args.n_instances:
        M, N, T = sample_scale(rng)
        V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
        P = np.asarray(P)

        obj, solution_3d, _, status, gap = solve_wta_scip(M, N, T, V, P, A=AMM, W=PREP, tw=TW, Time_Limit=SCIP_TIME_LIMIT)
        if solution_3d is None:
            continue  # infeasible/failed, skip and resample

        # actions[t, m] = target index (0..N-1) fired by weapon m at round t, or N for no-op
        actions = np.full((T, M), N, dtype=np.int64)
        for m in range(M):
            for t in range(T):
                for n in range(N):
                    if round(solution_3d[m, n, t]) == 1:
                        actions[t, m] = n

        dataset.append({
            'M': M, 'N': N, 'T': T,
            'V': V, 'P': P.tolist(), 'TW': TW, 'AMM': AMM, 'PREP': PREP,
            'actions': actions.tolist(),
            'status': str(status),
        })
        if status == 'optimal':
            n_optimal += 1
        i += 1

        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"[{i}/{args.n_instances}] optimal={n_optimal}/{i} elapsed={elapsed:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(dataset, args.out)
    print(f"Saved {len(dataset)} instances ({n_optimal} proven optimal) to {args.out} "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
