"""Generate small-scale test instances deliberately designed to contain
genuine hold-vs-fire temporal dilemmas: ammo scarce relative to the number of
EARLY, low-value targets, with a smaller number of substantially
higher-value targets that only become engageable later. Unlike the main
tiered benchmark (randomly generated, found to have almost no such
structure -- fire_rate identical across all methods), this is a stress test
for whether "hold ammo now for a better target later" actually pays off, and
whether trained RL captures that better than round-local methods
(Greedy/Auction) which cannot see future rounds at all.
"""
import json

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CONFIGS = [
    ("5M_5N_5T_temporal", 5, 5, 5),
    ("10M_10N_5T_temporal", 10, 10, 5),
    ("20M_20N_10T_temporal", 20, 20, 10),
]
N_INSTANCES = 10


def gen_instance(M, N, T):
    # Half the targets are LOW value, engageable from round 0.
    # Half the targets are HIGH value, only engageable from the back half of
    # the horizon onward -- a weapon that spends ammo early on a low-value
    # target forecloses one of these.
    n_low = N // 2
    n_high = N - n_low
    V_low = RNG.uniform(1, 3, n_low)
    V_high = RNG.uniform(7, 10, n_high)
    V = np.concatenate([V_low, V_high])
    order = RNG.permutation(N)
    V = V[order]
    is_high = np.zeros(N, dtype=bool)
    is_high[order >= n_low] = True  # track which shuffled slots are "high"

    TW = []
    late_start = max(1, T // 2)
    for n in range(N):
        if is_high[n]:
            start = int(RNG.integers(late_start, T))
            TW.append([start, T - 1])
        else:
            TW.append([0, T - 1])

    # Ammo: deliberately scarce -- roughly enough for HALF the rounds' worth
    # of firing opportunities per weapon, forcing real fire-or-hold choices.
    AMM = [int(RNG.integers(1, max(2, T // 3 + 1))) for _ in range(M)]
    PREP = [int(RNG.integers(0, 2)) for _ in range(M)]
    COST = [1 for _ in range(M)]
    P = RNG.uniform(0.3, 0.9, (M, N)).tolist()

    return {
        "V": json.dumps(V.tolist()),
        "P": json.dumps(P),
        "TW": json.dumps(TW),
        "AMM": json.dumps(AMM),
        "PREP": json.dumps(PREP),
        "COST": json.dumps(COST),
        "M": M, "N": N, "T": T,
    }


for name, M, N, T in CONFIGS:
    rows = [gen_instance(M, N, T) for _ in range(N_INSTANCES)]
    df = pd.DataFrame(rows)
    out_path = f"TEST_INSTANCE/{name}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"wrote {out_path} ({N_INSTANCES} instances, {M}x{N}x{T})")
