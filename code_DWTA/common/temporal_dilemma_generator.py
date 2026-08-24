"""
Instance generator for training/testing WITH genuine hold-vs-fire temporal
dilemmas -- ammo scarce relative to firing opportunities, with a subset of
substantially higher-value targets that only become engageable partway
through the horizon. Unlike the main multi-scale training curriculum
(common/Dynamic_Instance_generation.py's random generator), which was found
to produce almost no such structure (fire_rate identical across Greedy/RL/
Sequential on the main tiered benchmark -- "whether to fire" was rarely a
live question there), this is deliberately built so that a myopic,
round-local policy (Greedy, or the auction refinement alone) provably loses
value by firing too early, and only a policy with real temporal judgment can
do well. See memory brerla_rl_auction_hybrid_decomposition.md.

Produces the same (V, P, TW, AMM, PREP, COST) format eval scripts and
`input_generation`'s explicit-override path already consume -- no changes
to common/Dynamic_Instance_generation.py needed.
"""
import numpy as np


def generate_temporal_dilemma_instance(M, N, T, rng=None):
    """Single random instance with staggered high-value target emergence and
    scarce ammo. Returns (V, P, TW, AMM, PREP, COST) in the same list/array
    format used throughout the eval scripts."""
    if rng is None:
        rng = np.random.default_rng()

    n_low = max(1, N // 2)
    n_high = N - n_low
    V_low = rng.uniform(1, 3, n_low)
    V_high = rng.uniform(7, 10, n_high) if n_high > 0 else np.array([])
    V = np.concatenate([V_low, V_high])
    order = rng.permutation(N)
    V = V[order]
    is_high = np.zeros(N, dtype=bool)
    is_high[order >= n_low] = True

    late_start = max(1, T // 2)
    TW = []
    for n in range(N):
        if is_high[n]:
            start = int(rng.integers(late_start, T)) if late_start < T else T - 1
            TW.append([start, T - 1])
        else:
            TW.append([0, T - 1])

    AMM = [int(rng.integers(1, max(2, T // 3 + 1))) for _ in range(M)]
    PREP = [int(rng.integers(0, 2)) for _ in range(M)]
    COST = [1 for _ in range(M)]
    P = rng.uniform(0.3, 0.9, (M, N))

    return V.tolist(), P, TW, AMM, PREP, COST
