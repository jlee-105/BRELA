"""
Moderate-difficulty temporal-dilemma instance generator -- a middle point
between the main tiered benchmark (found to have almost no hold-vs-fire
structure) and temporal_dilemma_generator.py's deliberately extreme version
(sharp bimodal value split, hard emergence cutoff, very tight ammo -- SCIP
itself could only prove optimality on 6/10 instances at 10x10x5 within
120s, vs. 10/10 for the main benchmark's comparable-scale configs, meaning
that version is a materially harder combinatorial problem, not just an
adversarial case for myopic heuristics specifically).

Design: instead of a hard low/high value split with a hard emergence-time
cutoff, target value is CONTINUOUSLY correlated with emergence time (later
targets tend to be somewhat more valuable, with real variance/noise rather
than a deterministic split), and ammo is scarce but less severely so.
"""
import numpy as np


def generate_moderate_temporal_instance(M, N, T, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Emergence time spread across the whole horizon (not a hard early/late
    # split); value has a mild positive correlation with emergence time plus
    # substantial noise, so early targets are *often* but not *always* worse.
    start_times = rng.integers(0, T, size=N)
    base_value = rng.uniform(2, 6, size=N)
    time_bonus = (start_times / max(T - 1, 1)) * rng.uniform(2, 5, size=N)
    V = base_value + time_bonus  # roughly in [2, 11], smoothly increasing with start time

    TW = [[int(start_times[n]), T - 1] for n in range(N)]

    # Ammo: moderately scarce -- enough for a bit under half the horizon's
    # worth of firing opportunities per weapon (looser than the extreme
    # generator's ~1/3, tighter than "always enough").
    AMM = [int(rng.integers(1, max(2, T // 2 + 1))) for _ in range(M)]
    PREP = [int(rng.integers(0, 2)) for _ in range(M)]
    COST = [1 for _ in range(M)]
    P = rng.uniform(0.3, 0.9, (M, N))

    return V.tolist(), P, TW, AMM, PREP, COST
