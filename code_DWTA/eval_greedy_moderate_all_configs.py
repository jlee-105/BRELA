"""Greedy baseline on the MODERATE temporal-dilemma curriculum, across all 12
tiered-benchmark scales (M, N, T) -- reuses
common.rl4co_eval.load_moderate_fixed_instances_as_td's on-the-fly instance
generation (same seed=123 generator used to validate the RL+Auction hybrid
and the 5x5x5 SCIP reference), so Greedy/AM/POMO all get evaluated on
identical fixed instances per config. SCIP is deferred (slow) -- only 5x5x5
has a precomputed reference for now.
"""
import time

from common.rl4co_eval import load_moderate_fixed_instances_as_td, _MODERATE_SCIP_REFERENCE

ALL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]

if __name__ == "__main__":
    print(f"{'config':<18}{'greedy':>10}{'scip':>10}")
    for M, N, T in ALL_CONFIGS:
        t0 = time.time()
        _, scip_mean, greedy_mean, n = load_moderate_fixed_instances_as_td("cpu", M=M, N=N, T=T)
        elapsed = time.time() - t0
        scip_str = f"{scip_mean:.4f}" if scip_mean is not None else "-"
        print(f"{M}M_{N}N_{T}T{'':<8}{greedy_mean:>10.4f}{scip_str:>10}  ({elapsed:.1f}s, n={n})", flush=True)
