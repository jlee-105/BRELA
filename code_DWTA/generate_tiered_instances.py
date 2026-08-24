import pandas as pd
import random
import numpy as np
import json

random.seed(42)
np.random.seed(42)

N_INSTANCES = 50
EXCEL_CELL_LIMIT = 32767

configs = [
    # (n_weapon, n_target, e_time)
    (5, 5, 5),      # Small config1 (2026-08-06: included in the ammo-scaling regeneration too,
    (5, 7, 5),      # Small config2  per user request -- invalidates the earlier SCIP (5,5,5) run,
                    # which must be rerun on the new file before trusting any comparison against it)
    (10, 15, 5),    # Small config3 (legacy file had old schema, missing T/AMM/PREP/COST -- regenerate)
    (20, 30, 5),    # Medium config3
    (30, 40, 10),   # Large config2
    (40, 50, 10),   # Large config3
    (50, 50, 15),   # Battlefield config1
    (50, 70, 15),   # Battlefield config2
    (70, 100, 15),  # Battlefield config3
]

def dumps(obj):
    return json.dumps(obj, separators=(',', ':'))

for n_weapon, n_target, e_time in configs:
    for p_decimals in (2, 1):
        rows = []
        for _ in range(N_INSTANCES):
            V = [random.randint(1, 10) for _ in range(n_target)]
            P = np.round(np.random.uniform(0.2, 0.9, size=(n_weapon, n_target)), p_decimals).tolist()
            max_start = e_time // 2
            TW = [[random.randint(0, max_start), e_time] for _ in range(n_target)]
            # Scale ammo with the episode horizon (2026-08-06 fix, v2): the original
            # T=5 configs used AMM~U(1,3), reused unscaled for T=10/15, which made
            # ammo far scarcer relative to rounds than intended (84-87% of
            # weapon-decisions forced/choiceless at Large/Battlefield -- no room to
            # differentiate a learned policy from simple greedy). First attempt
            # (proportional to T alone, ~U(3,9) at T=15) overshot the other way --
            # ammo became roughly as large as the number of reload-eligible firing
            # opportunities, so ammo stopped binding at all (greedy destroyed
            # ~99.9%, trivial). This v2 targets real-but-not-total scarcity:
            # roughly half the reload-eligible opportunity count, so a weapon can
            # fire at some but not all of its opportunities -- forcing an actual
            # choice of when, not just whether.
            # Anchor exactly at the original U(1,3) for T=5 (avoid round()'s
            # banker's-rounding drifting T=5 to U(1,2) via the general formula).
            if e_time <= 5:
                amm_lo, amm_hi = 1, 3
            else:
                amm_lo = max(1, round(0.5 * e_time / 5))
                amm_hi = max(amm_lo, round(1.5 * e_time / 5))
            AMM = [random.randint(amm_lo, amm_hi) for _ in range(n_weapon)]
            PREP = [random.randint(1, 2) for _ in range(n_weapon)]
            COST = [random.choice(range(20, 101, 10)) for _ in range(n_weapon)]
            rows.append([n_weapon, n_target, e_time,
                         dumps(V), dumps(P), dumps(TW), dumps(AMM), dumps(PREP), dumps(COST)])

        max_cell_len = max(len(r[4]) for r in rows)  # P is the largest cell
        if max_cell_len <= EXCEL_CELL_LIMIT:
            break
        print(f"  {n_weapon}M_{n_target}N_{e_time}T: P cell len {max_cell_len} > limit at {p_decimals} decimals, retrying with fewer decimals")
    else:
        raise RuntimeError(f"{n_weapon}M_{n_target}N_{e_time}T: still exceeds cell limit at 1 decimal")

    df = pd.DataFrame(rows, columns=['M', 'N', 'T', 'V', 'P', 'TW', 'AMM', 'PREP', 'COST'])
    output_path = f"./TEST_INSTANCE/{n_weapon}M_{n_target}N_{e_time}T.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Saved {output_path} (P decimals={p_decimals}, max cell len={max_cell_len})")
