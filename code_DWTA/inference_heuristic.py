"""
Cost-aware Greedy Heuristic Inference.
At each (time, weapon): pick target with highest alpha*expected_damage - (1-alpha)*fire_cost.
If no positive score, skip (no-action).
"""
import sys, os, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

EVAL_ALPHA = 1.0  # destruction only, no cost
TEST_DIR = "TEST_INSTANCE"

TEST_FILES = [
    "5M_5N_5T.xlsx", "5M_7N_5T.xlsx", "7M_5N_5T.xlsx",
    "15M_15N_5T.xlsx", "15M_20N_5T.xlsx", "20M_20N_5T.xlsx",
    "30M_30N_5T.xlsx", "30M_30N_10T.xlsx", "50M_50N_10T.xlsx",
]

SCALE_MAP = {
    "5M_5N_5T": "Small", "5M_7N_5T": "Small", "7M_5N_5T": "Small",
    "15M_15N_5T": "Medium", "15M_20N_5T": "Medium", "20M_20N_5T": "Medium",
    "30M_30N_5T": "Large", "30M_30N_10T": "Large", "50M_50N_10T": "Large",
}


def solve_heuristic(V, P, TW, amm, prep, cost_list, nw, nt, mt, alpha):
    remaining = np.array(V, dtype=float)
    init_value = float(sum(V))
    ammo_left = list(amm)
    cooldown = [0] * nw
    total_cost = 0.0
    max_realistic_cost = sum(cost_list[w] * amm[w] for w in range(nw))
    total_fires = 0

    t0 = time.time()

    for t in range(mt):
        for w in range(nw):
            if ammo_left[w] <= 0 or cooldown[w] > 0:
                continue

            best_score = 0.0
            best_target = -1

            for n in range(nt):
                tw_start, tw_end = TW[n]
                if t < tw_start or t > tw_end:
                    continue

                expected_damage = remaining[n] * P[w][n]
                fire_cost_norm = cost_list[w] / max(max_realistic_cost, 1e-8)
                score = alpha * expected_damage - (1 - alpha) * fire_cost_norm

                if score > best_score:
                    best_score = score
                    best_target = n

            if best_target >= 0:
                remaining[best_target] *= (1 - P[w][best_target])
                ammo_left[w] -= 1
                cooldown[w] = prep[w]
                total_cost += cost_list[w]
                total_fires += 1

        cooldown = [max(0, c - 1) for c in cooldown]

    elapsed = time.time() - t0
    final_remaining = float(sum(remaining))
    remaining_norm = final_remaining / max(init_value, 1e-8)
    cost_norm = total_cost / max(max_realistic_cost, 1e-8)
    objective = alpha * remaining_norm + (1 - alpha) * cost_norm
    destruction = 1.0 - remaining_norm

    return {
        'init_value': round(init_value, 4),
        'remaining_value': round(final_remaining, 4),
        'remaining_norm': round(remaining_norm, 4),
        'total_cost': round(total_cost, 2),
        'cost_norm': round(cost_norm, 4),
        'objective': round(objective, 4),
        'destruction': round(destruction, 4),
        'fires': total_fires,
        'time_s': round(elapsed, 6),
    }


def main():
    print(f"Heuristic Inference | alpha={EVAL_ALPHA}\n")

    all_results = []

    for fname in TEST_FILES:
        fpath = os.path.join(TEST_DIR, fname)
        if not os.path.exists(fpath):
            print(f"SKIP: {fpath}")
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
            P = json.loads(row['P'])
            TW = json.loads(row['TW'])
            amm_list = json.loads(row['AMM'])
            prep_list = json.loads(row['PREP'])
            cost_list = json.loads(row['COST'])

            result = solve_heuristic(V, P, TW, amm_list, prep_list, cost_list, nw, nt, mt, EVAL_ALPHA)
            result['size'] = label
            result['scale'] = scale
            result['instance'] = i
            result['method'] = 'heuristic'
            file_results.append(result)

        all_results.extend(file_results)

        objs = [r['objective'] for r in file_results]
        destrs = [r['destruction'] for r in file_results]
        costs = [r['cost_norm'] for r in file_results]
        times = [r['time_s'] for r in file_results]

        print(f"  Objective:    {np.mean(objs):.4f} (+/- {np.std(objs):.4f})")
        print(f"  Destruction:  {np.mean(destrs):.2%} (+/- {np.std(destrs):.2%})")
        print(f"  Cost (norm):  {np.mean(costs):.4f} (+/- {np.std(costs):.4f})")
        print(f"  Time:         {np.mean(times):.6f}s")
        print()

    results_df = pd.DataFrame(all_results)
    out_path = "result/heuristic_inference_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

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
