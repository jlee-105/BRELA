"""Exhaustive constraint-violation check for the auction refinement across
ALL 12 tiered-benchmark configs -- checks every single fire decision against
(a) the environment's own legality mask, (b) the target's time window, and
(c) the weapon's ammo budget, tracked independently of the environment."""
import json

import numpy as np
import pandas as pd
import torch

from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action
from eval_tiered_benchmark import patch_globals, TIERS, TEST_DIR, N_EVAL

total_decisions = 0
total_fires = 0
total_violations = 0

for tier, files in TIERS.items():
    for fname in files:
        df = pd.read_excel(f"{TEST_DIR}/{fname}")
        nw, nt, mt = int(df.iloc[0]["M"]), int(df.iloc[0]["N"]), int(df.iloc[0]["T"])
        config_violations = 0
        config_fires = 0

        for i in range(min(N_EVAL, len(df))):
            row = df.iloc[i]
            V = json.loads(row["V"]); P = np.array(json.loads(row["P"])); TW = json.loads(row["TW"])
            amm = json.loads(row["AMM"]); prep = json.loads(row["PREP"]); cost = json.loads(row["COST"])
            patch_globals(nw, nt, mt, amm, prep, cost)
            ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                        max_time=mt, batch_size=1, alpha=1.0, amm=amm)
            ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
            env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)

            ammo_used = [0] * nw
            reload_until = [-1] * nw  # last round each weapon is still on cooldown

            for t in range(mt):
                remaining_value = env.current_target_value[:, :, 0:nt]
                legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
                prob = env.weapon_to_target_prob[:, :, :nw, :nt]
                with torch.no_grad():
                    action = auction_round_action(remaining_value, prob, legal_mask)
                flat = action.view(-1).tolist()
                for m, n in enumerate(flat):
                    total_decisions += 1
                    if n == nt:
                        continue
                    total_fires += 1
                    config_fires += 1
                    is_legal = bool(legal_mask[0, 0, m, n].item())
                    tw_ok = TW[n][0] <= t <= TW[n][1]
                    ammo_used[m] += 1
                    ammo_ok = ammo_used[m] <= amm[m]
                    reload_ok = t > reload_until[m]
                    if not (is_legal and tw_ok and ammo_ok and reload_ok):
                        config_violations += 1
                        total_violations += 1
                        print(f"VIOLATION [{fname} inst{i} t={t}] weapon{m}->target{n}: "
                              f"env_legal={is_legal} tw_ok={tw_ok}(TW={TW[n]}) "
                              f"ammo_ok={ammo_ok}(used={ammo_used[m]}/{amm[m]}) reload_ok={reload_ok}")
                    if n != nt:
                        reload_until[m] = t + prep[m]
                env.update_internal_variables_parallel(selected_actions=action)
                env.time_update()

        print(f"[{tier}] {fname}: fires={config_fires} violations={config_violations}", flush=True)

print(f"\nTOTAL decisions={total_decisions} fires={total_fires} violations={total_violations}")
