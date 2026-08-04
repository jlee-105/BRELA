import os
import sys
import numpy as np
from pyscipopt import Model, quicksum
import pandas as pd
import ast
import time
import gc
import copy
import json

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# from common.Dynamic_Instance_generation import *
# from common.DWTA_Simulator import Environment
from common.utilities import Average_Meter, Get_Logger

SAVE_FOLDER_NAME = "SCIP_5-10-7"
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

from math import prod
def solve_wta_scip(M, N, T, v, p_matrix, A, W, tw, MAX_T, Time_Limit):
    """
    Solves the Weapon Target Assignment (WTA) problem using SCIP with a proper product constraint.
    """
    model = Model("WTA")
    model.setParam("display/verblevel", 0)

    # Decision variables
    x = {(m, n, t): model.addVar(vtype="BINARY", name=f"x_{m}_{n}_{t}")
         for m in range(M) for n in range(N) for t in range(T)}

    y = {(n, t): model.addVar(vtype="CONTINUOUS", lb=0.0000000000001, ub=1, name=f"y_{n}_{t}")
         for n in range(N) for t in range(T)}

    z = {(m, n, t): model.addVar(vtype="CONTINUOUS", lb=0.0000001, ub=1.0, name=f"z_{m}_{n}_{t}")
         for m in range(M) for n in range(N) for t in range(T)}

    w = {(m, n, t): model.addVar(vtype="CONTINUOUS", lb=0.0000, ub=1.0, name=f"w_{m}_{n}_{t}")
          for m in range(M) for n in range(N) for t in range(T)}

    # w_1 = {(n, t): model.addVar(vtype="CONTINUOUS", lb=0.0000, ub=1.0, name=f"w_{m}_{n}_{t}")
    #      for n in range(N) for t in range(T)}

    k = {(m, t): model.addVar(vtype="INTEGER", lb=0, ub=10, name=f"w_{m}_{t}")
         for m in range(M) for t in range(T)}

    l = {(m, t): model.addVar(vtype="BINARY", name=f"w_{m}_{t}")
         for m in range(M) for t in range(T)}

    # (1): ok
    for m in range(M):
        for t in range(T):
            model.addCons(quicksum(x[m, n, t] for n in range(N)) <= 1)  # One target per weapon per time step
    # (2): ok
    for m in range(M):
        model.addCons(quicksum(x[m, n, t] for n in range(N) for t in range(T)) <= A[m])  # Ammunition constraint

    # Define z_m,n,t = x_m,n,t * (1 - p_m,n) + (1 - x_m,n,t)
    for m in range(M):
        for n in range(N):
            for t in range(T):
                model.addCons(z[m, n, t] == x[m, n, t] * (1 - p_matrix[m, n]) + (1 - x[m, n, t]))


    for n in range(N):
        model.addCons(w[0, n, 0] == z[0, n, 0])  # First weapon's impact at t=0

        for m in range(1, M):
            model.addCons(w[m, n, 0] == w[m - 1, n, 0] * z[m, n, 0])

    # Carry forward the impact from the previous time step and multiply by new impacts at current time
    for t in range(1, T):
        for n in range(N):
            model.addCons(w[0, n, t] == w[M - 1, n, t - 1] * z[
                0, n, t])  # Start new time step by continuing the product from the end of the last time step

            for m in range(1, M):
                model.addCons(w[m, n, t] == w[m - 1, n, t] * z[m, n, t])

    # Set final survival probability at each time step
    for n in range(N):
        for t in range(T):
            model.addCons(y[n, t] == w[M - 1, n, t])


    # for n in range(N):
    #     for t in range(1, T):
    #         model.addCons(y[n, t] == w[m, n, t])  # Final product constraint



    for m in range(M):
        for t in range(MAX_T-1):
            model.addCons(k[m, t+1] ==  k[m, t] - 1 + W[m]*l[m, t])

    for m in range(M):
        model.addCons(k[m, 0] == 0)

    for m in range(M):
        for t in range(MAX_T):
            model.addCons(quicksum(x[m,n,t]  for n in range(N)) <= l[m, t])


    for m in range(M):
        for t in range(MAX_T):
            model.addCons(quicksum(x[m,n,t]  for n in range(N)) <= l[m, t])

    for m in range(M):
        for n in range(N):
            for t in range(T):
                if t< tw[n][0] or t>tw[n][1]:
                    model.addCons(x[m, n, t] == 0)


    #Time constraint for weapon re-engagement
    # for m in range(M):
    #     for t in range(T):
    #         for delta in range(1, min(W[m] + 1, T - t)):
    #             model.addCons(
    #                 quicksum(x[m, n, t] for n in range(N)) + quicksum(x[m, n, t + delta] for n in range(N)) <= 1)



    # for m in range(M):
    #     for t in range(T - W[m]):  # Ensure we don't go out of bounds
    #         model.addCons(
    #             quicksum(x[m, n, t] for n in range(N)) + quicksum(
    #                 x[m, n, t + delta] for n in range(N) for delta in range(1, W[m] + 1)) <= 1
    #         )

    for n in range(N):
        for t in range(1, T):  # Start from t=1 to avoid out-of-bounds error
            model.addCons(y[n, t] <= y[n, t - 1])

    # for m in range(M):
    #     max_possible_firings = 1 + (T - 1) // (W[m] + 1)  # Calculate the maximum number of times weapon m can fire
    #     print(max_possible_firings)
    #     print(min(max_possible_firings, A[m]))
    #     a = input()
    #     model.addCons(
    #         quicksum(x[m, n, t] for n in range(N) for t in range(T)) >= min(max_possible_firings, A[m])
    #     )

    # for n in range(N):
    #     model.addCons(quicksum(x[m, n, t] for m in range(M) for t in range(T)) >= 1)

    # Objective function: minimize engagement cost
    # obj_expr = quicksum(v[n] * quicksum(y[n, 4]  for n in range(N)  )
    obj_expr = quicksum(v[n] * y[n, T - 1] for n in range(N))
    model.setObjective(obj_expr, "minimize")
    # obj_expr = quicksum(v[n] * quicksum(1 - z[m, n, t] for m in range(M)) for n in range(N) for t in range(T))


    # Solver settings - improved for better solution finding
    model.setParam("numerics/feastol", 1e-6)
    model.setParam("numerics/dualfeastol", 1e-6)
    model.setParam("limits/time", Time_Limit)
    model.setParam("limits/gap", 0.05)  # Accept 5% optimality gap
    model.setParam("presolving/maxrounds", 10)  # More presolving
    model.setParam("separating/maxrounds", 5)   # More cutting planes
    model.setIntParam("display/verblevel", 3)
    # model.setIntParam("display/freq", 1)
    # model.setBoolParam("display/lpinfo", True)

    # Solve model
    model.optimize()

    status = model.getStatus()
    
    if model.getNSols() == 0:
        print(f"No feasible solution found! Status: {status}")
        if str(status) == "infeasible":
            print("❌ Problem is mathematically INFEASIBLE")
        elif str(status) == "timelimit":  
            print("⏰ TIME LIMIT reached before finding any solution")
        return None, None, None, status

    best_sol = model.getBestSol()
    obj_value = model.getObjVal()

    # Extract solution
    solution_3d = np.zeros((M, N, T), dtype=int)
    solution_2d = np.zeros((N, T), dtype=float)

    for m in range(M):
        for n in range(N):
            for t in range(T):
                solution_3d[m, n, t] = round(model.getSolVal(best_sol, x[m, n, t]))

    for n in range(N):
        for t in range(T):
            solution_2d[n, t] = round(model.getSolVal(best_sol, y[n, t]), 3)

    manual_y = np.ones((N, T))
    #
    for n in range(N):
        for t in range(T):
            for m in range(M):
                x_val = round(model.getSolVal(best_sol, x[m, n, t]))  # Get x values from SCIP
                manual_y[n, t] *= x_val * (1 - p_matrix[m, n]) + (1 - x_val)
                
    print(f"✅ Status: {status}, Objective: {obj_value:.4f}")            

        # Compute the manually corrected objective function
    # manual_obj_value = prod(v[n] * (x[m, n, t]  for m in range (M) for t in range(T)) for n in range(N))

    # Print manually computed values
    # print("\n=== Manual Verification of Solution ===")
    # print("Manually Computed y[n,t]:")
    # print(manual_y)
    # print("\nManually Computed Objective Value:", manual_obj_value)
    # print("SCIP Reported Objective Value:", obj_value)

    # Compute the remaining importance per target
    # Compute the remaining importance per target (using the last y[n,T-1])
    # remaining_importance = [v[n] * y_opt[n, -1] for n in range(N)]
    #
    # # Print the remaining importance values
    # print("\n=== Corrected Remaining Target Importance ===")
    # for n in range(N):
    #     print(f"Target {n}: Initial = {v[n]}, Remaining = {remaining_importance[n]}")

    return obj_value, solution_3d, solution_2d, status

# === Sample Usage with Inputs and Solution Printing ===
import json
obj_list=list()
SAVE_FOLDER_NAME = "SCIP_30-30-10"
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

print(logger)
print(result_folder_path)

df = pd.read_excel("./TEST_INSTANCE/5M_5N_5T.xlsx")
obj = list()
solution_status = list()  # Track solution status
optimal_count = 0
feasible_count = 0

start_time = time.time()
for i in range(1):
    start = time.time()
    print("Instance number == ===================", i)
    i = i
    M = 5
    N = 5
    T = 5
    Time_limit = 600
    # reload time
    W = [1, 2, 1, 2, 1 ]#, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1 , 1, 2, 1, 2, 1 , 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    A  = [2, 2, 2, 2, 2] #   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   4, 4, 4, 4, 4 , 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 , 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3]# 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    V = ast.literal_eval(df.loc[i]['V'])
    df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
    P = df['P'][i]
    V = np.array(V, dtype=float).tolist()
    TW = ast.literal_eval(df.loc[i]['TW'])
    TW = np.array(TW, dtype=float).tolist()

    # print("=== Problem Parameters ===")
    # print("Weapon count (M):", M)
    # print("Target count (N):", N)
    # print("Time steps (T):", T)
    # print("Importance values (v):", V)
    # print("Ammunition limits (A):", A)
    # print("Cooldown limits (W):", W)
    # print("Probability matrix (p_matrix):")


    obj_val, x_opt, y_opt, status = solve_wta_scip(M, N, T, V, P, A=A, W=W, tw=TW, MAX_T= T, Time_Limit=Time_limit)

    # Track solution status with proper categorization
    status_str = str(status)
    
    # SCIP status categorization - 4 clear categories
    if status_str == "optimal":
        optimal_count += 1
        status_category = "1_optimal"
    elif status_str == "bestsollimit" or (status_str == "timelimit" and obj_val is not None):
        feasible_count += 1  
        status_category = "2_feasible"
    elif status_str == "timelimit" and obj_val is None:
        status_category = "3_timelimit_no_solution"
        print(f"⏰ Instance {i}: TIME LIMIT - no feasible solution found")
    elif status_str == "infeasible":
        status_category = "4_infeasible" 
        print(f"❌ Instance {i}: INFEASIBLE problem!")
    else:
        status_category = f"unknown_{status_str}"
        print(f"❓ Instance {i}: Unknown status - {status_str}")
    
    # Store categorized status
    solution_status.append(status_category)
    
    # Handle cases where no solution was found (obj_val is None)
    if obj_val is None:
        print(f"❌ Instance {i}: No solution found - {status_category}")
        logger.info(f'Instance {i}: No solution found - {status_category}')
        # Don't add to obj list since no objective value
    elif x_opt is not None:
        # print("\n=== Solution ===")
        # print("Objective Value:", obj_val)
        # print("\nWeapon-Target-Time Assignment (x[m,n,t]):")
        # for m in range(M):
        #     print(f"Weapon {m} assignment:\n", x_opt[m])

        # print("\nTarget Engagement Probability (y[n,t]):")
        print(y_opt)
        # a = input()

        final_survival_prob = np.ones(N)
        final_value = np.ones(N)
        # Extract y_opt from solution
        for n in range(N):
            for t in range(T):
                for m in range(M):
                    if x_opt[m, n, t] == 1:
                        final_survival_prob[n] *= (1 - P[m, n])  # Multiply survival probability step-by-step

        # Compute objective value (importance-weighted final survival probability)
        for n in range(N):
            final_value[n] = V[n]*final_survival_prob[n]
        print(final_value)

        # corrected_obj_value = np.sum(V * final_survival_prob)
        # print(corrected_obj_value)
        # a = input()

        # Print the corrected remaining importance
        # print("\n=== Corrected Remaining Target Importance ===")
        # for n in range(N):
        #     print(f"Target {n}: Initial = {V[n]}, Remaining = {final_value[n]}")
        # print(sum(final_value))

        # SAVE_FOLDER_NAME = "SCIP-5-5-5"
     
        obj.append(sum(final_value))

        print("solution_so_far ===========", sum(obj)/len(obj))

        logger.info('---------------------------------------------------')
        logger.info(f'Instance {i}: value = {sum(final_value):.4f}')
        logger.info(f'Status: {status_category}')
        logger.info(f'Average so far = {sum(obj) / len(obj):.4f}')
        logger.info('---------------------------------------------------')

end_time = time.time()

# Count different status categories
timelimit_no_solution_count = sum(1 for s in solution_status if s == "3_timelimit_no_solution")
infeasible_count = sum(1 for s in solution_status if s == "4_infeasible")

# Print solution status summary
print("\n" + "="*50)
print("SOLUTION STATUS SUMMARY")
print("="*50)
print(f"Total instances: {len(solution_status)}")
print(f"Optimal solutions: {optimal_count}")
print(f"Feasible solutions: {feasible_count}")
print(f"Solution not found: {timelimit_no_solution_count}")
print(f"Infeasible: {infeasible_count}")
print(f"Optimal percentage: {optimal_count/len(solution_status)*100:.1f}%")
print("Status details:", solution_status)
print("="*50)

logger.info('='*50)
logger.info('FINAL RESULTS SUMMARY')
logger.info('='*50)
logger.info(f'Total instances: {len(solution_status)}')
logger.info(f'Average objective: {sum(obj) / len(obj):.4f}')
logger.info(f'Optimal solutions: {optimal_count}')
logger.info(f'Feasible solutions: {feasible_count}')
logger.info(f'Solution not found: {timelimit_no_solution_count}')
logger.info(f'Infeasible: {infeasible_count}')
logger.info(f'Optimal percentage: {optimal_count/len(solution_status)*100:.1f}%')
logger.info(f'Average time per instance: {(end_time - start_time) / len(solution_status):.2f}s')

# Log detailed status breakdown
from collections import Counter
status_counts = Counter(solution_status)
logger.info('Status breakdown:')
for status, count in status_counts.items():
    logger.info(f'  {status}: {count} instances ({count/len(solution_status)*100:.1f}%)')
    
logger.info('='*50)

# Save the DataFrame to a CSV file
csv_file_path = './result/opt-5-5-5.csv'

# Create comprehensive results DataFrame with ALL instances (including failed ones)
all_instances = []
for i in range(len(solution_status)):
    instance_data = {
        'instance': i,
        'objective': obj[i] if i < len(obj) else None,
        'status': solution_status[i],
        'status_category': solution_status[i].split('_')[0] if '_' in solution_status[i] else 'unknown'
    }
    all_instances.append(instance_data)

df = pd.DataFrame(all_instances)

# Add summary statistics as additional columns
df['average_obj'] = sum(obj) / len(obj) if len(obj) > 0 else 0
df['total_instances'] = len(solution_status)
df['optimal_count'] = optimal_count
df['feasible_count'] = feasible_count
df['solution_not_found_count'] = timelimit_no_solution_count
df['infeasible_count'] = infeasible_count
df['optimal_percentage'] = optimal_count / len(solution_status) * 100 if len(solution_status) > 0 else 0
df['feasible_percentage'] = feasible_count / len(solution_status) * 100 if len(solution_status) > 0 else 0
df['solution_not_found_percentage'] = timelimit_no_solution_count / len(solution_status) * 100 if len(solution_status) > 0 else 0
df['infeasible_percentage'] = infeasible_count / len(solution_status) * 100 if len(solution_status) > 0 else 0

# Add status breakdown columns
from collections import Counter
status_counts = Counter(solution_status)
for status, count in status_counts.items():
    df[f'{status}_count'] = count
    df[f'{status}_percentage'] = count / len(solution_status) * 100

df.to_csv(csv_file_path, index=False)
print(f"Comprehensive DataFrame saved to {csv_file_path}")
print(f"Summary: Avg obj = {sum(obj)/len(obj):.4f}, Optimal = {optimal_count}/{len(solution_status)} ({optimal_count/len(solution_status)*100:.1f}%)")



