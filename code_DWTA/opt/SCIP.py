import os
import sys
import numpy as np
from pyscipopt import Model, quicksum

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def solve_wta_scip(M, N, T, v, p_matrix, A, W, tw, Time_Limit):
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



    # Reload-time constraint: after firing at t, weapon m cannot fire again for W[m] steps
    # (matches Eq. weapon_readiness in the manuscript: x_{m,n,t} <= 1 - w_{m,t}/D_m)
    for m in range(M):
        for t in range(T):
            for delta in range(1, min(W[m] + 1, T - t)):
                model.addCons(
                    quicksum(x[m, n, t] for n in range(N)) + quicksum(x[m, n, t + delta] for n in range(N)) <= 1)

    for m in range(M):
        for n in range(N):
            for t in range(T):
                if t< tw[n][0] or t>tw[n][1]:
                    model.addCons(x[m, n, t] == 0)


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


    # Solver settings: time limit only (SCIP's default gap target is already 0 --
    # i.e. it runs the full budget trying to prove optimality, no early stop)
    model.setParam("limits/time", Time_Limit)
    model.setParam("presolving/maxrounds", 10)  # More presolving
    model.setIntParam("display/verblevel", 3)

    # Solve model
    model.optimize()

    status = model.getStatus()
    gap = model.getGap()

    if model.getNSols() == 0:
        print(f"No feasible solution found! Status: {status}")
        if str(status) == "infeasible":
            print("Problem is mathematically INFEASIBLE")
        elif str(status) == "timelimit":
            print("TIME LIMIT reached before finding any solution")
        return None, None, None, status, gap

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
                
    print(f"Status: {status}, Objective: {obj_value:.4f}")

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

    return obj_value, solution_3d, solution_2d, status, gap



