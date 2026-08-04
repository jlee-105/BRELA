"""
Simple greedy baseline method for DWTA.
Selects targets with highest value-to-probability ratio.
"""
import os
import sys
import pandas as pd
import numpy as np
import ast
import json

# Add common directory to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from common.Dynamic_HYPER_PARAMETER import NUM_WEAPONS, NUM_TARGETS, MAX_TIME
from common.Dynamic_Instance_generation import input_generation


def greedy_solve(V, P, TW):
    """
    Simple greedy solution: assign weapons to targets with highest value/probability ratio.
    
    Args:
        V: Target values
        P: Hit probabilities  
        TW: Time windows
    
    Returns:
        Total objective value
    """
    # Create assignment matrix: weapon -> target
    assignments = np.zeros((NUM_WEAPONS, NUM_TARGETS))
    remaining_value = np.array(V)
    
    for weapon in range(NUM_WEAPONS):
        # Calculate value/probability ratios for all targets
        ratios = remaining_value / (P[weapon] + 1e-8)  # Avoid division by zero
        
        # Select target with highest ratio
        best_target = np.argmax(ratios)
        assignments[weapon, best_target] = 1
        
        # Update remaining value (reduce by expected damage)
        remaining_value[best_target] *= (1 - P[weapon, best_target])
    
    # Calculate total objective
    total_value = 0
    for target in range(NUM_TARGETS):
        survival_prob = 1.0
        for weapon in range(NUM_WEAPONS):
            if assignments[weapon, target] == 1:
                survival_prob *= (1 - P[weapon, target])
        total_value += V[target] * survival_prob
    
    return total_value


def main():
    """Run greedy baseline on test instances."""
    file_name = './TEST_INSTANCE/5M_5N_5T.xlsx'
    df = pd.read_excel(file_name)
    
    results = []
    
    for i in range(len(df)):  # Test on first 5 instances
        print(f"Processing instance {i}")
        
        V = ast.literal_eval(df.loc[i]['V'])
        df['P'] = df['P'].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else x)
        P = df['P'][i]
        TW = ast.literal_eval(df.loc[i]['TW'])
        TW = np.array(TW)
        
        obj_value = greedy_solve(V, P, TW)
        print(f"Instance {i} objective: {obj_value}")
        results.append(obj_value)
    
    print(f"Average objective: {np.mean(results)}")


if __name__ == "__main__":
    main()
