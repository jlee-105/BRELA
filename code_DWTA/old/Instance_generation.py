import pandas as pd
import random
import numpy as np
import json

# Create instance_list
instance_list = []
for test_index in range(1):
    n_weapon = 30
    n_target = 30
    s_time = 3
    e_time = 10
    # Random target values
    t_value = [random.randint(1, 10) for _ in range(n_target)]
    # Generate random probabilities matrix
    P = np.random.uniform(low=0.1, high=0.5, size=(n_weapon, n_target))
    P = np.round(P, 4)
    # Create random time window
    TIME_START = [random.randint(0, s_time) for _ in range(n_target)]
    time_window = [(start, e_time) for start in TIME_START]   
    
    instance_tuple = [n_weapon, n_target, t_value, P, time_window]
    instance_list.append(instance_tuple)

# Convert list to DataFrame
df = pd.DataFrame(instance_list, columns=['M', 'N', 'V', 'P', 'TW'])

def list_to_json(obj):
    """
    Convert a Python list (or other JSON-serializable object) to a JSON string.
    """
    return json.dumps(obj)

def matrix_to_json(matrix):
    """
    Convert a NumPy array to a JSON string by first converting it to a Python list.
    """
    return json.dumps(matrix.tolist())

# Convert each column to a JSON string
df['V'] = df['V'].apply(list_to_json)          # t_value is a list of ints
df['TW'] = df['TW'].apply(list_to_json)        # time_window is a list of tuples
df['P'] = df['P'].apply(matrix_to_json)        # P is a NumPy array

# Save DataFrame to Excel
output_path = "./TEST_INSTANCE/30M_30N_10T.xlsx"
df.to_excel(output_path, index=False)
print(f"Data saved to {output_path}")