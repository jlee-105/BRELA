# DWTA Method Organization

This directory is organized by method type, with each method folder containing all necessary files.

## Folder Structure

### `common/`
Shared files used by all methods:
- `Dynamic_HYPER_PARAMETER.py` - Configuration parameters
- `Dynamic_Instance_generation.py` - Instance generation utilities
- `TORCH_OBJECTS.py` - PyTorch device configuration
- `utilities.py` - Common utility functions
- `DWTA_Simulator.py` - Base simulator

### `opt/`
Optimization methods (SCIP):
- `SCIP.py` - SCIP optimization solver
- All common files (copied)

### `greedy/`
Simple greedy methods:
- `greedy_baseline.py` - Greedy baseline algorithm
- All common files (copied)

### `rl/`
Basic RL methods (GNN actor-critic):
- `DWTA_GNN.py` - GNN model definitions
- `Dynamic_Sampling_GNN.py` - GNN training
- `DWTA_GNN_INFERENCE_FLEXIBLE.py` - GNN inference
- `TRAIN/` - Training checkpoints and logs
- All common files (copied)

### `rl_rollout/`
RL with rollout methods:
- `DWTA_Rollout_Improvement.py` - Rollout improvement
- `DWTA_Rollout_TruncCritic.py` - Truncated critic rollout
- `BEAM_WITH_SIMULATION_NOT_TRUNC.py` - Beam search without truncation
- `BEAM_WITH_SIMULATION_TRUNC_CRITIC.py` - Beam search with truncated critic
- `DWTA_Simulator_rollout.py` - Rollout-specific simulator
- All common files (copied)

### `unused/`
Unused/debug files:
- `debug_gnn.py` - Debug scripts
- `DWTA_Evaluation.py` - Evaluation scripts
- `DWTA_PPO_TRAIN_GNN.py` - Old training scripts
- `GNN_BASELINE_5-5-5.csv` - Old baseline results

### `TEST_INSTANCE/`
Test data files:
- `30M_30N.xlsx` - 30 weapons, 30 targets instances
- `5M_5N.xlsx` - 5 weapons, 5 targets instances
- `5M_5N_5T.xlsx` - 5 weapons, 5 targets, 5 time steps instances

## Usage

Each method folder is self-contained. To run a specific method:

```bash
# Run greedy baseline
cd greedy/
python greedy_baseline.py

# Run SCIP optimization
cd opt/
python SCIP.py

# Run GNN inference
cd rl/
python DWTA_GNN_INFERENCE_FLEXIBLE.py

# Run rollout methods
cd rl_rollout/
python DWTA_Rollout_TruncCritic.py
```

## Note on Imports

Each method folder contains copies of common files to ensure self-contained execution. The import paths in the files may need to be updated to work with the new structure.
