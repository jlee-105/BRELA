#!/bin/bash
for ablation in no_critic no_shaping no_reward_to_go; do
  for seed in 0 1 2 3 4; do
    echo "=== ABLATION $ablation SEED $seed ==="
    python -u rl/DWTA_GNN_TRAIN.py --seed $seed --ablation $ablation
  done
done
echo "ALL_ABLATIONS_COMPLETE"
