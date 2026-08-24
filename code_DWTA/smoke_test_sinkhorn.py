"""
Smoke test for EdgeAwareGNN_ACTOR_COMM_SINKHORN + self_play_gnn_moderate.
Mirrors smoke_test_parallel.py / smoke_test_sequential.py. Runs a tiny number
of epochs/episodes to verify the full actor <-> environment <-> Sinkhorn
bonus <-> REINFORCE loss pipeline executes without shape/NaN errors on a
random multi-scale (M,N,T~U[5,7]) curriculum. Not a real training run.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rl'))

import torch

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from Dynamic_Sampling_GNN_moderate import self_play_gnn_moderate

if __name__ == "__main__":
    torch.manual_seed(0)
    actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    print(f"sinkhorn_scale initial value: {actor.sinkhorn_scale.item()} (expect 0.0)")

    print("Running 3 tiny 'epochs' (2 episodes each, random 5-7 x 5-7 x 5-7 scale)...")
    for epoch in range(3):
        actor_loss, info = self_play_gnn_moderate(actor, episode=2, epoch=epoch)
        print(f"Epoch {epoch}: actor_loss={actor_loss}, info={info}, "
              f"sinkhorn_scale={actor.sinkhorn_scale.item():.6f}")
        assert torch.isfinite(torch.tensor(actor_loss)).all(), "actor_loss is NaN/inf"
        for k, v in info.items():
            assert v == v, f"{k} is NaN"  # NaN != NaN

    for name, p in actor.named_parameters():
        assert torch.isfinite(p).all(), f"parameter {name} has NaN/inf after training"
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"grad of {name} has NaN/inf"

    print("sinkhorn_scale moved from init:", actor.sinkhorn_scale.item())
    print("SMOKE TEST PASSED: full comm+sinkhorn self_play_gnn_moderate loop ran without errors.")
