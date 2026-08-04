"""
Smoke test for the parallel multi-pointer decoder + shaped REINFORCE training loop.
Runs a tiny number of epochs/episodes on a small instance size to verify the full
actor <-> environment <-> critic <-> loss pipeline executes without shape/logic errors.
Not a real training run -- just an end-to-end correctness check.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from common.DWTA_GNN import create_gnn_actor, create_gnn_critic
from rl.Dynamic_Sampling_GNN import self_play_gnn

if __name__ == "__main__":
    actor = create_gnn_actor()
    critic = create_gnn_critic()

    print("Running 2 tiny epochs (2 episodes each) as a smoke test...")
    for epoch in range(2):
        actor_loss, critic_loss, info = self_play_gnn(
            old_actor=None,
            actor=actor,
            critic=critic,
            episode=2,
            temp=1.0,
            epoch=epoch,
            logger=None,
        )
        print(f"Epoch {epoch}: actor_loss={actor_loss}, critic_loss={critic_loss}, info={info}")

    print("SMOKE TEST PASSED: full self_play_gnn loop ran without errors.")
