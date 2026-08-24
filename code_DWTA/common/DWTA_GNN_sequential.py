"""
Sequential (originally-published BReRLA) decoder, rebuilt as the comparison
point for the parallel multi-pointer decoder in DWTA_GNN.py.

Per BReRLA_revision_plan.md:112, the original architecture produced ONE
global softmax over all W*T+1 (weapon,target,no-op) options per forward
call. The training loop called the actor once per (time_step, weapon_idx)
pair -- M*T calls per episode -- but the weapon_idx loop variable did not
restrict the choice: each call could pick any still-valid (weapon,target)
pair globally. So this is M unrestricted decision opportunities per time
step, each one immediately updating environment state, not M weapon-
restricted turns and not one simultaneous joint decision.

The shared GNN message-passing stack (EdgeAwareGNNLayer) is unchanged from
DWTA_GNN.py -- only the output head differs: one flat joint softmax instead
of a per-weapon softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import EdgeAwareGNNLayer, ResidualBlock, ReplayMemory


class EdgeAwareGNN_ACTOR_Sequential(nn.Module):
    """
    Edge-aware GNN Actor with the original SEQUENTIAL single global-softmax
    decoding: one joint distribution over all (weapon,target) edges plus a
    single global no-op, per forward call. The caller samples ONE edge,
    applies it to the environment, and calls forward() again for the next
    decision -- state changes between calls, so coordination (if any) must
    emerge from re-observing updated state each time, not from a single
    shared snapshot (contrast with EdgeAwareGNN_ACTOR's parallel decoding).
    """
    def __init__(self, num_layers=3):
        super().__init__()

        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGNNLayer() for _ in range(num_layers)
        ])

        # Residual edge scorer -- identical head shape to the parallel actor,
        # just flattened+joint-softmaxed instead of per-weapon-softmaxed.
        self.edge_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.edge_score = nn.Linear(EMBEDDING_DIM, 1)

        # Single global no-op scorer: pooled context only (no per-weapon
        # identity), since the original architecture had one system-wide
        # no-op option, not one per weapon.
        self.no_action_proj = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)
        self.no_action_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.no_action_score = nn.Linear(EMBEDDING_DIM, 1)

        self.assignment_embedding = None
        self.current_state = None
        self.replay_memory = ReplayMemory(capacity=BUFFER_SIZE)

        import torch.optim as optim
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=ACTOR_LEARNING_RATE,
            weight_decay=ACTOR_WEIGHT_DECAY
        )
        self.lr_stepper = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,
            gamma=0.9
        )

    def forward(self, assignment_embedding, prob, mask):
        """
        Args:
            assignment_embedding: [batch, para, NUM_WEAPONS*NUM_TARGETS+1, 9]
            prob: [batch, para, num_weapons, num_targets] or None
            mask: [batch, para, num_weapons*num_targets+1] -- flat legality
                  mask (env.mask), last index is the single global no-op.
        Returns:
            policy: [batch, para, num_weapons*num_targets+1] -- one joint
                    softmax over every (weapon,target) edge plus no-op.
            assignment_embedding: for compatibility (per weapon-target edge embeddings)
        """
        batch_size, para_size = assignment_embedding.shape[:2]

        if prob is None:
            num_weapons, num_targets = NUM_WEAPONS, NUM_TARGETS
        else:
            num_weapons, num_targets = prob.shape[2], prob.shape[3]

        n_feat = assignment_embedding.size(-1)
        assignments = assignment_embedding[:, :, :-1, :]  # [batch, para, W*T, F]
        features_reshaped = assignments.view(batch_size, para_size, num_weapons, num_targets, n_feat)

        weapon_features = features_reshaped[:, :, :, 0, :NUM_WEAPON_FEATURES]
        target_features = features_reshaped[:, :, 0, :, NUM_WEAPON_FEATURES:NUM_WEAPON_FEATURES+NUM_TARGET_FEATURES]
        edge_features = features_reshaped[:, :, :, :, -NUM_EDGE_FEATURES:]

        weapon_h, target_h, edge_h = weapon_features, target_features, edge_features
        for layer in self.gnn_layers:
            weapon_h, target_h, edge_h = layer(weapon_h, target_h, edge_h)

        edge_h_res = self.edge_residual(edge_h)
        edge_scores = self.edge_score(edge_h_res).squeeze(-1)  # [batch, para, W, T]
        edge_scores_flat = edge_scores.view(batch_size, para_size, num_weapons * num_targets)

        # Single global no-op score from pooled context only.
        global_weapon = weapon_h.mean(dim=2)  # [batch, para, hidden]
        global_target = target_h.mean(dim=2)  # [batch, para, hidden]
        global_context = torch.cat([global_weapon, global_target], dim=-1)  # [batch, para, 2*hidden]
        no_action_hidden = self.no_action_proj(global_context)
        no_action_hidden = self.no_action_residual(no_action_hidden)
        no_action_score = self.no_action_score(no_action_hidden)  # [batch, para, 1]

        all_scores = torch.cat([edge_scores_flat, no_action_score], dim=-1)  # [batch, para, W*T+1]

        all_scores = all_scores.masked_fill(~mask.bool(), float('-inf'))
        policy = F.softmax(all_scores, dim=-1)  # [batch, para, W*T+1], ONE joint distribution

        self.assignment_embedding = edge_h_res.view(batch_size, para_size, num_weapons * num_targets, -1)
        no_action_emb = torch.zeros(batch_size, para_size, 1, EMBEDDING_DIM, device=edge_h_res.device)
        self.assignment_embedding = torch.cat([self.assignment_embedding, no_action_emb], dim=2)
        self.current_state = self.assignment_embedding

        return policy, self.assignment_embedding


def create_gnn_actor_sequential():
    """Create the sequential (original BReRLA) GNN actor model."""
    return EdgeAwareGNN_ACTOR_Sequential()


if __name__ == "__main__":
    print("Testing sequential (single global softmax) GNN actor...")

    actor = create_gnn_actor_sequential()
    print(f"Sequential Actor created - Parameters: {sum(p.numel() for p in actor.parameters()):,}")

    batch_size, para_size = 2, 1
    test_assignment = torch.randn(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS + 1, NUM_FEATURES)
    test_prob = torch.randn(batch_size, para_size, NUM_WEAPONS, NUM_TARGETS)
    test_mask = torch.ones(batch_size, para_size, NUM_WEAPONS * NUM_TARGETS + 1)

    with torch.no_grad():
        policy, embeddings = actor(test_assignment, test_prob, test_mask)

    print(f"Policy output shape: {policy.shape}  (expected [{batch_size},{para_size},{NUM_WEAPONS*NUM_TARGETS+1}])")
    assert policy.shape == (batch_size, para_size, NUM_WEAPONS * NUM_TARGETS + 1)
    assert torch.allclose(policy.sum(dim=-1), torch.ones(batch_size, para_size), atol=1e-5)
    print("Sequential GNN actor test completed successfully!")
