"""
IMPROVEMENT-policy actor: scores, for each weapon at a given round state,
how desirable it would be to FLIP that weapon's current fire/hold decision.

This is not a construction policy (it never decides what to do from
scratch). It operates on top of an already-complete solution produced by a
frozen base pipeline (trained actor + auction), in the
learning-to-improve / neural local search paradigm (Chen & Tian, NeurIPS
2019 "Learning to Perform Local Rewriting"; Lu, Zhang & Yang, ICLR 2020;
Ma et al., NeurIPS 2021 "DACT") -- a family this project has never tried,
structurally different from every construction-style attempt so far.

Motivation (2026-08-21): every construction-style approach tried
(REINFORCE on N+1-way / binary fire-hold, supervised rollout labels, ES,
Q-value regression + auction) either collapsed or underperformed -- see
brerla_sinkhorn_coordination_experiment memory. All shared one structure:
decide each round's action from scratch under a sparse episode-level
reward. Improvement-based search instead gives a DENSE, directly
attributable reward (the objective delta caused by one specific edit) and
starts from a known-good solution, so it cannot structurally regress below
its base.

Architecture is deliberately identical to EdgeAwareGNN_ACTOR_BINARY_COMM
(common/DWTA_GNN_binary_comm.py) -- GNN + weapon-to-weapon comm layer --
except the head returns RAW per-weapon flip scores (logits) with no
sigmoid, since these are pooled across all (round, weapon) slots into one
softmax over the whole edit space, not treated as independent Bernoullis.

New file -- does not modify DWTA_GNN_binary_comm.py.
"""
import torch
import torch.nn as nn

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import EdgeAwareGNNLayer, ResidualBlock, ReplayMemory


class EdgeAwareGNN_ACTOR_IMPROVE(nn.Module):
    def __init__(self, num_layers=3, num_heads=HEAD_NUM):
        super().__init__()

        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGNNLayer() for _ in range(num_layers)
        ])

        self.weapon_comm_attn = nn.MultiheadAttention(
            embed_dim=EMBEDDING_DIM, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.weapon_comm_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.weapon_comm_ffn = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)

        self.flip_proj = nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM)
        self.flip_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.flip_score = nn.Linear(EMBEDDING_DIM, 1)

        self.replay_memory = ReplayMemory(capacity=BUFFER_SIZE)

        import torch.optim as optim
        self.optimizer = optim.Adam(
            self.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY
        )

    def forward(self, assignment_embedding, prob, mask):
        """Returns flip_score [batch, para, W] -- raw logits, higher = more
        desirable to flip this weapon's current decision at this round."""
        batch_size, para_size = assignment_embedding.shape[:2]
        if prob is None:
            num_weapons, num_targets = NUM_WEAPONS, NUM_TARGETS
        else:
            num_weapons, num_targets = prob.shape[2], prob.shape[3]

        n_feat = assignment_embedding.size(-1)
        assignments = assignment_embedding[:, :, :-1, :]
        features_reshaped = assignments.view(batch_size, para_size, num_weapons, num_targets, n_feat)

        weapon_features = features_reshaped[:, :, :, 0, :NUM_WEAPON_FEATURES]
        target_features = features_reshaped[:, :, 0, :, NUM_WEAPON_FEATURES:NUM_WEAPON_FEATURES + NUM_TARGET_FEATURES]
        edge_features = features_reshaped[:, :, :, :, -NUM_EDGE_FEATURES:]

        weapon_h, target_h, edge_h = weapon_features, target_features, edge_features
        for layer in self.gnn_layers:
            weapon_h, target_h, edge_h = layer(weapon_h, target_h, edge_h)

        bp = batch_size * para_size
        weapon_flat = weapon_h.reshape(bp, num_weapons, EMBEDDING_DIM)
        comm_out, _ = self.weapon_comm_attn(weapon_flat, weapon_flat, weapon_flat, need_weights=False)
        weapon_comm = self.weapon_comm_norm(weapon_flat + comm_out)
        weapon_comm = self.weapon_comm_ffn(weapon_comm)
        weapon_comm = weapon_comm.view(batch_size, para_size, num_weapons, EMBEDDING_DIM)

        global_weapon = weapon_comm.mean(dim=2)
        global_target = target_h.mean(dim=2)
        global_context = torch.cat([global_weapon, global_target], dim=-1)
        global_context_exp = global_context.unsqueeze(2).expand(-1, -1, num_weapons, -1)
        flip_input = torch.cat([weapon_comm, global_context_exp], dim=-1)
        flip_hidden = self.flip_proj(flip_input)
        flip_hidden = self.flip_residual(flip_hidden)
        flip_logit = self.flip_score(flip_hidden).squeeze(-1)  # [batch, para, W]

        return flip_logit, None


def create_gnn_actor_improve():
    return EdgeAwareGNN_ACTOR_IMPROVE()
