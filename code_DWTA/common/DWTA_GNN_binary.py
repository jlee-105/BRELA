"""
Binary fire/hold actor for the RL+Auction hybrid decomposition (see memory
brerla_rl_auction_hybrid_decomposition.md / discussion 2026-08-07).

Motivation: SCoPE's original parallel decoder (common/DWTA_GNN.py) learns
two different kinds of reasoning jointly in one N+1-way softmax per weapon:
(1) WHETHER to fire this round vs. hold ammo for a better future
opportunity (a temporal-credit-assignment problem -- the no-op action's
immediate reward is always exactly 0, so its credit depends entirely on
downstream outcomes), and (2) WHICH target to fire at (a combinatorial
coordination problem across simultaneously-deciding weapons). (2) has a
known, exact, closed-form-computable answer that does not require learning
at all (see common/auction_refinement.py); only (1) genuinely requires
learning. This actor learns ONLY (1): a per-weapon binary fire probability,
discarding target selection entirely -- that is handled downstream by the
auction, using the environment's exact remaining_value x P formula, not
this network's judgment.

Reuses the existing GNN message-passing encoder (EdgeAwareGNNLayer,
ResidualBlock) from common/DWTA_GNN.py unchanged/unmodified -- only the
output head differs (this file does not edit DWTA_GNN.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import EdgeAwareGNNLayer, ResidualBlock, ReplayMemory


class EdgeAwareGNN_ACTOR_BINARY(nn.Module):
    """
    Same GNN encoder as EdgeAwareGNN_ACTOR, but the output head is a single
    per-weapon fire probability (sigmoid), not a per-weapon N+1-way softmax
    over targets. Structurally this is exactly EdgeAwareGNN_ACTOR's
    no_action_score sub-head in isolation (weapon's own embedding + pooled
    global context -> scalar), just interpreted as P(fire) instead of
    P(no-op) -- reusing that same architectural pattern since it already
    proved effective at producing a per-weapon, context-aware scalar
    judgment (see DWTA_GNN.py's no_action_score docstring).
    """
    def __init__(self, num_layers=3):
        super().__init__()

        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGNNLayer() for _ in range(num_layers)
        ])

        self.fire_proj = nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM)
        self.fire_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.fire_score = nn.Linear(EMBEDDING_DIM, 1)

        self.replay_memory = ReplayMemory(capacity=BUFFER_SIZE)

        import torch.optim as optim
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=ACTOR_LEARNING_RATE,
            weight_decay=ACTOR_WEIGHT_DECAY
        )
        self.lr_stepper = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.9
        )

    def forward(self, assignment_embedding, prob, mask):
        """
        Args:
            assignment_embedding: [batch, para, NUM_WEAPONS*NUM_TARGETS+1, 9]
            prob: [batch, para, num_weapons, num_targets] or None
            mask: [batch, para, num_weapons, num_targets+1] -- per-weapon
                legality mask; a weapon with NO legal target (mask[...,m,:-1]
                all False) is forced to hold regardless of the network's own
                score, matching how illegal targets are masked in the
                original actor.

        Returns:
            fire_prob: [batch, para, num_weapons] -- P(fire) per weapon,
                already zeroed out for weapons with no legal target.
            None: kept for call-signature parity with EdgeAwareGNN_ACTOR
                (which returns assignment_embedding here); unused downstream
                since target selection is not this network's job.
        """
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

        global_weapon = weapon_h.mean(dim=2)
        global_target = target_h.mean(dim=2)
        global_context = torch.cat([global_weapon, global_target], dim=-1)
        global_context_exp = global_context.unsqueeze(2).expand(-1, -1, num_weapons, -1)
        fire_input = torch.cat([weapon_h, global_context_exp], dim=-1)
        fire_hidden = self.fire_proj(fire_input)
        fire_hidden = self.fire_residual(fire_hidden)
        fire_logit = self.fire_score(fire_hidden).squeeze(-1)  # [batch, para, W]

        # A weapon with no legal target at all cannot fire regardless of the
        # network's own score.
        has_legal_target = mask[:, :, :, :-1].any(dim=-1)  # [batch, para, W]
        fire_logit = fire_logit.masked_fill(~has_legal_target.bool(), float('-inf'))
        fire_prob = torch.sigmoid(fire_logit)
        fire_prob = torch.where(has_legal_target.bool(), fire_prob, torch.zeros_like(fire_prob))

        return fire_prob, None


def create_gnn_actor_binary():
    return EdgeAwareGNN_ACTOR_BINARY()
