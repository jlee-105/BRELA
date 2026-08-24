"""Binary fire/hold actor (see common/DWTA_GNN_binary.py's docstring for the
RL+Auction hybrid rationale) plus the weapon-to-weapon self-attention
"communication" layer (see common/DWTA_GNN_comm.py's docstring for why the
naive mean-pool global context was insufficient).

Combines three findings from the same investigation:
  1. Target assignment doesn't need to be learned -- delegate it entirely to
     the auction (DWTA_GNN_binary.py's original rationale), which also gives
     REINFORCE a clean, unambiguous log-prob (the Bernoulli fire/hold
     decision) with no train/inference mismatch -- unlike the full N+1-way
     actor, whose target choice is discarded by the auction at inference
     but IS part of what it's trained to predict, a mismatch identified as
     the likely reason a policy-guided auction blend gave no real gain (see
     memory brerla_auction_train_inference_mismatch.md).
  2. A naive mean-pool global context is a weak substitute for genuine
     weapon-to-weapon awareness when deciding fire-vs-hold (DWTA_GNN_comm.py).
  3. The auction used downstream should be the capacitated many-to-one
     variant (auction_round_action_multifire), not the 1:1 eviction one --
     see brerla_auction_train_inference_mismatch.md Finding 1-2.

This file only changes the fire-probability head; it does not touch target
assignment (still 100% auction's job, at both train and inference time).
"""
import torch
import torch.nn as nn

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import EdgeAwareGNNLayer, ResidualBlock, ReplayMemory


class EdgeAwareGNN_ACTOR_BINARY_COMM(nn.Module):
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

        self.fire_proj = nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM)
        self.fire_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.fire_score = nn.Linear(EMBEDDING_DIM, 1)
        # Bias toward firing at initialization -- mirrors the original N+1-way
        # actor's skip_scorer.bias=-2.0 (biased AGAINST early no-op, see
        # DISCUSSION_NOTES.md "Skip bias" row), but in the opposite direction
        # since this head's own "do nothing" is fire_prob<0.5. Found empirically
        # (2026-08-20) that default PyTorch init for this architecture happens
        # to land fire_prob around 0.27-0.31 at random init (verified directly,
        # not assumed) -- combined with eval's deterministic fire_prob>0.5
        # threshold, this starts training from an already-collapsed "never
        # fire" policy at small/medium scale, which then reads as a training
        # failure at low step counts when it is really an initialization bias.
        nn.init.constant_(self.fire_score.bias, 2.0)

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
        """Same signature/shapes as EdgeAwareGNN_ACTOR_BINARY.forward."""
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
        fire_input = torch.cat([weapon_comm, global_context_exp], dim=-1)
        fire_hidden = self.fire_proj(fire_input)
        fire_hidden = self.fire_residual(fire_hidden)
        fire_logit = self.fire_score(fire_hidden).squeeze(-1)  # [batch, para, W]

        has_legal_target = mask[:, :, :, :-1].any(dim=-1)
        fire_logit = fire_logit.masked_fill(~has_legal_target.bool(), float('-inf'))
        fire_prob = torch.sigmoid(fire_logit)
        fire_prob = torch.where(has_legal_target.bool(), fire_prob, torch.zeros_like(fire_prob))

        return fire_prob, None


def create_gnn_actor_binary_comm():
    return EdgeAwareGNN_ACTOR_BINARY_COMM()
