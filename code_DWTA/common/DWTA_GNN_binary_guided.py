"""RL+Auction hybrid, LEARNED-guidance variant: the actor outputs THREE
things instead of one --
  1. fire_prob [B,P,W]: Bernoulli fire/hold probability per weapon, as in
     DWTA_GNN_binary_comm.py.
  2. target_pref [B,P,W,N]: softmax target preference per weapon (over
     legal targets), computed from an edge-scoring head (same pattern as
     DWTA_GNN_comm.py's edge_score, but not merged into a single N+1-way
     softmax with no-op -- fire/hold is handled separately by (1)).
  3. guide_mu [B,P]: a learned, per-instance scalar controlling how much
     the capacitated auction's marginal-value ranking should defer to (2)
     -- see common/auction_refinement.py::auction_round_action_multifire_guided.

Motivation: a fixed, hand-tuned guide_weight (tried in
eval_guided_auction_sweep.py, settled on 0.5 by a coarse sweep) is not
principled -- it should be learned like everything else the policy decides.
Since the auction itself is a hard discrete argmax loop (no gradient path),
guide_weight is treated as a SAMPLED action too (Normal(guide_mu, fixed
std)), with its own REINFORCE log-prob term, exactly like fire_prob's
Bernoulli. Similarly, (2)'s contribution to the loss is NOT "log-prob of
whatever target the policy would have argmaxed" (that reintroduces the
train/inference mismatch documented in brerla_auction_train_inference_mismatch
memory) -- instead, credit is computed AFTER the auction has decided the
REAL executed target for each firing weapon, as log(target_pref[executed
target]) under the policy's own current distribution. This way the gradient
always reflects what actually happened, never a discarded hypothetical.

Reuses the shared GNN encoder + weapon-to-weapon comm layer from
DWTA_GNN_comm.py's pattern (see that file's docstring for why a naive
mean-pool was insufficient); does not modify DWTA_GNN.py, DWTA_GNN_comm.py,
or DWTA_GNN_binary_comm.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import EdgeAwareGNNLayer, ResidualBlock, ReplayMemory

GUIDE_WEIGHT_MAX = 4.0  # matches the top of the hand-tuned sweep's tested range
GUIDE_WEIGHT_STD = 0.3  # fixed exploration std for the learned guide_weight's Normal


class EdgeAwareGNN_ACTOR_BINARY_GUIDED(nn.Module):
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

        # Head 1: fire/hold (unchanged from DWTA_GNN_binary_comm.py)
        self.fire_proj = nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM)
        self.fire_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.fire_score = nn.Linear(EMBEDDING_DIM, 1)

        # Head 2: target preference (edge scorer, DWTA_GNN_comm.py pattern)
        self.edge_comm_proj = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)
        self.edge_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.edge_score = nn.Linear(EMBEDDING_DIM, 1)

        # Head 3: learned guide_weight (per-instance scalar, from pooled global context)
        self.guide_proj = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)
        self.guide_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.guide_score = nn.Linear(EMBEDDING_DIM, 1)

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
        mask: [batch, para, W, N+1] -- same per-weapon legality convention
            as EdgeAwareGNN_ACTOR_BINARY_COMM (mask[...,:-1] = legal targets).

        Returns:
            fire_prob: [batch, para, W]
            target_pref: [batch, para, W, N] -- softmax over LEGAL targets
                only (illegal targets get exactly 0 probability); weapons
                with no legal target get a uniform-over-nothing (all-zero)
                row, never consumed since fire_prob is already 0 there.
            guide_mu: [batch, para] -- mean of the learned guide_weight
                distribution, already scaled to [0, GUIDE_WEIGHT_MAX].
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

        # Head 1: fire/hold
        fire_input = torch.cat([weapon_comm, global_context_exp], dim=-1)
        fire_hidden = self.fire_proj(fire_input)
        fire_hidden = self.fire_residual(fire_hidden)
        fire_logit = self.fire_score(fire_hidden).squeeze(-1)

        has_legal_target = mask[:, :, :, :-1].any(dim=-1)
        fire_logit = fire_logit.masked_fill(~has_legal_target.bool(), float('-inf'))
        fire_prob = torch.sigmoid(fire_logit)
        fire_prob = torch.where(has_legal_target.bool(), fire_prob, torch.zeros_like(fire_prob))

        # Head 2: target preference
        weapon_comm_exp = weapon_comm.unsqueeze(3).expand(-1, -1, -1, num_targets, -1)
        edge_comm_input = torch.cat([edge_h, weapon_comm_exp], dim=-1)
        edge_h_comm = self.edge_comm_proj(edge_comm_input)
        edge_h_res = self.edge_residual(edge_h_comm)
        edge_scores = self.edge_score(edge_h_res).squeeze(-1)  # [batch, para, W, T]

        legal_targets = mask[:, :, :, :-1].bool()  # [batch, para, W, T]
        edge_scores = edge_scores.masked_fill(~legal_targets, float('-inf'))
        target_pref = F.softmax(edge_scores, dim=-1)
        # Weapons with no legal target: softmax of all -inf is nan -- zero it out
        # (never consumed, since fire_prob is already 0 for these weapons).
        target_pref = torch.nan_to_num(target_pref, nan=0.0)

        # Head 3: learned guide_weight (per-instance, from pooled context)
        guide_hidden = self.guide_proj(global_context)
        guide_hidden = self.guide_residual(guide_hidden)
        guide_raw = self.guide_score(guide_hidden).squeeze(-1)  # [batch, para]
        guide_mu = torch.sigmoid(guide_raw) * GUIDE_WEIGHT_MAX

        return fire_prob, target_pref, guide_mu


def create_gnn_actor_binary_guided():
    return EdgeAwareGNN_ACTOR_BINARY_GUIDED()
