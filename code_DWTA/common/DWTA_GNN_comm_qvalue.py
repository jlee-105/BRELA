"""
Q-VALUE variant of EdgeAwareGNN_ACTOR_COMM (common/DWTA_GNN_comm.py): same
GNN + weapon-to-weapon comm layer, but the final head returns RAW per-
(weapon,target) and per-weapon-no-op SCORES directly as Q-values -- no
softmax, no categorical distribution, nothing sampled.

Motivation (2026-08-20, end-of-session pivot): inspired by Boyko/Bakolas-
style "REDA" multi-agent RL for sequential assignment problems (satellite
task assignment) -- see brerla_sinkhorn_coordination_experiment memory --
which trains per-agent Q-VALUES fed as a benefit matrix into a classical
assignment solver (Hungarian, for their 1:1 setting), via TD/regression
loss, NOT policy-gradient sampling. This sidesteps every instability this
whole session's REINFORCE-based and rollout-supervised binary attempts hit,
because there is no sampled action / log-prob anywhere in the loss.

DWTAP is many-to-one (not REDA's 1:1), so the downstream solver here is
the CAPACITATED auction (auction_round_action_multifire_qvalue, in
common/auction_refinement_qvalue.py) instead of Hungarian -- the direct
generalization already grounded in Bertsekas & Castanon (1989)'s
transportation-problem auction theory, same citation already used for the
existing multifire auction variant.

New file -- does not modify common/DWTA_GNN_comm.py.
"""
import torch
import torch.nn as nn

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import ResidualBlock, ReplayMemory, EdgeAwareGNNLayer


class EdgeAwareGNN_ACTOR_COMM_QVALUE(nn.Module):
    """Same architecture as EdgeAwareGNN_ACTOR_COMM, but returns raw
    (weapon,target)+no-op Q-values instead of a softmax policy."""

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

        self.edge_comm_proj = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)
        self.edge_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.edge_score = nn.Linear(EMBEDDING_DIM, 1)

        self.no_action_proj = nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM)
        self.no_action_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.no_action_score = nn.Linear(EMBEDDING_DIM, 1)
        # Bias no-op's Q-value LOW at init (mirrors the original N+1-way
        # actor's skip_scorer.bias=-2.0) -- without this, the auction (which
        # picks greedily from whatever the network's CURRENT Q-values say,
        # with NO exploration mechanism at all) can get stuck never firing
        # from step 1 if no-op's random-init score happens to edge out
        # firing for most weapons: it never fires, so it never observes a
        # non-zero reward to learn "firing is good" from -- a cold-start
        # trap, confirmed empirically (2026-08-20): destruction=0.0000 at
        # both step 10 and step 20 of an unbiased run.
        nn.init.constant_(self.no_action_score.bias, -2.0)

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
        """Same signature/shapes as EdgeAwareGNN_ACTOR_COMM.forward, but
        returns q_values [B,P,M,N+1] (raw, unmasked-softmax scores; illegal
        entries set to a large negative number, NOT -inf, so they remain
        finite/differentiable) instead of a softmax policy."""
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

        weapon_comm_exp = weapon_comm.unsqueeze(3).expand(-1, -1, -1, num_targets, -1)
        edge_comm_input = torch.cat([edge_h, weapon_comm_exp], dim=-1)
        edge_h_comm = self.edge_comm_proj(edge_comm_input)
        edge_h_res = self.edge_residual(edge_h_comm)
        edge_q = self.edge_score(edge_h_res).squeeze(-1)  # [batch, para, W, T]

        global_weapon = weapon_comm.mean(dim=2)
        global_target = target_h.mean(dim=2)
        global_context = torch.cat([global_weapon, global_target], dim=-1)
        global_context_exp = global_context.unsqueeze(2).expand(-1, -1, num_weapons, -1)
        no_action_input = torch.cat([weapon_comm, global_context_exp], dim=-1)
        no_action_hidden = self.no_action_proj(no_action_input)
        no_action_hidden = self.no_action_residual(no_action_hidden)
        no_action_q = self.no_action_score(no_action_hidden).squeeze(-1)  # [batch, para, W]

        q_values = torch.cat([edge_q, no_action_q.unsqueeze(-1)], dim=-1)  # [batch, para, W, T+1]
        q_values = q_values.masked_fill(~mask.bool(), -1e4)

        return q_values, None


def create_gnn_actor_comm_qvalue():
    return EdgeAwareGNN_ACTOR_COMM_QVALUE()
