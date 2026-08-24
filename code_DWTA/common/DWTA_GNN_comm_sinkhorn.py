"""
EdgeAwareGNN_ACTOR_COMM (common/DWTA_GNN_comm.py) plus a differentiable
masked partial-Sinkhorn coordination bonus (common/sinkhorn_coordination.py)
on the edge (target-choice) scores.

Motivation (2026-08-20): the comm layer (weapon-to-weapon attention over
EMBEDDINGS) closed most, not all, of the quality gap to the Sequential
decoder on the moderate curriculum (brerla_comm_layer_and_multifire_auction
memory: CommSCoPE 0.1465 vs Sequential 0.1432). Every attempt to close the
rest by putting the (discrete, non-differentiable) auction inside the
training loop made things WORSE and less stable (brerla_auction_train_
inference_mismatch memory, Findings 4-6). This variant instead adds a fully
DIFFERENTIABLE coordination signal directly on the (weapon, target) SCORE
matrix -- partial Sinkhorn row/column log-normalization, blended in as a
learnable-weight additive bonus -- so gradients flow through it natively
during REINFORCE and the exact same forward pass is used for both training
and inference (no train/inference mismatch by construction).

New file, does not modify DWTA_GNN.py or DWTA_GNN_comm.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import ResidualBlock, ReplayMemory, EdgeAwareGNNLayer
from .sinkhorn_coordination import masked_sinkhorn_log_bonus


class EdgeAwareGNN_ACTOR_COMM_SINKHORN(nn.Module):
    """EdgeAwareGNN_ACTOR_COMM + a differentiable partial-Sinkhorn
    coordination bonus on the edge (target-choice) scores."""

    def __init__(self, num_layers=3, num_heads=HEAD_NUM, sinkhorn_iters=3, sinkhorn_temperature=1.0):
        super().__init__()

        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temperature = sinkhorn_temperature
        # Learnable, starts at 0 so early training behaves EXACTLY like the
        # plain comm actor (no coordination bonus at all) until there is a
        # gradient reason to trust the signal -- avoids destabilizing early
        # training with an untuned coordination term (same caution as this
        # project's other additive-signal designs, e.g. auction_refinement's
        # guide_weight sweep).
        self.sinkhorn_scale = nn.Parameter(torch.tensor(0.0))

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
            self.optimizer, step_size=50, gamma=0.9
        )

    def forward(self, assignment_embedding, prob, mask):
        """Same signature/shapes as EdgeAwareGNN_ACTOR_COMM.forward."""
        batch_size, para_size = assignment_embedding.shape[:2]

        if prob is None:
            num_weapons, num_targets = NUM_WEAPONS, NUM_TARGETS
        else:
            num_weapons, num_targets = prob.shape[2], prob.shape[3]

        n_feat = assignment_embedding.size(-1)
        assignments = assignment_embedding[:, :, :-1, :]
        features_reshaped = assignments.view(batch_size, para_size, num_weapons, num_targets, n_feat)

        weapon_features = features_reshaped[:, :, :, 0, :NUM_WEAPON_FEATURES]
        target_features = features_reshaped[:, :, 0, :, NUM_WEAPON_FEATURES:NUM_WEAPON_FEATURES+NUM_TARGET_FEATURES]
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
        edge_scores = self.edge_score(edge_h_res).squeeze(-1)  # [batch, para, W, T]

        # Differentiable coordination bonus: nudge scores away from
        # redundant same-round, same-target concentration across weapons,
        # using the SAME per-(weapon,target) legality this round already
        # carries in `mask` (its first num_targets columns -- see
        # common/DWTA_Simulator.py::mask_per_weapon, which concatenates the
        # legal-edge tensor with an always-True no-op column in exactly this
        # order).
        edge_legal_mask = mask[:, :, :, :num_targets].bool()
        sinkhorn_bonus = masked_sinkhorn_log_bonus(
            edge_scores, edge_legal_mask,
            n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temperature,
        )
        edge_scores = edge_scores + self.sinkhorn_scale * sinkhorn_bonus

        global_weapon = weapon_comm.mean(dim=2)
        global_target = target_h.mean(dim=2)
        global_context = torch.cat([global_weapon, global_target], dim=-1)
        global_context_exp = global_context.unsqueeze(2).expand(-1, -1, num_weapons, -1)
        no_action_input = torch.cat([weapon_comm, global_context_exp], dim=-1)
        no_action_hidden = self.no_action_proj(no_action_input)
        no_action_hidden = self.no_action_residual(no_action_hidden)
        no_action_score = self.no_action_score(no_action_hidden).squeeze(-1)

        all_scores = torch.cat([edge_scores, no_action_score.unsqueeze(-1)], dim=-1)
        all_scores = all_scores.masked_fill(~mask.bool(), float('-inf'))
        policy = F.softmax(all_scores, dim=-1)

        self.assignment_embedding = edge_h_res.view(batch_size, para_size, num_weapons * num_targets, -1)
        no_action_emb = torch.zeros(batch_size, para_size, 1, EMBEDDING_DIM, device=edge_h_res.device)
        self.assignment_embedding = torch.cat([self.assignment_embedding, no_action_emb], dim=2)
        self.current_state = self.assignment_embedding

        return policy, self.assignment_embedding


def create_gnn_actor_comm_sinkhorn():
    return EdgeAwareGNN_ACTOR_COMM_SINKHORN()
