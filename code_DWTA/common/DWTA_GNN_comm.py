"""
Parallel multi-pointer GNN actor with an added WEAPON-TO-WEAPON
self-attention ("communication") layer -- new variant, does not modify
common/DWTA_GNN.py at all.

Motivation (2026-08-11): on the MODERATE temporal-dilemma curriculum, the
original EdgeAwareGNN_ACTOR (see its own docstring: "no learned conflict
handler is used ... coordination is expected to emerge from the shared
embeddings alone -- the soft-coupling hypothesis under test") plus auction
refinement (SCoPE) lost to the plain Sequential decoder on 11/12 held-out
configs (see brerla_moderate_full_baseline_table memory). Diagnosis:
auction only coordinates WHICH TARGET already-firing weapons should hit --
it cannot coordinate WHETHER a weapon should fire in the first place, since
that's decided independently per weapon in one simultaneous forward pass.
Sequential gets this coordination for free (each weapon's decision is
conditioned on every earlier weapon's ALREADY-COMMITTED choice this same
round); Parallel's only substitute was a crude mean-pool "global context"
feeding the no-op score (see EdgeAwareGNN_ACTOR.forward -- global_weapon =
weapon_h.mean(dim=2)), which is permutation-invariant and much weaker than
genuine per-weapon-pair awareness, and it never touched the edge (target-
choice) scores at all.

Fix here: one extra self-attention layer over the W weapon embeddings
(nn.MultiheadAttention, weapon-to-weapon, NOT weapon-to-target), inserted
after the existing EdgeAwareGNNLayer stack, feeding a genuinely
weapon-aware "communicated" embedding into BOTH the no-op score (replacing
the old naive mean-pool) AND the edge (target-choice) scores (previously
edge_scores came from edge_h alone, with zero direct weapon_h information)
-- this is the PARCO-style "conflict handler" the original docstring
explicitly opted out of; adding it here directly targets the identified gap
while staying a single forward pass (no Sequential-style M-step loop).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Dynamic_HYPER_PARAMETER import *
from .TORCH_OBJECTS import *
from .DWTA_GNN import ResidualBlock, ReplayMemory, EdgeAwareGNNLayer


class EdgeAwareGNN_ACTOR_COMM(nn.Module):
    """Same as EdgeAwareGNN_ACTOR, plus a weapon-to-weapon self-attention
    ("communication") layer before the final scoring heads."""

    def __init__(self, num_layers=3, num_heads=HEAD_NUM):
        super().__init__()

        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGNNLayer() for _ in range(num_layers)
        ])

        # Weapon-to-weapon communication: standard Transformer-encoder-style
        # self-attention + residual block, operating purely over the W
        # weapon nodes (not weapon-target edges) -- this is what was
        # missing: a way for each weapon's final decision to depend on a
        # learned, per-weapon-pair-aware summary of what OTHER weapons in
        # this same round are about to do, not just a flat average.
        self.weapon_comm_attn = nn.MultiheadAttention(
            embed_dim=EMBEDDING_DIM, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.weapon_comm_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.weapon_comm_ffn = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)

        # Edge (target-choice) scorer now takes edge_h AND the communicated
        # weapon embedding for that edge's weapon -- previously edge_scores
        # came from edge_h alone with no direct weapon_h signal at all.
        self.edge_comm_proj = nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM)
        self.edge_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.1)
        self.edge_score = nn.Linear(EMBEDDING_DIM, 1)

        # No-op scorer: same structure as the original, but global_context
        # is now built from the COMMUNICATED weapon embeddings, not a naive
        # mean pool of the pre-communication ones.
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
        """Same signature/shapes as EdgeAwareGNN_ACTOR.forward."""
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

        # Weapon-to-weapon communication: flatten (batch, para) into one
        # attention-batch dimension, attend over the W weapon axis.
        bp = batch_size * para_size
        weapon_flat = weapon_h.reshape(bp, num_weapons, EMBEDDING_DIM)
        comm_out, _ = self.weapon_comm_attn(weapon_flat, weapon_flat, weapon_flat, need_weights=False)
        weapon_comm = self.weapon_comm_norm(weapon_flat + comm_out)
        weapon_comm = self.weapon_comm_ffn(weapon_comm)
        weapon_comm = weapon_comm.view(batch_size, para_size, num_weapons, EMBEDDING_DIM)

        # Edge (target-choice) scores: edge_h combined with ITS weapon's
        # communicated embedding (previously edge_h alone).
        weapon_comm_exp = weapon_comm.unsqueeze(3).expand(-1, -1, -1, num_targets, -1)
        edge_comm_input = torch.cat([edge_h, weapon_comm_exp], dim=-1)  # [batch, para, W, T, 2*hidden]
        edge_h_comm = self.edge_comm_proj(edge_comm_input)
        edge_h_res = self.edge_residual(edge_h_comm)
        edge_scores = self.edge_score(edge_h_res).squeeze(-1)  # [batch, para, W, T]

        # No-op score: communicated weapon embedding + pooled global context
        # (kept for parity with the original signal, now built from the
        # communicated embeddings).
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


def create_gnn_actor_comm():
    return EdgeAwareGNN_ACTOR_COMM()
