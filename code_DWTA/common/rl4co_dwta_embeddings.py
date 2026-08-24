"""
Custom RL4CO embeddings for DWTAEnv (common/rl4co_dwta_env.py).

Deliberately implements a genuine DYNAMIC embedding for remaining target
value (updated at every decode step, not computed once at reset) --
directly informed by tonight's Bug 4 finding elsewhere in this codebase
(common/DWTA_Simulator.py silently never refreshed the actor's own view of
target value, leaving the policy blind to accumulated damage for an entire
episode). RL4CO's AM/POMO defaults (StaticEmbedding, `return 0,0,0`) exist
precisely for problems where node features don't change during rollout
(TSP, CVRP, OP, ...); DWTAP is not such a problem, and reusing that default
here would silently reproduce the exact same class of bug in a different
codebase.
"""
import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index


class DWTAInitEmbedding(nn.Module):
    """Static per-target features (value, time window) -- computed once at
    reset, since these do not change during the episode. Remaining value
    (which DOES change) is intentionally excluded here and handled by
    DWTADynamicEmbedding instead. An extra learned "no-op" node embedding is
    appended (index N, matching DWTAEnv's action convention: 0..N-1 =
    targets, N = no-op) so the node count matches the action space size
    (N+1) -- analogous to how OP/CVRP append a depot embedding."""

    def __init__(self, embed_dim, linear_bias=True):
        super().__init__()
        node_dim = 3  # value, tw_start, tw_end (all normalized)
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.noop_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)

    def forward(self, td):
        bs = td["value"].size(0)
        T = max(1.0, float(td["tw_end"].max().item()) + 1.0)
        feat = torch.stack(
            [
                td["value"] / 10.0,
                td["tw_start"] / T,
                td["tw_end"] / T,
            ],
            dim=-1,
        )
        target_emb = self.init_embed(feat)  # [batch, N, embed_dim]
        noop_emb = self.noop_embed.expand(bs, 1, -1)  # [batch, 1, embed_dim]
        return torch.cat([target_emb, noop_emb], dim=1)  # [batch, N+1, embed_dim]


class DWTADynamicEmbedding(nn.Module):
    """Refreshes each target's remaining-value ratio into the attention
    mechanism's key/value/logit projections at EVERY decode step -- the
    piece that must never go stale (see module docstring)."""

    def __init__(self, embed_dim, linear_bias=False):
        super().__init__()
        self.projection = nn.Linear(1, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        bs = td["value"].size(0)
        remaining_ratio = (td["remaining_value"] / (td["value"] + 1e-8)).unsqueeze(-1)  # [batch, N, 1]
        # No-op node's "remaining ratio" is a fixed neutral 1.0 -- it never
        # depletes (there is nothing to deplete), included only so the
        # dynamic projection's node count matches init embedding's N+1.
        noop_ratio = torch.ones(bs, 1, 1, device=td.device, dtype=remaining_ratio.dtype)
        full_ratio = torch.cat([remaining_ratio, noop_ratio], dim=1)  # [batch, N+1, 1]
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            full_ratio
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class DWTAContext(nn.Module):
    """Decoder query context: which weapon is deciding this step (its own
    ammo/reload state and its compatibility with every target), plus a
    round-progress feature. DWTAP has no "current node" concept (unlike
    routing problems, where the query naturally depends on where the
    vehicle currently is) -- the query here is inherently about the
    upcoming DECIDER (a weapon), not a location, so this does not reuse
    EnvContext's `_cur_node_embedding` gather-by-current-node pattern.

    Size-agnostic by construction (a earlier version was NOT: it directly
    concatenated the weapon's raw N-length kill-probability row into a
    fixed-width nn.Linear(N+3, embed_dim), which baked the number of targets
    into the weight matrix's shape and made a trained policy unusable at any
    other N -- defeating the entire point of an attention-based constructive
    policy, which is supposed to generalize across problem sizes the same
    way it does in TSP/CVRP. Fixed here by using the weapon's per-target
    kill-probability row as ATTENTION WEIGHTS over the already-computed
    (size-invariant) target embeddings, producing one embed_dim-sized vector
    regardless of N -- the same pattern CVRP/OP use for their own context.
    M (num_weapon) and T (max_time) are ALSO read dynamically from `td`
    (never stored on self) -- needed for multi-scale training
    (train_rl4co_am_multiscale.py), where M/N/T all vary batch to batch and
    a single policy is reused across every size."""

    def __init__(self, embed_dim, linear_bias=False):
        super().__init__()
        # weighted_target_context (embed_dim) + ammo_left (1) + reload_ready (1) + round_frac (1)
        self.project_context = nn.Linear(embed_dim + 3, embed_dim, bias=linear_bias)

    def forward(self, embeddings, td):
        bs = td.batch_size[0]
        idx = torch.arange(bs, device=td.device)
        M = td["ammo_left"].size(-1)
        T = td["tw_end"].max().item() + 1  # every target's tw_end == T-1 by construction
        weapon = td["step_idx"] % M
        rnd = td["step_idx"] // M
        N = td["prob"].size(-1)

        weapon_prob_row = td["prob"][idx, weapon]  # [batch, N]
        target_embed = embeddings[:, :N, :]  # [batch, N, embed_dim] -- exclude the no-op node
        attn_weights = weapon_prob_row / (weapon_prob_row.sum(-1, keepdim=True) + 1e-8)
        weighted_target_context = torch.einsum('bn,bnd->bd', attn_weights, target_embed)  # [batch, embed_dim]

        ammo_left = td["ammo_left"][idx, weapon].unsqueeze(-1) / max(1.0, float(td["ammo_left"].max().item()))
        reload_ready = (rnd.float() > td["reload_until"][idx, weapon]).float().unsqueeze(-1)
        round_frac = (rnd.float() / max(1, T)).unsqueeze(-1)

        feat = torch.cat([weighted_target_context, ammo_left, reload_ready, round_frac], dim=-1)
        return self.project_context(feat)
