"""
Masked partial-Sinkhorn coordination for the parallel multi-pointer actor's
per-round (weapon, target) scores.

Motivation: the weapon-to-weapon comm layer (DWTA_GNN_comm.py) coordinates
via attention over WEAPON EMBEDDINGS -- an indirect, representation-level
signal. Auction refinement (auction_refinement.py) coordinates via a
DISCRETE, non-differentiable, post-hoc reassignment of the sampled action;
every attempt to put that inside the training loop made results WORSE and
less stable (brerla_auction_train_inference_mismatch memory, Findings 4-6:
binary+comm+multifire-in-loop training consistently underperformed and was
more unstable than inference-only auction swaps on an already-trained
checkpoint).

This module instead nudges the raw (weapon, target) SCORE MATRIX itself
toward anti-concentration -- multiple weapons independently scoring the same
target very high get pushed apart -- via a few (deliberately NOT converged)
Sinkhorn row/column log-normalization iterations. Because it is just
elementwise/logsumexp ops on the existing score tensor, it is fully
differentiable end-to-end: the exact same forward pass (and therefore the
exact same coordination behavior) is used during both training and
inference, so there is no train/inference mismatch to diagnose.

IMPORTANT: DWTAP is a MANY-TO-ONE problem (multiple weapons CAN legally
engage the same target in the same round -- see auction_refinement.py's own
module docstring). Full Sinkhorn convergence drives a matrix toward a
doubly-stochastic / near-permutation structure, which is the WRONG limit
here (it would forbid legitimate multi-fire on one high-value target).
`masked_sinkhorn_log_bonus` therefore runs only a small, fixed number of
iterations (n_iters, default 3) -- a soft nudge, not a hard 1:1 assignment.

`masked_sinkhorn_log_bonus_capacitated` takes a different route to the same
many-to-one requirement: it reduces the round to a genuine 1:1 assignment
problem via node-splitting (replicate each target into `capacity` slots --
same theory as Bertsekas & Castanon 1989's transportation-problem auction,
already cited in auction_refinement.py for its capacity=2 default), which
makes it legitimate to run Sinkhorn much closer to convergence. IMPORTANT
CAVEAT, found by working through the math before implementing (see
brerla_sinkhorn_coordination_experiment memory): naively replicating a
target's score identically across its `capacity` slots adds NO new
information over the plain (uncapacitated) function -- Sinkhorn cannot
distinguish identical duplicate columns, so folding them back via
logsumexp exactly reproduces the plain function's result at matching
n_iters (verified: this is an EXACT reduction, not merely "close to zero"
-- an earlier draft of this docstring wrongly guessed the duplicated
signal would cancel to ~0 instead; see the capacity=2,slot_decay=0 test in
this file's __main__ block, which checks equality against the plain
function rather than near-zero). A fixed, small per-slot decay
(`slot_decay`, subtracted as `k * slot_decay` from slot k's score,
k=0..capacity-1) breaks that symmetry -- a cheap, non-learned approximation
of the diminishing-marginal-value logic `auction_round_action_multifire`
computes exactly (via a `survival` variable updated after each sequential
assignment), suitable for a single parallel Sinkhorn pass where a truly
order-dependent computation isn't available.
"""
import torch

NEG = -1e9


def _sinkhorn_log_alpha(scores, mask, n_iters, temperature):
    """Shared core: alternating masked column/row log-normalization.
    `scores` is assumed already appropriately scaled by the caller (both
    public functions below divide by temperature themselves, in different
    places, before calling this)."""
    log_alpha = scores.masked_fill(~mask, NEG)
    for _ in range(n_iters):
        col_norm = torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = (log_alpha - col_norm).masked_fill(~mask, NEG)
        row_norm = torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = (log_alpha - row_norm).masked_fill(~mask, NEG)
    return log_alpha


def _recenter(values, mask):
    """Zero-mean the legal entries of `values` over its trailing (M,N) dims
    and hard-zero the illegal ones -- shared final step for both bonus
    functions below."""
    values = values.masked_fill(~mask, 0.0)
    legal_count = mask.float().sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = values.sum(dim=(-2, -1), keepdim=True) / legal_count
    return (values - mean).masked_fill(~mask, 0.0)


def masked_sinkhorn_log_bonus(edge_scores, edge_mask, n_iters=3, temperature=1.0):
    """
    Args:
        edge_scores: [..., M, N] raw (weapon, target) scores (any leading
            batch/para dims).
        edge_mask: [..., M, N] bool, True where (weapon, target) is a legal
            edge this round.
        n_iters: number of alternating column/row log-normalization passes.
            Deliberately small / not run to convergence -- see module
            docstring.
        temperature: divides edge_scores before normalizing; lower values
            sharpen the redistribution.

    Returns:
        bonus: [..., M, N] float tensor, zero-mean over the legal entries of
            each (weapon,target) score matrix and exactly 0 on illegal
            entries -- safe to ADD to edge_scores. The caller is still
            responsible for the usual -inf legality mask applied after
            concatenation with the no-op score; this function never touches
            that (illegal entries here are 0, an inert additive identity,
            not a large negative number that would double-count).
    """
    mask = edge_mask.bool()
    log_alpha = _sinkhorn_log_alpha(edge_scores / temperature, mask, n_iters, temperature)
    return _recenter(log_alpha, mask)


def masked_sinkhorn_log_bonus_capacitated(edge_scores, edge_mask, capacity=2, n_iters=10,
                                           temperature=1.0, slot_decay=1.0):
    """
    Node-splitting variant: replicate each target into `capacity`
    interchangeable-but-decayed slots, run Sinkhorn on the resulting
    [..., M, N*capacity] genuinely-1:1-structured matrix (safe to run much
    closer to convergence than the plain function above), then fold the
    capacity slots back into one per-target signal via logsumexp ("log
    probability of landing in ANY of this target's slots").

    Args:
        capacity: number of weapons a target can accept before Sinkhorn
            starts actively discouraging further piling-on. Matches
            auction_round_action_multifire's max_per_target=2 default.
        slot_decay: FIXED (not learned) per-slot penalty, in log_alpha
            units. Required to avoid the identical-slot collapse described
            in the module docstring -- capacity=1 (only k=0 exists, penalty
            irrelevant) reduces to (numerically matches) plain
            `masked_sinkhorn_log_bonus`.

            IMPLEMENTATION NOTE (found the hard way -- an earlier version
            applied `k*slot_decay` once to the initial scores and it had
            EXACTLY ZERO effect on the output, verified numerically, not
            just "small"): column-normalization is mathematically invariant
            to any additive constant applied uniformly across the WEAPON
            axis within one column (`logsumexp(x - c) = logsumexp(x) - c`
            cancels exactly in the following subtraction), so a per-slot
            constant baked into the score ONCE is erased by the very first
            column-norm step regardless of the score distribution -- not a
            toy-example artifact, a general property of this construction.
            Fix: the penalty is re-injected after EVERY column-norm step
            inside the loop below (not baked in once before it), so it
            survives into that same iteration's row-norm (where slot-decay
            differences are exactly the signal meant to be visible) before
            the next iteration's column-norm erases it again -- and gets
            reinjected again next iteration.
        n_iters: default higher than the plain function's (10 vs 3) since
            this reduction is legitimately 1:1 at the slot level, so
            running closer to convergence is theoretically sound here
            (unlike the plain many-to-one case).

    Returns: bonus, same shape/semantics as masked_sinkhorn_log_bonus.
    """
    *lead, M, N = edge_scores.shape
    K = capacity
    device = edge_scores.device
    slot_penalty = torch.arange(K, device=device, dtype=edge_scores.dtype) * slot_decay  # [K]
    slot_penalty_flat = slot_penalty.repeat(N)  # [N*K], matches the (n,k)-flatten order below

    scaled_scores = edge_scores / temperature  # [..., M, N]
    expanded_scores = scaled_scores.unsqueeze(-1).expand(*lead, M, N, K).reshape(*lead, M, N * K)
    expanded_mask = edge_mask.bool().unsqueeze(-1).expand(*lead, M, N, K).reshape(*lead, M, N * K)

    log_alpha = expanded_scores.masked_fill(~expanded_mask, NEG)
    for _ in range(n_iters):
        col_norm = torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = (log_alpha - col_norm - slot_penalty_flat).masked_fill(~expanded_mask, NEG)
        row_norm = torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = (log_alpha - row_norm).masked_fill(~expanded_mask, NEG)

    folded = torch.logsumexp(log_alpha.view(*lead, M, N, K), dim=-1)  # [..., M, N]

    return _recenter(folded, edge_mask.bool())


if __name__ == "__main__":
    # Quick numerical sanity check (CPU, no GNN/network involved): verify
    # (a) no NaN/inf leaks into the bonus, (b) illegal entries stay exactly
    # 0, (c) the bonus actually pushes a weapon WITH a viable alternative
    # away from a target another weapon has NO alternative for -- the
    # anti-concentration effect only shows up under this kind of asymmetry;
    # a purely symmetric setup (every weapon rating every target identically)
    # has no differentiating information for Sinkhorn to redistribute and is
    # correctly left ~unchanged (verified this is NOT a bug, see inline note
    # below where it's tried first).
    torch.manual_seed(0)
    B, P, M, N = 1, 1, 2, 2
    scores = torch.tensor([[[[5.0, -5.0],   # weapon A: target0 great, target1 terrible (no real alternative)
                              [5.0, 4.0]]]])  # weapon B: target0 slightly better, target1 nearly as good
    mask = torch.ones(B, P, M, N, dtype=torch.bool)

    bonus = masked_sinkhorn_log_bonus(scores, mask, n_iters=3, temperature=1.0)
    assert torch.isfinite(bonus).all(), "bonus contains NaN/inf"
    assert (bonus[~mask] == 0).all(), "illegal entries must be exactly 0"

    adjusted = scores + bonus
    raw_probs = torch.softmax(scores, dim=-1)
    adj_probs = torch.softmax(adjusted, dim=-1)
    print("raw probs   [A, B] x [target0, target1]:\n", raw_probs[0, 0])
    print("adjusted probs:\n", adj_probs[0, 0])
    # Both weapons put ~all mass on target0 under raw scores (classic
    # over-concentration). Weapon B has a viable alternative (target1 is
    # nearly as good) -- the bonus should shift B's mass toward target1,
    # while weapon A (no real alternative) should stay on target0.
    assert adj_probs[0, 0, 1, 1] > raw_probs[0, 0, 1, 1] + 0.05, \
        "expected weapon B (has a viable alternative) to shift toward target1"
    assert adj_probs[0, 0, 0, 0] > 0.9, \
        "expected weapon A (no viable alternative) to stay on target0"

    # Fully symmetric case (every weapon identical over every target) has no
    # differentiating information -- bonus should stay near zero, confirming
    # this is principled behavior, not a degenerate/broken case.
    sym_scores = torch.zeros(2, 3, 5, 4)
    sym_scores[..., 0] = 5.0
    sym_mask = torch.ones(2, 3, 5, 4, dtype=torch.bool)
    sym_bonus = masked_sinkhorn_log_bonus(sym_scores, sym_mask, n_iters=3)
    assert torch.isfinite(sym_bonus).all()
    assert sym_bonus.abs().max() < 1e-4, \
        "fully symmetric scores should produce ~zero bonus (no info to redistribute)"

    # Illegal-edge case: mask out target 0 entirely for one weapon in a
    # larger, otherwise-random instance.
    torch.manual_seed(1)
    big_scores = torch.randn(2, 3, 5, 4)
    mask2 = torch.ones(2, 3, 5, 4, dtype=torch.bool)
    mask2[:, :, 2, 0] = False
    bonus2 = masked_sinkhorn_log_bonus(big_scores, mask2, n_iters=3)
    assert torch.isfinite(bonus2).all()
    assert (bonus2[:, :, 2, 0] == 0).all()

    # All-illegal row (weapon fully out of legal targets this round, e.g.
    # reloading) must not produce NaN.
    mask3 = mask2.clone()
    mask3[:, :, 4, :] = False
    bonus3 = masked_sinkhorn_log_bonus(big_scores, mask3, n_iters=3)
    assert torch.isfinite(bonus3).all()
    assert (bonus3[:, :, 4, :] == 0).all()

    # --- capacitated (slot-duplication) variant ---
    from sinkhorn_coordination import masked_sinkhorn_log_bonus_capacitated  # noqa: E402

    # capacity=1 must numerically match the plain function (no slots to
    # differentiate, slot_decay is irrelevant with only k=0).
    cap1_bonus = masked_sinkhorn_log_bonus_capacitated(big_scores, mask2, capacity=1, n_iters=3, slot_decay=1.0)
    plain_bonus = masked_sinkhorn_log_bonus(big_scores, mask2, n_iters=3)
    assert torch.allclose(cap1_bonus, plain_bonus, atol=1e-5), \
        "capacity=1 should reduce to the plain (uncapacitated) function"

    # Naive identical-replication (slot_decay=0) adds NO new information
    # beyond the plain function -- with every slot an exact duplicate,
    # column-norm produces identical per-slot values, row-norm shifts every
    # slot by the same weapon-specific constant, and logsumexp-folding K
    # identical copies exactly cancels that shift -- so capacity=2,
    # slot_decay=0 reduces EXACTLY to the plain function at matching
    # n_iters, for ANY capacity, not just capacity=1 (verified analytically
    # after an earlier, wrong guess that it would collapse to ~0 instead --
    # see module docstring). This confirms slot_decay is the ONLY thing
    # that makes the capacitated variant do something new.
    cap2_nodecay = masked_sinkhorn_log_bonus_capacitated(big_scores, mask2, capacity=2, n_iters=5, slot_decay=0.0)
    plain_5iter = masked_sinkhorn_log_bonus(big_scores, mask2, n_iters=5)
    assert torch.allclose(cap2_nodecay, plain_5iter, atol=1e-4), \
        "slot_decay=0 (identical slots) should reduce exactly to the plain function, confirming slot_decay is what matters"

    # Capacity-awareness demonstration: 3 weapons all strongly prefer
    # target0, with DIFFERENT quality alternatives at target1 (A: no real
    # alternative, B: decent alternative, C: near-as-good alternative).
    # capacity=2 should leave weapon B comfortably on target0 (2 slots
    # available, A has priority via no alternative, B is next-best-off
    # among those with a worse alternative than C) far more than the plain
    # (effectively capacity=1) function does, which tries to clear target0
    # down toward a single occupant.
    torch.manual_seed(2)
    # NOTE: target0's column must NOT be perfectly weapon-symmetric (5.0 for
    # all three) -- a column uniform across weapons always collapses to
    # exactly -log(M) after column-norm regardless of its absolute level
    # (logsumexp of M identical copies of v, minus itself, is v-(v+logM)),
    # which would erase slot_decay's effect too (both slots' target0 levels
    # would still be weapon-uniform, just at different levels, and BOTH
    # collapse to the same -log(M) either way). Small per-weapon variation
    # breaks that degeneracy so slot_decay has something real to act on.
    cap_scores = torch.tensor([[[[5.0, -5.0],   # A: no real alternative
                                  [5.1, 3.0],    # B: decent alternative
                                  [4.9, 4.5]]]])  # C: near-as-good alternative
    cap_mask = torch.ones(1, 1, 3, 2, dtype=torch.bool)

    plain_cap_bonus = masked_sinkhorn_log_bonus(cap_scores, cap_mask, n_iters=10)
    plain_probs = torch.softmax(cap_scores + plain_cap_bonus, dim=-1)

    cap2_bonus = masked_sinkhorn_log_bonus_capacitated(cap_scores, cap_mask, capacity=2, n_iters=10, slot_decay=1.5)
    cap2_probs = torch.softmax(cap_scores + cap2_bonus, dim=-1)

    print("\ncapacity-awareness demo (A/B/C x target0/target1):")
    print("  plain (~capacity=1) probs:\n", plain_probs[0, 0])
    print("  capacity=2 probs:\n", cap2_probs[0, 0])
    assert cap2_probs[0, 0, 1, 0] > plain_probs[0, 0, 1, 0] + 0.05, \
        "expected weapon B to stay on target0 MORE under capacity=2 than under the plain (capacity=1-like) function"

    assert torch.isfinite(cap2_bonus).all()
    assert (cap2_bonus[~cap_mask] == 0).all()

    print("\nAll sinkhorn_coordination sanity checks passed.")
