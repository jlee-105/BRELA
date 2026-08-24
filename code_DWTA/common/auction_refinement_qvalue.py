"""
Q-VALUE-DRIVEN capacitated auction: same greedy, survival-adjusted,
many-to-one assignment structure as auction_round_action_multifire
(auction_refinement.py), but candidates are RANKED by the network's own
Q-values instead of a hand-computed marginal-value formula -- this is what
makes the assignment decision itself part of what the network controls
(REDA-inspired: Q-values ARE the benefit matrix the assignment solver
consumes), while the REWARD used to train those Q-values still comes from
the TRUE game-mechanic marginal value (returned alongside the action),
independent of whatever the network currently believes.

See common/DWTA_GNN_comm_qvalue.py's module docstring for the full
motivation. New file -- does not modify auction_refinement.py.
"""
import torch


@torch.no_grad()
def auction_round_action_multifire_qvalue(q_values, remaining_value, prob, legal_mask, max_per_target=2):
    """
    Args:
        q_values: [B, P, M, N+1] network Q-values (edges + no-op), used
            ONLY to rank candidates -- the assignment decision follows
            whatever the network currently believes is best.
        remaining_value: [B, P, N] current remaining value per target.
        prob: [B, P, M, N] weapon-target damage probability.
        legal_mask: [B, P, M, N] bool, True where (weapon,target) legal
            this round. No-op is always legal (not part of this tensor).
        max_per_target: capacity cap, matches auction_round_action_multifire.

    Returns:
        action: [B, P, M] long, values 0..N-1 (target) or N (no-op).
        realized_value: [B, P, M] float, each assigned weapon's TRUE
            survival-adjusted marginal value (remaining_value * survival *
            prob, same accounting the game's multiplicative damage model
            uses) -- 0 for no-op weapons. Independent of q_values; this is
            what the caller should use as the TD/regression training
            reward, NOT the network's own Q prediction.
    """
    B, P, M, N = prob.shape
    device = prob.device

    edge_q = q_values[:, :, :, :N]
    noop_q = q_values[:, :, :, N]

    action = torch.full((B, P, M), N, dtype=torch.long, device=device)
    realized_value = torch.zeros(B, P, M, device=device)

    for b in range(B):
        for p in range(P):
            rv = remaining_value[b, p]
            pr = prob[b, p]
            legal = legal_mask[b, p]
            eq = edge_q[b, p]
            nq = noop_q[b, p]

            survival = torch.ones(N, device=device)
            target_count = torch.zeros(N, dtype=torch.long, device=device)
            action_bp = torch.full((M,), N, dtype=torch.long, device=device)
            realized_bp = torch.zeros(M, device=device)

            remaining_weapons = list(range(M))
            while remaining_weapons:
                at_cap = target_count >= max_per_target
                best_m, best_n, best_q = None, None, float("-inf")
                for m in remaining_weapons:
                    cand_q = eq[m].masked_fill(~legal[m].bool(), float("-inf")).masked_fill(at_cap, float("-inf"))
                    if not torch.isfinite(cand_q).any():
                        continue
                    n = int(cand_q.argmax().item())
                    q = float(cand_q[n].item())
                    if q > float(nq[m].item()) and q > best_q:
                        best_m, best_n, best_q = m, n, q

                if best_m is None:
                    break  # every remaining weapon's own Q prefers no-op (or has no legal/under-cap target)

                action_bp[best_m] = best_n
                marginal = float((rv[best_n] * survival[best_n] * pr[best_m, best_n]).item())
                realized_bp[best_m] = marginal
                survival[best_n] = survival[best_n] * (1 - pr[best_m, best_n])
                target_count[best_n] = target_count[best_n] + 1
                remaining_weapons.remove(best_m)

            action[b, p] = action_bp
            realized_value[b, p] = realized_bp

    return action, realized_value
