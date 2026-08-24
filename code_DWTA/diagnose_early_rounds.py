"""
Why does the parallel pipeline destroy so little in the FIRST few rounds at
Battlefield scale?

`diagnose_battlefield_multifire.py` isolated the residual 70M_100N_15T gap
to early rounds: over rounds 1-3 Sequential destroys 0.114 of initial value
while the parallel pipeline destroys 0.047, and the deficit is never
recovered -- even though the parallel pipeline BURNS MORE AMMO overall
(0.924 vs 0.803) and, with multifire, already matches Sequential's
dispersion (0.891 vs 0.918) with zero wasteful redundancy. So this is not
the coordination failure multifire fixed; it is about WHICH targets get
engaged early.

Candidate explanations this separates, per round:

  fires            how many weapons actually fire
                   -- are we simply not shooting early (holding), or
                   shooting and getting less for it?
  legal_frac       fraction of weapon-decisions with any legal target
                   -- an environment-side ceiling: if few targets are in
                   their time window early, low destruction is forced and
                   Sequential must be exploiting something else.
  value_hit        total (remaining_value * kill_prob) of the engaged
                   (weapon,target) pairs -- expected damage the choices
                   were WORTH, independent of realized randomness.
  value_available  the same quantity for the best legal pairing available
                   that round (greedy upper reference)
                   -- value_hit / value_available is a direct "did we pick
                   the good targets that were on the table" score.

If value_hit/value_available is much lower than Sequential's early, the
proposer is picking weak targets while good ones sit available. If instead
`fires` is low, it is a fire/hold timing issue. If `legal_frac` is low for
both, the early rounds are environment-limited and the gap lives elsewhere.
"""
import sys

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

CONFIGS = [(70, 100, 15)]
N_EVAL = 10
SEED = 123
BASE_CKPT = "result/Joint_base_seed5_simlr_best.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"


def _round_stats(env, nw, nt, fired_pairs):
    """fired_pairs: list of (weapon, target) engaged this round, recorded
    BEFORE the environment applied them."""
    rv = env.current_target_value[0, 0, 0:nt]
    pr = env.weapon_to_target_prob[0, 0, :nw, :nt]
    legal = env.mask_per_weapon[0, 0, :nw, :nt] > 0

    value_hit = sum(float(rv[n] * pr[m, n]) for (m, n) in fired_pairs)

    # Best legal pairing available this round, greedy and non-repeating on
    # weapons -- an upper reference for "what was on the table".
    marginal = (rv.unsqueeze(0) * pr).masked_fill(~legal, float('-inf'))
    value_available = 0.0
    taken = torch.zeros(nw, dtype=torch.bool)
    flat = marginal.flatten()
    order = torch.argsort(flat, descending=True)
    for idx in order[:nw * 2]:
        v = float(flat[idx])
        if not np.isfinite(v) or v <= 0:
            break
        m, n = int(idx) // nt, int(idx) % nt
        if taken[m]:
            continue
        taken[m] = True
        value_available += v

    return {
        'fires': len(fired_pairs),
        'legal_frac': float(legal.any(dim=-1).float().mean()),
        'value_hit': value_hit,
        'value_available': value_available,
    }


@torch.no_grad()
def run_parallel(base_actor, ae, wtp, nw, nt, mt):
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    per_round = []
    for _ in range(mt):
        policy, _ = base_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = policy[:, :, :nw, :].argmax(dim=-1) < nt
        action = auction_round_action_multifire(
            env.current_target_value[:, :, 0:nt],
            env.weapon_to_target_prob[:, :, :nw, :nt],
            env.mask_per_weapon[:, :, :nw, :nt] > 0,
            must_fire=must_fire)
        pairs = [(m, int(a)) for m, a in enumerate(action[0, 0].tolist()) if a < nt]
        per_round.append(_round_stats(env, nw, nt, pairs))
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    return per_round


@torch.no_grad()
def run_sequential(seq_actor, ae, wtp, nw, nt, mt):
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    n_edges = nw * nt
    per_round = []
    for _ in range(mt):
        # Stats must be read from the round-start state to be comparable
        # with the parallel arm, but sequential mutates state within the
        # round -- so record the pairs as they are chosen and score them
        # against the round-start snapshot.
        rv0 = env.current_target_value.clone()
        pr0 = env.weapon_to_target_prob.clone()
        mask0 = env.mask_per_weapon.clone()
        snapshot = Environment(assignment_encoding=env.assignment_encoding.clone(),
                                weapon_to_target_prob=pr0.clone(), max_time=mt)
        snapshot.current_target_value = rv0.clone()
        snapshot.mask_per_weapon = mask0.clone()

        pairs = []
        for _ in range(nw):
            policy, _ = seq_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask.clone())
            idx = int(policy.view(-1, n_edges + 1).argmax(dim=1).item())
            if idx < n_edges:
                pairs.append((idx // nt, idx % nt))
            env.update_internal_variables(selected_action=torch.tensor([[idx]], device=DEVICE))

        per_round.append(_round_stats(snapshot, nw, nt, pairs))
        env.time_update()
    return per_round


if __name__ == "__main__":
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    seq_actor.eval()

    for (M, N, T) in CONFIGS:
        print(f"\n{'='*70}\n{M}M_{N}N_{T}T\n{'='*70}", flush=True)
        rng = np.random.default_rng(SEED)
        acc = {'parallel': [], 'Sequential': []}

        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            patch_globals(M, N, T, AMM, PREP, COST)
            ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=np.asarray(P), TW=TW,
                                        max_time=T, batch_size=1, alpha=1.0, amm=AMM)
            ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
            acc['parallel'].append(run_parallel(base_actor, ae, wtp, M, N, T))
            acc['Sequential'].append(run_sequential(seq_actor, ae, wtp, M, N, T))
            print(f"  instance {i+1}/{N_EVAL}", flush=True)

        for name, runs in acc.items():
            print(f"\n{name}:")
            print(f"  {'round':<7}{'fires':>8}{'legal%':>9}{'val_hit':>10}{'val_avail':>11}{'ratio':>8}")
            for t in range(T):
                rows = [r[t] for r in runs]
                vh = np.mean([r['value_hit'] for r in rows])
                va = np.mean([r['value_available'] for r in rows])
                print(f"  {t:<7}{np.mean([r['fires'] for r in rows]):>8.1f}"
                      f"{100*np.mean([r['legal_frac'] for r in rows]):>8.1f}%"
                      f"{vh:>10.2f}{va:>11.2f}{(vh/va if va > 0 else float('nan')):>8.2f}")
