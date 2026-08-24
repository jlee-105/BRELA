"""
Why does the multifire auction HELP at Large tier but HURT at Battlefield?

Measured (eval_multifire_pipeline.py, joint-sim checkpoint, K=10):
  30M_30N_10T   1:1 0.0245 -> multifire 0.0125   (decisive win, beats Seq 0.0162)
  40M_50N_10T   1:1 0.0350 -> multifire 0.0274   (win, beats Seq 0.0283)
  70M_100N_15T  1:1 0.1100 -> multifire 0.1183   (WORSE, Seq 0.1003)

The Large-tier gain matched the diagnosis exactly: the 1:1 eviction auction
pins per-round dispersion at 1.000 (no target can ever be double-teamed),
Sequential runs at ~0.89-0.92 dispersion with ~0.001 wasteful redundancy,
and lifting the constraint closed the gap. If that mechanism were the whole
story, Battlefield should improve too. It does not, so something else
dominates at 50-70 weapons.

Hypotheses this separates:
  (a) multifire concentrates but WASTEFULLY at Battlefield -- redundant
      hits landing on already-nearly-dead targets. Signature: dispersion
      drops AND wasteful_redundancy rises sharply vs Sequential's ~0.001.
  (b) the cap is mis-set for the scale -- with 70 weapons and 100 targets
      there may be enough good targets that pairing is simply unnecessary,
      so multifire spends shots on second-best pairings. Signature:
      dispersion drops but waste stays low, while ammo_used rises without
      a matching destruction gain.
  (c) concentration is fine but MIS-TIMED at this horizon (T=15).
      Signature: per-round destruction shifts earlier/later vs Sequential.

Also sweeps max_per_target (2 vs 3) at Battlefield, since the cap was tuned
by inspecting SCIP solutions at much smaller scale.
"""
import sys
from functools import partial

import numpy as np
import torch

sys.path.append('rl')

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action, auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals, _round_dispersion, _round_wasteful_redundancy

CONFIGS = [(30, 30, 10), (70, 100, 15)]   # one where multifire wins, one where it loses
N_EVAL = 10
SEED = 123
BASE_CKPT = "result/Joint_base_seed5_simlr_best.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"


@torch.no_grad()
def run_parallel(base_actor, ae, wtp, nw, nt, mt, auction_fn):
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()
    prev = init_value
    fires = decisions = 0
    disps, wastes, destr = [], [], []

    for _ in range(mt):
        orig = env.original_target_value[0, 0, 0:nt]
        survival_before = (env.current_target_value[0, 0, 0:nt] / orig.clamp_min(1e-8)).tolist()
        policy, _ = base_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        must_fire = policy[:, :, :nw, :].argmax(dim=-1) < nt
        action = auction_fn(env.current_target_value[:, :, 0:nt],
                            env.weapon_to_target_prob[:, :, :nw, :nt],
                            env.mask_per_weapon[:, :, :nw, :nt] > 0,
                            must_fire=must_fire)
        fired = [int(a) for a in action[0, 0].tolist() if a < nt]
        fires += len(fired)
        decisions += nw
        disps.append(_round_dispersion(fired))
        wastes.append(_round_wasteful_redundancy(fired, survival_before))
        env.update_internal_variables_parallel(selected_actions=action)
        cur = env.current_target_value[:, :, 0:nt].sum().item()
        destr.append((prev - cur) / max(init_value, 1e-8))
        prev = cur
        env.time_update()
    return init_value, prev, fires, decisions, disps, wastes, destr


@torch.no_grad()
def run_sequential(seq_actor, ae, wtp, nw, nt, mt):
    env = Environment(assignment_encoding=ae.clone(), weapon_to_target_prob=wtp.clone(), max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()
    prev = init_value
    fires = decisions = 0
    disps, wastes, destr = [], [], []
    n_edges = nw * nt

    for _ in range(mt):
        orig = env.original_target_value[0, 0, 0:nt]
        survival_before = (env.current_target_value[0, 0, 0:nt] / orig.clamp_min(1e-8)).tolist()
        fired = []
        for _ in range(nw):
            policy, _ = seq_actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask.clone())
            idx = int(policy.view(-1, n_edges + 1).argmax(dim=1).item())
            if idx < n_edges:
                fired.append(idx % nt)
            env.update_internal_variables(selected_action=torch.tensor([[idx]], device=DEVICE))
        fires += len(fired)
        decisions += nw
        disps.append(_round_dispersion(fired))
        wastes.append(_round_wasteful_redundancy(fired, survival_before))
        cur = env.current_target_value[:, :, 0:nt].sum().item()
        destr.append((prev - cur) / max(init_value, 1e-8))
        prev = cur
        env.time_update()
    return init_value, prev, fires, decisions, disps, wastes, destr


def summarize(runs, ammo_total):
    init_v, remaining, fires, decisions, disps, wastes, destr = runs
    d = [x for x in disps if x is not None]
    w = [x for x in wastes if x is not None]
    return {
        'obj': remaining / max(init_v, 1e-8),
        'fire_rate': fires / max(decisions, 1),
        'ammo_used': fires / max(ammo_total, 1),
        'dispersion': float(np.mean(d)) if d else float('nan'),
        'waste': float(np.mean(w)) if w else float('nan'),
        'destr': destr,
    }


if __name__ == "__main__":
    base_actor = create_gnn_actor_comm_sinkhorn().to(DEVICE)
    base_actor.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False))
    base_actor.eval()
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    seq_actor.eval()

    # The cap is a hand-set constant, which is exactly the kind of thing the
    # learned component should be deciding instead. Note the multifire
    # auction ALREADY has diminishing returns built in (survival[n] shrinks
    # each time a weapon is assigned to n), so the cap is a patch on top of
    # that for residual myopia. Sweeping it to unlimited tests whether the
    # patch is still needed -- and if the auditor can absorb the difference
    # by cancelling over-concentrated shots, a hand-tuned constant gets
    # replaced by something learned.
    arms = {
        'Sequential': None,
        '1:1': auction_round_action,
        'multifire(cap2)': partial(auction_round_action_multifire, max_per_target=2),
        'multifire(cap3)': partial(auction_round_action_multifire, max_per_target=3),
        'multifire(uncapped)': partial(auction_round_action_multifire, max_per_target=10 ** 9),
    }

    for (M, N, T) in CONFIGS:
        print(f"\n{'='*76}\n{M}M_{N}N_{T}T\n{'='*76}", flush=True)
        rng = np.random.default_rng(SEED)
        acc = {k: [] for k in arms}

        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            patch_globals(M, N, T, AMM, PREP, COST)
            ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=np.asarray(P), TW=TW,
                                        max_time=T, batch_size=1, alpha=1.0, amm=AMM)
            ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
            ammo_total = float(sum(AMM))

            for name, fn in arms.items():
                runs = (run_sequential(seq_actor, ae, wtp, M, N, T) if fn is None
                        else run_parallel(base_actor, ae, wtp, M, N, T, fn))
                acc[name].append(summarize(runs, ammo_total))
            print(f"  instance {i+1}/{N_EVAL}", flush=True)

        print(f"\n{'arm':<20}{'obj':>9}{'fire_rate':>11}{'ammo_used':>11}{'dispersion':>12}{'waste':>9}")
        for name, rows in acc.items():
            print(f"{name:<20}"
                  f"{np.mean([r['obj'] for r in rows]):>9.4f}"
                  f"{np.mean([r['fire_rate'] for r in rows]):>11.3f}"
                  f"{np.mean([r['ammo_used'] for r in rows]):>11.3f}"
                  f"{np.mean([r['dispersion'] for r in rows]):>12.3f}"
                  f"{np.mean([r['waste'] for r in rows]):>9.3f}", flush=True)

        print("\ndestruction by round:")
        for name, rows in acc.items():
            per_round = np.mean(np.array([r['destr'] for r in rows]), axis=0)
            print(f"  {name:<18}" + " ".join(f"{v:.3f}" for v in per_round))
