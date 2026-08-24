"""Post-hoc comparison (no retraining): does swapping the auction mechanism
used at eval time (1:1 eviction vs many-to-one marginal-value) change the
score of an ALREADY-TRAINED actor's SCoPE-style (policy fire/hold + auction
target assignment) evaluation? Auction is inference-only, so this is a fair
like-for-like test of the mechanism alone, holding the actor fixed.
"""
import argparse

import numpy as np
import torch

from common.DWTA_GNN import create_gnn_actor
from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action, auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

ALL_EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
N_EVAL = 10
SEED = 123


@torch.no_grad()
def eval_instance(actor, auction_fn, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]
        prob = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :nw, :].argmax(dim=-1)
        must_fire = policy_choice < nt
        action = auction_fn(remaining_value, prob, legal_mask, must_fire=must_fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


def eval_all(actor, auction_fn):
    actor.eval()
    results = []
    for M, N, T in ALL_EVAL_CONFIGS:
        rng = np.random.default_rng(SEED)
        objs = []
        for i in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            objs.append(eval_instance(actor, auction_fn, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        results.append((M, N, T, float(np.mean(objs))))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--comm', action='store_true', help='use EdgeAwareGNN_ACTOR_COMM instead of the plain actor')
    args = parser.parse_args()

    actor = (create_gnn_actor_comm() if args.comm else create_gnn_actor()).to(DEVICE)
    actor.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=False))

    print(f"Checkpoint: {args.ckpt}")
    print(f"{'config':<18}{'auction(1:1)':>14}{'auction(multi)':>16}")
    r1 = eval_all(actor, auction_round_action)
    r2 = eval_all(actor, auction_round_action_multifire)
    scores1, scores2 = [], []
    for (M, N, T, s1), (_, _, _, s2) in zip(r1, r2):
        scores1.append(s1)
        scores2.append(s2)
        print(f"{M}M_{N}N_{T}T{'':<8}{s1:>14.4f}{s2:>16.4f}")
    print(f"{'MEAN':<18}{sum(scores1)/len(scores1):>14.4f}{sum(scores2)/len(scores2):>16.4f}")
