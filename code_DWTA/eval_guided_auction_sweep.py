"""Sweep guide_weight for auction_round_action_multifire_guided (policy's own
per-target preference boosts the auction's marginal-value ranking, instead of
the auction re-deriving target choice from scratch with zero network input)
-- against Sequential, on a few representative configs (Small/Medium where
plain multifire regressed, Large where it won decisively), same CommSCoPE
checkpoint, no retraining.
"""
import numpy as np
import torch

from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action, auction_round_action_multifire, auction_round_action_multifire_guided
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

N_EVAL = 10
SEED = 123
COMM_CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"
CONFIGS = [(5, 5, 5), (15, 15, 5), (30, 30, 10), (70, 100, 15)]
GUIDE_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0]


@torch.no_grad()
def eval_guided(actor, V, P, TW, nw, nt, mt, amm, prep, cost, mode, guide_weight=0.0):
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

        if mode == "1:1":
            action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        elif mode == "multi":
            action = auction_round_action_multifire(remaining_value, prob, legal_mask, must_fire=must_fire)
        else:  # guided
            target_scores = policy[:, :, :nw, :nt]
            legal_f = legal_mask.float()
            denom = (target_scores * legal_f).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            policy_target_prob = (target_scores * legal_f) / denom
            action = auction_round_action_multifire_guided(
                remaining_value, prob, legal_mask, policy_target_prob,
                must_fire=must_fire, guide_weight=guide_weight)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


@torch.no_grad()
def eval_seq(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                                max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)
    init_value = env.current_target_value[:, :, 0:nt].sum().item()

    num_edges = nw * nt
    for _ in range(mt):
        for _ in range(nw):
            mask = env.mask.clone()
            policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
            flat = policy.view(-1, num_edges + 1)
            action_idx = int(flat.argmax(dim=1).item())
            action = torch.tensor([[action_idx]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:nt].sum().item()
    return remaining / max(init_value, 1e-8)


if __name__ == "__main__":
    comm_actor = create_gnn_actor_comm().to(DEVICE)
    comm_actor.load_state_dict(torch.load(COMM_CKPT, map_location=DEVICE, weights_only=False))
    comm_actor.eval()
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    seq_actor.eval()

    header = f"{'config':<16}{'Seq':>8}{'1:1':>8}" + "".join(f"{'gw='+str(g):>9}" for g in GUIDE_WEIGHTS)
    print(header)
    for M, N, T in CONFIGS:
        rng = np.random.default_rng(SEED)
        instances = [generate_moderate_temporal_instance(M, N, T, rng=rng) for _ in range(N_EVAL)]

        seq_vals = [eval_seq(seq_actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST)
                    for V, P, TW, AMM, PREP, COST in instances]
        c11_vals = [eval_guided(comm_actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST, mode="1:1")
                    for V, P, TW, AMM, PREP, COST in instances]

        row = f"{M}M_{N}N_{T}T{'':<7}{np.mean(seq_vals):>8.4f}{np.mean(c11_vals):>8.4f}"
        for gw in GUIDE_WEIGHTS:
            vals = [eval_guided(comm_actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST,
                                 mode="guided", guide_weight=gw)
                    for V, P, TW, AMM, PREP, COST in instances]
            row += f"{np.mean(vals):>9.4f}"
        print(row, flush=True)
