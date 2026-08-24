"""Round-by-round coordination stats (dispersion / wasteful redundancy) for
Sequential vs SCoPE-Comm at LARGE scale (30M_30N_10T), where the aggregate
table shows the biggest relative gap (Seq 0.016 vs Comm 0.025, +56%
relative). The 5M_5N_5T instance-by-instance reading (diagnose_scip_seq_comm.py)
found no single clean bug -- both decoders make different, locally-reasonable
marginal-value trade-offs, roughly 50/50 at that scale. This checks whether a
systematic "many weapons deciding blind to each other -> pile-up" signature
(the original motivation for the comm layer) actually shows up in the
aggregate stats once there are enough SIMULTANEOUS weapons per round for it
to matter, rather than reading individual schedules by eye.
"""
import numpy as np
import torch

from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action, auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals, _round_dispersion, _round_wasteful_redundancy, _pack_result

M, N, T = 30, 30, 10
SEED = 123
N_EVAL = 10
COMM_CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"
WASTE_SURVIVAL_THRESHOLD = 0.2


@torch.no_grad()
def eval_comm(actor, V, P, TW, AMM, PREP, multifire=False):
    patch_globals(M, N, T, AMM, PREP, [1] * M)
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=P, TW=TW,
                                max_time=T, batch_size=1, alpha=1.0, amm=AMM)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=T)
    init_value = env.current_target_value[:, :, 0:N].sum().item()

    round_dispersions, round_waste = [], []
    for _ in range(T):
        survival_before = (env.current_target_value[0, 0, :N] / env.original_target_value[0, 0, :N]).tolist()
        remaining_value = env.current_target_value[:, :, 0:N]
        prob = env.weapon_to_target_prob[:, :, :M, :N]
        legal_mask = env.mask_per_weapon[:, :, :M, :N] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :M, :].argmax(dim=-1)
        must_fire = policy_choice < N
        if multifire:
            action = auction_round_action_multifire(remaining_value, prob, legal_mask, must_fire=must_fire)
        else:
            action = auction_round_action(remaining_value, prob, legal_mask, must_fire=must_fire)
        flat = action.view(-1).tolist()
        fired_targets = [n for n in flat if n < N]
        round_dispersions.append(_round_dispersion(fired_targets))
        round_waste.append(_round_wasteful_redundancy(fired_targets, survival_before))
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()

    remaining = env.current_target_value[:, :, 0:N].sum().item()
    return _pack_result(init_value, remaining, 0, 0, 0.0, round_dispersions, round_waste), round_dispersions, round_waste


@torch.no_grad()
def eval_seq(actor, V, P, TW, AMM, PREP):
    patch_globals(M, N, T, AMM, PREP, [1] * M)
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=P, TW=TW,
                                max_time=T, batch_size=1, alpha=1.0, amm=AMM)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=T)
    init_value = env.current_target_value[:, :, 0:N].sum().item()

    num_edges = M * N
    round_dispersions, round_waste = [], []
    for _ in range(T):
        survival_before = (env.current_target_value[0, 0, :N] / env.original_target_value[0, 0, :N]).tolist()
        fired_targets = []
        for _ in range(M):
            mask = env.mask.clone()
            policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, mask)
            flat = policy.view(-1, num_edges + 1)
            action_idx = int(flat.argmax(dim=1).item())
            if action_idx < num_edges:
                fired_targets.append(action_idx % N)
            action = torch.tensor([[action_idx]], device=DEVICE)
            env.update_internal_variables(selected_action=action)
        round_dispersions.append(_round_dispersion(fired_targets))
        round_waste.append(_round_wasteful_redundancy(fired_targets, survival_before))
        env.time_update()

    remaining = env.current_target_value[:, :, 0:N].sum().item()
    return _pack_result(init_value, remaining, 0, 0, 0.0, round_dispersions, round_waste), round_dispersions, round_waste


if __name__ == "__main__":
    comm_actor = create_gnn_actor_comm().to(DEVICE)
    comm_actor.load_state_dict(torch.load(COMM_CKPT, map_location=DEVICE, weights_only=False))
    comm_actor.eval()
    seq_actor = create_gnn_actor_sequential().to(DEVICE)
    seq_actor.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE, weights_only=False))
    seq_actor.eval()

    rng = np.random.default_rng(SEED)
    comm_obj, seq_obj = [], []
    comm_disp_by_round = [[] for _ in range(T)]
    seq_disp_by_round = [[] for _ in range(T)]
    comm_waste_by_round = [[] for _ in range(T)]
    seq_waste_by_round = [[] for _ in range(T)]

    import sys
    multifire = '--multifire' in sys.argv

    for i in range(N_EVAL):
        V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
        P = np.asarray(P)
        cres, cdisp, cwaste = eval_comm(comm_actor, V, P, TW, AMM, PREP, multifire=multifire)
        sres, sdisp, swaste = eval_seq(seq_actor, V, P, TW, AMM, PREP)
        comm_obj.append(cres["objective"])
        seq_obj.append(sres["objective"])
        for t in range(T):
            if cdisp[t] is not None:
                comm_disp_by_round[t].append(cdisp[t])
            if sdisp[t] is not None:
                seq_disp_by_round[t].append(sdisp[t])
            if cwaste[t] is not None:
                comm_waste_by_round[t].append(cwaste[t])
            if swaste[t] is not None:
                seq_waste_by_round[t].append(swaste[t])
        print(f"instance {i}: Seq_obj={sres['objective']:.4f} Comm_obj={cres['objective']:.4f} "
              f"Seq_disp={sres['dispersion']} Comm_disp={cres['dispersion']} "
              f"Seq_waste={sres['wasteful_redundancy']} Comm_waste={cres['wasteful_redundancy']}", flush=True)

    print(f"\nMEAN objective: Seq={np.mean(seq_obj):.4f}  Comm={np.mean(comm_obj):.4f}")
    print("\nPer-round dispersion (1.0=every shot different target, low=piling on same target), mean across instances with >=1 fire that round:")
    print(f"{'round':<8}{'Seq disp':>12}{'Comm disp':>12}{'Seq waste':>12}{'Comm waste':>12}")
    for t in range(T):
        sd = np.mean(seq_disp_by_round[t]) if seq_disp_by_round[t] else float('nan')
        cd = np.mean(comm_disp_by_round[t]) if comm_disp_by_round[t] else float('nan')
        sw = np.mean(seq_waste_by_round[t]) if seq_waste_by_round[t] else float('nan')
        cw = np.mean(comm_waste_by_round[t]) if comm_waste_by_round[t] else float('nan')
        print(f"{t:<8}{sd:>12.3f}{cd:>12.3f}{sw:>12.3f}{cw:>12.3f}")
