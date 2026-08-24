"""Diagnostic: can the actor overfit a SINGLE SCIP-teacher instance, given
many gradient steps? If not, something structural is blocking learning
(not just class imbalance / insufficient data)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

from common.Dynamic_HYPER_PARAMETER import *
from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN import create_gnn_actor
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from eval_tiered_benchmark import patch_globals

dataset = torch.load('result/scip_teacher_dataset.pt', weights_only=False)
item = dataset[0]
M, N, T = item['M'], item['N'], item['T']
V, P, TW, AMM, PREP = item['V'], np.asarray(item['P']), item['TW'], item['AMM'], item['PREP']
actions = item['actions']
print(f"Instance: {M}M_{N}N_{T}T, actions={actions}")

actor = create_gnn_actor().to(DEVICE)
optimizer = torch.optim.SGD(actor.parameters(), lr=0.1, momentum=0.9)

for step in range(200):
    patch_globals(M, N, T, AMM, PREP, [1] * M)
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=P, TW=TW,
                                max_time=T, batch_size=1, alpha=1.0, amm=AMM)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=T)

    total_loss = 0.0
    correct, total = 0, 0
    for t in range(T):
        current_state = env.assignment_encoding.clone()
        current_prob = env.weapon_to_target_prob.clone()
        current_mask = env.mask_per_weapon.clone()
        policy, _ = actor(current_state, current_prob, current_mask)
        label = torch.tensor(actions[t], device=DEVICE, dtype=torch.long).view(1, 1, M)
        pred = policy.view(M, N + 1).argmax(dim=-1)
        correct += (pred == label.view(M)).sum().item()
        total += M
        weight = torch.ones(N + 1, device=DEVICE)
        weight[N] = 3268 / 7517
        loss_t = F.cross_entropy(policy.view(M, N + 1), label.view(M), weight=weight)
        total_loss = total_loss + loss_t
        env.update_internal_variables_parallel(selected_actions=label)
        env.time_update()

    loss = total_loss / T
    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step}: loss={loss.item():.4f} acc={correct}/{total} grad_norm={grad_norm:.6f}")
