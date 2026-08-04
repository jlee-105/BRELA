# BReRLA Revision Plan — Working Notes

Last updated: 2026-07-20 (session 2: implementation)

## Background

- **BReRLA**: "Beam Search and Rollout Enhanced Reinforcement Learning for Dynamic Weapon-Target Assignment" — sole-author PhD dissertation chapter (Jaejin Lee, Intel Corp / ASU PhD 2025). Transformer policy + REINFORCE, inference-time beam search + rollout (pure rollout for small instances, truncated neural rollout via value network for larger). Trained on (5,5,5), tested on 12 configs up to (10,10,10). Gaps vs CPLEX: 6–17%.
- Submitted to **IEEE Access**, decision 2026-07-15: **major revision**, one resubmission allowed (if concerns not addressed, final rejection with no further resubmission).
- Separate paper, **"Heterogeneous Graph RL with Edge Pruning and Rollout Enhancement for Dual-Resource Constrained Lithography Scheduling"** (Lee, Cho, Jang, Runger) — just submitted to IEEE Access, currently under review. Same author, different domain (semiconductor lithography scheduling), same general recipe (RL + rollout + beam search) but done more rigorously (see below).
- Both papers are intended as first-author evidence for an **NIW (National Interest Waiver)** petition. Priority: get a first-author paper published, reasonably fast, at a legitimate (non-MDPI) Q2–Q3 journal.

## IEEE Access reviewer comments on BReRLA (5 reviewers)

Common threads:
- CPLEX comparison validity: 600s cutoff on larger instances means CPLEX may not be optimal — reframe gaps as "vs. best-found," not "optimality gap." (R1, R2, R3)
- Statistical rigor: need multiple seeds, mean ± std, significance testing. (R1, R2, R3)
- Ablation coverage: beam width/rollout depth ablations only on 2 of 12 configs; R3 also wants module-wise ablation (encoder, value net, rollout variants, beam search individually).
- Baselines too weak: only CPLEX/greedy/REINFORCE — need metaheuristics, MIP variants, recent neural CO methods. (R1)
- Realism/scale: synthetic-only data, no uncertainty/stochastic behavior, max scale 10×10×10. (R1, R5)
- Reproducibility: report episodes, training time, optimizer, discount factor, seeds, hardware/software. (R3)
- Compute reporting: params, FLOPs, memory, training cost. (R3)
- Novelty positioning: read as "technical integration" rather than new mechanism. (R3, R4, R5)
- Reference list: R4 flagged refs 1, 2, 16, 34, 35, 37–39 as inappropriate; R2 wants non-peer-reviewed sources (encyclopedia, thesis) swapped.
- Math/MDP formulation inconsistencies flagged by R4 (most concrete "this is wrong" signal).
- Minor: Table 1 overlaps text (R5).

## Concrete issues found in the manuscript (verified against text)

1. **Notation inconsistency**: Table 1 defines sets W, E, T with elements W_m, E_n — never used again; rest of paper uses M, N, T throughout. Easy fix (rewrite Table 1).
2. **Math bug in constraint (3)**: `w_{m,t+1} = w_{m,t} - 1 + D_m·Σx_{m,n,t}` has no floor at 0. If weapon is ready (w=0) and doesn't fire, w_{m,t+1} = -1, violating constraints (9)(10) which require w ∈ [0, D_m]. Needs `max(·, 0)`.
3. **Reference [35] mismatch**: text says "Li et al. [35] decomposed a DWTAP into sequential SWTAPs..." but ref [35] is actually S.E. Li, "Deep reinforcement learning" — a generic RL textbook chapter, unrelated to the claim.
4. **Refs [1], [2]**: Britannica and "The OR Society" website — low-quality/non-peer-reviewed, should be swapped for academic OR-history sources.
5. **Ref [39] (Bellman)**: malformed citation format ("...Princeton University Press Princeton, New Jersey Google Scholar...").
6. **Refs [34], [38]**: use "et al." in the reference-list author field itself (improper formatting).

## Critical novelty finding

- BReRLA's core inference technique (beam search + rollout guiding a learned policy) is **methodologically very close to Simulation-Guided Beam Search (SGBS)** — Choo, Kwon, Kim, Jae, Hottung, Tierney, Gwon, "Simulation-guided Beam Search for Neural Combinatorial Optimization," NeurIPS 2022 (arXiv:2207.06190). **Not cited anywhere in BReRLA's reference list.** This is very likely the real source of R3/R4/R5's "just a technical integration" skepticism, even though none of them named SGBS explicitly.
- Bertsekas rollout (Bertsekas, Tsitsiklis, Wu, "Rollout algorithms for combinatorial optimization," Journal of Heuristics, 1997) is the origin of the rollout idea itself — also uncited in BReRLA.
- **PARCO** (Berto et al., "Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization," arXiv:2409.03811, NeurIPS 2025) is directly relevant because WTA is inherently a multi-agent (multi-weapon) parallel assignment problem, but BReRLA's action formulation (Eq. 16, K = M×N+1) appears to decode one (weapon, target) pair per step — sequential, not parallel. PARCO's three components: (1) transformer-based communication layers between agents, (2) multiple pointer mechanism for simultaneous per-agent decisions, (3) priority-based conflict handler for resolving contention over shared/exclusive resources (e.g., VRP: one customer, one vehicle).
- **RL4CO** (Berto et al., KDD 2025, github.com/ai4co/rl4co) is a benchmark library/ecosystem (27 CO environments, 23 baselines incl. AM, POMO), not an algorithm — signals that "REINFORCE-only baseline" is now below field standard, and offers a path to get strong baselines (AM, POMO) largely for free by wrapping WTA as a custom environment (RL4CO explicitly supports adding custom envs).

## Comparison case: the Litho paper does this right

The lithography scheduling paper (already submitted, same author) is the template for "how to do this properly":
- Explicitly cites Bertsekas rollout [48] and SGBS [18], and clearly differentiates from POMO's permutation-symmetry trick ("DRCLSP has no such symmetry... we instead manufacture diversity by anchoring...").
- Real, ablation-proven contributions: **Heuristic-Anchored Parallel Episodic Rollout** (solves REINFORCE trajectory collapse for problems without permutation symmetry) and **Reticle-Edge Pruning** (ablation: high-mix N=200 gap goes from +30.7% → +0.8% with pruning — a decisive, quantified contribution).
- Uses an established benchmark (Ham 2018 CP formulation) rather than only self-generated synthetic data.
- Releases code, data, and checkpoints on GitHub for reproducibility.

BReRLA should be revised to match this standard of honesty and rigor.

## Strategic decision: what counts as "the contribution"

Beam search + rollout is **a tool, not the contribution** (it's SGBS/Bertsekas, properly cited). The candidate real contributions, in order of strength:

1. **(To be built/validated) Core architectural contribution — hard-coupling vs. soft-coupling insight for multi-agent NCO:**
   PARCO's conflict handler exists to resolve **hard** feasibility conflicts (e.g., VRP: one customer can only be visited by one vehicle — a discrete, binary infeasibility). WTA has no such hard exclusivity — multiple weapons *can* legally strike the same target simultaneously. But WTA's objective is nonlinear/multiplicative (Eq. 1: Π(1−P_{m,n})^x — diminishing marginal returns per additional hit on the same target), which creates a **soft coordination problem**: independently-greedy agents may wastefully over-concentrate fire on one target. Research question: does a communication-only parallel decoder (transformer communication layers + multiple pointer mechanism, **no explicit conflict handler**) learn to solve this soft-coordination problem on its own? This reframes the contribution as a general taxonomy claim about multi-agent combinatorial optimization structure (hard-coupled vs. soft-coupled problems need different architectures), with WTA as the validating case study — not just "PARCO applied to a new domain."
   - Needs a concrete test: e.g., a "target-dispersion" metric comparing parallel vs. sequential decoding, to empirically show coordination is/isn't emerging.
   - Practical payoff already visible in BReRLA's own Table 6: Beam-Rollout inference time explodes from 6.3s at (5,5,5) to 208s at (10,10,10) — plausibly because decoding is sequential per-weapon-per-step (~M×T steps). Parallel decoding directly targets this bottleneck and connects to the paper's own stated goal of "real-time decision-making."
2. **Problem formulation contribution (already have the data):** first learned, size-generalizing DWTA solver jointly modeling reload time + ammunition + target time windows, benchmarked systematically vs. CPLEX across 12 configs. Legitimate but modest — application-paper-grade, not top-venue-grade on its own.
3. **Empirical rigor contribution (buildable via RL4CO integration):** first systematic, statistically rigorous (multi-seed, mean±std), standardized benchmark of neural CO methods (incl. AM/POMO baselines) on dynamic WTA, built on the RL4CO ecosystem for credibility and reproducibility.

If (1) is successfully built and validated, the paper has a genuine, defensible algorithmic contribution and could aim higher than a pure applications journal. If (1) doesn't pan out empirically, contributions (2)+(3) alone are still honest and sufficient for a Q2 applications/OR journal — not a wasted effort either way.

## Venue shortlist (non-MDPI, verified quartiles as of search date)

| Journal | Publisher | Quartile | IF | Notes |
|---|---|---|---|---|
| Journal of the Operational Research Society | Taylor & Francis | Q2 | 2.7 | WTAP's home community; reviewers less likely to know SGBS/PARCO, lower risk of the novelty objection recurring |
| Journal of Combinatorial Optimization | Springer | Q2 | — | Title matches topic exactly |
| Soft Computing | Springer | Q2 | 2.5 | Broad AI/heuristics applications |
| Applied Intelligence | Springer | Q2 | 3.5 | AI applications community |
| Naval Research Logistics | Wiley | Q1 (Modeling & Sim, Ocean Eng) / Q2 (Mgmt Sci & OR) | — | Direct defense-OR fit, but reviewer pool may be more rigorous on WTAP specifics |
| IEEE Trans. Aerospace and Electronic Systems | IEEE | Q1 | 5.7 | Too high a bar given current honest scope; revisit if (1) above succeeds |
| Neural Computing and Applications | Springer | Q1 | 6.5 | Too high a bar currently |
| MDPI options (Applied Sciences, etc.) | MDPI | Q2 | 2.9 | Ruled out — reputational concerns raised by user |

Current leaning: **Journal of the Operational Research Society**, revisit if the parallel-decoding contribution (item 1 above) succeeds and justifies aiming higher.

## IEEE Access resubmission vs. new venue — risk tradeoff

- Resubmitting to IEEE Access: reviewers likely reassigned (same 5), and per the decision letter this is the **last chance** — if concerns aren't fully addressed, permanent rejection with no further resubmission. R3 alone listed 11 discrete requests (module-wise ablation, error analysis, distribution-shift testing, FLOPs/memory reporting, etc.) — a high bar for one revision cycle.
- New Q2/Q3 venue: slower (fresh review cycle), but no accumulated skepticism from reviewers who already suspect "just technical integration"; can target a review community (OR) less likely to know SGBS/PARCO, and doesn't require satisfying the full IEEE Access laundry list — just the moderate, well-defined fixes below.
- **Decision: pursue a new venue**, not IEEE Access resubmission.

## Session 2: manuscript finalization + real implementation

### Manuscript changes (`paper/BReRLA_manuscript.tex`, `paper/references.bib`)

- **Title changed** to "Soft-Coupled Multi-Agent Reinforcement Learning for Dynamic Weapon-Target Assignment" (dropped "Beam Search and Rollout Enhanced" from the headline since that's a tool, not the contribution).
- **Contribution bullets rewritten** around the hard-coupling vs. soft-coupling distinction (see below), with beam+rollout explicitly reframed as adapted prior art, not a claim.
- **Related Work restructured** with new subsections: "Inference-Time Search for NCO" (Bertsekas rollout, SGBS, EAS), "Multi-Agent Combinatorial Optimization" (CommNet, TarMAC, PARCO -- PARCO repositioned as the *comparison point* for hard-coupled problems, not a source we extend), "Neural Construction Architectures for Assignment-Structured and Large-Scale Problems" (MatNet, LEHD, RL4CO, Grinsztajn et al. population training).
- **Problem formulation**: added weapon-target engagement compatibility set $Q_m \subseteq N$ (mirrors the Litho paper's qualified-pairs $Q_j$), with corresponding constraint and notation entry.
- **Baselines section** updated to name planned comparisons: CPLEX/SCIP (see open item below), GA (own prior work, `lee2003genetic`), AM, POMO, MatNet -- all to be run inside RL4CO for a fair/standardized comparison.
- **references.bib**: added SGBS, Bertsekas rollout, EAS, CommNet, TarMAC, PARCO, MatNet, LEHD, RL4CO, Grinsztajn et al., Mazyavkina survey, Bogyrbayeva survey, and the companion Litho paper (`lee2026litho`). Fixed the Bellman citation (was garbled via a Google-Scholar copy-paste artifact), fixed the Hodson/"M. W. T. H. Ill" author-field transposition bug, flagged Britannica/OR-Society for replacement with `kirby2003operational`, fixed the ref-[35] mismatch (replaced with Zhang et al.'s receding-horizon papers, which actually match the claimed methodology), filled in full author lists for Gaudet et al. and Hu et al., proposed `yoon2024reinforcement` as the peer-reviewed JDMS replacement for the ASU Master's thesis citation (Yoon, **Lee** [=paper's own author], Cho -- pending user confirmation this is the right paper).
- Added then **reverted** a paragraph motivating short/repeated decision cycles that used the Litho paper's 120s CP time budget as a numeric anchor for DWTAP -- user correctly flagged this as transplanting a semiconductor-specific number into defense context without justification. Left as a `%% TODO` comment: find/justify a DWTA-appropriate real-time budget from the defense literature (OODA-loop framing via the existing `zhang2020efficient`/`zhang2020dynamic` receding-horizon citations is a reasonable starting point) before submission. **Deferred, not urgent.**

### The core technical narrative (for anyone picking this up cold)

1. **Novelty framing** (see "Strategic decision" section above, refined further this session): BReRLA's real claim is that DWTAP is a *soft-coupled* multi-agent problem (coordination emerges from the nonlinear/multiplicative objective, Eq. 1 -- diminishing returns from over-concentrating fire on one target) rather than a *hard-coupled* one (PARCO's setting: exclusive resource contention, e.g. one vehicle per customer). We test whether a parallel multi-pointer decoder *without* PARCO's learned conflict handler is sufficient to induce coordinated, non-wasteful targeting.
2. **Important correction during discussion**: PARCO's own components 1 (transformer communication) and 2 (multiple-pointer simultaneous decoding) are *not* PARCO's novel contribution either -- both predate PARCO by years (CommNet, Sukhbaatar & Fergus 2016; TarMAC, Das et al. 2019; pointer networks, Vinyals et al. 2015). PARCO's actual addition is component 3, the conflict handler. So the paper's real framing is: **we build on the standard communication+pointer lineage directly, and show PARCO's conflict-handler addition is unnecessary for soft-coupled problems** -- PARCO is a comparison point, not something we "extend."
3. **Second independent motivation for the graph+pruning architecture** (beyond the soft-coupling argument): full self-attention over all $M \times N$ weapon-target pairs is $O((MN)^2)$, which plausibly explains the inference-time blowup already visible in the original paper's Table 6 (6.3s at (5,5,5) -> 208s at (10,10,10)). A compatibility-pruned bipartite graph (via $Q_m$) reduces this to the number of actually-compatible pairs. This ties the architecture choice to already-collected empirical evidence, not just a conceptual argument.
4. **Skip/no-op action as a second analytical thread supporting the same soft-coupling claim** (not a separate headline contribution): the no-op is a genuine strategic lever in DWTAP (ammunition is finite over the whole horizon, so firing now forecloses a possibly-better future opportunity -- an intertemporal tradeoff routing/VRP problems don't have). Verified via literature search that this is a real, under-studied gap: Tassel et al. 2021 (`tassel2021reinforcement`, already in bib) include a No-Op action for job-shop scheduling RL with heuristic eligibility rules, but explicitly do **not** ablate its effect (only observe that the agent "learns to use it sparingly"). Planned ablation: **Always-Act** (no-op only as pure feasibility fallback, like PARCO's passive fallback) vs. **Skip-Enabled** (no-op as a first-class strategic choice).
5. **Reward-signal difficulty for no-op, and its fix**: the user had independently struggled to find a good reward signal for no-op (immediate reward is always exactly 0 for a no-op step). Diagnosed as a credit-assignment problem, not a fundamental flaw in using pure REINFORCE (the user's design rationale for avoiding actor-critic bootstrapping -- avoiding bias from an inaccurate critic -- is sound and was preserved). Fix implemented: **potential-based reward shaping** (Ng, Harada & Russell, ICML 1999) using the existing critic $V(s)$ as the potential $\Phi$. This is provably policy-invariant (does not introduce the bias actor-critic bootstrapping would), because it telescopes to a constant over a complete trajectory -- it only reduces variance / densifies credit assignment. Combined with **reward-to-go** (standard, unbiased REINFORCE variance reduction: a step's credit shouldn't depend on reward realized before it was taken).
6. **Deferred idea, explicitly not pursued yet**: combining truncated rollout (empirical, "partial" value for the near horizon) with the critic (model-based estimate of the remainder) for training-time value estimation, potentially via variance-weighted ("Bayesian") fusion. User correctly noted classical Bertsekas rollout uses a single deterministic greedy-heuristic pass (no variance estimate); genuine Bayesian fusion would need multiple stochastic rollouts, which is architecturally closer to MCTS than to single-pass Bertsekas rollout. **Deferred until after the core contribution is validated.**

### Code changes (local clone at `code_DWTA/`, GitHub: `jlee-105/dwta-revision-after-ieee-access`, not yet pushed -- see below)

Investigated two repos: `jlee-105/DWTA_VER2/GNN_MODEL` (an earlier, abandoned attempt at moving to a GNN) and `jlee-105/DWTA` (the current one, per user). Key findings before any edits:
- **Already has a working heterogeneous GNN encoder** (`common/DWTA_GNN.py`, `EdgeAwareGNNLayer`): weapon nodes, target nodes, edge = damage rate $P_{m,n}$, with proper message passing. This is directly reusable -- did not need to design from scratch.
- **Decoding was sequential**: `EdgeAwareGNN_ACTOR.forward()` produced ONE global softmax over all $W \times T + 1$ options per call; the training loop (`Dynamic_Sampling_GNN.py`) looped `for time_step: for weapon_idx:` calling the actor once per iteration (confirmed this is $M \times T$ actor calls per episode, matching the hypothesized complexity bottleneck). Note: the `weapon_idx` loop variable was **not actually used to restrict the choice** -- each call could pick any still-valid (weapon,target) pair globally, so it was "$M$ decision opportunities per time step," not literally per-weapon turns.
- **Exact solver is SCIP, not CPLEX** (`opt/SCIP.py` in both repos). The manuscript says CPLEX throughout. **Unresolved -- needs user confirmation**: was CPLEX actually run separately for the reported numbers, or should the manuscript be corrected to SCIP throughout (Tables, Fig. 2, Section VI text)?
- **Training loop is "parallel multi-episodic REINFORCE"** as described in the paper, using `NUM_PAR` parallel rollouts of the *same* instance with a POMO-style shared baseline (`returns.mean(dim=1)`) -- but with **no anchoring/diversity mechanism**, i.e. naive i.i.d. stochastic sampling. Since DWTAP (like the Litho paper's DRCLSP) lacks permutation symmetry, this risks the same trajectory-collapse problem the Litho paper found and fixed with Heuristic-Anchored Rollout. **Not yet fixed here -- flagged as a candidate follow-up, not done this session** (scope was kept to the parallel-decoder + reward-shaping changes below).
- **Critic existed but was never used in the actor's advantage computation** -- confirmed exactly the gap that made the no-op credit-assignment problem hard (see point 5 above).
- User separately tried **Expert Iteration (EXIT) training** (`rl/DWTA_EXIT_TRAIN.py`, Bertsekas-style policy improvement via multi-policy beam search + hybrid imitation/REINFORCE loss) -- reported as **unstable**. Explicit decision: **do not pursue EXIT this round**; training stays plain REINFORCE (with the reward-to-go + shaping fix below), which is already validated (produced the existing Table 5/6 results).

**Changes made** (all in `code_DWTA/`, committed as `1caae7f` on `main`, local only -- push blocked on GitHub auth):

1. `common/DWTA_GNN.py` -- `EdgeAwareGNN_ACTOR`: replaced the single global $W{\times}T{+}1$ softmax with **parallel multi-pointer decoding** -- $M$ simultaneous, independent softmaxes (one per weapon) over {that weapon's targets, no-op}, computed in one forward pass. No-op score is now per-weapon (weapon's own embedding + pooled global context), not one system-wide no-op. GNN message passing (global context/communication) is unchanged -- only the output head changed.
2. `common/DWTA_GNN.py` -- `EdgeAwareGNN_CRITIC`: now actually uses the mask (previously accepted but ignored). Uses **masked pooling** over edges (illegal weapon-target pairs excluded from the pooled global state, since they carry no actionable information that round) and recovers exact $(W,T)$ from the mask shape instead of a fragile square-root/factorization guess that could silently pick the wrong split for ambiguous non-square sizes.
3. `common/DWTA_Simulator.py` -- `Environment`: added `mask_per_weapon` ($[B,P,W,T{+}1]$, built from the same underlying legality tensor as the legacy flat mask, so it can't drift out of sync) and `update_internal_variables_parallel()` (applies all $M$ weapons' simultaneous decisions in one call, reusing the existing scalar per-(weapon,target) update helpers -- correct even when multiple weapons hit the same target in the same round, since the multiplicative damage update is order-independent).
4. `rl/Dynamic_Sampling_GNN.py` -- `self_play_gnn`: switched `Environment` import from `rl_rollout.DWTA_Simulator_rollout` (a separate, less-refactored duplicate with a possibly-inconsistent damage formula -- flagged, not used) to `common.DWTA_Simulator` (the one just fixed). Removed the per-weapon inner loop; one actor forward pass per time step now decides all weapons simultaneously. Implemented **reward-to-go + potential-based shaping** (see point 5 above) replacing the old single whole-episode-return advantage applied uniformly to every step.
5. Careful dimension audit done at user's request (tracked batch $B$ / para $P$ / weapons $W$ / targets $T$ through every reshape): fixed an entropy-logging bug (`total_steps` was incorrectly incremented by $W$ per call when `step_entropy.mean()` was already averaged over $B,P,W$ -- logging-only, did not affect gradients) and made the multinomial-sampling reshape use actual tensor shape (`current_state.shape[:2]`) instead of the global `TRAIN_BATCH`/`NUM_PAR` constants, for robustness.
6. Fixed `common/DWTA_GNN.py`'s own `__main__` self-test to use the new per-weapon mask shape.
7. Added `requirements.txt` (`torch>=2.0`, `numpy`, `pandas`, `pytz`) -- none existed before, a reproducibility gap.

**Verified working** (local smoke tests, CPU-only, Python 3.10 venv at `NEURAL_CO/venv_dwta/`):
- `python -m common.DWTA_GNN`: actor/critic forward pass, confirms policy shape `[2,1,5,6]` = `[batch,para,W=5,T+1=6]` as expected for the new parallel decoder.
- `code_DWTA/smoke_test_parallel.py` (new file, not part of the original repo): runs 2 tiny epochs of the full `self_play_gnn` loop end-to-end (multi-scale random sizes each episode, e.g. 5x5x5, 5x6x7, 6x6x5, 5x7x6) -- ran without errors, losses finite, destruction_ratio moved 0.37 -> 0.57 across the 2 epochs (not meaningful evidence of anything, just confirms no NaN/divergence in a minimal run).
- **Not yet verified**: correctness at larger scales, actual training convergence over a real number of epochs, or GPU execution (user has a desktop GPU available for this later).

### Local environment

- Python 3.10 venv at `NEURAL_CO/venv_dwta/` (system default was Python 3.7.2, too old for current PyTorch). CPU-only `torch==2.13.0+cpu`, `numpy`, `pandas`, `pytz`, `openpyxl`.
- New GitHub repo created by user: `https://github.com/jlee-105/dwta-revision-after-ieee-access` (separate from the original `jlee-105/DWTA`, kept as the stable/working baseline). Local `code_DWTA/` has both remotes: `origin` (still points at the original `jlee-105/DWTA` -- do not push there) and `new-origin` (the new repo). Commit `1caae7f` is ready but **push failed on GitHub auth** (no credentials configured in this environment) -- user needs to either push manually from their own terminal, set up `gh auth login`, or configure a credential helper.

## Action plan / TODO status

Done:
1. [x] Locate/obtain BReRLA source code (`jlee-105/DWTA`)
2. [x] Fix citations/positioning: SGBS, Bertsekas rollout, PARCO (repositioned as comparison point), CommNet/TarMAC honestly cited
3. [x] Fix math bugs: constraint (3) floor-at-0; unify Table 1 notation (M/N/T)
4. [x] Formalize hard-coupling vs. soft-coupling taxonomy in Related Work
5. [x] Implement parallel multi-pointer decoder in code (actor, critic, environment, training loop) -- verified via smoke test on CPU
6. [x] Reward-to-go + potential-based shaping for the no-op credit-assignment problem
7. [x] New GitHub repo created, changes committed locally (push pending auth)
8. [x] Rewrite manuscript Introduction/Related Work/Problem Formulation/title/baselines sections

Still open:
9. [ ] Push local commit to `jlee-105/dwta-revision-after-ieee-access` (blocked on GitHub auth)
10. [ ] Resolve SCIP vs. CPLEX naming throughout the manuscript
11. [ ] Define target-dispersion and skip-usage coordination metrics precisely (code + manuscript)
12. [ ] Add Always-Act vs. Skip-Enabled ablation
13. [ ] Integrate WTA as a custom RL4CO environment; wire up AM/POMO/MatNet baselines
14. [ ] Run full experiments: parallel vs. sequential decoding, all 12 configs, multi-seed with mean±std (needs GPU for realistic turnaround)
15. [x] Write the Methodology section's equations to match the now-implemented architecture -- Section V rewritten: V-A heterogeneous graph encoding, V-B parallel multi-pointer policy (Eq. `parallel-policy`), V-C masked-pooling critic, V-D reward-to-go + potential-based shaping (cites `ng1999shaping`), V-E inference beam/rollout unchanged but flagged with a TODO that the beam search still needs adapting to score joint (not single-edge) candidates for the parallel decoder -- not yet implemented, only the training-loop side is smoke-tested.
16. [ ] Find a properly-justified DWTA-specific real-time decision-budget reference (do not reuse the Litho paper's 120s number)
17. [ ] Confirm `yoon2024reinforcement` (JDMS, Yoon/Lee/Cho) is really the peer-reviewed counterpart to the ASU thesis originally cited
18. [ ] Consider applying Heuristic-Anchored Rollout (from the Litho paper) here too, since the parallel-rollout baseline currently has no diversity mechanism and DWTAP lacks permutation symmetry (same risk the Litho paper found and fixed)

## Reviewer comment → fix traceability

Manuscript source: `paper/BReRLA_manuscript.tex`. Bibliography: `paper/references.bib`.

| Reviewer comment | Status | What was actually changed |
|---|---|---|
| R4: math/MDP formulation inconsistencies | **Fixed** | Eq. (3) (`eq:wait_time_update`) now has `max(0, w_{m,t}-1)` floor — previously could go negative when a ready weapon (w=0) didn't fire, violating the stated bounds in Eq. (9)-(10). |
| R4/R5: Table 1 notation doesn't match rest of paper (W/E/T vs M/N/T) | **Fixed** | Table 1 (`tab:notation`) rewritten to use $M$, $N$, $T$ consistently as both set names and indices, matching Section III body text. Removed the unused $W_m$/$E_n$ element notation. |
| R4: references 1, 2, 16, 34, 35, 37, 38, 39 inappropriate | **Mostly fixed, 2 items still need manual verification before submission** | See breakdown below. |
| — ref [1] Britannica | Fixed (pending final check) | Flagged in bib for replacement with `kirby2003operational` (Kirby, *Operational Research in War and Peace*, Imperial College Press, 2003) — a real academic source for the same historical claim. Not yet swapped into the \cite{} call in the text — do this before submission. |
| — ref [2] The OR Society website | Fixed (pending final check) | Same as above; currently still cited via `orhistory` key in text intro sentence — replace with `kirby2003operational` at the same time as ref [1]. |
| — ref [16] "M. W. T. H. Ill" | **Fixed** | Root cause found: bib entry had `author={Ill, MAJOR WILLIAM T HODSON}` — BibTeX parsed "Ill" as the surname. Corrected to `author={Hodson, William T., III}` (key: `hodson1971linear`). Original source not independently re-verified — confirm before submission. |
| — ref [34] Gaudet et al. | Partially fixed | Confirmed real full author list via search: Brian Gaudet, Kris Drozd, Roberto Furfaro, "Deep Reinforcement Learning for Weapons to Targets Assignment in a Hypersonic Strike," arXiv:2310.18509 (2023). Bib entry still needs updating from "et al." to the full list (todo). |
| — ref [35] Li et al. mismatch | **Fixed** | The claim ("decomposed a DWTAP into sequential SWTAPs at each time stage" via receding-horizon-style independent per-stage solving) did not match what the cited Li reference actually contains — likely a reference-manager mixup from thesis writing, not fabrication. Replaced with `zhang2020efficient` + `zhang2020dynamic` (Zhang et al., receding-horizon heuristic decomposition of DWTAP into per-stage SWTAPs), which actually matches the described methodology. Rewrote the surrounding sentence to (a) cite correctly and (b) not claim this decomposition work is learning-based (it's a classical heuristic) since the original paragraph's framing implied a learning-based method. |
| — ref [37] Yoon Master's thesis | **Not yet fixed** | Still cited both as substantive related work and (oddly) as the source of a US Army doctrine quote. Text says "Yoon and Kim" but the ref is solo-authored (C. Yoon). Need to: (1) fix the author mismatch, (2) find a non-thesis source for the doctrine quote, (3) check whether a peer-reviewed version of Yoon's thesis exists to cite instead (addresses R2's "replace thesis/non-peer-reviewed sources" comment too). |
| — ref [38] Hu et al. | Partially fixed | Confirmed full author list via search: Hu, Tao; Zhang, Xiaoxue; Luo, Xueshan; Chen, Tao, "Dynamic Target Assignment by Unmanned Surface Vehicles Based on Reinforcement Learning," *Mathematics* (MDPI), 2024. Bib entry still needs updating from "et al." to full list (todo). Note: this is an MDPI journal (Mathematics) — flag for awareness given the user's MDPI concerns, though this is citing it, not publishing in it. |
| — ref [39] Bellman | **Fixed** | Bib entry was malformed (`journal={New Jersey Google Scholar}` — a Google Scholar copy-paste artifact). This is the foundational MDP reference and should NOT be removed despite R4 flagging it; only the citation formatting was broken. Corrected to a proper `@book` entry (Bellman, *Dynamic Programming*, Princeton University Press, 1957). |
| R1/R2/R3: novelty framed as "technical integration" (beam search + rollout not cited/differentiated from prior art) | **Fixed in prose, not yet empirically validated** | Related Work now has dedicated subsections citing Bertsekas rollout, SGBS, EAS, PARCO, MatNet, LEHD, RL4CO, and explicitly reframes beam search + rollout as an adapted tool, not a contribution. The actual algorithmic contribution claim (hard-coupling vs. soft-coupling, communication-only parallel decoder) is written into the Introduction/Related Work but **not yet implemented or tested** — this is the main remaining risk (see action plan below). |
| R2/R3: multi-seed statistics, mean±std, significance testing | Not yet done | Requires re-running experiments; planned via RL4CO integration (action item 9 below). |
| R1/R3: ablation only on 2/12 configs; module-wise ablation | Not yet done | Requires re-running experiments. |
| R3: reproducibility (episodes, hardware, optimizer, seeds) | Not yet done | Needs the actual training run info — blocked on same info as the Python code. |
| R1: realism (synthetic-only, no compatibility constraints) | **Fixed in formulation** | Added weapon-target engagement compatibility set $Q_m \subseteq N$ to the problem formulation (mirrors the "qualified pairs" $Q_j$ concept from the companion lithography-scheduling paper), with a corresponding constraint Eq. (compatibility_constraint) and notation table entry. Not yet reflected in experimental data generation (todo, depends on Python code). |
| R5: Table 1 text overlap (layout bug) | Not checked | Requires rendering the PDF; revisit once other edits stabilize. |

## Deferred / future work

- **Rollout-critic Bayesian fusion for training-time value estimation**: use truncated rollout (already planned for inference, Section V-E) to also inform the training-time potential Φ(s) — rollout gives an empirical "partial" return for the near horizon (α steps), critic estimates the remainder; simple n-step bootstrap first. IMPORTANT DISTINCTION (user correction): classic Bertsekas rollout uses a single deterministic pass with a *greedy* base heuristic to estimate cost-to-go -- it does not itself produce a variance estimate. Doing true Bayesian/precision-weighted fusion of rollout vs. critic estimates requires *multiple* stochastic rollouts to characterize each estimator's uncertainty, which is architecturally closer to MCTS (tree search with repeated stochastic simulation, e.g. UCT-style exploration) than to single-pass Bertsekas rollout. If pursued, decide explicitly whether to (a) stay within the cheaper single-greedy-rollout Bertsekas paradigm (simple n-step bootstrap, no variance weighting), or (b) adopt an actual MCTS-style multi-rollout scheme to get the variance estimates Bayesian fusion needs (more expensive, bigger scope change). Deferred until after the core parallel-decoder contribution is implemented and validated.

## Open questions for next session

1. Push the local commit -- set up GitHub auth (`gh auth login`, credential helper, or push manually) so `git push new-origin main` from `code_DWTA/` succeeds.
2. Was CPLEX actually run separately for the manuscript's reported numbers, or should all CPLEX references be corrected to SCIP?
3. Confirm `yoon2024reinforcement` is the right peer-reviewed replacement for the ASU thesis citation.
4. Decide whether to apply Heuristic-Anchored Rollout here too (item 18 above) before or after the first real training run.
5. When GPU (desktop) is available: run a real training pass (not just the 2-epoch smoke test) to see if the parallel decoder actually converges and whether the no-op/skip behavior looks sane before investing in the full 12-config multi-seed experiment matrix.
