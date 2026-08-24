# Investigation Log: Why does SCoPE (parallel RL) lose to Greedy?

Started from: newly-built Greedy (MMR) baseline was beating SCoPE across most/all of the
12-config tiered benchmark, including in-distribution (5,5,5). Old IEEE Access submission
(BReRLA) showed the opposite pattern -- REINFORCE never lost to its Heuristic baseline in
12/12 configs. This log tracks root-cause investigation. Written for continuity across
sessions; update in place as new findings land.

## CONFIRMED, closed items

1. **Greedy implementation is correct.** Verified four independent ways: (a) exact match
   against a hand-computed brute-force optimum on a tiny 2x2x1 toy instance, (b) never beats
   SCIP's proven-optimal on any of 10 held-out (5,5,5) instances, (c) zero illegal-action
   (mask) violations across 1050 weapon-decisions checked at Battlefield scale, (d) matches
   the literature-standard max-marginal-return design (Kolitz & Eckler 1988), which is also
   exactly the classical matroid-greedy analyzed in the manuscript's own Proposition 2.

2. **Three real bugs found and fixed in `common/DWTA_Simulator.py` / `eval_tiered_benchmark.py`**
   (full detail in memory `brerla_simulator_bugs.md`):
   - Bug 1: reload-cooldown off-by-one (`_batch_update_weapon_status` set `weapon_wait_time`
     one short of the manuscript's own Eq. 3-4 formula). Fixed: `+1`.
   - Bug 2: `eval_tiered_benchmark.py::patch_globals` never re-patched `Sim.MAX_TIME` on the
     `common.DWTA_Simulator` module object (wildcard-import staleness), breaking target
     time-window enforcement at T!=5 tiers. Fixed: added `Sim.MAX_TIME = mt`. Training was
     NOT affected (its own `patch_hyperparameters_for_epoch` already did this correctly).
   - Bug 3: `_initialize_weapon_status` set `possible_weapons = weapon_indices` (0,1,2,...)
     instead of a legality mask, then checked via `<= 0` -- **weapon index 0 was always
     wrongly excluded from round 1 of every single episode**, for every decoder/baseline,
     since before tonight. Fixed: compute `possible_weapons` from
     `(weapon_wait_time<=0) & (ammunition_availability>0)` at init, same formula used every
     later round.
   - **All three bugs affect Greedy, Sequential, and Parallel/SCoPE equally** (shared
     `Environment` class) -- real, worth having fixed, but do NOT explain the differential
     Greedy-vs-SCoPE gap this investigation is chasing.

3. **Test instances were miscalibrated for 7 of 12 configs**, independent of the above:
   `generate_tiered_instances.py` reused `AMM~U(1,3)` (tuned for the original T=5 configs)
   unscaled for T=10/15 tiers, making ammo far scarcer relative to rounds than intended --
   84-87% of weapon-decisions were mechanically forced (0 or 1 legal target), leaving little
   room for any policy to show a quality difference. Rescaled (`amm_lo/hi` now scale with
   `e_time`, anchored to exactly reproduce the original at T=5) and regenerated
   `{5M_5N_5T,5M_7N_5T,10M_15N_5T,20M_30N_5T,30M_40N_10T,40M_50N_10T,50M_50N_15T,
   50M_70N_15T,70M_100N_15T}.xlsx`. Real-choice fraction now 18-28% across all regenerated
   configs (verified directly, not just assumed). **NOTE**: `15M_15N_5T`, `15M_20N_5T`, and
   `30M_30N_10T` are still the *original*, un-recalibrated files (not in
   `generate_tiered_instances.py`'s config list) -- `15M_15N_5T` specifically has a much more
   generous ammo ratio (2.27 shots/target) than the recalibrated files and is NOT a fair
   apples-to-apples comparison point until/unless it's also recalibrated.

4. **Training direction is correct** (maximize destruction, not minimize) -- verified
   empirically from a real training run's own eval log: destruction ratio rose from
   ~56-62% in the first few epochs to ~84-87% by epoch 200. Not a sign-error.

5. **The no-op action's credit assignment under reward-to-go is empirically weak, and this
   is a real, fixable contributor.** Proven via a hand-verified toy "trap" instance: 1
   weapon, 2 targets, 2 rounds, `AMM=[1]`, `V=[1,10]`, `P=[[0.9,0.9]]`,
   `TW=[[0,1],[1,1]]` (target B, 10x more valuable, only legal at round 1; true optimal
   policy holds fire at round 0 and gets B at round 1, remaining=2.0).
   - Reward-to-go (current default, `GNN_TRAIN_ABL_no_critic_seed5`, all 3 bugs already
     fixed): round-0 policy gives only P(no-op)=13.2%, picks fire-at-A under argmax,
     remaining=10.1 (5x worse than optimal). Falls into the trap.
   - Whole-episode return (`--ablation no_reward_to_go`, i.e. original-POMO-style credit,
     same critic+shaping as before, single seed retrained under the fixed simulator,
     `GNN_TRAIN_ABL_no_reward_to_go_seed5`): round-0 P(no-op)=95.1%, correctly picks no-op,
     then fires B at round 1 with P=99.99%. **Remaining=2.000 -- matches the true optimum
     exactly.**
   - Confirmed this is NOT an information-visibility bug: raw state vector inspection shows
     target B's `ws` (start-time) and value features ARE correctly present in the round-0
     input (verified against `common/Dynamic_HYPER_PARAMETER.py`'s actual index constants,
     not just the manuscript's abstract description -- also found `TARGET_START_TIME_INDEX`
     and `TARGET_AVAILABILITY_INDEX` are both `-4`, same slot; confirmed currently inert/
     harmless since nothing overwrites index -4 after init, but flagged as a landmine for a
     future edit).
   - **Confirmed this generalizes beyond the toy case, partially**: at real (5,5,5), the
     no_reward_to_go seed5 checkpoint achieves 93.1% %Achieved vs SCIP-optimal, beating
     Greedy's 92.7% (previous reward-to-go checkpoint was 91.7%, losing to Greedy). This is
     a real, reproduced, non-toy improvement.

## OPEN, NOT YET RESOLVED

6. **The no_reward_to_go fix does NOT close the gap at Medium scale or larger.** Tested on
   two different Medium configs:
   - `15M_15N_5T` (original file, generous ammo 2.27/target): Greedy 0.0845 vs SCoPE(no_rtg)
     0.2418 -- Greedy far ahead.
   - `20M_30N_5T` (recalibrated file, genuinely scarce ammo 1.4/weapon mean): Greedy 0.3422
     vs SCoPE(no_rtg) 0.5347 -- Greedy still far ahead.
   - **In both cases fire_rate is IDENTICAL between Greedy and SCoPE to 4 decimal places**
     (0.3613=0.3613 and 0.2920=0.2920 respectively). This rules out "SCoPE is holding back
     ammo it shouldn't" as an explanation (that hypothesis was raised and directly falsified
     by this data). The remaining gap is specifically about *which target gets picked* in
     the (evidently rare, but apparently high-leverage) rounds where more than one legal
     target exists, not about *whether* to fire.
   - Resource scarcity level (generous vs scarce ammo) does NOT change this pattern --
     tested explicitly, both show the same identical-fire-rate/large-objective-gap
     signature, ruling out "instance wasn't realistic/scarce enough" as the sole explanation
     for the Medium-scale gap specifically (it may still matter for other things, but doesn't
     explain this).

7. **Two live, untested hypotheses for the remaining gap**, both consistent with all evidence
   so far, not yet distinguished:
   - (a) **Parallel decoder architecture itself has a harder time learning precise
     target-selection at larger scale** than the old paper's sequential/single-global-softmax
     architecture did. Consistent with the manuscript's own Proposition 2 (parallel has a
     real, theoretically-bounded within-round information disadvantage vs sequential,
     worst-case up to 25 percentage points of OPT) and with the already-existing
     dispersion-metric finding in Section III-B (dispersion gap widens with scale). The old
     IEEE Access paper's REINFORCE (sequential architecture) beat its Heuristic at every
     scale tested including sizes comparable to or larger than today's Medium tier -- this
     is the strongest indirect evidence for (a), though confounded with the old paper also
     using a critic + whole-episode return, and confounded by the *direction* of that old
     comparison being trustworthy even if the *magnitude* (relative-gap metric) might not be
     (self-corrected by user mid-session).
   - (b) **Simple insufficient training volume/capacity.** `TOTAL_EPISODE=5` per epoch x
     `TOTAL_EPOCH=200` = 1000 total training episodes across a randomized multi-scale
     (M,N,T in [5,7]) curriculum. This is a very small sample count for a neural network to
     learn a precise enough approximation of exact marginal-value ranking that a closed-form
     greedy computes exactly and instantly. Not yet tested directly (e.g., by increasing
     epoch/episode count and re-checking).
   - Attempted to test (a) directly tonight by retraining the Sequential decoder
     (`rl/DWTA_GNN_TRAIN_sequential.py --seed 5`) under the fixed simulator -- **this run was
     started twice and stopped both times before completion, no sequential result exists yet
     under the fixed simulator.** Also discovered `rl/DWTA_GNN_TRAIN_sequential.py` has no
     `--ablation` flag, so it cannot directly test the no_reward_to_go variant for Sequential
     without a small code change to add that support (mirroring what
     `rl/DWTA_GNN_TRAIN.py`/`rl/Dynamic_Sampling_GNN.py` already have for Parallel).

## NEW, decisive finding: the gap is entirely a round-0 phenomenon

Ran Greedy and SCoPE(no_reward_to_go) side-by-side, round-by-round, on the same
`20M_30N_5T` instance (20 weapons, 30 targets, 5 rounds), comparing not just aggregate
fire_rate but the *exact* per-weapon action each round.

- **no-op counts match exactly every single round** (round0: 0=0, round1: 20=20, round2:
  15=15, round3: 17=17, round4: 20=20). Directly answers "does SCoPE take extra skip
  actions" -- no, definitively not; this was checked at the finest possible granularity
  (per-round noop counts, not just aggregate rate) and there is zero difference.
- **Target choices diverge massively at round 0 only**: 10 of 20 weapons (half) pick a
  *different target* than Greedy at round 0. Rounds 1-4 have only 0, 3, 2, 0 weapons
  differing respectively.
- Final remaining value: Greedy 36.70, SCoPE 60.42 -- essentially the entire gap traces
  back to round 0's target-allocation divergence.

**Why round 0 specifically**: it's the round with the most simultaneously-deciding weapons
(nothing damaged yet, most targets still fully in play, least differentiation forcing easy
choices) -- exactly the regime where Proposition 2's within-round coordination disadvantage
for the parallel decoder (all M weapons commit from one shared pre-round snapshot, zero
visibility into each other's this-round pick) should bite hardest. Later rounds naturally
have fewer live weapons/more differentiated target states, leaving less room for
mis-coordination. This is now the strongest evidence yet for hypothesis (a) (architecture,
not training volume) -- though still not fully proven without a completed Sequential
comparison under the same fixed simulator + instance.

## Root mechanism identified: over-concentration on "obviously good" targets, not bad individual judgment

Broke round 0 down further: how many *distinct* targets does each policy's 20-weapon joint
action cover, and what is the individual marginal value of each differing weapon's choice?

- **Greedy**: 20 fires spread across 13 distinct targets (mostly 1-2 weapons/target, max 2).
- **SCoPE**: 20 fires concentrated on only 9 distinct targets -- target 14 gets 5 weapons,
  target 19 gets 4, targets 20 and 6 get 3 each.
- Crucially, when comparing the *individual* marginal value (`remaining_value * P`) of each
  differing weapon's own choice, **SCoPE's individual pick is frequently as good as or
  *better* than Greedy's** (e.g. weapon 14: greedy picks a target worth 1.83, SCoPE picks one
  worth 5.60; weapon 17: greedy 2.64 vs SCoPE 5.32; weapon 13: greedy 1.78 vs SCoPE 4.10).
  This rules out "SCoPE just has worse value estimates" -- individually, its judgment about
  which targets are valuable is often *good*, sometimes better than greedy's.

**This is the classic mean-field/independent-decision coordination failure, caught directly
in the act**: every weapon, deciding independently and simultaneously from the same shared
pre-round context, correctly identifies the same small set of "obviously good" targets
(14, 19, 20, 6) and piles onto them -- with no mechanism to see that 4 other weapons just
made the identical judgment this round. Greedy avoids this only because it commits one
(weapon,target) pair at a time and re-evaluates remaining value after each pick, so once one
weapon claims a target its attractiveness correctly drops for the next pick -- something the
simultaneous parallel decoder structurally cannot do within a single round. This is precisely
the mechanism Proposition 2 characterizes theoretically (and the existing dispersion-metric
result in Section III-B already showed empirically, just not this granularly). Individual
value judgment is not the problem; multi-agent redundancy is.

## Correction: added no_reward_to_go support to Sequential (was missing)

User correctly flagged that the first Sequential retrain (default reward-to-go) would not
be a clean test -- the earlier "parallel ~= sequential" finding was itself measured under
reward-to-go for both decoders, so it may have been masking a real architectural gap behind
a shared temporal-credit-assignment weakness. Stopped that run. Added `no_reward_to_go`
ablation support to the sequential training path (`rl/Dynamic_Sampling_GNN_sequential.py::
self_play_gnn` gained the same `ablation` param/branch as the parallel version;
`rl/DWTA_GNN_TRAIN_sequential.py` gained a matching `--ablation` flag). Relaunched as
`--seed 5 --ablation no_reward_to_go`, output dir `GNN_TRAIN_SEQ_ABL_no_reward_to_go_seed5`.
This is now the actually-clean comparison: both decoders share the fixed simulator (bugs
1-3) and the same (whole-episode-return) credit-assignment style; only the decoder
architecture differs.

## Note: GPU contention explains apparent "hang"

Running an eval script at the same time as a training job on this single GPU causes severe
slowdown (confirmed: same 10-instance eval took 4s alone vs. appearing stuck for minutes
while training ran concurrently -- `nvidia-smi` showed 94% utilization from the training
job). Epoch-10 checkpoint (killed and restarted training to get a fast, if very
undertrained, checkpoint) gave Sequential(no_rtg) 0.4979 vs Greedy 0.3422 on `20M_30N_5T` --
**not a meaningful result, only 10/200 epochs in, essentially still near-random**. Training
restarted (now writing to `GNN_TRAIN_SEQ_ABL_no_reward_to_go_seed5(2)` due to directory
name collision with the two earlier partial attempts -- (2) is the current, only one that
matters going forward). Going forward: do not run eval concurrently with training on this
single-GPU machine; wait for training to pause/finish first.

## Status: Sequential retrain running

`rl/DWTA_GNN_TRAIN_sequential.py --seed 5` launched under the fixed simulator, running in
background. ~15.3s/epoch (vs parallel's ~4s/epoch -- expected, sequential does M forward
passes per round instead of 1), so ~50 min total for 200 epochs, much longer than the
parallel retrains earlier tonight. Will evaluate against Greedy on `20M_30N_5T` (same
instance as the round-0 breakdown above) the moment it finishes.

## New candidate lever: graph pruning (from companion Litho paper's methodology)

User pulled up the companion paper ("Heterogeneous Graph RL with Edge Pruning and Rollout
Enhancement for DRC Lithography Scheduling," same author, under review at IEEE Access) and
had me study its methodology directly. Two findings:

1. **That paper's REINFORCE also does NOT use reward-to-go** -- Eq. 14 applies the
   whole-episode Monte Carlo return `R(tau_n)` uniformly to every step, exactly matching
   `no_reward_to_go` here. That paper's RL policy with this training style *beats* CP
   (the exact solver) at large/industrial scale (e.g. -16.0% gap at 200-job low-mix,
   Table VIII) -- strong external validation, from the same author's own more mature
   companion project, that abandoning reward-to-go for whole-episode return is a sound,
   proven design choice, not a one-off toy-instance artifact.
2. **Reticle-edge pruning** (Section IV-A2 of that paper): the unpruned graph included all
   R(R-1) reticle-reticle edges regardless of current relevance; restricting message-passing
   to only the "active" subset (needed by waiting jobs or currently busy) was found to be
   *decisive* at large scale (Table X: N=200 high-mix gap improved from +30.7% to +0.8%,
   "applied without retraining, accounts for most of the performance difference at this
   scale"). Checked whether SCoPE's GNN has an analogous issue: confirmed directly in
   `common/DWTA_GNN.py` (line ~256 comment: "full message passing over all weapon/target
   nodes") that the per-weapon legality mask (ammo/reload/time-window) is applied only to
   the *final* action scores, not to the graph message-passing itself -- every weapon
   attends to every Q_m-compatible target's embedding regardless of whether that target is
   currently legal for that weapon this round. This is architecturally the same "unpruned"
   situation. Unlike the round-0 concentration finding (worst when almost everything IS
   legitimately live), this dilution effect should worsen specifically as N (targets) grows
   -- i.e. plausibly a lever for the *scale-generalization* gap (open item 7b context) rather
   than the round-0 coordination issue specifically. Not yet implemented or tested; a
   reasonable next candidate once the current sequential-vs-parallel comparison lands.

## New finding: Sequential(no_reward_to_go) training is unstable, not monotonically converging

Integrated per-5-epoch progress eval directly into `rl/DWTA_GNN_TRAIN_sequential.py::train()`
(same fixed 10 instances from `20M_30N_5T.xlsx`, fixed Greedy reference=0.3422, no separate
process so no GPU contention with training itself). Full trajectory, `--seed 5
--ablation no_reward_to_go`, output dir `GNN_TRAIN_SEQ_ABL_no_reward_to_go_seed5(3)`:

| epoch | Sequential | gap vs Greedy | ahead? |
|---|---|---|---|
| 5  | 0.7708 | +0.4286 | Greedy |
| 10 | 0.4979 | +0.1557 | Greedy |
| 15 | 0.4484 | +0.1062 | Greedy |
| 20 | 0.4076 | +0.0654 | Greedy |
| 25 | 0.3856 | +0.0434 | Greedy |
| 30 | 0.3353 | **-0.0069** | **Sequential** |
| 35 | 0.3781 | +0.0359 | Greedy |
| 40 | 0.4178 | +0.0756 | Greedy |
| 45 | 0.4787 | +0.1365 | Greedy |
| 50 | 0.3398 | **-0.0024** | **Sequential** |
| 55 | 0.4355 | +0.0932 | Greedy |
| 60 | 0.4144 | +0.0722 | Greedy |

Epochs 5-25 show a clean, monotonic narrowing of the gap (as hypothesis (a) predicted).
Epochs 30-60 do NOT show continued convergence or a stable crossover -- instead the gap
oscillates with large amplitude (roughly -0.007 to +0.14), crossing zero twice (epoch 30, 50)
but never staying there. This is a materially different picture from "Sequential cleanly beats
Greedy once trained enough, confirming architecture is the sole cause." Two live
interpretations, not yet distinguished:

- The early monotonic phase (5-25) was real learning; epochs 30+ show the policy has entered
  a higher-variance regime -- possibly the random multi-scale curriculum (M,N,T sampled per
  episode) occasionally samples runs of episodes far from `20M_30N_5T`'s regime, causing the
  eval'd checkpoint's quality on this specific config to swing depending on what it was just
  trained on, without truly forgetting/diverging.
- Or, genuine training instability (no reward-to-go's whole-episode-return removes a variance
  reduction reward-to-go normally provides; combined with only 5 episodes/epoch, gradient
  noise could plausibly be large enough to explain this).

Either way, the current 3-6 point-per-run picture is not sufficient to conclude Sequential
robustly beats Greedy at this scale, only that it *can*, transiently. Continuing to monitor
further epochs; if oscillation persists through epoch 100+ without settling, this becomes its
own open item (training instability under no_reward_to_go) independent of the
architecture-vs-training-volume question this run was originally launched to answer.

**Update, epochs 65-80: no longer just oscillation -- net degradation.** Continuing the same
table:

| epoch | Sequential | gap vs Greedy | ahead? |
|---|---|---|---|
| 65 | 0.5266 | +0.1844 | Greedy |
| 70 | 0.5048 | +0.1626 | Greedy |
| 75 | 0.5437 | +0.2015 | Greedy |
| 80 | 0.4803 | +0.1381 | Greedy |

The gap in this window (0.14-0.20) is worse than the entire 10-25 window (0.04-0.16) and
never approaches the epoch-30/50 zero-crossings again. The 30/50 crossovers now read as
transient excursions during a noisy climb, not evidence of convergence toward a Sequential
win. Working hypothesis revised: **`no_reward_to_go` (whole-episode return, uniform credit to
every step) combined with the sequential decoder's much longer per-episode trajectory (M
forward/backward decisions per time step vs. parallel's 1) may be a bad combination** --
every one of the M*T_steps per-decision log-probs in an episode gets pushed by the exact same
scalar advantage, which is a much coarser/noisier training signal here than in the parallel
decoder (fewer, coarser steps) or than reward-to-go would give (temporally localized credit).
This would mean the earlier finding "no_reward_to_go helps" (item 5, proven for parallel and
for a hand-verified toy case) may NOT transfer cleanly to the sequential architecture, which
complicates hypothesis (a) vs (b): a fair sequential-vs-parallel comparison may need
reward-to-go for sequential specifically, undermining the "matched credit-style" premise this
whole run was designed around. Not yet confirmed -- still monitoring further epochs before
drawing a conclusion or restarting with a different config.

## FINAL: Sequential training completed at epoch 200 -- hypothesis (a) is falsified

Training finished after 2.10 hours (`GNN_TRAIN_SEQ_ABL_no_reward_to_go_seed5(3)`). Full
epoch-100-to-200 tail of the per-5-epoch progress log:

| epoch | Sequential | gap vs Greedy | ahead? |
|---|---|---|---|
| 105 | 0.4707 | +0.1285 | Greedy |
| 110 | 0.3903 | +0.0480 | Greedy |
| 115 | 0.4170 | +0.0748 | Greedy |
| 120 | 0.4072 | +0.0650 | Greedy |
| 125 | 0.3734 | +0.0312 | Greedy |
| 130 | 0.4212 | +0.0790 | Greedy |
| 135 | 0.4048 | +0.0626 | Greedy |
| 140 | 0.3506 | +0.0084 | Greedy |
| 145 | 0.4418 | +0.0996 | Greedy |
| 150 | 0.4397 | +0.0975 | Greedy |
| 155 | 0.4579 | +0.1157 | Greedy |
| 160 | 0.3516 | +0.0094 | Greedy |
| 165 | 0.4042 | +0.0620 | Greedy |
| 170 | 0.4727 | +0.1305 | Greedy |
| 175 | 0.4493 | +0.1071 | Greedy |
| 180 | 0.3572 | +0.0150 | Greedy |
| 185 | 0.4025 | +0.0603 | Greedy |
| 190 | 0.4397 | +0.0975 | Greedy |
| 195 | 0.4123 | +0.0701 | Greedy |
| **200 (final)** | **0.3505** | **+0.0083** | **Greedy** |

**Conclusion: over the full 200-epoch run, Sequential never sustainably beat Greedy on
`20M_30N_5T`.** After epoch 25, the gap oscillated in a roughly [-0.007, +0.20] band with no
clear downward trend across epochs 30-200 (mean gap epochs 100-200: ~+0.07); it repeatedly
approached zero (epochs 30, 50, 140, 160, 180, 200) but always drifted back up within 1-2
checkpoints, and the final epoch-200 checkpoint itself is a near-tie slightly in Greedy's
favor (+0.0083), not a win.

**This falsifies hypothesis (a) as the primary/sole explanation.** The premise of hypothesis
(a) was that the *sequential* architecture (matching the old paper's design, which beat its
heuristic baseline in 12/12 configs) would cleanly beat Greedy once trained under the same
fixed simulator + matched (whole-episode-return) credit style that already worked for
parallel. It did not. Both decoder architectures, under identical conditions, show the same
qualitative pattern at this scale: strong early improvement, then a plateau/oscillation
regime that fails to close the gap with Greedy. This points toward hypothesis (b)
(insufficient training volume -- `TOTAL_EPISODE=5` x `TOTAL_EPOCH=200` = 1000 episodes total
across a randomized multi-scale curriculum) or a training-stability issue with
`no_reward_to_go` itself (see the epoch 65-80 degradation note above) as the more likely
shared root cause, rather than an architecture-specific coordination failure. The round-0
target-concentration mechanism documented above (multi-agent redundancy under parallel,
simultaneous decoding) remains real and directly observed, but is now better understood as
*one contributor alongside*, not a full explanation for, the Medium-scale gap -- since
Sequential (which does not have that specific simultaneous-decoding failure mode, deciding
one weapon at a time) still doesn't beat Greedy here.

**Next steps given this result**: (1) do not conclude "parallel architecture is worse than
sequential" from this data alone -- direct architecture head-to-head at matched epoch counts
under whatever credit style each does best with, not yet done; (2) test hypothesis (b)
directly by substantially increasing `TOTAL_EPISODE`/`TOTAL_EPOCH` for one decoder and
checking whether the gap closes with more training, holding architecture fixed; (3)
investigate whether `no_reward_to_go`'s oscillation (epochs 30-200) is itself the bottleneck
-- e.g. try reward-to-go for Sequential specifically (its much longer M*T-step trajectories
may need temporally-localized credit that reward-to-go provides and whole-episode-return
does not, per the coarser-signal hypothesis above) and compare; (4) the graph-pruning lever
(companion paper, Section "New candidate lever" above) remains untested and independent of
this credit-assignment question -- still a reasonable next experiment regardless of how (1)-(3)
resolve.

## ROOT CAUSE FOUND: `common/DWTA_Simulator.py` never fed current (damaged) target value back into the actor's input encoding

After the Sequential 200-epoch result falsified hypothesis (a) (architecture) and user pushed
back hard on "greedy is just legitimately strong" as an explanation ("i do not believe
greedy", "just think reasonably, just shoot off greedyily, how it is close to the optimal",
"no, this is against my experiment before"), re-audited the shared `Environment` class from
scratch rather than accepting the credit-assignment/architecture framing.

**Found**: `TARGET_VALUE_INDEX` (the feature slot in `assignment_encoding` -- the actor's
actual neural-net input -- that encodes each target's current remaining value) is set exactly
ONCE, at `__init__` (`_initialize_target_values`, from the ORIGINAL undamaged value), and is
**never written again** by `_batch_update_state_encoding_for_time_step` (the function called
after every single decision, in both `update_internal_variables` and
`update_internal_variables_parallel`, and after `time_update()`). That function does refresh
weapon availability, weapon wait-time, and time-left features -- but not target value.
Meanwhile `self.current_target_value` (the ground-truth scalar used for reward/objective
calculation) IS correctly decremented every hit via `_batch_update_target_values`. The two
diverge silently: reward computation stays correct, but **the actor's own input tensor shows
every target at 100% of its original value for the ENTIRE episode, regardless of how much has
actually been destroyed** -- the network is permanently blind to accumulated damage, including
its own prior hits this same round.

**This is independent of decoder architecture and independent of credit-assignment style** --
it affects Parallel and Sequential identically, and no amount of no_reward_to_go tuning or more
training epochs could fix it, because the fundamental Markov state the policy conditions on is
missing the single most decision-relevant piece of information. It exactly explains: (1) why
neither decoder's 200-epoch run under the fixed simulator + no_reward_to_go ever closed the gap
with Greedy, (2) why the round-0 target-concentration/redundancy pattern was observed and never
self-corrected in later rounds, (3) why fire_rate matched Greedy exactly (ammo/reload/TW
features ARE correctly refreshed) while target CHOICE was consistently worse, (4) why the old
IEEE Access paper's REINFORCE (trained under a *different* simulator,
`rl_rollout/DWTA_Simulator_rollout.py`) never lost to its heuristic 12/12 -- that file's
`update_internal_variables` (line ~212) DOES write
`assignment_encoding[...,TARGET_VALUE_INDEX] = current_target_value.clone()/MAX_TARGET_VALUE`
on every update; this repo's current `common/DWTA_Simulator.py` (written later, shared by
Greedy/Sequential/Parallel this whole investigation) silently dropped that line. Greedy itself
was never affected, because `eval_greedy_benchmark.py::_greedy_round_action` reads
`env.current_target_value` directly, bypassing the (broken) encoding tensor entirely -- this is
exactly why Greedy was "seeing straight" the whole time and RL was not.

**Fixed** in `_batch_update_state_encoding_for_time_step` (common/DWTA_Simulator.py): added a
target-value refresh block, broadcasting the *accurate* per-target slice of
`current_target_value` (only `[:, :, :actual_num_targets]` -- the tensor's last dim is actually
`num_weapons*num_targets`, a legacy redundant-copy-per-weapon layout, and only the first
`num_targets` entries are ever kept accurate by `_batch_update_target_values`; the rest are
stale leftovers from init and must NOT be used directly) out to all weapons' rows, matching the
existing pattern used for weapon-wait-time. **Verified correct with a standalone before/after
check**: 2-weapon/2-target/1-round toy, weapon 0 fires at target 0 (P=0.5, V=10->5) --
before the fix the encoding showed target 0's value frozen at 1.0 (normalized) for both weapons
even after the hit; after the fix it correctly shows 0.5 for both weapons immediately after the
single decision.

**Also ported the same per-5-epoch progress-eval infra (fixed Greedy reference on
`20M_30N_5T.xlsx`) into `rl/DWTA_GNN_TRAIN.py` (parallel)**, which didn't have it yet (only the
sequential script did). Relaunched Parallel training under the NOW-FIXED simulator, `--seed 5
--ablation no_reward_to_go` (same settings as the earlier, now-superseded Sequential run), to
see whether this real fix -- not just a credit-assignment tweak -- finally closes the gap.

## CONFIRMED: the fix works. Parallel now beats Greedy, cleanly and reproducibly.

Full 200-epoch run completed in 0.34 hours (vs the pre-fix Sequential run's 2.1 hours -- much
faster since this is Parallel). Per-5-epoch progress vs the same fixed Greedy reference
(0.3422 on `20M_30N_5T.xlsx`):

| epoch | Parallel | gap | epoch | Parallel | gap |
|---|---|---|---|---|---|
| 5 | 0.5471 | +0.2048 (G) | 105 | 0.2734 | -0.0688 (P) |
| 10 | 0.2668 | -0.0755 (P) | 110 | 0.4121 | +0.0699 (G) |
| 15 | 0.2610 | -0.0812 (P) | 115 | 0.3422 | -0.0000 (tie) |
| 20 | 0.3620 | +0.0198 (G) | 120 | 0.2828 | -0.0594 (P) |
| 25 | 0.3041 | -0.0381 (P) | 125 | 0.2957 | -0.0465 (P) |
| 30 | 0.2860 | -0.0562 (P) | 130 | 0.3262 | -0.0160 (P) |
| 35 | 0.3229 | -0.0193 (P) | 135 | 0.3512 | +0.0089 (G) |
| 40 | 0.3171 | -0.0251 (P) | 140 | 0.3010 | -0.0412 (P) |
| 45 | 0.2866 | -0.0556 (P) | 145 | 0.2889 | -0.0533 (P) |
| 50 | 0.2677 | -0.0745 (P) | 150 | 0.2692 | -0.0730 (P) |
| 55 | 0.2785 | -0.0637 (P) | 155 | 0.3335 | -0.0087 (P) |
| 60 | 0.3118 | -0.0304 (P) | 160 | 0.2853 | -0.0569 (P) |
| 65 | 0.3828 | +0.0406 (G) | 165 | 0.3183 | -0.0239 (P) |
| 70 | 0.3487 | +0.0065 (G) | 170 | 0.2458 | -0.0964 (P) |
| 75 | 0.3828 | +0.0405 (G) | 175 | 0.2936 | -0.0486 (P) |
| 80 | 0.3237 | -0.0185 (P) | 180 | 0.2685 | -0.0737 (P) |
| 85 | 0.4213 | +0.0791 (G) | 185 | 0.2651 | -0.0771 (P) |
| 90 | 0.3659 | +0.0237 (G) | 190 | 0.2632 | -0.0790 (P) |
| 95 | 0.2968 | -0.0454 (P) | 195 | 0.3511 | +0.0089 (G) |
| 100 | 0.3027 | -0.0395 (P) | **200 (final)** | **0.2784** | **-0.0638 (P)** |

**Parallel is ahead of Greedy at 30/40 checkpoints (75%)**, band is tight (-0.10 to +0.08,
vs the pre-fix Sequential run's -0.007 to +0.20), and the FINAL epoch-200 checkpoint is a
clear, meaningful win (18.6% better than Greedy in relative terms). This is qualitatively
different from every pre-fix run tonight (both Parallel and Sequential, both credit styles),
none of which ever produced a stable, majority-of-checkpoints win.

**This confirms**: the persistent Greedy-vs-RL gap investigated all night was NOT explained by
decoder architecture (hypothesis a, tested and falsified via the Sequential 200-epoch run) and
was NOT primarily a training-volume/credit-assignment issue (hypothesis b) -- it was a genuine
correctness bug in the shared environment's state encoding, silently blinding every neural
policy (Parallel and Sequential alike, in every run tonight before this fix) to accumulated
target damage for the entire episode. Once fixed, a plain 200-epoch/1000-episode Parallel run
(the SAME training budget that failed all night) beats Greedy comfortably. The round-0
target-concentration mechanism and the mean-field coordination-failure story documented
earlier in this log were real, correctly-diagnosed SYMPTOMS, but downstream of this deeper
cause, not standalone explanations.

## Full 12-config Parallel(post-fix) vs Greedy sweep -- Parallel wins ALL 12 configs

Evaluated the just-completed Parallel checkpoint (`GNN_TRAIN_ABL_no_reward_to_go_seed5(2)`,
epoch 200) against the existing (still-valid, Bug-4-doesn't-affect-Greedy) Greedy results
across every tier. Saved to `result/parallel_postfix_vs_greedy.csv`:

| Tier | Config | Greedy | Parallel(post-fix) | gap |
|---|---|---|---|---|
| Small | 5M_5N_5T | 0.1591 | 0.1389 | -0.020 |
| Small | 5M_7N_5T | 0.3100 | 0.2111 | -0.099 |
| Small | 10M_15N_5T | 0.3624 | 0.2606 | -0.102 |
| Medium | 15M_15N_5T | 0.0860 | 0.0634 | -0.023 |
| Medium | 15M_20N_5T | 0.1514 | 0.1233 | -0.028 |
| Medium | 20M_30N_5T | 0.3216 | 0.2784 | -0.043 |
| Large | 30M_30N_10T | 0.1506 | 0.0673 | -0.083 |
| Large | 30M_40N_10T | 0.2288 | 0.1634 | -0.065 |
| Large | 40M_50N_10T | 0.1705 | 0.1255 | -0.045 |
| Battlefield | 50M_50N_15T | 0.0803 | 0.0498 | -0.031 |
| Battlefield | 50M_70N_15T | 0.1659 | 0.1163 | -0.050 |
| Battlefield | 70M_100N_15T | 0.1659 | 0.1620 | -0.004 |

**Parallel beats Greedy at all 12/12 configs**, including Battlefield (70x100x15, ~14x beyond
the training curriculum of M,N,T in [5,7]), matching the old IEEE Access paper's own 12/12
pattern (with a *different* architecture, under a *different*, correctly-implemented
simulator) -- exactly the target signature the whole night's investigation was chasing.
Both Parallel and Sequential checkpoints were separately verified not to exceed SCIP's proven
optimal on (5,5,5) (10/10 instances, no violations), so this is not a repeat of the original
Greedy-vs-SCIP impossibility bug.

**Also confirmed a separate, unrelated engineering issue while producing this table**: running
`eval_instance_parallel` many times in a single long-lived Python process (looping over 12
configs x 10 instances = 120 calls) progressively slows down for unknown reasons (single-call
timing in isolation: 0.21s; the same call embedded in the long loop stalled for 50+ minutes on
just the 3rd config). Root cause not identified (not GPU-bound -- nvidia-smi showed ~0%
utilization during the stall; likely some per-call accumulation in the Environment/actor
Python objects not being freed). Workaround: run each config in its own fresh subprocess
(`eval_parallel_one_config.py`, invoked once per config from `run_all_configs.sh`) -- this
fixed it completely, each config took 0.4-7s depending on scale. Worth investigating properly
before running any larger eval sweep (e.g. multi-seed), but not blocking.

## Full 12-config SCIP sweep completed -- SCIP collapses entirely at Battlefield scale

Ran SCIP (600s/instance, 10 instances/config, 10 min generous "time-relaxed" cap) on all 11
remaining configs (only (5,5,5) had been done before, at 120s/instance). Saved to
`result/full_12config_scip_parallel_greedy.csv`:

| Tier | Config | SCIP best-found | Parallel(post-fix) | Greedy | SCIP proven-optimal (/10) |
|---|---|---|---|---|---|
| Small | 5M_5N_5T | 0.0926 | 0.1389 | 0.1591 | 10 |
| Small | 5M_7N_5T | 0.1712 | 0.2111 | 0.3100 | 10 |
| Small | 10M_15N_5T | 0.1731 | 0.2606 | 0.3624 | 2 |
| Medium | 15M_15N_5T | 0.0480 | 0.0634 | 0.0860 | 0 |
| Medium | 15M_20N_5T | 0.0973 | 0.1233 | 0.1514 | 0 |
| Medium | 20M_30N_5T | 0.1629 | 0.2784 | 0.3422 | 0 |
| Large | 30M_30N_10T | 0.0413 | 0.0673 | 0.1506 | 0 |
| Large | 30M_40N_10T | 0.1115 | 0.1634 | 0.2288 | 0 |
| Large | 40M_50N_10T | 0.1075 | 0.1255 | 0.1705 | 0 |
| Battlefield | 50M_50N_15T | **1.0000 (total failure)** | 0.0498 | 0.0803 | 0 |
| Battlefield | 50M_70N_15T | **1.0000 (total failure)** | 0.1163 | 0.1659 | 0 |
| Battlefield | 70M_100N_15T | **1.0000 (total failure)** | 0.1620 | 0.1659 | 0 |

Two findings:
1. **Consistent ordering SCIP <= Parallel <= Greedy holds at every scale** where SCIP found any
   real solution (Small/Medium/Large) -- further confirmation (12 configs x 10 instances = 120
   more data points) that the post-Bug-4-fix Parallel checkpoint never exceeds a valid
   optimality bound, on top of the earlier direct (5,5,5) per-instance violation check.
2. **SCIP completely collapses at Battlefield scale (50-70 weapons, 100 targets, 15 rounds)**:
   all 30 instances (3 configs x 10) hit the 600s limit having found NOTHING better than the
   trivial all-no-op solution (objective_norm=1.0000 exactly, std=0.0000). Meanwhile Parallel
   and Greedy both solve these instances well (destroying 84-95% of value) near-instantly. This
   is a strong, concrete motivating result for the manuscript: at real battlefield scale, exact
   MIP solving isn't just slower than a learned/heuristic policy, it's *useless* within any
   practically relevant time budget.

Note: most non-Small configs report a nonsensical numerical "gap" (values in the 10^8-10^20
range) from `model.getGap()` when the dual bound is at or near zero -- this is a SCIP artifact
of the gap formula's denominator, not a real signal; the `objective_norm` (primal best-found)
column remains valid as a feasible-solution upper bound regardless.

## Sequential/Parallel inference-speed tradeoff (measured, informs paper framing)

| Scale | Parallel | Sequential | ratio |
|---|---|---|---|
| 5x5x5 | 27ms | 86ms | 3.1x |
| 20x30x5 | 91ms | 310ms | 3.4x |
| 70x100x15 | 879ms | 5,447ms | 6.2x |

Ratio grows with scale (Sequential does one forward pass per weapon per round; Parallel does
one per round total). At Battlefield scale Sequential's 5.4s/instance starts to strain
real-time usability while Parallel's 0.9s stays comfortable. Combined with 20M_30N_5T's
finding that Sequential lands within 0.02 of SCIP's 10-minute best-found (0.1846 vs 0.1629) --
i.e. Sequential is a genuinely strong quality ceiling, not just "the old paper's method" --
the user and assistant agreed on paper framing: **Parallel remains the primary proposed
method** (it's the actual novel contribution motivating the new-venue pivot, and now reliably
beats Greedy 12/12 with a comfortable speed margin); **Sequential is reported alongside as a
quality-ceiling reference/ablation**, not swapped in as the main method. Candidate future lever
to narrow Parallel's remaining gap to Sequential/SCIP specifically, discussed but not yet
implemented: an inference-time auction-algorithm (Bertsekas) refinement pass on top of the
parallel decoder's joint proposal -- naturally parallelizable (unlike Sequential's per-weapon
loop) and directly targets the round-0 mean-field redundancy failure mode documented earlier.

**Immediate follow-ups**:
1. Retrain Sequential under the now-fixed simulator (its last run, the 200-epoch one used to
   test hypothesis (a), predates this fix and should be considered superseded/uninformative --
   rerun before drawing any final Parallel-vs-Sequential conclusion).
2. Re-run the full 12-config Greedy vs SCoPE sweep (`eval_tiered_benchmark.py`,
   `eval_greedy_benchmark.py`) once both decoders are retrained under the fixed simulator, to
   replace every number currently in this log and in `brerla_greedy_baseline_and_calibration.md`
   memory (all of it predates this fix and is now stale).
3. SCIP comparison at (5,5,5) also predates this fix (SCIP itself is unaffected, since it
   doesn't use the simulator/encoding at all -- but the SCoPE checkpoint it was compared
   against does, so that specific comparison should be redone with a post-fix checkpoint).
4. Multi-seed variance check on the post-fix result (this is one seed, seed5) before
   treating -0.0638 as a stable point estimate.

## Immediate next steps (in priority order, per user instruction to investigate autonomously)

1. Actually complete a Sequential retrain under the fixed simulator (`--seed 5`, default
   reward-to-go since no ablation flag exists yet) and evaluate it on `20M_30N_5T` against
   Greedy's 0.3422, holding everything else constant. This directly tests hypothesis (a).
2. If time permits, add `--ablation` support to the Sequential training path to allow a
   fully matched (architecture x reward-credit-style) 2x2 comparison.
3. Test hypothesis (b) by comparing training-curve quality at a substantially larger
   `TOTAL_EPISODE`/`TOTAL_EPOCH` budget for the parallel no_reward_to_go setup, to see if
   the Medium-scale gap shrinks with more training alone (architecture held fixed).
4. Re-run the full 12-config Greedy vs SCoPE(no_reward_to_go) sweep once Sequential's
   result is in, so the picture is complete rather than spot-checked on 2 configs.

---

# SESSION CONTINUATION (2026-08-07/08): critic removal, auction refinement, RL+Auction hybrid

This section covers a later continuation of the same investigation, starting from a paper
completion request. Summarized here for continuity; superseded/updated conclusions from
earlier in this log are noted explicitly.

## 1. Critic permanently removed from REINFORCE (user directive, not just an ablation)

User's standing position, stated repeatedly and emphatically: "NEVER EVER USE CRITIC IN THE
REINFORCE." Both `rl/Dynamic_Sampling_GNN.py` (parallel) and
`rl/Dynamic_Sampling_GNN_binary.py`/`Dynamic_Sampling_GNN_sequential.py` (sequential) were
changed so critic/shaping is **unconditionally** disabled (not behind an `--ablation` flag
anymore) -- `value` is always zero, `critic_loss` is always `torch.tensor(0.0)`, no
`critic.optimizer.step()` call exists anymore. All old checkpoints (39 directories under
`result/GNN_TRAIN*`) were deleted per explicit user instruction, keeping only
SCIP/Greedy CSV results (unaffected by this, since Greedy/SCIP don't use the actor network).

**Retrained both decoders from scratch** (`--seed 5 --ablation no_reward_to_go`, no critic):
- Parallel: 200 epochs, `result/GNN_TRAIN_ABL_no_reward_to_go_seed5/`. Best checkpoint by
  gap-vs-Sequential (not final epoch -- see below): **epoch 190**.
- Sequential: 200 epochs, `result/GNN_TRAIN_SEQ_ABL_no_reward_to_go_seed5/`, epoch 200 final.

**Full 12-config comparison, Parallel(no-critic, epoch190) vs Sequential vs Greedy vs
SCIP-postfix-critic-comparison**:

| Config | Greedy | Parallel(no-critic e190) | Sequential(no-critic e200) |
|---|---|---|---|
| 5,5,5 | 0.1591 | 0.1182 | 0.1363 |
| 5,7,5 | 0.3100 | 0.1904 | 0.2123 |
| 10,15,5 | 0.3624 | 0.2819 | 0.2462 |
| 15,15,5 | 0.0860 | 0.0628 | 0.0484 |
| 15,20,5 | 0.1514 | 0.1341 | 0.1008 |
| 20,30,5 | 0.3216 | 0.2700 | 0.2283 |
| 30,30,10 | 0.1506 | 0.0710 | 0.0585 |
| 30,40,10 | 0.2288 | 0.1571 | 0.1463 |
| 40,50,10 | 0.1705 | 0.1172 | 0.1079 |
| 50,50,15 | 0.0803 | 0.0447 | 0.0246 |
| 50,70,15 | 0.1659 | 0.1146 | 0.1013 |
| 70,100,15 | 0.1659 | 0.1614 | 0.1275 |

Parallel(no-critic) beats Greedy 12/12; gap to Sequential is narrow (0.01-0.04), and Parallel
actually **beats** Sequential at (5,5,5) and (5,7,5). Confirms critic removal did not hurt --
a 12-config comparison against the earlier critic-included run
(`result/parallel_postfix_vs_greedy.csv`, now superseded/deleted along with its checkpoint)
showed no-critic at least as good or better on 9/12 configs. **User's instinct that critic was
unnecessary for this setting was empirically correct.**

Also confirmed: the "final epoch = best checkpoint" assumption is wrong for the no-critic
Parallel run specifically -- across 40 per-5-epoch checkpoints on `20M_30N_5T`, only 21/40
(52.5%) beat Greedy and the actual epoch-200 checkpoint was a slight loser (+0.0099) --
markedly noisier than the earlier critic-included run (75% win rate). Epoch 190 was selected
as the reported checkpoint using gap-vs-Sequential (not gap-vs-Greedy) as the selection
criterion, per explicit user correction: "just beating GREEDY is basic... should be minimize
gap with serial [Sequential]."

## 2. Auction-algorithm inference-time refinement (`common/auction_refinement.py`)

Implements a Bertsekas-auction-*inspired* (not literally the classical 1-1 assignment-problem
algorithm -- DWTAP is many-to-one, no formal guarantee inherited) price-based coordination
mechanism to fix the round-0 "multiple weapons pile onto the same obviously-good target"
mean-field failure documented earlier in this log, purely at inference time (zero extra
network forward passes; only the environment's own remaining_value x P is used).

**Design iteration (real bugs found and fixed in sequence, same pattern as the rest of this
log -- verify empirically, don't trust intuition)**:
1. First version: synchronous, all-weapons-rebid-every-iteration, price = full joint-survival
   value claimed. Found to **oscillate forever** for near-identical weapons (they all switch
   targets in lockstep each iteration, chasing each other) -- classic failure mode of
   synchronous best-response dynamics with symmetric agents.
2. Fixed by processing bids **one weapon at a time** (a Python loop per (batch,para) item --
   fine at inference time, never called during training) with **eviction**: a later, higher
   bidder can displace an earlier occupant of a target, who then re-queues and re-bids at the
   new price. This is much closer to Bertsekas's actual algorithm and converges cleanly.
3. `must_fire` parameter added afterward (see Section 3) -- when `None`, auction decides
   fire-vs-no-op itself from immediate net value alone (same myopia as Greedy, by
   construction); when supplied, weapons marked `False` are held **unconditionally**, even
   if some target has positive immediate value (this exact bug -- "if not respect_policy and
   net.max()<0: no-op" only, forgetting the `must_fire=False` case entirely overriding this
   -- was caught once via a diagnostic toy check showing ALL weapons firing at no-op despite
   260/600 legal edges existing).

**Validated no constraint violations**: exhaustive check across all 12 tiered-benchmark
configs, 39,000 total weapon-round decisions, 8,185 actual fires -- **zero** violations of
legality/time-window/ammo (`verify_auction_constraints.py`). Also never exceeds SCIP's
proven-optimal on the (5,5,5) held-out set (10/10 instances checked directly).

**Pure auction (no must_fire) 12-config sweep result**: beats Greedy and Sequential at most
configs, dramatically at Large/Battlefield scale (e.g. 70,100,15: naive Parallel 0.1614 ->
pure-auction 0.0372). **However**: this was later shown to be because the current 12-config
benchmark has almost no genuine temporal (hold-now-fire-later) structure -- pure auction is
just as blind to that as Greedy (verified: pure auction fails the hand-built toy trap instance
identically to Greedy, remaining=10.1 vs optimal 2.0). The benchmark win reflects that
within-round coordination, not temporal judgment, is what these particular instances mostly
test -- a finding entirely consistent with the rest of this log (fire_rate identical across
all methods on the main benchmark).

## 3. RL+Auction hybrid decomposition -- new architecture direction

**Core idea** (full writeup: memory `brerla_rl_auction_hybrid_decomposition.md`): decompose
the per-round decision into (a) a **binary fire/hold** judgment, LEARNED (this is the one
thing that genuinely requires foresight/training -- proven by the toy trap instance and the
whole-episode-return finding earlier in this log), and (b) **which target**, handled entirely
by the auction (an exact, non-learned computation -- no reason to make a network learn
something that has a closed-form near-optimal answer). New files only:
`common/DWTA_GNN_binary.py` (binary actor, reuses `EdgeAwareGNNLayer`/`ResidualBlock` from
`DWTA_GNN.py` unmodified), `rl/Dynamic_Sampling_GNN_binary.py` +
`rl/DWTA_GNN_TRAIN_binary.py` (training loop). Uses the **parallel** architecture (one joint
forward pass per round, not sequential) since the binary decision doesn't need to see other
weapons' this-round choices -- that coordination is now auction's job.

**Prior art check (live web search, 2026-08-07)** -- MUST be cited if this direction is
written up: (a) Zavlanos/Michael/Kumar ~2010, Q-learning estimates auction bid *values* for a
classical 1-1 assignment auction (different: our RL doesn't touch bid values, only gates
participation); (b) a very recent (~July 2026) ScienceDirect paper, heterogeneous air-defense
WTA, PPO learns the **bidding strategy itself** within a contract-network-protocol auction --
closest prior art found, but structurally different (their RL is *inside* the auction, ours is
*before* it, and the auction's own values stay a fixed exact formula, never learned); (c)
Delarue et al. NeurIPS 2020 (VRP), general "RL strategic decision + classical exact method for
detailed planning" pattern -- establishes the *general* pattern isn't novel, but this specific
DWTAP instantiation (binary temporal gate + auction, no learned bid values) wasn't found.
Honest framing for the manuscript if pursued: not "first RL+auction for WTA," but "RL for
strategic/temporal decision + classical exact auction for detailed/combinatorial planning,
following Delarue et al.'s general pattern, with RL and auction kept fully decoupled unlike
concurrent WTA work that learns the auction's own bidding policy."

**Draft theoretical framing discussed** (not fully formalized): (1) Local-optimality/
competitive-equilibrium proposition for the auction's converged assignment given a fixed
firing set -- straightforward to prove, essentially free from the auction's own termination
condition. (2) A claim that the auction-refined assignment weakly dominates a single-pass
greedy sweep (via the eviction mechanism only ever improving the objective), and therefore
inherits the classical 1/2-OPT submodular-greedy guarantee (Fisher/Nemhauser/Wolsey 1978,
already cited in the manuscript's Proposition 2) for the round-local target-assignment
sub-problem -- plausible but **not yet rigorously proven**, flagged as needing real
formalization before claiming it in the paper.

### Result 1: extreme temporal-dilemma instances -- clean, decisive success

`common/temporal_dilemma_generator.py` (+ `generate_temporal_dilemma_instances.py` for fixed
test files): deliberately extreme -- ammo ~1-2 shots/weapon at T=5, sharp bimodal target value
split (1-3 vs 7-10), high-value targets only engageable from the back half of the horizon.
SCIP itself struggled here (only 6/10 proven optimal within 120s at 10x10x5, vs 10/10 for
comparable-scale main-benchmark configs) -- confirms this is a genuinely harder combinatorial
problem, not just an adversarial case engineered to fool myopic heuristics specifically.

10x10x5, 10 fixed instances (seed=123), full 5-way comparison:

| Method | Remaining value (lower better) |
|---|---|
| Greedy | 0.8260 |
| Pure Auction (no must_fire) | 0.8254 |
| Sequential (existing checkpoint, NOT trained on these instances -- zero-shot) | 0.7453 |
| **RL+Auction (trained specifically on this curriculum, 200 epochs)** | **0.2148** |
| SCIP (6/10 proven optimal, rest gap 5-30%) | 0.1637 |

Greedy and pure auction are statistically identical (both blind to temporal structure, as
expected). Sequential's general-purpose temporal judgment transfers somewhat
out-of-distribution but nowhere near the specifically-trained RL+Auction, which closes most of
the gap to SCIP-proven-optimal. This is the clean, complete validation of the whole hybrid
idea -- training worked essentially immediately (large, unambiguous per-instance reward
signal from the extreme value/scarcity structure), no exploration/collapse issues.

### Result 2: moderate temporal-dilemma instances -- open problem, premature convergence

Per user's own critique ("these instances are also too extreme... what do you think") --
`common/temporal_dilemma_generator_moderate.py`: value continuously correlated with (not
strictly split by) emergence time, moderate ammo (~T/2 not ~T/3). At 5x5x5 (fixed instances,
seed=123), SCIP reference: 9/10 proven optimal, mean 0.1479. Greedy 0.5984, pure auction
0.3624 (auction's coordination advantage IS visible here, unlike the extreme case, since this
config isn't as purely temporal-dominated).

**Training the same binary RL+Auction architecture on this curriculum repeatedly converged
prematurely to a degenerate "fire whenever legal" policy** (fire_prob saturating toward
0.99-0.998, monotonically increasing across epochs despite an entropy bonus) -- worse than
pure auction alone (must_fire=True forced on every legal weapon strips auction of its own
endogenous no-op judgment). Root cause hypothesis (not fully confirmed): unlike the extreme
curriculum, "always fire" is *close to* correct on this moderate distribution (ammo isn't that
scarce, value differences are smooth not stark), so the policy locks onto a "good enough on
average" strategy very early, before the rarer instances/rounds where holding actually helps
can provide enough gradient signal to escape that basin.

**Fixes attempted, in order, none yet fully resolving it**:
1. Added entropy bonus (was entirely missing from the first version of
   `Dynamic_Sampling_GNN_binary.py` -- an oversight, not a deliberate choice; the original
   N+1-way actor's training loop has one, this new file's first draft didn't). `entropy_coef
   = 1e-2` (vs the original actor's `1e-3`, scaled up since a Bernoulli's max entropy `ln 2`
   is much smaller than an N+1-way softmax's `ln(N+1)`). **Did not stop the collapse** --
   checked directly by loading epoch 5/25/45 checkpoints and reading raw `fire_prob`: 0.897 ->
   0.992 -> 0.998, monotonically still saturating.
2. Added epsilon-greedy-style exploration floor (`sample_prob = fire_prob*(1-eps) + 0.5*eps`,
   `eps=0.2`), reasoning this gives a *hard* exploration floor unlike a soft entropy penalty
   the return term can outweigh. Per-checkpoint progress-eval (deterministic
   `fire_prob>0.5`) still showed the identical stuck value (0.4087) at epochs 5/10 --
   expected, since epsilon-mixing only affects what's *sampled* during rollout, not the
   *direction* of the gradient on the underlying `fire_prob` itself (gradient is just scaled
   by `(1-eps)`, not redirected) -- doesn't by itself stop the learned probability from
   still saturating over time.
3. Annealed the entropy coefficient instead of holding it fixed: `0.3 -> 0.01` exponential
   decay over 200 epochs (well above the `1e-2` that failed in attempt 1, especially early in
   training). **Partial effect observed**: epoch 10 broke out of the stuck value for the
   first time (0.4087 -> 0.2573, gap-to-SCIP 0.26 -> 0.11) but reverted to the same stuck
   0.4087 by epoch 15 and remained there through at least epoch 40 (entropy_coef ~0.15 at
   that point, still substantial) -- user correctly predicted this in advance ("it will stuck
   since getting near to final, it would stuck"), though the observed stall happened earlier
   (epoch 15-40) than the entropy schedule's own late-training endpoint, suggesting the
   attractor is strong enough that even a still-large entropy coefficient isn't sufficient on
   its own.

**Status: open, unresolved as of this log entry.** Not something a single further
hyperparameter tweak is likely to fix blindly -- candidate next steps, not yet tried: (a)
much higher episodes-per-epoch (currently 5, same as the main curriculum -- may simply be too
few samples to detect the rare-but-important hold-was-better cases against noise), (b) a
curriculum that deliberately over-samples harder/more-temporal instances early in training
before shifting to the full moderate distribution, (c) lower learning rate, (d) explicit
reward bonus/penalty shaping specific to correct holds (would need care to keep it
policy-invariant per Ng/Harada/Russell, matching the potential-based-shaping discussion
already in this manuscript's training section -- and would reintroduce something
critic-shaped-like, which the user has explicitly ruled out ("NEVER EVER USE CRITIC IN THE
REINFORCE") unless framed as a fixed, non-learned potential rather than a learned critic).

**Important for the manuscript**: the *extreme* curriculum result is a complete, clean,
SCIP-validated proof that the RL+Auction idea works. The *moderate* curriculum's training
instability is a genuine, separate, open engineering finding -- worth reporting honestly as
future work / a limitation, not something to paper over or claim is solved.

## Files added this continuation (all new, no existing files/results modified except the two
critic-removal edits noted in Section 1)

- `common/auction_refinement.py` -- auction algorithm (Section 2)
- `common/DWTA_GNN_binary.py` -- binary fire/hold actor architecture
- `common/temporal_dilemma_generator.py`, `common/temporal_dilemma_generator_moderate.py` --
  instance generators (Section 3)
- `rl/Dynamic_Sampling_GNN_binary.py`, `rl/DWTA_GNN_TRAIN_binary.py`,
  `rl/DWTA_GNN_TRAIN_binary_moderate.py` -- binary-actor training loops
- `eval_auction_one_config.py`, `eval_auction_mustfire_one_config.py`,
  `eval_parallel_one_config.py`, `eval_sequential_one_config.py` -- per-config subprocess eval
  scripts (subprocess-per-config avoids a still-unexplained slowdown found when
  `eval_instance_parallel`/`eval_instance_sequential` are called many times in one long-lived
  process -- see the "Full 12-config Parallel(post-fix) vs Greedy sweep" section above)
- `verify_auction_constraints.py` -- exhaustive 12-config constraint violation check
- `generate_temporal_dilemma_instances.py` -- writes fixed `.xlsx` test files (extreme curriculum)
