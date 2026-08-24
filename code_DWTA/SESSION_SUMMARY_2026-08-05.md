# SCoPE Paper — Session Summary (2026-08-05)

## What we did this session

### Table 5 / main results
- Removed the "Sequential" column from Table 5 (`tab:main_results`) — Sequential comparison lives only in the Efficiency section now, framed as "the ablation of parallel decoding," not mixed into the baseline comparison table.
- Decided (via "neutral" discussion): **greedy SCoPE is the base/reported model in Table 5**, not SCoPE+beam+rollout. Beam+rollout stays a separate, later "fixed decision-budget headroom" result (Efficiency section), consistent with Related Work's existing disclaimer that beam+rollout "is a tool we employ, not a claimed contribution."

### SCIP baseline — real bug fixes in `opt/SCIP.py`
Found and fixed two real correctness bugs before trusting any SCIP numbers:
1. **Reload-time constraint was never enforced.** Dead `k`/`l` variables existed but nothing gated firing on cooldown state (verified by derivation: the constraint math never actually restricted `x`). The *correct* constraint was sitting right there, commented out (`for delta in range(1, min(W[m]+1, T-t))...`). Re-enabled it; removed the dead `k`/`l` code.
2. **Ammo (`A`) and reload (`W`) were hardcoded constants** in the old demo script instead of read per-instance from the `AMM`/`PREP` columns — meant SCIP would've solved a different, easier problem than what SCoPE was evaluated on.

Also cleaned up `opt/SCIP.py` into a clean, side-effect-free importable module (removed the old demo block that ran at import time, removed unused imports/`MAX_T` param, fixed a Windows `UnicodeEncodeError` crash caused by emoji in `print()` statements — **user has since asked for a hard rule: no emoji anywhere, chat or code**).

Installed `pyscipopt` and `openpyxl` into the project venv (`after_review_ieee_access/venv`).

### New eval script: `eval_scip_benchmark.py`
- Runs SCIP on the same 10 held-out instances (rows 0–9) per config that `eval_tiered_benchmark.py` uses for SCoPE, with real per-instance AMM/PREP/TW/V/P.
- Time limit: **600s (10 min) per instance** — settled after discussion (see below). No artificial `limits/gap` early-stop; `presolving/maxrounds=10` kept, nothing else tuned.
- Writes per-instance results + a running summary (mean/std objective, mean gap, optimal/timelimit counts) to `result/scip_<config>_<limit>s.csv` and a progress log.

### (5,5,5) SCIP result
All 10/10 instances solved to **proven optimality** (gap = 0.00%), max solve time 52.3s (well under even the original 120s test limit). Mean normalized objective = **0.093 ± 0.028** (instance-to-instance std — SCIP has no seed axis). Filled into Table 5.

### Metric discussion — this took a while, ended with a real decision
- SCoPE's reported 0.177 ± .018 for (5,5,5): verified precisely how it's computed — mean/std **over the 5 seeds' own 10-instance averages** (seed-level variation), *not* pooled over all 50 (seed×instance) points. Confirmed this is standard/correct for RL papers and matches what the original IEEE Access reviewers asked for — not something to change.
- Comparing SCoPE (0.177) vs SCIP-optimal (0.093) as a **relative gap** ((0.177−0.093)/0.093) gives **90.3%**, which looked alarmingly bad for a config inside the training distribution.
- Traced this down: it's not a modeling failure so much as a **metric artifact** — DWTAP's objective (remaining value) can approach zero at the optimum, so relative-gap's denominator is small and the ratio is volatile. In absolute terms the picture is much more moderate: SCoPE destroys 82.3% of value vs SCIP's 90.7% — an **8.4-percentage-point** absolute gap.
- Researched how other WTA RL papers report this (read the full Gaudet, Drozd & Furfaro 2023 hypersonic-WTA paper, arXiv:2310.18509, already cited in Related Work). They report a **ratio of achieved (destroyed) value**, not remaining value — `% NLIP = 100 × RL_median / NLIP_median` — which sidesteps the near-zero-denominator problem by construction, and their NLIP benchmark also uses no fixed time budget (mean ~2s, up to 265s in some cases, one pathological instance ran 24hrs before being killed — confirmed 24hrs is not a sane budget to emulate).
- **Final decision**: report a single metric, `% Achieved = avg(1 − Obj_model) / avg(1 − Obj_best-found) × 100%` (Eq. `eq:achieved`), matching Gaudet et al.'s convention exactly. Dropped both the original relative-gap (`eq:gap`) and a separately-considered absolute-gap (`eq:abs_gap`) metric entirely — decided they were redundant with each other and less literature-standard than `% Achieved`. For (5,5,5): **90.7% Achieved**.
- Manuscript updated: added `eq:achieved` with justification text (citing `gaudet2023`), removed `eq:gap`/`eq:abs_gap` and all references to them, simplified Table 5 to one extra column (`% Achieved`) instead of two. Verified no dangling `\eqref` references remain.

### SCIP time budget for larger tiers
- Settled on **600s (10 min)** as the relaxed cap for all future SCIP runs (Medium/Large/Battlefield), reasoning: Gaudet's 24hr outlier is not sane to emulate, and "over 10 min isn't realistically usable in a battlefield decision context anyway." `eval_scip_benchmark.py`'s `TIME_LIMIT` constant updated to 600 (from the original 120 used for the already-complete (5,5,5) run — no rerun needed there, all instances converged well under 120s already).
- Noted explicitly: our DWTAP is structurally much harder than Gaudet's NLIP (which is *static*, no time dimension, capped at 20×12=240 variables and already ran out of memory beyond that). Our Large tier (30,30,10)=9,000 and Battlefield (70,100,15)=105,000 variable-scale is 37×–400× larger — so we should **expect** SCIP to not reach proven optimality within 600s at Large/Battlefield scale, and need to report residual gap honestly when that happens, not treat non-convergence as a bug.

## Open / in-progress when session paused

- **Convergence graph (gap vs. time) for Large and Battlefield tiers** — was about to check `pyscipopt`'s log-capture API (likely `Model.setLogfile(...)` or stdout capture) to parse SCIP's live B&B log (`time | dualbound | primalbound | gap` rows) into a time-series, then plot for one representative hard config per tier. **Not yet started** — need to pick exact configs (candidates discussed: Large → (40,50,10), Battlefield → (70,100,15), 1 instance each) and confirm scope before running (up to 600s per instance).

## TODO (not yet done)

1. **Gap-vs-time convergence graph** for Large/Battlefield (see above) — figure out log capture, pick configs, run, plot.
2. **Run SCIP (600s cap) on the remaining 11 configs**: (5,7,5), (10,15,5), (15,15,5), (15,20,5), (20,30,5), (30,30,10), (30,40,10), (40,50,10), (50,50,15), (50,70,15), (70,100,15) — fill in Table 5's SCIP + % Achieved columns for all rows.
3. **Greedy heuristic baseline** (max marginal-return dispatch) — need to find or build an implementation. Was going to use this to diagnose whether the large relative-gap we saw is a DWTAP-metric property (shared by all methods) or SCoPE-specific — this diagnostic got interrupted (agent spawn was rejected) and hasn't been redone.
4. **GA baseline** (genetic algorithm, per own prior work `lee2003genetic`) — not started.
5. **Beam search + truncated rollout for the parallel decoder** — only exists for the old sequential decoder; needs adapting to score joint (not per-edge) candidates across the M simultaneous per-weapon choices (flagged as TODO in `sec:method`-D). This is now higher-priority than before, since it's needed to eventually report the "SCoPE+search" fixed-budget headroom result in the Efficiency section.
6. **RL4CO AM/POMO/MatNet baselines** — not started; deliberately lowest priority (requires building 3 training pipelines from scratch vs. call-and-done SCIP/GA/Greedy).
7. **Abstract rewrite** — still describes the old sequential-decoder era, has a stale "6–17% gap" claim that doesn't match anything we've measured. Rewrite last, once real numbers are in.
8. **Conclusion rewrite** — still pre-revision (old CPLEX/scale claims); flagged TODO in manuscript.
9. **Statements and Declarations** (Competing Interests, Funding, Data Availability) — currently placeholder TODOs.
10. Two reference entries (Gaudet et al., Hu et al.) need "et al." expanded to full author lists in the bib.
11. Verify `yoontransformer` citation's description matches the peer-reviewed journal version, not the old Master's-thesis version (TODO already in manuscript text).
12. Real-time decision-budget citation still needed to justify whatever fixed budget eventually anchors the Efficiency section's "SCoPE+search vs. Sequential, same budget" headroom comparison — this is a *separate* number from the 600s SCIP-relaxed cap above, still unresolved.
13. Table 1 / text layout overlap (an R5 reviewer comment from the original submission) — needs a PDF render pass to check, not yet done.

## Standing preferences (for continuity)
- Respond in Korean.
- Never use emoji — in chat or in any code/output.
- Confirm before nontrivial code changes, reruns, or scope changes — don't jump ahead. Fast/cheap eval-only reruns (seconds to minutes) are generally fine to just do; anything that could take a while or commits to a framing decision should be checked first.
