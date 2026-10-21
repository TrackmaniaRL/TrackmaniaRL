> Update 2026-10-08 07:22 UTC — repair search reorganized around decisions; calibration trajectory COMPLETE.
>
> Central entry point: [SD_SAC_REPAIR_DECISIONS.md](../../SD_SAC_REPAIR_DECISIONS.md), with closed hypotheses, fixed budgets and decision/stop rules. Existing scheduler instructions shortened to current state and durable constraints; no more unrelated audits or grids at each wakeup. One checkpoint load now collects complementary trajectory metrics and reason groups.
>
> Fixed weight0/.01, seeds17/29,48steps each:192 copied steps,212.36s; all four endpoints EXACTLY reproduce prior calibration. Both weight0.01 seeds first fail held nonterminal MAE guard at sampled step8. Final held MAE0.113856/0.115229, MSE0.095405/0.090436 versus initial0.071385/0.033756; fit nonterminal MSE also worsens. Held clipped loss worsens alongside MAE. Terminal MAE improves to4.550828/4.597501. This supports a finite calibration tradeoff, not merely metric disagreement or ordinary train-only overfit; no optimizer/clipping bug is established. Close further generic weighting/initial-optimizer tweaks; no early promotion.
>
> Original actual CP SHA/model/target/all optimizers unchanged; pins/STOP valid, both captured exact processes closed. Ruff/mypy PASS. Evidence calibration-trajectory.json plus receipts/verification and repair-state.json under evidence/sd-projlr08-final. No runtime learner/controller/pilot; full SD BLOCKED.
>
> Concrete next hypothesis and causal/control/scope contract prepared in [SD_SAC_OBSERVABILITY_PROPOSAL.md](../../SD_SAC_OBSERVABILITY_PROPOSAL.md). SD-SAC-only remaining-clock adapter requires explicit widening of the current no-observation/model/architecture-change scope. No dependent experiment before that decision; keep heartbeat ACTIVE5min and quiet on unchanged pending state.

> Update 2026-10-08 07:14 UTC — critic Adam-direction audit COMPLETE; zero optimizer/model/runtime steps.
>
> Same calibration episode split/frozen targets as the prior gradient audit. Non-mutating saved/fresh/zero-first-moment Adam algebra for empirical-frequency baseline and baseline plus previously tested normalized terminal weight0.01. Reason-specific additions weighted by their fit-terminal prevalence sum to the total terminal addition (invariant PASS). Eight independent float64 scalar Adam algebra controls PASS, including bias correction, weight decay and AMSGrad; no synthetic optimizer step was executed. Frozen SD-SAC _optimize has no gradient clipping to reproduce here.
>
> All15 initial analytic tangent directions have negative derivatives for held nonterminal clipped loss/MSE/MAE and held terminal MSE/MAE. Saved baseline nonterminal-MAE derivative -0.00031489; with terminal0.01 -0.00070291; terminal-MAE -0.00068902 -> -0.01854663. Fresh/zero-first-moment directions also improve locally. Thus neither initial raw gradient conflict nor initial Adam momentum reversal explains the observed later calibration drift; no optimizer reset or runtime change is justified. These are infinitesimal directional predictions, not executed finite steps or driving evidence.
>
> Evidence critic-adam-directions.json plus process receipt/verification under evidence/sd-projlr08-final. CPU IDLE/single-thread compute25.56s; both captured exact PID/ctime identities closed. Actual checkpoint SHA, model/target/all optimizer digests unchanged; frozen pins/STOP valid. Ruff/mypy PASS. Full SD BLOCKED, no runtime pilot.
>
> Next justified bounded offline diagnostic: trajectory of the existing calibration baseline versus terminal weight0.01, saved Adam/projected disposable q1/q2, frozen actor/targets/alpha, identical fit/held episodes and random draws. Seeds17/29,48steps each,192total, fixed300s compute cap; trace0/1/4/8/16/32/48 fit/held nonterminal clipped loss/MSE/MAE and terminal errors. Require final endpoints reproduce projected-calibration.json within1e-6. Purpose is locating onset and objective/MAE disagreement, not promoting an early checkpoint. No new coefficient, reset, runtime candidate, architecture/shared model/reward/GNN/replay change, timeout relabeling or guard waiver. ACTIVE5min.

> Update 2026-10-08 07:09 UTC — critic parameter-gradient audit COMPLETE; zero optimizer/runtime updates.
>
> Exact calibration episode split:105/25 fit/held terminals,1024/1024 nonterminals; reasons56/13 no_progress,30/7 slow_progress,19/5 time_limit. Frozen actor/target critics/alpha/discounts; endpoint reasons label groups only, never model inputs. Raw MSE and clipped Bellman gradients, spherical tangent projection and per-parameter comparisons are recorded.
>
> Terminal-MSE tangent cosine with fit/held nonterminal clipped loss is +0.39168/+0.66611. Held alignment by reason: +0.49030 no_progress, +0.71055 slow_progress, +0.17171 time_limit; globally positive despite72/20/62 negative individual parameter dots. Terminal-MSE tangent norm22.71267 versus nonterminal-MSE0.21923, so normalized terminal weighting can materially alter gradient scale. Terminal fit/held agreement0.72189 overall,0.26055 for time_limit (only5 held episodes). Initial global conflict does not explain the later finite-step drift; these are not Adam steps, critic truth or driving evidence.
>
> Evidence critic-gradient-conflict.json plus process receipt/verification under evidence/sd-projlr08-final. CPU IDLE/single-thread computation19.22s; both captured exact identities closed. Actual CP SHA, model/target/all optimizer digests unchanged; frozen pins/STOP valid. Ruff/mypy PASS. No runtime pilot; full SD BLOCKED.
>
> Next zero-update audit: saved critic-Adam momentum/preconditioning directions for the empirical baseline and previously tested terminal weight0.01, with reason-specific decomposition and held nonterminal derivatives. Numerical Adam controls, no optimizer/model steps; no blind grid, architecture/shared model/reward/GNN/replay changes, automatic timeout relabeling or guard waiver. ACTIVE5min.

> Update 2026-10-08 07:02 UTC — clock/timeout semantics control COMPLETE; zero optimizer/runtime updates.
>
> Frozen reward and environment intentionally implement maximum_race_time_s as a penalized terminal; collection/telemetry interruptions are separate truncations. Three existing focused tests PASS: penalized race deadline, terminal zero bootstrap, truncation preserving bootstrap. SD-SAC target construction respects the replay discount. Automatic time-limit-to-truncation relabeling would change the tested task objective and is rejected.
>
> Actual frozen V5 feature transform on two synthetic identical-motion histories with clock pairs0/149900ms then50/149950ms produces exactly identical context/physics/recovery/track tensors (all max errors0). This establishes clock-shift information loss for this control, not an exact alias observed in replay and not failure causality. Absolute race time is used by termination but is not represented explicitly by this pipeline. No shared feature/model/reward/GNN/replay changes are authorized by this finding alone.
>
> Evidence clock-semantics.json and clock-semantics-verification.json under evidence/sd-projlr08-final; source audit_sd_sac_clock_semantics.py, Ruff/mypy PASS. Actual CP SHA unchanged, immutable pins/STOP valid, no remaining process. Checkpoint models/optimizers were not loaded or updated. Short foreground control ended before capture; no exact PID/ctime claim. Pytest emitted only the documented unknown cache_dir option warning with cacheprovider disabled;3passed.
>
> Next scoped zero-update diagnostic: critic parameter-gradient alignment by terminal reason (including time_limit) versus ordinary nonterminal Bellman loss, to understand why earlier terminal calibration worsened general estimates. Inspect existing harness before loading; no optimizer updates, runtime promotion, architecture changes or blind coefficient grid. ACTIVE5min; full SD BLOCKED.

> Update 2026-10-08 06:57 UTC — terminal-history observability audit COMPLETE; zero optimizer/runtime updates.
>
> Frozen coordinator intentionally retains only behavior entropy in online replay info. First diagnostic found all terminal-reason/history metadata absent; its output is preserved. Extended diagnostic joins all130 terminal rows to train/episode events by exact episode name and asserts episode steps=row step+1. Previous transition links/episode/step continuity also PASS, but causal pre-action progress counters are absent from replay, so no counter-conditioned fit is claimed.
>
> Matched endpoint reasons:69 no_progress,37 slow_progress,24 time_limit. Both-critic terminal-reward MAE: no_progress4.2454/4.2379, slow_progress4.6690/4.6947, time_limit10.5076/10.5670. Time-limit rows contribute35.18% of total absolute meanQ terminal error despite18.46% of terminal rows. No-progress endpoints all have steps_since=100; slow-progress window predicates PASS. Endpoint metadata is post-action and cannot be used as current-observation input.
>
> Frozen V5 observation shapes/context/recovery layouts and V2 motion construction omit explicit reward no-progress age, reward progress-window aggregate and absolute race time. They include frame advance and decaying incident memory, which are different signals. This omission does not establish non-identifiability from other features, a causal explanation of failed driving, or permission to alter shared model/reward/GNN/replay.
>
> Evidence terminal-history.json, terminal-history-events.json and terminal-history-verification.json preserved in evidence/sd-projlr08-final. Actual CP SHA/fingerprint/model/all-optimizer invariants PASS, immutable pins/STOP valid, no offline process remains. Short foreground runs ended before capture; no exact PID/ctime claim. Source audit_sd_sac_terminal_history.py, Ruff/mypy PASS. No new pilot; full SD BLOCKED.
>
> Next: inspect time-limit termination versus truncation semantics and clock observability in the intended SD-SAC objective before any finite-step or runtime change. Do not automatically relabel time limits or add shared features. ACTIVE5min.

> Update 2026-10-08 06:50 UTC — terminal observation alias / neighbor audit COMPLETE; zero updates.
>
> Exact decoded current-observation hashes cover all145424 current and145466 predecessor replay rows. No terminal/nonterminal exact aliases in either snapshot (130/181 terminal rows). Nearest-neighbor audit uses fixed4096 random nonterminals plus all immediate terminal predecessors, giving4224/4269 candidates. Episode/step/action/reward/termination metadata and per-branch raw RMS differences are preserved. Current median branch-scaled distance .20932 for any neighbor, .25456 across episodes, .65917 for the same action across episodes; predecessor .17414/.21311/.50606. These are different populations with separately scaled distances, not paired causal evidence or a classifier.
>
> Similar states exist, including near-adjacent episode steps, but no exact observation conflict or incorrect terminal label is established. Current observations precede actions; termination can depend on next state and progress history. No architecture, label, replay or runtime change is justified by proximity alone. Existing return audit remains sufficient; do not repeat it.
>
> Both actual CP SHA/fingerprint/model/all-optimizer invariants PASS; immutable pins valid, no new STOP, no remaining offline process. Short foreground audits completed before capture; no exact PID/ctime claim. Runtime9.36/11.30s, CPU IDLE single thread, optimizer0/controller false. Source audit_sd_sac_terminal_alias.py and current/predecessor JSON plus verifications saved under evidence/sd-projlr08-final; Ruff/mypy PASS.
>
> Next: inspect terminal-reason / progress-history observability using existing replay metadata and observation schema before another critic optimization probe. No further actor-loss grid or guard waiver. Full SD BLOCKED; ACTIVE5min, no new pilot.

> Update 2026-10-08 06:43 UTC — critic feature/head geometry audit COMPLETE; no optimizer/runtime updates.
>
> Current and predecessor actual checkpoint SHA/fingerprint/model/all-optimizer invariants PASS; immutable pins/STOP valid. Linear-head reconstruction and Cauchy output bounds PASS on all sampled rows/actions. All terminal rewards lie inside the observed feature-norm head envelopes. No output-range impossibility or architecture fix is established. Short foreground audits completed before identity capture; no exactID receipt claimed. First current attempt aborted on encoded replay actions and was corrected with the existing decode_tree helper, with zero updates throughout.
>
> Current nonterminal feature pair-cosine .88951/.89923, centered effective rank2.3634/1.4392, taken-action head alignment .93672/.93273. Predecessor: cosine .73066/.72783, rank1.3973/1.3959, alignment .77923/.77946. Low rank also exists in the predecessor, so do not claim new representation collapse. Replay populations differ; this is descriptive, not paired causal evidence. Head mean norms current8.4611/8.4927, predecessor8.5781/8.6004; terminal targets remain representable within observed norm envelopes.
>
> Evidence: evidence/sd-projlr08-final/critic-geometry-current.json and critic-geometry-predecessor.json plus verifications; source audit_sd_sac_critic_geometry.py, Ruff/mypy PASS. Original models and replay unchanged; no controller or pilot. Full SD remains BLOCKED.
>
> Next: zero-update terminal/nonterminal nearest-neighbor and exact decoded-observation alias audit, including episode/step/action/termination metadata, to assess separability before another critic optimization probe. Reuse existing return audit; no architecture/shared-model/GNN/replay changes. ACTIVE5min, no offline process remains.

> Update 2026-10-08 06:37 UTC — centered-score hypothesis rejected; SD-SAC remains BLOCKED.
>
> Zero-update gradient audit COMPLETE13.69s; five shift/scale/optimum/gradient invariants PASS. Fit-only score normalizer10.41447. Score-fit tangent cosine with score-held .91281, with forwardKL-held .01883, with forwardKL-held-starts .11963. Both forward/score local descent worsen35-vs56 start margin; this was not claimed as a start fix or Adam-step evidence.
>
> Matched FRESH-Adam projected copies COMPLETE256steps/58.08s, forwardKL versus normalized centered-score regression, LR.0009, seeds17/29,64steps, frozen meanQ/alpha.01/anchor/episodes/ordinary sampling unchanged. Score held error improves .258 to .178/.177, but forwardKL .137607/.146395, reverseKL .205478/.212842, greedy regret .004639/.005727 and both held-start agreements0 FAIL all guards. Forward controls reproduce earlier fresh-Adam probe within1e-6. Reject objective; no runtime promotion or new pilot.
>
> OriginalCP/model/Adam actual SHA f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53 unchanged, immutable pins/STOP valid, all three finite-probe exact identities closed. Gradient audit completed before identity capture and explicitly makes no exactID claim. Evidence: evidence/sd-projlr08-final/centered-score-gradients.json and centered-score.json plus verifications. Ruff/mypy PASS. CPU copies are not driving/critic-truth evidence.
>
> Existing recorded-return audit already covers behavior soft-return proxies: current nonterminal Bellman MAE .06185 versus proxy MAE3.869, terminal MAE5.529. Do not repeat it or call off-policy returns current-policy truth. Next zero-update diagnostic: critic feature/head geometry, normalized-feature variation/effective rank and head reachable intervals on current/predecessor checkpoints; inspect harness first. No architecture/shared-model/replay edits or further actor-loss probe without evidence. ACTIVE5min; no offline process remains.

> Update 2026-10-08 06:31 UTC — SD-SAC remains BLOCKED; no new driving pilot.
>
> Zero-update distribution-geometry audit: held target entropy3.60254 versus sharp fitted actor .885/.872 and inverse-scaled actor4.33995/4.33980. Compensation loses expectedQ .02369/.02358 versus target. The ideal inverse-temperature identity holds in float64 (max error7.1e-15); the fitted actor is not that ideal target. Initial float32 assertion failure is preserved in verification; no optimizer updates occurred. FrozenQ is not ground truth.
>
> Fixed two-stage control COMPLETE512copy steps/115.17s: sharp64 then original64 versus ordinary128, savedAdam/projection/LR.0009, identical episodes/ordinary sampling/anchor/frozenQ. Both seeds retain100% held-start agreement. Final general forwardKL .083571/.073379, reverseKL .120566/.113120, greedy regret .00056664/.00045306. Prior64 guards all PASS, but equal128 ordinary KL guards FAIL (ordinary forward .037866/.032359, reverse .056223/.046170). Reject runtime promotion; do not extend or select a favorable partial endpoint.64step controls reproduce prior results within1e-6.
>
> OriginalCP actual SHA f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53, saved model/Adam, immutable pins and STOP unchanged. All three captured exact probe identities closed. Evidence: evidence/sd-projlr08-final/distribution-geometry.json and two-stage-distillation.json plus verifications. No controller/runtime/replay changes; no driving claim.
>
> Next: inspect centered-logit regression history and zero-update objective geometry before another finite-step experiment. This may address relative-score calibration rather than another scalar-temperature search; no implementation selected yet. ACTIVE5min; all previous probes COMPLETE, never restart.

> Update 2026-10-08 06:22 UTC — SD-SAC remains BLOCKED. No new driving pilot.
>
> Zero-update paired metadata audit found all 62/71 worsened held rows are moving, nonterminal states; 41/51 changes are action56 to gas action35. This identifies an observed tradeoff, not true action values. Value-gap margin control failed (held starts100%/0%, regret .001411/.000725).
>
> Actor-only target temperature .001 (saved alpha/critics .01 unchanged) on ordinary fit rows gave both seeds100% held starts and lower greedy regret .000389/.000340, but forward/reverse KL failed. Fixed inverse-logit scaling .1 preserves that ranking but still fails KL: forward .583804/.580610, reverse1.631637/1.620626. Both rejected; no temperature grid or waived guards. Uncompensated endpoints reproduce the sharpening control within1e-6.
>
> All probes are CPU idle/single-thread disposable copies. Original checkpoint actual SHA f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53, saved model/Adam, immutable pins and STOP checks pass; all captured exact processes closed. Evidence: evidence/sd-projlr08-final/{tradeoff-metadata,value-gap-margin,actor-target-sharpening,temperature-compensation}.json and corresponding verifications. No runtime/controller/replay changes. These are fitting results, not driving or critic-truth evidence.
>
> Next: zero-update held-distribution error/target-margin audit before any further optimizer experiment. Existing probes COMPLETE; never restart/extend. Monitor ACTIVE5min.

## 2026-10-08 — fixed combination fits starts but fails greedy-regret protection

The standalone all-state margin probe completed 384 steps in87.47s: coefficient0.1 improves general held reverse-KL to0.09108/0.09931, but both seeds still miss all starts. The subsequent fixed0.25-start-loss plus0.1-margin combination fits all27 held-episode starts in both seeds. General held forward-KL0.06339/0.06397 and reverse-KL0.09081/0.09135 pass the matched-control limits, but greedy regret0.00082685/0.00091310 exceeds both ordinary and margin-only 20% drift guards. The combination is rejected for runtime promotion; start success alone is insufficient. This is frozen-Q diagnostic evidence, not driving or a validated critic.

All original checkpoint/model/Adam pins, actual SHA, STOP and exact owned-process closure checks pass. Evidence includes `greedy-margin.json`, `start-margin-combination.json`, their verification files, and `greedy-tradeoff-rows.json` in `evidence/sd-projlr08-final`. The paired row analysis identifies where greedy regret worsens without rereading or updating any model. Next inspect replay metadata for those rows and recurring action/context tradeoffs before another candidate. Full SD remains BLOCKED; no new pilot is launched.

## 2026-10-08 — repeated greedy-margin reversal; auxiliary probe pending

The matched trajectory audit completed 384 disposable actor steps in 202.56 seconds. Both ordinary-fit seeds reach 100% held-start agreement at step48 and lose it by step64, while general held reverse-KL improves from about0.20 to0.125. The 0.25 mixture has no correct start mode at the sampled trace points; start-only fits starts but fails general calibration. Ordinary endpoints reproduce prior metrics within1e-6. Original checkpoint/model/Adam, exact process closure, pins and STOP verified. This supports a soft-distribution-versus-greedy-mode fitting tradeoff on these saved data, not critic correctness or a driving fix.

A new bounded offline hypothesis is PENDING: an all-state greedy-margin hinge added to the unchanged ordinary forward-KL plus behavior entropy anchor. It uses only the original ordinary fit rows (no added start sampling), detached mean-Q top action and soft-target top-two gap capped at0.1 log-probability, ignores tied Q maxima, coefficients0/.01/.1, seeds17/29,64steps (384total), cap300s. Five numerical hinge invariants and Ruff/mypy pass. Require both seeds 100% held-start agreement AND general held forward-KL/reverse-KL/regret <=1.2xmatched ordinary baseline; preserve all results. No runtime change or new pilot authorized by partial CPU fit; Full remains BLOCKED.

## 2026-10-08 — intermediate actor-margin trajectories pending

A bounded CPU single-thread diagnostic is running on disposable saved-Adam actor copies: ordinary fitting, 0.25 start-loss mixture and start-only fitting, seeds17/29, 64 steps each (384 total), fixed 300-second copy-compute cap. It records intermediate held start margins against actions8/56 and general held calibration at steps0/1/4/8/16/32/48/64. The endpoints already have evidence; the new purpose is to locate the reversal that the zero-update direction audit cannot explain. No new mixture fraction, runtime setting, controller, critic update or persisted model is introduced. Result PENDING; no pilot promotion from partial fitting.

## 2026-10-08 — saved-Adam direction and start-margin audit

The zero-update CPU audit completed in 8.67 seconds. The analytic Adam displacement matches PyTorch on eight numerical controls (saved/fresh state, weight decay and AMSGrad). With ordinary fitting, saved Adam gives a negative local target35-versus8 margin derivative (-0.000995); fresh Adam and clearing only the first moment make it positive (+0.04402/+0.00740). However, all six combinations of ordinary/start-only fitting and saved/fresh/cleared-first-moment Adam worsen target35 versus action56 locally. Even start-only fitting does so initially, although the completed 64-step start-only probe eventually selects35. Lower held start loss therefore does not imply an immediate greedy-margin improvement. These are first-order directional measurements, not actual finite-step outcomes, and do not justify resetting Adam or launching a pilot.

Original checkpoint SHA, model and Adam remain unchanged; all captured exact identities are closed, episode rows match, pins and STOP pass. Evidence: `evidence/sd-projlr08-final/adam-directions.json` and verification. Full SD remains BLOCKED. Next inspect intermediate finite-step margins on matched ordinary, 0.25-mixture and start-only disposable copies to locate the reversal and interference; do not expand the mixture grid blindly.

## 2026-10-08 — zero-update actor gradient audit

The CPU single-thread audit completed in 7.05 seconds with zero optimizer steps. At the original saved actor, tangent gradient cosine is +0.44116 for ordinary fit versus fit starts, +0.42375 for fit starts versus general held rows, and approximately 1 for fit versus held starts. Nine parameter tensors nevertheless have negative tangent dot products (listed in verification); the aggregate result does not support a global initial gradient-conflict explanation. These local gradients are not saved-Adam steps and do not establish compatibility after finite updates or driving quality.

Episode rows match the completed mixture probe. Actual checkpoint SHA, original model and Adam are unchanged; immutable pins and STOP checks pass. The audit exited normally before identity capture; verification records session exit 0 and an empty matching-process inventory, without claiming captured exact identities. Evidence: `evidence/sd-projlr08-final/gradient-conflict.json` and its verification. No new pilot is justified; Full SD remains BLOCKED. Next inspect saved-Adam preconditioning and action-margin directional derivatives without optimizer steps before choosing further fitting experiments.

## 2026-10-08 — completed start-loss mixture diagnostic

The bounded offline probe completed all 384 disposable actor steps in 160.48 seconds. Fractions 0.01, 0.1 and 0.25 passed all general held forward-KL, reverse-KL and greedy-regret limits against their matched ordinary-fit baseline, but every fraction and both seeds retained 0% held-start agreement. No tested fraction qualifies. Lower start fitting error alone does not justify a runtime change or a new pilot.

Original checkpoint SHA, models and saved Adam remain unchanged; both exact owned processes are closed, all immutable pins are valid, and no new STOP is present. Evidence: `evidence/sd-projlr08-final/actor-start-mixture.json` and its verification. Full SD training remains BLOCKED. Next: inspect parameter-gradient conflict between start and ordinary fitting losses with a zero-update CPU diagnostic before selecting any further fitting experiment. Do not repeat or extend the completed grid.

## 8 October 2026 05:41 UTC - bounded start/general fitting tradeoff running

New explicit --fit-scope mixture compares fixed start-loss fractions.01/.1/.25,seeds17/29,64 projected disposable saved-Adam actor steps each,LR.0009,384 total,300s copy-compute cap. Loss=(1-fraction)*ordinary mixed forward-KL+behavior-anchor loss + fraction*fit-start forward-KL+behavior-anchor loss. Each component draws128 rows with replacement;ordinary mixed RNG is paired with the completed baseline,start RNG is separate. Frozen original mean-Q,alpha.01 and behavior references stay fixed;no critic/target update or persistent policy.

Same episode split and2048 mixed fit/512 general held rows,104 fit starts/27 held starts. Predeclared fit eligibility requires100% held-start agreement and every general-held forwardKL/reverseKL/greedy-regret metric<=1.2 times the matched seed ordinary-mixed result from actor-objectives.json,in both seeds. Report all outcomes;no promotion from only favorable start fit. These reused-replay/frozen-Q criteria do not establish driving quality or critic truth. No runtime replay sampling or model/reward/GNN changes.

Probe executes on CPUidle1thread:child65620ctime1791438024.3426428,wrapper74432ctime1791438024.3286068;exec94880,receipt BASE/sd-projlr08-audit/actor-start-mixture-processes.json,output actor-start-mixture.json. Ruff and boundary-scoped mypy pass;immutable pins/no changed STOP checked before execution. Results pending,original CP/models/Adam must remain unchanged. AutomationACTIVE5min/fullBLOCKED,no new pilot. Wait existing process,never duplicate or extend.

## 8 October 2026 05:36 UTC - start-fit capacity/exposure control complete

Dedicated fit to104 fit-episode starts completed128 projected saved-Adam actor steps in27.30s,seeds17/29,LR.0009,forward-KL,unchanged alpha.01 and behavior entropy anchor. Both copies reach100% mean-Q greedy agreement(action35,gas1) on all27 held-episode starts,zero greedy regret,forwardKL.010035/reverseKL.016348. The actor can fit this saved initial-state preference;short mixed-probe failure does not establish an architecture/capacity bottleneck. Held starts are similar observations already encountered during original training,not unseen real driving.

General held replay reverseKL increases from original.471377 to.716892/.716887 despite forwardKL declining.375595 to.329989/.329986 and greedy regret declining.008611 to.005998. Start-only fitting therefore loses general distribution calibration and is not a runtime candidate. No proof that frozen Q values are correct,no driving claim,no model/reward/GNN/replay change.

Original checkpoint/model/Adam unchanged,actualSHA f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53,all exact owned probe processes closed,immutable pins valid,no changed STOP. Evidence: evidence/sd-projlr08-final/actor-start-fit.json and actor-start-fit-verification.json. Explicit --fit-scope starts mode is bounded180s and128steps;default mixed behavior unchanged. Ruff and boundary-scoped mypy pass;real split disjointness and immutable-input assertions pass. No offline process remains,fullBLOCKED,automationACTIVE5min.

Next offline diagnostic may test a fixed mixed/start loss-mixture grid .01/.1/.25 on disposable projected actor/savedAdam copies,paired seeds64steps,against the completed ordinary mixed fit. Require held-start agreement and both-seed general-held forwardKL/reverseKL/greedy-regret no more than20% worse than the matched ordinary-mixed result. This tests a fitting tradeoff only;no automatic runtime replay reweighting or fresh pilot until scoped evidence is reviewed.

## 8 October 2026 05:32 UTC - zero-update start coverage audit

The diagnostic fit sample contains3 episode-first observations out of2048(.1465%),held sample1/512. With128 draws per step and64 steps,the fit distribution yields12 expected start draws per copy. This quantifies limited exposure in the short offline probe;it does not reconstruct or establish insufficient start sampling in actual training. Saved replay has131 starts among145424 transitions. All131 starts form9 exact decoded observation hashes with small context/physics differences;none of the sampled nonstart rows exactly aliases any start. This is a sampled exact-identity check,not proof of a fully Markov representation or absence of near-aliases.

Original actor start top action8 probability.032694 with top-two gap.0005497. Mean-Q soft target top action35 probability.058295,second action56 probability.047109,gap.011187;Q gap.002131,alpha.01,soft entropy4.05041. Thus the target has an identifiable gas-action preference despite high entropy. Shared no-gas choices after short mixed-sample copy fitting cannot alone establish inability to represent the start preference.

New audit_sd_sac_start_coverage.py: Ruff/boundary-scoped mypy pass,real CPUidle1thread inference completed,zero optimizer/controller updates,checkpoint fingerprint/SHA/model immutability verified. Immutable source pins valid/no changed STOP. Evidence: evidence/sd-projlr08-final/start-coverage.json. No offline process running,no new pilot,fullBLOCKED,automationACTIVE5min. Next bounded offline capacity/exposure control: fit disposable projected actor copies to fit-episode start rows directly against unchanged mean-Q/alpha/behavior anchor,compare held-episode starts and general held rows;paired seeds,savedAdam,strict fixed cap. This is a representation/fitting diagnostic,not permission to alter replay sampling or claim driving improvement.

## 8 October 2026 05:25 UTC - fresh-Adam control complete; no objective promotion

The paired fresh-Adam control completed512 projected actor-copy steps in113.53s. AtLR.0009,SAC held reverseKL.078282/.072759 is close to forward-KL.075557/.074223; SAC greedy agreement64.65/64.84% remains below forward76.76/76.37%,and greedy regret.000963/.000879 exceeds forward.000522/.000596. Resetting disposable optimizer history removes much of the saved-Adam SAC deficit,so the earlier inherited-Adam comparison cannot establish that SAC is intrinsically worse.

Both fresh objectives atLR.0009 choose action56(no gas) on all131 saved starts,with zero agreement against mean-Q greedy action35(gas1) and regret.002131; atLR.0001 both choose action3,also zero agreement. Neither comparison supports an objective-only runtime fix or another identical fast-actor pilot. Reused replay,frozen critics and only64 steps limit interpretation; no capacity/architecture conclusion or claim about driving.

Original CP/model/Adam unchanged,all exact probe processes closed,actual CP SHA f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53,immutable pins valid,no changed STOP. Evidence: evidence/sd-projlr08-final/actor-objectives-fresh-adam.json and actor-objectives-fresh-adam-verification.json. No offline process remains,no new pilot launched,fullBLOCKED,automationACTIVE5min. Next zero-update diagnostic: quantify start-state fit coverage,observation aliasing and actor/soft-target top-action probability margins; distinguish sampling/generalization from fitting ability before another optimizer probe.

## 8 October 2026 05:20 UTC - saved-Adam actor objective result and fresh control

All512 projected actor-copy steps completed in113.14s; original checkpoint/model/saved Adam unchanged,all exact probe identities closed,actual CP SHA and immutable pins valid,no changed STOP. On held episodes,forward-KL atLR.0009 gives agreement76.95/77.15%,regret.000536/.000518,reverseKL.124906/.124686. Canonical SAC at the same LR and inherited Adam gives agreement21.68/22.27%,regret.007336/.007167,reverseKL.653937/.608725 versus baseline.471377. AtLR.0001,SAC reverseKL.469171/.468223 barely improves baseline. All variants still have zero mean-Q greedy agreement on saved starts; no objective promoted and no new runtime pilot.

This is not a clean comparison of objectives trained from scratch: inherited Adam moments were accumulated under forward-KL,whose current local logit gradients are much larger. A paired fresh-Adam control is therefore now running with identical original actor,Q,alpha,entropy anchor,episode split,seeds,LR grid,64 steps per cell and projection. Explicit --adam-state fresh changes only disposable optimizer initialization; saved mode remains default. Output actor-objectives-fresh-adam.json is new,never overwrites saved-Adam evidence. Fixed300s cap/512steps,no runtime learner/controller. Exec3005,CPUchild77448ctime1791436820.5188873 IDLE64,wrapper77376ctime1791436820.5062568. Wait this existing probe;never duplicate or extend.

Evidence saved-Adam: evidence/sd-projlr08-final/actor-objectives.json and actor-objectives-verification.json. Fresh-Adam results pending. Ruff and boundary-scoped mypy pass. No claim about critic truth,architecture bottleneck or driving from this reused-replay frozen-Q diagnostic. FullBLOCKED;automationACTIVE5min.

## 8 October 2026 05:16 UTC - actor objective copy comparison running

New diagnostic probe_sd_sac_actor_objectives.py compares forward-KL versus canonical SAC on identical frozen mean-Q,alpha,behavior entropy references and whole-episode fit/held rows. Each objective uses actorLR.0001/.0009 and seeds17/29,64 projected saved-Adam actor-copy steps per cell,512 total,300s copy-compute cap. Saved checkpoint/model/optimizer remain originals; no controller/runtime learner/new pilot. Fixed-Q fitting cannot establish critic correctness or driving.

Initial held logit gradients differ in magnitude and direction: forward norm.006390,SAC.00009552,cosine.76136. Unchanged entropy-anchor gradient relative norm is.00213 versus.14257. These are local logit gradients,not full parameter gradients or Adam step sizes; no conclusion about learning rate follows directly. Metrics include both KL directions,SAC objective,expected Q,greedy agreement/regret,entropy and per-row actions. Compare matched-seed/LR results only after completion; no objective promotion from partial results.

Ruff and boundary-scoped mypy pass. Existing objective helpers match runtime forward cross-entropy and canonical SAC up to a state-only Q baseline. Active CPU child80212 created1791436551.8764615 IDLE64,wrapper39824 created1791436551.862483. Exec66735,BASE/sd-projlr08-audit/actor-objectives.json and actor-objectives-processes.json. No duplicate execution or cap extension. Immutable source pins valid/no changed STOP. AutomationACTIVE5min; fullBLOCKED.

## 8 October 2026 05:12 UTC - projected terminal calibration rejected

Offline projected critic/saved-Adam copy probe completed all 384 steps in336.33s within600s cap. Both lambda0 baselines pass the held nonterminal drift guard. Every tested extra terminal weight(.01,.1,1) fails that guard in both seeds. At lambda.01,held terminal MAE improves from lambda0 4.994/5.013 to4.551/4.598,while held nonterminal MAE worsens to.113856/.115229 against the fixed limit.085662. Higher weights worsen nonterminal error further(.429/.434 and.913/1.159).

Whole-episode paired bootstrap with three-candidate family adjustment gives positive terminal-error reductions versus lambda0: .01 interval[.3456,.5213],.1[1.6123,2.2928],1[2.9242,4.7702]. Both-seed terminal gains also beat the initial baseline,but none qualifies because all nonzero weights fail the predeclared drift guard. This frozen-target reused-replay diagnostic does not establish driving quality; no terminal weighting promoted and no new pilot launched.

Final assertions confirm original checkpoint,models and saved Adam unchanged. Independent actual checkpoint SHA f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53,all probe exactPID/ctime closed,immutable pins valid,no new/changed STOP. Evidence: evidence/sd-projlr08-final/projected-calibration.json and projected-calibration-verification.json. Full BLOCKED;automationACTIVE5min. Next scoped offline hypothesis: inspect actor objective gradients and compare projected forward-KL/canonical SAC copies against identical frozen mean-Q targets with episode-held rows and saved Adam. No runtime setting change justified yet.

## 8 October 2026 05:08 UTC - projected critic copy probe running

New standalone probe_sd_sac_projected_calibration.py is executing offline on CPU idle/single-thread against immutable CP33856 f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53. Only independent q1/q2 and saved critic Adam copies receive updates; saved actor,target critics,alpha and Bellman targets remain frozen. Projection after each copied Adam step; no Polyak update. No game controller/runtime learner/new training run.

Fixed whole-episode fit/held split, all available eligible terminal rows plus1024 sampled nonterminals per split. Stratified128+128 sampling restores empirical fit terminal prevalence for the clipped Bellman baseline; extra terminal MSE lambda0/.01/.1/1,seeds17/29,48 steps each,384 total,600s copy-compute cap. Baseline held terminal MAE5.01630,held nonterminal0.071385; relative20% guard<=0.085662. Both completed lambda0 copies pass guard and leave terminal error near baseline. Nonzero-grid result pending; do not infer success or launch a candidate from partial output.

Three focused numerical tests pass (empirical class weighting,explicit terminal contribution,clipped-versus-extra gradient). Ruff and boundary-scoped mypy pass. Process receipts and incremental results: BASE/sd-projlr08-audit/projected-calibration-processes.json and projected-calibration.json. AutomationACTIVE5min; poll existing execsession37505 or inspect recorded exactPID/ctime,never start a duplicate. Candidate interpretation requires both seed guards,held-terminal paired comparison against lambda0 and initial baseline,and original checkpoint/model/Adam immutability. Full remains BLOCKED.

## 8 October 2026 05:01 UTC - mean/min and target-clipping audit

Completed zero-update CPU idle/single-thread inference on both immutable projected checkpoints. The current mean-Q actor target ranks action35 (gas1) first on all131 saved starts; Q1 and Q2 individually rank no-gas actions56/62 first. Therefore the preceding no-gas finding for the individual critics does not mean their averaged actor target also prefers no gas. Mean-Q gas mass56.03%,best-gas advantage0.00213; min-Q gas mass51.00%,best-gas advantage-0.00176. Predecessor mean-Q gas mass93.77%,advantage0.02486. These are different saved replay states,not a paired causal comparison. Switching actor targets to min-Q is not supported as a fix for this failed start behavior.

Terminal clipping blocks the selected loss branch for only1/130 samples per critic (0.77%) in the current checkpoint,versus0/181 and1/181 previously. Nonterminal blocked fractions0%. This checkpoint snapshot does not explain terminal MAE5.53 through pervasive clipped gradients and cannot reconstruct historical gradients. Terminal bootstrap and mean-minus-min target gap are exactly0; terminal targets match immediate rewards. Nonterminal mean-minus-min target gap0.01718 on1024 rows,entropy bootstrap0.03785. These counterfactual targets do not establish true Q values.

New runnable audit: audit_sd_sac_target_calibration.py. Ruff and boundary-scoped mypy pass; two real CPU inference executions pass fingerprint/SHA/model immutability checks. Terminal-zero-bootstrap,target-equals-reward,and mean>=min invariants pass. No optimizer/controller/training run created. Evidence: [current](../../evidence/sd-projlr08-final/target-calibration.json),[predecessor](../../evidence/sd-projlr08-final/predecessor-target-calibration.json).

Next diagnostic remains offline: test whether projected disposable critic+savedAdam copies can reduce episode-held terminal error while preserving fixed nonterminal Bellman targets under a relative20% drift guard; include an unmodified-loss baseline and paired seeds. Earlier unprojected checkpoint weighting probes failed the guard; do not promote terminal weighting or claim an architecture bottleneck. Only a positive,repeatable scoped result can justify a fresh pilot. Current sd-projlr08 remains closed;full BLOCKED;automationACTIVE5min. No new controller launched.

## 8 October 2026 04:55 UTC - fast actor pilot failed; offline diagnosis

sd-projlr08 completed normally at04:50:06 UTC. Real evaluation: **0/10 finishes, mean progress1.060849%**, versus predecessor2/10 and51.558708%. All other gate checks passed: ten complete correct-UID trials,1446 valid timing measurements,max50ms,no controller/telemetry errors; skipped frames2906,max6. Full training remains BLOCKED. No next pilot launched.

Normal guard closure at04:50:07 UTC,forced[]; external post-closure verification confirms all43 observed exactPID/ctime closed,mutex free,1883 immutable pins valid,no new/changed STOP. Final CP33856 SHA f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53:145424 transitions,33856 updates,credit0,finite/accounted/drained/complete. Frozen9821c23e and prior checkpoints preserved.

Completed offline CPU idle/single-thread audits,zero runtime learner/controller updates:130 terminals MAE5.52922 (predecessor2.08577);944 analyzed nonterminals Bellman MAE0.06185 (predecessor0.08369). These are different saved replay samples,not a paired causal comparison. Critic pair greedy agreement on the sampled replay falls to5.20% from70.79%. On131 saved starts actor selects8,critic1 selects56,critic2/min selects62; all three are no-throttle timed-brake actions. Actor entropy4.25073; minQ margin0.06533; actor/minQ agreement0%. This is evidence against faster actor tracking as a sufficient repair,not proof of a capacity/architecture bottleneck.

Completed disposable saved-Adam actor probe64 steps per LR/seed,384 total,86.77s,checkpoint/model/Adam unchanged. Fixed averageQ held agreement23.24% initially; LR0.0001 gives25.00/25.00%,0.0003 gives25.20/24.61%,0.0009 gives76.95/77.15%. For0.0009 held KL0.37559->0.08744/0.08687 and regret0.00861->0.000536/0.000518,while saved-start top1 agreement remains0% for every variant. Frozen-Q fitting improvements did not establish useful driving. Next step: offline inspect critic target/value calibration and mean-versus-min actor target before selecting a justified fresh plan. No terminal-weighting promotion,shared reward/model/GNN changes,resume,or reused-queue restart.

Evidence: [closure](../../evidence/sd-projlr08-final/post-closure-verification.json),[critic](../../evidence/sd-projlr08-final/critic-audit.json),[starts](../../evidence/sd-projlr08-final/initial-landscape.json),[actor copies](../../evidence/sd-projlr08-final/actor-lr-probe.json). Automation ACTIVE every5min during offline diagnosis; earlier active-training states below are historical.

## Aktualizacja 2026-10-08 02:20UTC: nowy projected actor-LR pilot działa

sd-projlr08 wystartował02:18:45.468189UTC=04:18Warsaw, run tmrl-sd-sac-projected-fastactor-s17,W&Bryzo37yc. Aktor registered i świeży ingest205/0updates02:19:23UTC; warmup nie ocenia jazdy. SAVE06:08:45UTC/HARD06:18:45UTC,cap14400s,targetCAŁKOWITY145408/max11400+10greedy/max2100. Runner74772/guard8472/learner7632/actor66896 exactctimes w receipts. Preflight10ms/33fields/57reads/correctUID. Frozen projection runtime9821c23e unchanged;1883pins,fingerprintc022d63558596fe1a341d616f4d83681b2d9435ad90d966ea645cd6fe2cc0fb7.

Jedyna zmiana learning setting względem sd-proj08: actorLR.0001→.0009. Fresh odzera,noresume; reward/model/GNN/replay/UTD/alpha.01/beta.0005/criticLR/objective/projection bez zmian. Uzasadnienie: completed paired64step disposable savedAdam actorcopies,fixedaverageQ,CPUidle1thread. HeldKL .6118/.6125→.2123/.2094,regret .001408/.001304→.000575/.000330,agreement85.16/85.74%→89.26/88.87%; wszystkie LR variants poprawiły starts agreement0→100%. LR.0003 również poprawiał fitting; wybór.0009 bada szybsze tracking movingQ,nie izoluje prawdziwości Q i nie dowodzi poprawy jazdy. Saved model/Adam/CP unchanged; brak zapisania kopii policy.

27testówPASS:8nowychscope/window +19applicableguards; validate-only,Ruff i mypy z pominięciem zewnętrznych/importowanych boundariesPASS. Pierwsze wywołanie obejmowało legacy validate_plan oczekujący auth-path zamiast aktualnego prose-auth (27PASS/1FAIL),zachowano diagnostykę;łącznie5legacy actorLR/auth-plan przypadków nie stosuje się do nowego launchera. Nowe scope tests weryfikują dokładną zmianęactorLR,zakazreward/training/entropy/full/stale/reuse. Bez edycji starych helpers/frozen. MonitoringACTIVE5min do>=15min zdrowych learned checks,potemgodzinny tylko daleko odSAVE. Gate>=8/10 i pełne kontrole,fullBLOCKED; nigdyrestart/extend/nowySTOPoverride.

## Aktualizacja 2026-10-08: sd-proj08 zakończony, dalsza diagnoza offline

Świeży projected pilot zakończył się normalnie 02:07:37UTC; guard closure 02:07:38UTC, forced[]. Wszystkie zaobserwowane exactPID/ctime zamknięte, mutex wolny, 1865 immutable pins i poprzedni CP niezmienione. Final CP33866 SHA333522bce19b998118da2ca3acde864c17e092f2d821f82d54d861d16256e8dc:145466 transitions/33866 updates/credit0.5, finite/accounted/drained/complete.

Rzeczywista ocena FAIL2/10, średni postęp51.558708%, mediana dwóch met87.675s. Dziesięć kompletnych prób z prawidłowym UID,10404/10404 pomiarów timing valid,max50ms,bez błędów kontrolera/telemetrii; skips2262,max7. Wszystkie kontrole poza>=8finishes PASS. Lepszy wynik niż bezpośredni actorfit0/10 nie oznacza lepszej niezawodności od wcześniejszego slow3/10; pełne treningi nadal BLOCKED.

Offline CPU idle/single-thread,zero learner/controller updates:181 terminali MAE2.0858,978 analizowanych nonterminali Bellman MAE0.08369; behavior-return proxy ma ograniczenia off-policy. Na182 zapisanych startach oba krytyki wybierają akcję5,aktor akcję65 (agreement0%,regret minQ0.02737),entropia aktora3.9276 vs soft-target2.7918. Normy hyperspherical0.99999982–1.00000012: projekcja działa,nie dowodzi poprawności krytyka. Rozpoczęto bounded disposable actor+savedAdam LR probe64steps,seeds17/29,LR.0001/.0003/.0009 przy fixed averageQ/alpha.01/beta unchanged/projection. Wynik jeszcze nieznany; brak nowego pilota lub zmiany savedpolicy. Autoryzowana pętla pozostaje ACTIVE5min; kolejny świeży pilot wymaga konkretnego uzasadnienia,nowego planu/predecessor/pins/STOP,bez restartu sd-proj08.

## 8 October 2026 01:32 Warsaw — projection pilot running

Following the direct human go, sd-proj08 started at01:30:24 Warsaw on frozen9821c23e.
Actual preflight passed (10ms,33fields,57reads,correctUID); actor registered and
new ingest confirmed. Start verification:1149 transitions,0 updates (warmup),
1865 immutable pins valid, no new STOP/run failure. Monitoring ACTIVE every5min;
hourly cadence requires sustained verified health. SAVE05:20:24/HARD05:30:24
Warsaw; no extension/restart. Driving remains unqualified, full BLOCKED.
See [scheduled handoff](SD_SAC_SCHEDULE_20261008.md) and sd-proj08/start-verification.json.
Earlier PREPARED_NOT_LAUNCHED statements are historical preparation receipts.

## 8 October 2026 — pilots scheduled from 01:30 Warsaw

New direct human authorization resumes justified bounded SD-SAC tests and monitoring.
First queue sd-proj08 is PREPARED_NOT_LAUNCHED on frozen9821c23e, fresh projection
control; live game readiness and driving remain unverified. Existing heartbeat is
ACTIVE for01:30 and will switch to5-minute checks, then60-minute checks after three
healthy observations spanning15 minutes. Actual guard deadlines remain binding.
Full training remains BLOCKED; no automatic full launch. See
[scheduled pilot handoff](SD_SAC_SCHEDULE_20261008.md). Earlier PAUSED/offline-only
statements below describe historical scopes, superseded only by this authorization.

Offline optimizer repair 2026-10-07: SD-SAC now projects hyperspherical weights after
critic/actor Adam and before Polyak; old unprojected Adam contract cannot resume.
Saved raw row norms reached4.20. Fixed-Q actor copies at unchangedalpha.01 improved
held agreement84.59/88.58% to92.20/93.53% with projection, not proof of driving/Q truth.
Small terminal-weight/head-only controls selected no statistically supported repair
under the nonterminal drift guard. Fresh sd-sac-projected-s17.yaml PREPARED_NOT_LAUNCHED;
no new run/controller/game/network publication.176 tests/Ruff/mypy passed,1260pins
and original CP SHA unchanged, automationPAUSED, full17/29/43 BLOCKED; QR preserved.
See [SD_SAC_PROJECTION_REPAIR_20261007.md](SD_SAC_PROJECTION_REPAIR_20261007.md).

Offline joint-calibration follow-up 2026-10-07: SHA-pinned CP33866 actorfit,
138 starts/78 actions, both critics choose71 in138/138; actor21 ranks11/5/8
(Q1/Q2/average).384 disposable steps, lambda0.1/1/5/10, seeds17/29,48 each.
All held-terminal reductions have positive family-adjusted paired intervals,
but ALL weights violate held-nonterminal MAE<=0.145 in both seeds. No viable
joint calibration under this guard; capacity/interference conclusion conditional
on frozen targets/split/budget. No live run/resume/networking; immutable CP/model/
Adam unchanged, automationPAUSED/fullBLOCKED. Local offline deliverables only.
See [SD_SAC_JOINT_CALIBRATION_20261007.md](SD_SAC_JOINT_CALIBRATION_20261007.md).

Offline follow-up 2026-10-07: terminal materialization/targets/gradient sign checked;
137 exact terminal reward targets, only1/274 clipped gradients blocked.384 disposable
critic-copy optimizer steps, no runtime learner/controller. Corrected oversampling
showed no consistent held-episode terminal improvement; terminal-only MSE reduced
held terminal MAE3.3102 to0.2463/0.2500 but raised nonterminal residual0.1210 to
11.8800/11.9412, so it is rejected as a repair. Separate scalar-tensor replay-read
bug fixed with regression tests; ordinary saved action indices were unaffected.
87 tests/Ruff/mypy passed; original CP and frozen pins unchanged. No live launch,
no qualified learning-setting change; full BLOCKED/automation PAUSED/QR preserved.
See [SD_SAC_CRITIC_PROBE_20261007.md](SD_SAC_CRITIC_PROBE_20261007.md).

Final 2026-10-07 20:04UTC: sd-fit07d CLOSED / FAIL, automation PAUSED.
Real evaluation0/10 finishes, mean progress5.770238%, median0.599639%; nine
slow_progress failures and one51.66% time_limit. All ten correct-UID trials valid,
4477/4477 timing measurements/max50ms, no controller/telemetry errors; skips1248/max5.
FinalCP33866:145464 transitions/33866updates/credit0, finite/accounted/drained,
SHA a243b4f50d210d882860b74e4626ff42b369c3f1c2fd78a25fe81068d05906d7.
Normal closure/forced[], all51 exact identities closed/mutexfree/1260pins unchanged.
Final offline audit still finds terminal MAE3.3182 on137 states. No new live test or
full training; further live work requires a new direct human instruction. FullSD
BLOCKED; QR PASS preserved. Earlier ACTIVE sections below are historical snapshots.
[SD_SAC_ACTORFIT_RESULT_20261007.md](SD_SAC_ACTORFIT_RESULT_20261007.md).

Update 2026-10-07 19:46UTC: terminal MAE aggregation repaired in editable source;
50 regression/contract/stability tests, Ruff and mypy passed. Frozen sd-fit07d
continues unchanged; no new live test/full training. Read-only CP25000 audit finds
known terminal target miscalibration: mean Q+3.0702 vs reward-2.0465, MAE5.1168 on
all106 terminal failures. Off-policy return proxies do not establish action rankings
or driving readiness. Full remains BLOCKED pending actual evaluation. Actual actor/
checkpoint/frozen-runtime fingerprint1f4d2613… supersedes the differing historical
launch-confirmation snapshot; no override. Details/evidence:
[SD_SAC_CRITIC_AUDIT_20261007.md](SD_SAC_CRITIC_AUDIT_20261007.md).

Actor-fit actual start 2026-10-07 17:22:36.869341UTC (19:22 Warsaw): NEW sd-fit07d is RUNNING, fresh tmrl-sd-sac-actorfit-s17/W&B6d20j27j, frozen281332d99fdb8bc1edb70d590260731699ab40bf. Actualpreflight ready10ms/33fields/58reads/correctUID, learner83588 and actor71412 registered/collecting, initialupdates0 and freshingest confirmed. Objective-only experimentalforwardKL; noresume, unchangedreward/model/GNN/replay/UTD. Train145408/max11400s then10greedy/max2100s, whole14400s, SAVE21:12UTC/HARD21:22UTC (23:12/23:22Warsaw). ExactPIDctime inlive-ownedreceipt. AutomationACTIVE15min, newSTOPwins, noautonexttest/full. Warmup is not driving evidence; fullBLOCKED until>=8/10 plus all integrity/timing checks. Earlier sd-fit07 failedforeground/normalclosed; sd-fit07b/c preparationonly preserved. See sd-fit07d/launch-confirmation.json and actualstatus, not older snapshots.

> Prepared only after human STOP: new actor-fit candidate `configs/diagnostic/sd-sac-actorfit-s17.yaml` changes actor objective SAC→experimental forwardKL; all other learning/reward/model/GNN/replay/UTD settings remain as stopped alpha-floor. CPU actor-copy probe: heldout Q-argmax agreement13.28% SAC vs91.80% forwardKL after300steps; episode-first action0→57 for forwardKL. Frozen-Q fitting is not proof of driving or critic correctness. Added greedy-margin diagnostics and strict preparation/scope checks. See SD_SAC_ACTORFIT_PREPARATION_20261007.md. No launch, no checkpoint resume, no STOP exemptions; automationPAUSED/fullBLOCKED.

> Offline analysis after human STOP (2026-10-07): same 1024 replay observations across both saved models show current alpha-floor greedy gas-off66.89%, mean steer-0.970. All256 sampled episode-first observations choose action0 (gas0/brake0/steer-1); top probability only2.76%, so broad stochastic entropy masks an unfavorable greedy maximum. Actor feature std0.0385 does not support global representation collapse. Comparable-budget last30 training mean progress8.37% vs8.87%, median7.33% vs1.25%, finishes0/30 both; these are training episodes, not evaluation. Evidence: artifacts/tmrl-test-comparison/sd-af07-stopped-analysis-20261007/{analysis.json,RAPORT.md}. CPU inference only,0updates/noController/models and checkpoints unchanged. Improvement unproven, full BLOCKED, human STOP and automation PAUSED remain in force.

> Current state — 2026-10-07 14:01 Europe/Warsaw: HUMAN STOP / PAUSED. Direct instruction: "przerwij i tak na ten moment trening". sd-af07 closed normally at 12:00:26 UTC, forced processes [], all 19 observed exact PID/creation-time identities closed, controller mutex free. No evaluation or subsequent training launched. Full SD-SAC remains BLOCKED. Earlier ACTIVE sections below are historical snapshots.
>
> Preserved alpha-floor candidate CP25735: 114821/145408 transitions, 25735 updates, remaining update credit 470.25; finite, fingerprint matched and accounting valid, but budget incomplete and credit not drained. SHA256: 38279862c9c56417dc9804869c66f2bb8b06c894341ccaf01aecb8b198487938. Immutable source/helpers and four previous saved checkpoint hashes verified unchanged. Evidence: artifacts/tmrl-test-comparison/sd-af07/human-stop-verification.json and human-stop-checkpoint.json. New STOP preserved and automation PAUSED; do not launch until a new direct human instruction.
>
> Driving improvement remains unproven for this candidate: evaluation NEVER_LAUNCHED (not 0/10). Latest completed predecessor scored 3/10 finishes. Current training entropy metrics are not a driving qualification.

## 2026-10-07 12:13 Warsaw — completed slow pilot FAIL, fresh alpha-floor-only pilot ACTIVE

Completed exact continuation sd-c07:145408 transitions/33852 updates/credit0,
finite/accounted/drained, CP33852 SHA90705fda277c26536c2c7d08a7e501d2b2dc751c92e457935befcdf6e2827fa6.
Actual evaluation3/10 finishes, median143.22s, mean progress51.365148%.
Ten complete correct-UID trials, timing maxima60ms, no controller/telemetry errors,
skips9278/max5. All gate checks passed except>=8 finishes. This fails readiness.
Compared with the incomplete checkpoint's2/10 and57.836355% mean progress, extra
training did not establish a consistent improvement. No full training has started.
Normal closure10:09:35UTC, forced[], all41 observed exact PID/ctime identities
closed, mutex free, source/helper/original evidence and final checkpoint unchanged.

Offline temperature probe of this final checkpoint: savedalpha0.00027483, frozen-Q
target entropy0.0152 on1024 replay states; atalpha0.01 target entropy1.805.
No learner updates, saved-policy changes or game input. This suggests excessively
sharp targets as a testable mechanism; it does not prove critic correctness or
that a floor improves driving. Evidence BASE/sd-temp08/temperature-sensitivity.json.

Fresh bounded queue BASE/sd-af07 is ACTIVE, run tmrl-sd-sac-alphafloor02-s17,
W&B gdfg75cl. Same frozen70d3eb0a runtime, NEVER EDIT/PULL; fresh replay/zero initial
updates, no checkpoint resume. The only algorithm setting changed from actorslow is
entropy_coefficient_min0.01. ActorLR0.0001,critic/entropyLR0.0003,target0.8,beta0.0005,
SAC objective,alpha init0.2,terminalcoef0,qclip0.5,reward/model/GNN/replay/UTD unchanged.
Fingerprint e629a9f7d08deb923a19c14c4c53982a8e0b8ee761595cf3ac101eab42523d14.
27 plan/predecessor tests and validate-only passed. Actual preflight30ms/33fields/
54reads/correctUID. Runner82556,guard71832,learner76756,actor50956 exact identities
persisted in actual-launch-receipt.json and live-owned-identity-receipt.json.
ActualSTART10:11:53.139355UTC,SAVE14:01:53UTC/HARD14:11:53UTC (16:01/16:11Warsaw).
Cap14400s,train145408/max11400+10greedy/max2100; never extend or restart.
Historical STOP exceptions are inherited exactly, never automatically re-pinned.
New/changed STOP wins. No UI or second controller while learner/evaluation is active.
FullSD remains BLOCKED until actual gate and matching full configurations; QR PASS
29/30 median54.3s preserved. Older ACTIVE sections below are historical snapshots.

## 2026-10-07 11:55 Warsaw — slow training COMPLETE, ten-trial evaluation ACTIVE

The new human instruction reauthorizes SD-SAC repair/test loops and prioritizes SD-SAC
before other algorithms. Historical STOP files remain preserved with exact SHA/mtime
exemptions; a new or changed STOP always wins.

Actual saved-policy evaluation sd-e08 completed normally: **2/10 finishes**, median
149.24s, mean progress57.836355%. Ten complete correct-UID trials; timing, SHA and
controller/telemetry checks passed, skips reported. All observed owned identities
closed, mutex free, immutable pins unchanged. This is improved driving versus earlier
0/10 results, but it fails the>=8/10 gate and uses incomplete training.
Evidence: BASE/sd-e08/post-closure-verification.json and its evaluation artifact.

New bounded attempt BASE/sd-c07 completed training normally, run slow-complete,
W&B8dj5aeh9. Final145408 transitions/33852 updates/credit0, finite/accounted/drained.
CP33852 SHA90705fda277c26536c2c7d08a7e501d2b2dc751c92e457935befcdf6e2827fa6.
Checkpoint validation passed; the new ten-trial evaluation is now ACTIVE.
FULL restore confirmed140754 transitions/32256 updates, replay140754, credit432.5,
same frozen70d3eb0a source and fingerprint2be95ca917238a1fd5f90760e24c2763d899b3633784b5285acc1a914372b1bc.
Only run/artifact identity changed. Original checkpoint/journal remain byte-identical.
Total target remains145408 (4654 remaining); no target extension, replay reset or
changed-code/settings resume. Trainmax900s, then10greedy/max2100s, overallcap3600s.
ActualSTART09:47:33.022069UTC, SAVE10:42:33UTC/HARD10:47:33UTC (12:42/12:47Warsaw).
Preflight20ms/33fields/55reads/correctUID and19 completion/STOP/accounting tests passed.
Exact identities are persisted in sd-c07/live-owned-identity-receipt.json.
The separate alpha-floor-only candidate is PREPARED_NOT_LAUNCHED, not the active run.

Full SD-SAC remains BLOCKED pending complete/finite/accounted/drained checkpoint and
>=8/10 real finishes plus all identity/timing/error/skips checks. Existing full YAMLs
still use different defaults and must be aligned with the qualified candidate before
any three-seed start. QR PASS29/30 median54.3s is preserved. No other algorithm or
full training started. Earlier ACTIVE sections are historical snapshots.

## 2026-10-07 07:32 Warsaw — fresh actorslow pilot ACTIVE

Queue `queue-sd-sac-actorslow-20261007`, run `tmrl-sd-sac-actorslow-s17`, W&B olarxz70
API running. Fresh from zero using unchanged frozen refit70d3eb0a; no resume.
Only actorLR0.0009->0.0001; critic/entropyLR0.0003,target0.8,beta0.0005,SAC objective,
alpha init0.2/unbounded,terminalcoef0,qclip0.5,model/GNN/reward/replay/UTD unchanged.
Hypothesis: slower actor updates may prevent representation collapse. Offline evidence
is mixed: fresh fixed-Q copies100steps heldout agreement0.535/0.398 for LR0.0001/0.0009,
but forwardKL3.992/3.200 and entropy4.343/3.915. Both retain feature diversity.
This does not prove the cause, critic correctness or driving. Collapsed full-actor
copies remain poorly fitted after100steps; evidence BASE/sd-sac-refit-audit-20261007.
24 guard tests+validate-only passed; preparation10ms/33fields/60reads and actual
preflight10ms/33fields/55reads/correctUID. Runner56556,guard37024,learner63824,
actor64872 identities in actual-launch-receipt.json. ActualSTART05:31:45.606638UTC,
SAVE09:21:45.606638UTC/HARD09:31:45.606638UTC (11:21/11:31Warsaw),cap14400s.
Train145408/max11400+10greedy/max2100. UnderSAVE stayturn waits<=60s toclosure/HARD;
no owned learner/controller afterHARD. NewSTOP wins; never extend/restart/noUI.
Refit FAIL and all previous checkpoints/receipts remain preserved and verified.
FullSD BLOCKED until real>=8/10 plus all completion/identity/timing checks.
QR PASS29/30 median54.3s preserved. Authorized repair loops ACTIVE; no automatic full.
Older ACTIVE entries below are historical snapshots.

## 2026-10-07 07:22 Warsaw — refit COMPLETE, driving FAIL; offline diagnosis active

The fresh refit queue completed normally at05:22:14UTC; deadline closure forced[].
All27 exact owned PID/creation-time identities are closed and the shared mutex is free.
W&B aqsyvox9/6v7bhym0 API finished. Complete145433 transitions/33858 updates,
credit0.25, earned=accounted33858.25, drained and finite. CP33858 SHA
58f1c4db6bfc539988979a1d7a672b8ce4344538fc62ef4f5ebd4fb8faf993bc
and frozen source/helper/prior receipt pins verified unchanged, including prior CP33857.
Actual driving FAIL0/10; meanprogress4.787796%,
10 complete correctUID trials, timing maxima<=60ms, no controller/telemetry errors,
skips831/max6. FullSD remains BLOCKED; QR PASS29/30 median54.3s preserved.
The audit on exact frozen70d3 source is CPU-only, no controller or saved-policy changes.
Actor greedy action33 on1024/1024 replay states and128/128 starts, gas1/steer-1/6;
alpha0.001243, entropy0.2525 versus target0.8. Actor feature std0.000506 versus
critic feature std0.042807/0.041055; actor/Q agreement0.0957. This supports representation
collapse and poor actor fitting; it does not establish critic correctness. Independent
1000-step fixed-Q head fitting reaches only0.373 agreement at LR0.0009; full-actor
copy probes completed with limited recovery. Evidence: BASE/sd-sac-refit-audit-20261007.
No next game test has launched. Authorized repair/test/diagnosis loops remain ACTIVE;
never restart refit or extend its expired-as-completed attempt. Old ACTIVE entries below
are historical snapshots. New STOP wins. No changed-code resume or automatic full start.

## 2026-10-07 04:33 Warsaw — fresh refit pilot ACTIVE

Following the authorized repair loop, clipping5 was rejected: independent100-step
critic copies gave held-out nonterminal fixed-target MAE0.52267 with epsilon0.5
versus0.53640 with epsilon5; terminal MAE0.65203 versus0.65821. No saved policy changed.
A fresh bounded refit pilot now uses the earlier best-driving beta0.0005/target0.8
SAC objective with actorLR0.0009; the previous actorLR test was interrupted.
ForwardKL, separate entropyLR, alpha bounds and terminal auxiliary loss are explicitly
rolled back together. This does not isolate their individual effects or establish
critic correctness. Numerical/contracts/focus repairs remain; reward/model/GNN/replay/
UTD unchanged. No checkpoint resume. Exact new frozen source70d3eb0a, fingerprint
b9202ac959f155a529952b74b84d01b588e098156fbd75a841beaf27f04a1c85.
Queue `queue-sd-sac-refit-20261007`, run `tmrl-sd-sac-refit-s17`, W&Baqsyvox9 APIrunning.
Actual start02:32:54.190437UTC; SAVE06:22:54.190437UTC/HARD06:32:54.190437UTC
(08:22/08:32Warsaw), cap14400s,train145408/max11400+10greedy/max2100.
Runner30620,guard68232,learner36960,actor42340 exact identities in actual-launch-receipt.
24guard tests+18CPU learning tests+validate-only passed; actualpreflight10ms/33fields/
correctUID/57reads. Actor registered and collecting. These checks are not a driving gate.
Previous resume02 FAIL and all evidence preserved. New STOP wins; no UI/secondcontroller.
UnderSAVE remain in turn with waits<=60s untilclosure/HARD; never extend/restart.
FullSD BLOCKED,QR PASS preserved,monitorACTIVE for authorized bounded repair loops.


## 2026-10-07 04:20 Warsaw — resume02 COMPLETE, driving FAIL

This supersedes the ACTIVE snapshot below. Queue resume02 completed normally at
02:12:37 UTC; guard closure at 02:12:38 UTC has forced[]. Post-closure verification
confirmed all 31 exact owned PID/creation-time identities closed, controller mutex
free, 440 immutable pins unchanged, and training/evaluation W&B runs finished.
Checkpoint33857 SHA c3ebe43048f24bf709fb1993da432e308fb569ddc3f319c1ebb002c5c0b73502:
145430 transitions,33857 updates,credit0.5; finite, complete, drained and accounted.
Actual driving gate FAIL0/10, mean progress3.72697%; ten complete correct-map trials,
3626/3626 valid timing measurements,max60ms,p99=50ms,no controller/telemetry errors,
1728 skipped frames,max4. FullSD remains BLOCKED. QR PASS stays preserved.
Offline audit found greedy action28 (gas plus full brake) on128 saved start states,
actor/Q agreement7.62% on1024 replay states, alpha0.01, entropy3.279. Fixed-Q independent
actor-head fitting improves agreement but does not prove critic correctness or driving.
Recorded behavior soft-return proxy bias+8.981 on21220 complete-episode states is a
warning, not on-policy calibration. Independent critic clipping probe is diagnostic;
no saved weights changed and no new game test launched. Evidence:
`artifacts/tmrl-test-comparison/sd-sac-resume02-audit-20261007` and queue
`post-closure-verification.json`. Authorized repair loops and automation remain ACTIVE.
Never restart this completed queue or resume these weights on changed source.

# SD-SAC exact-source continuation and bounded repair loops — 2026-10-07

The latest direct human instruction authorizes sensible autonomous repair,
bounded-test and diagnosis loops until the SD-SAC driving gate and handoff are
complete. Full training never starts automatically. New or modified STOP wins.
Every attempt has a separate immutable plan and time limit; used queues never
restart and running attempts never gain extra time. Changed-code checkpoint
resume, replay reset and shared-reward changes remain forbidden.

## Actual continuation

Queue: H:/Studia/inzynierskie/inzynierkav2/AITrackmania/artifacts/tmrl-test-comparison/queue-sd-sac-comprehensive-resume02-20261007.
Run: tmrl-sd-sac-resume02-s17; config BASE/sd-r07/resume-sd-sac.yaml.
Frozen comprehensive runtime ab736f53b3b96f517be8c72dc85bdcefb50a0606 remains
unchanged. Full checkpoint fingerprint bb4a1819292e59a76cba405198cfd92ba05baee99a25bceabb6d339848cce652.
Initial checkpoint SHA bc68ab6231f3e2ff21235961f1337144690031bb7034acc143e261f8f8b5401d.
Original checkpoint and journal remain preserved in the old closed run.

Actual log confirms FULL state restored:23582 transitions,3272 updates;
replay23582 and credit123.5 continue, with new ingested rollouts and updates.
The total target remains145408, with121826 remaining at launch. Optimizers,
sampler, RNG, replay and durable journal are restored, with no reset.
The newer keyboard focus patch3cacbb57 is deliberately not substituted into
this checkpoint's source. Persistent foreground denial remains a possible
bounded failure; no security or focus-permission bypass is used.

START2026-10-06T23:48:20.701294UTC (7Oct01:48Warsaw).
SAVE2026-10-07T03:36:30.115902UTC (05:36Warsaw).
HARD2026-10-07T03:46:30.115902UTC (05:46Warsaw).
Effective cap14289.414608s, clamped to resume01's deadline without extension.
Train max11400s, then ten greedy trials max2100s. Under SAVE keep checking at
intervals <=60s until normal closure/HARD; no controller/learner survives HARD.

Actual-launch-receipt.json pins runner63992/ctime1791330490.7654765,
learner57576/1791330516.88723, actor49712/1791330516.8990867,
launcher53032, guard launcher78860 and actual guard44104 in effective-deadline.
W&B [5xgna8bn](https://wandb.ai/dsc-pjatk-warsaw/my-trackmania-agent/runs/5xgna8bn)
was verified running by API. Actor registered and collecting; this is launch
evidence, not a useful-driving result. No UI or second controller during the run.

## Preserved failure and verification

Resume01 failed before game controller or learner: Windows refused a temporary
checkpoint decompression path exceeding its length limit. Its failure log,
status and normal deadline closure (forced[]) are preserved; it is never
restarted. A separate short-path full-state clone was prepared directly from
the original saved checkpoint and journal. Its actual codec load passed.
The first new validator drafts rejected serialization defaults and an incorrect
import/runtime invocation; these external-helper issues were corrected before
launch. Final40 contract/preparation tests and validate-only passed on exact
frozen cwd/PYTHONPATH. No frozen/main-source edits were made for these failures.
Actual preflight ready:10ms race time,33fields,correctUID,58countdown reads.

Prepared receipts are historical preparation records, not live launch status.
Use actual launch/status/effective-deadline and events. Source freeze includes
the initial cloned checkpoint as a launch input pin; normal retention may later
remove that cloned input after full restore. Its initial SHA remains recorded;
the original checkpoint and immutable source/helper/evidence pins must stay
unchanged. New run checkpoints/journal are legitimate training outputs.

## Gate and next decision

Require complete145408 training, finite checkpoint, matching source fingerprint
and SHA, exact update accounting and drained credit; ten complete correct-map
trials, all timing maxima<=100ms, no controller/telemetry errors, reported skips
and at least eight finishes. FullSD stays BLOCKED until this actual gate.
Training/warmup progress and CPU tests do not establish improved driving.
Comparable earlier training data did not show a demonstrated improvement.
QR29/30median54.3PASS is preserved and never rerun for this repair.

On failure preserve evidence, save/close all owned PID/ctime identities and
release the shared mutex before diagnosing and considering a new bounded test.
Never resume a checkpoint under changed source. On PASS verify manual full
generator, seed17/29/43 YAML, guard and setup; publish an honest manual handoff
without starting full training. Monitor remains ACTIVE for the authorized loops,
silent while healthy; pause only after the completed objective or explicit STOP.
