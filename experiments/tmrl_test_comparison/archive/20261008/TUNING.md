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

## 2026-10-07 01:48 Warsaw — exact-source FULL resume ACTIVE

Human instruction “To dzialaj w sensownych pętlach aż będzie wszystko dopięte”
authorizes bounded repair/test/diagnose loops; older one-pilot-only restrictions
below are historical. Queue `queue-sd-sac-comprehensive-resume02-20261007`,
run `tmrl-sd-sac-resume02-s17`, W&B5xgna8bn API running. Actual FULL restore:
23582 transitions/3272 updates/replay23582/credit123.5, then new updates/ingest.
Frozen ab736f53 remains exact; no changed-code resume, replay reset or reward change.
SAVE 2026-10-07 05:36:30 Warsaw; HARD05:46:30, both retained from resume01.
No extension/restart. Resume01 failed before controller/learner on Windows temporary
path length and closed normally; the new shorter clone passed actual offline load,
40 contract tests, validate-only and actual preflight10ms/33fields/correctUID.
FullSD remains BLOCKED until actual complete/drained/accounted/finite training and
>=8/10 valid driving trials. QR29/30median54.3PASS preserved. MonitorACTIVE;
healthy unchanged state stays quiet. See [resume evidence and loop limits](SD_SAC_RESUME_LOOPS_20261007.md).

## 2026-10-07 — comprehensive interrupted; focus repair verified; resume not launched

This supersedes older ACTIVE snapshots below. Comprehensive closed normally at
2026-10-06 21:19 Warsaw after foreground activation was denied at an episode reset.
Saved CP3272:23582 transitions,3272updates,credit123.5,finite/accounted; training
incomplete and not drained. Evaluation NEVER LAUNCHED; no driving-gate result.
All checked owned identities closed; source/checkpoint preserved; automationPAUSED.

Editable focus recovery now waits at most5s per focus acquisition and rechecks
exact game HWND before keys.84 focused CPU/fake tests and two actual idle resets
passed (10ms,33fields,protocol2,correctUID); no policy/learner/evaluation started.
Persistent Windows focus denial remains bounded/fail-closed. Exact-source resume
is prepared separately using frozenab736f53 and cloned checkpoint/journal; the
new keyboard source cannot be substituted under that checkpoint. No old queue or
deadline is restarted/extended. See [focus recovery evidence and resume limits](SD_SAC_FOCUS_RECOVERY_20261007.md).
FullSD remains BLOCKED; QR29/30median54.3PASS preserved. Comparable training data
do not yet establish improved driving; the earlier train-vs-eval comparison was invalid.

## 2026-10-06 20:57 Warsaw — comprehensive repaired-source pilot ACTIVE

Direct instruction “no to odpalaj, chce to dzisiaj ogarnąć najlepiej” authorizes
one fresh bounded training and10 greedy trials, with no automatic following/full.
Queue `queue-sd-sac-comprehensive-20261006`, run `tmrl-repair-sd-sac-comprehensive-s17`.
Frozen source ab736f53b3b96f517be8c72dc85bdcefb50a0606 at
C:/Users/szulc/.codex/worktrees/tmrl-sd-sac-comprehensive-runtime/AITrackmania.
All terminal experimental settings are retained exactly; only audited code and
bounded preflight repairs change. No checkpoint/replay resume, reward/model/GNN/
replay/UTD changes, or old frozen edits. This remains an experimental variant.

START 2026-10-06T18:57:35.287822+00:00 = 2026-10-06 20:57:35 Warsaw.
SAVE 2026-10-06T22:47:35.287822+00:00 = 2026-10-07 00:47:35 Warsaw.
HARD 2026-10-06T22:57:35.287822+00:00 = 2026-10-07 00:57:35 Warsaw.
Cap14400s, train145408/max11400 then10greedy/max2100; never extend/restart.
Runner67324 created1791313053.5805008, guard59804,
learner58108 ready, actor57380 registered/collecting, W&Bh9q80yog API running.
Actual preflight20ms/33fields/correctUID, bounded countdown polling55frames.
First preparation attempt safely rejected denied foreground activation before
keyboard input; computer-use activated the returned game window, then a separate
prepared check and actual preflight passed. Failed attempt evidence is retained.

329 code tests and34 new queue guards plusvalidate-only passed. This is code and
launch evidence, not driving qualification. No UI/secondcontroller during learner
or evaluation. New/changedSTOP wins; only pinned unchanged historicalSTOP exempt.
Gate>=8/10 plus complete/finite/fingerprint/SHA/drained/accounted training and ten
validUID trials, timing maxima<=100ms/noerrors/skips. FullSD BLOCKED; QR29/30median
54.3PASS preserved. MonitorACTIVE; pause and report on completion/STOP; no nextpilot
without direct human instruction. `actual-launch-receipt.json` pins the evidence.

## 2026-10-06 20:17 Warsaw — terminal train complete; evaluation not launched; offline repairs

This snapshot supersedes older ACTIVE/deadline entries below. The authorized
terminal pilot completed training normally at18:17:35UTC:145481/145408
transitions,33870updates,credit0.25,earned=accounted33870.25,finite/drained/complete.
CP33870 SHA09507b686ffb7077b4ad1483ac9154c1e6907fd1a45272d7e5144924a3850505
and frozen7704b7f9 remain unchanged. Evaluation preflight failed18:17:50UTC:
reset did not return to the race start. NO driving evaluation, not0/10.
Normal closure18:17:50.898527UTC forced[];37 exactPID/ctime closed,mutexfree,
source/helper/previous/STOP pins valid,W&Bmygl7lrw API finished.
`final-closure-audit.json` preserves the verification. Automation PAUSED; no
following pilot/evaluation/controller without another direct human instruction.
QR PASS29/30median54.3s preserved; fullSD BLOCKED.

Human “Dobra to teraz na serio już napraw wszystkie problemy z SD SAC” authorized
comprehensive editable-code repairs and CPU/fake-backend checks. Contracts,
numerics, weight normalization, terminal continuation, metric availability and
reset/preflight defects are addressed; ordinary actor/critic math and gas/brake
mapping were not demonstrated faulty. No shared reward/model/replay change,
checkpoint resume or frozen edit. See
[SD_SAC_COMPREHENSIVE_REPAIR_20261006.md](SD_SAC_COMPREHENSIVE_REPAIR_20261006.md)
for changes, tests, unresolved partial observability and terminal reweighting.

Exact-frozen CPU inference finds gas1 at all182 preserved starts, brake tap and
steer+0.8333,actor/Qagreement0%;terminalMAE0.71417 on181 terminals. This does not
prove useful driving or isolate causes versus older different replay states.
Code checks and the delayed-goal Bellman oracle do not replace the real gate.
Final integrated329CPU/fake tests passed; Ruff/format/mypy clean. Full YAML
17/29/43 matches generator; synthetic model update/checkpoint roundtrip passed
with environment creation forbidden. Checks/evidence:
BASE/sd-sac-comprehensive-repair-20261006.

## 2026-10-06 17:56 Warsaw — fresh terminal-correction pilot ACTIVE

Latest human instruction “to ogarniaj, badz bardziej autonomiczny” authorizes one
fresh bounded pilot and ten greedy trials. This supersedes the previous paused
snapshot; it does not resume any saved checkpoint or authorize following/full runs.
Queue queue-sd-sac-terminal-20261006; run tmrl-repair-sd-sac-terminal-s17;
W&B mygl7lrw API running. START15:56:52.234211UTC, SAVE19:46:52.234211UTC
(21:46Warsaw), HARD19:56:52.234211UTC(21:56Warsaw). Total cap14400s;
train145408/max11400 then10greedy/max2100, never extend or restart.
Frozen runtime C:/Users/szulc/.codex/worktrees/tmrl-sd-sac-terminal-runtime/AITrackmania,
commit7704b7f9cfb6c7cdc5f32ad672136b9f8e01437d; never edit/update it.
Single algorithmic change vs stability: terminal_value_loss_coefficient1.
All prior actor/temperature/reward/model/GNN/replay/UTD settings retained.
This remains an experimental variant; no actual driving result yet; fullSD BLOCKED.
New evaluation persists issued controls/gas completeness, without claiming game acknowledgement.
113CPU tests/Ruff/mypy and33 guard/config/closure tests+validate-only passed.
Actual preflight260ms,33fields,correctUID; no new STOP, old hash/mtime pins unchanged;
previous exact owned identities closed, mutex free before launch. Source/helper,
previous training/evaluation/closure/checkpoint and offline audit receipts pinned.
Runner66408(created1791302211.215128),guard64200,learner54428,actor51704;
full identities and current startup evidence in actual-launch-receipt.json.
No UI or second controller during training/evaluation. QR PASS29/30 median54.3s retained.
After outcome verify checkpoint/source, all owned closures, report honestly and
pause monitoring. No automatic following pilot, changed-code resume or full start.

## 2026-10-06 — no-movement diagnosis and offline corrective variant

After “no to ogarnij to”, audited the preserved CP33866 in exact frozen e50b0b02,
CPU only, no controller or learner updates. Across128 recorded starts actor argmax
was always action24=[gas0,brake0,steer-1/3], maxprob0.03947, entropy4.16891;
critic preferred56, agreement0%. This explains why this greedy policy can remain
stationary; the old evaluation did not persist issued commands, so actual backend
execution cannot be reconstructed. It was still a valid driving FAIL0/10.

All137 replay terminal states had mean Q12.368 vs reward-2.046, MAE14.414;
terminal fraction0.09418%, clipped loss blocked9.489% of terminal gradients.
Added explicit optional terminal_value_loss_coefficient (default0 preserves SD-SAC)
to anchor selected true-terminal actions directly to known rewards, normalized
within terminal samples, without bootstrap or clipping. Truncations excluded;
importance weights honored; checkpoint options prohibit changing this objective
on resume. It is an experimental critic objective, NOT a proven driving fix.
Added terminal TD error sum/count diagnostics and issued-control histogram,
measurement completeness and gas mean in future evaluation artifacts. Missing
commands are unknown, not zero gas; commands do not prove game acknowledgement.

Offline independent critic copies,200 CPU regression steps, fixed old bootstrap:
coefficient0 terminalMAE13.921/nonterminalTDMAE0.138;
coefficient1 terminalMAE3.009/nonterminalTDMAE1.995. This tradeoff must remain visible.
Head-only300step probe improved terminalMAE14.105 to13.992 only. Training-data
calibration and frozen targets cannot establish ranking, policy quality or driving.
113 CPU tests, Ruff and mypy passed. No saved model, reward, replay or frozen source
was edited; no game test, evaluation or new pilot launched. Evidence in
BASE/sd-sac-stability-no-movement-audit-20261006. FullSD BLOCKED; automation stays
PAUSED. Fresh training/test requires a new direct human instruction.

## 2026-10-06 17:30 Warsaw — evaluation completed; driving gate FAIL

This supersedes the earlier pre-evaluation failure and paused snapshot below.
After the human instruction “no to ogarniaj”, supervision was repaired and the
already authorized ten trials were completed on the preserved checkpoint,
without repeating training. New queue: queue-sd-sac-stability-eval-recovery01-20261006.
Evaluation ended 15:27:46.118406 UTC; guard normal closure 15:27:46.483867 UTC,
forced[]. All recorded PID/creation-time identities closed; shared mutex free.
W&B wwvg5ikp API finished. Original SAVE16:45:15/HARD16:55:15 UTC were retained.

Actual driving gate FAIL: 0/10 finishes; all trials no_progress, progress0.0%.
All ten trials used the correct map UID, 1000/1000 race-clock measurements valid,
maximum/p99 50ms, no controller or telemetry errors. Skips162 total, maximum4.
Complete training145466/145408,33866 updates, earned=accounted33866.5 and
credit0.5, finite/fingerprint-valid/drained checkpoint remains unchanged:
CP33866 SHA1428c80edfc05f5b341040aa39e6d051893b8c32d3b2648a667c048f5a94cb73.
Recovery performed zero learner updates. Frozen e50b0b02 source and helper hashes verified.
Evidence: the new queue's sd-sac-repair-decision.json, post-evaluation-receipt.json,
deadline-closure.json and actual benchmark evaluation.json.

Supervision repair f68cd0e2 adds bounded identity-aware polling of cached process
objects, avoiding wait-handle reopening; descendants now persist PID plus creation
time. Nine module tests and eleven recovery/supervision regression tests passed;
Ruff passed; type checking passed with only missing psutil stubs ignored.
The original AccessDenied OS cause remains unproven; no system process was modified.
The recovery completed under the repaired supervisor, but did not repair driving.

The experimental soft_q_forward_kl stability variant is NOT qualified for full
SD-SAC or three seeds. QR PASS29/30 median54.3s remains preserved.
Automation PAUSED after completion. No new pilot, training or evaluation without
another direct human instruction. Offline diagnosis may inspect why a single
unchanging action produced no progress; its cause is not yet demonstrated.

## Poprawki kodu SD-SAC po nowym poleceniu człowieka — 6 października 2026

## 2026-10-06 17:12 Warsaw — training complete; queue failed before evaluation

This supersedes the earlier active-pilot snapshot. The repaired experimental
SD-SAC stability pilot completed145466/145408 transitions and33866 updates.
Final CP33866 SHA1428c80edfc05f5b341040aa39e6d051893b8c32d3b2648a667c048f5a94cb73
is preserved. Read-only CPU validation in the exact frozen runtime confirmed
fingerprint match, finite learner state, earned=accounted33866.5 and credit0.5,
drained and complete. No learner was instantiated or updated for this validation.

Queue failed2026-10-06T15:12:40.466247UTC with `psutil.AccessDenied(pid59824)`
from `psutil.wait_procs` while checking descendant closure after training exited.
The exact OS-level cause is unproven. Guard normal closure15:12:40.945327UTC,
forced[], all pinned owned processes closed and shared controller mutex free.
Current PID59824 belongs to svchost; the old descendant creation-time identity
was not persisted. PID reuse is plausible but unproven, and that system process
was not modified. W&B ufoxma8r API finished. Original evidence/helpers,
checkpoint and original deadlines are unchanged. This queue is never restarted.

Driving evaluation NEVER STARTED: no evaluation YAML/history/artifact or driving
score. This is neither0/10 nor a gate PASS. Full SD-SAC remains BLOCKED, QR
PASS29/30 median54.3s preserved. No new pilot/training/evaluation was started.
Monitoring is being paused on completion of this report; a further test requires
new direct human authorization. Recommended offline harness repair: bounded
PID-and-creation-time liveness checks instead of waiting on disappeared process
handles; unknown live identities must continue to block controller handoff.

Evidence: `artifacts/tmrl-test-comparison/queue-sd-sac-stability-20261006/`
`failure.json`, `runner-error.log`, `deadline-closure.json`,
`offline-checkpoint-validation.json`, `post-failure-receipt.json`.


## 2026-10-06 14:55 Warsaw — one repaired experimental pilot active

Latest direct instruction: “Jak wszystko naprawisz odpal pilot SD SAC na nowo”.
The earlier human STOP and incomplete checkpoint are preserved; this is a new run
from zero, with no checkpoint or replay resume. Only this pilot and its ten greedy
evaluations are authorized; no automatic following pilot or full training.

Frozen source `e50b0b0216a658737209b99df6c1515e5367e625` at
`C:/Users/szulc/.codex/worktrees/tmrl-sd-sac-stability-runtime/AITrackmania`.
The explicit experimental variant uses soft-Q forward-KL actor fitting, actor
LR0.0003, temperature LR0.0001 and alpha bounds[0.01,0.2]. Target entropy0.8,
beta0.0005, critic LR0.0003, reward/map/model/GNN/replay/UTD remain unchanged.
Multiple stability controls change together; this pilot cannot isolate their
individual effects or establish canonical SD-SAC performance.

Code regressions: 93 CPU checks including the formerly skipped synthetic loops
passed, then the expanded behavior-entropy module (including both canonical and
forward-KL terminal learning and checkpoint round-trip) passed16 checks. Ruff
passed; 30 external guard/candidate/closure checks and validate-only passed.
This is code evidence, not driving qualification.

Actual queue `artifacts/tmrl-test-comparison/queue-sd-sac-stability-20261006`;
run `tmrl-repair-sd-sac-stability-s17`, W&B `ufoxma8r` API running.
START12:55:15.425207UTC=14:55Warsaw, SAVE16:45:15.425207UTC=18:45Warsaw,
HARD16:55:15.425207UTC=18:55Warsaw. Runner61800(created1791291314.4029312),
guard38544; learner66424 ready, actor51072 registered and collecting.
Real preflight270ms/33fields/protocol2/correctUID.

Train145408/max11400 followed by10greedy/max2100 within14400 seconds;
never extend or restart. The gate still requires >=8/10 finishes plus complete,
finite/fingerprint/SHA-valid, drained/accounted training and ten complete trials
with correctUID, all timing maxima<=100ms, no errors and skip accounting.
No driving result exists yet; full SD-SAC remains BLOCKED. QR PASS29/30,
median54.3s remains preserved. Evidence includes plan/source-freeze,
AUTHORIZATION/RESUME-AUTHORIZATION and actual-launch-receipt.json.


Polecenie „to napraw to wszystko” wykonano w editable; trening, ewaluacja jazdy i nowe
piloty nadal wymagają osobnej zgody. Automatyzacja PAUSED, STOP zachowany, Full SD BLOCKED.
Naprawiono zapis tempa aktora, dodano stabilne log-softmax, osobną regulację temperatury
z opcjonalnymi granicami, kontrolę opcji/optimizerów checkpointu, jawny opcjonalny
wariant soft-Q forward-KL oraz diagnostykę błędu krytyka i zapadania polityki.
SD-SAC wymaga n_step1; poprawiono również generatory i przykład dokumentacji.
91 testów kodu CPU passed (2 długie pętle syntetyczne pominięte), generator7 ponownie
passed, Ruff i mypy passed. Nie uruchomiono żadnego nowego eksperymentu jazdy/treningu.

Odczyt istniejącego cp25112: 31 kompletnych epizodów, terminal Qmean1.6086 wobec
rzeczywistej ostatniej nagrody −2.0518 (bias3.6604, RMSE4.1004). Wagi pozostają bez zmian;
kod nie daje jeszcze dowodu naprawionej kalibracji lub jazdy. Analiza inference-only,
bez learnera/optimizer steps/kontrolera; SHA/fingerprint i frozen runtime zachowane.
Szczegóły: [SD_SAC_CODE_REPAIR_20261006.md](SD_SAC_CODE_REPAIR_20261006.md).
Poniższe „nie poprawiano algorytmu” dotyczy wcześniejszego etapu analizy; starsze
ACTIVE i zgody na autonomiczne starty pozostają odwołane.

## SD-SAC zatrzymany na żądanie człowieka — 6 października 14:08 Warsaw

Nadrzędna instrukcja: zatrzymać SD-SAC, przeanalizować poprawki, **nie uruchamiać
nowego treningu, testu jazdy, ewaluacji ani pilota bez wyraźnego nowego pozwolenia**.
Wcześniejsze upoważnienia do autonomicznych pilotów są odwołane. Automatyzacja PAUSED.

`queue-sd-sac-actorlr0009-20261006` otrzymała nowy STOP. Trening zamknął się normalnie
12:08:20 UTC, returncode0, guard forced[]. Wszystkie procesy należące do próby zamknięte;
W&B `1rd6oj0m` finished (API). Zachowano cp25112,
SHA `37188c52fc549c26f4c31b792678a10fa3308b1cd32666c76dd7d0342999577f`.
112021/145408 transitions, 25112 updates, credit393.25,
earned=accounted25505.25, finite i fingerprint match. Budżet NIE ukończony,
credit NIE drained. Ewaluacja jazdy NIE rozpoczęta, brak wyniku 0/10 lub PASS dla tej próby.
Full SD nadal BLOCKED; QR29/30, mediana54.3s pozostaje zachowanym wynikiem.

Dowody: BASE/queue-sd-sac-actorlr0009-20261006/human-stop-receipt.json oraz
BASE/sd-sac-stopped-analysis-20261006/saved-checkpoint-inspection.json.
Odczyt istniejącego checkpointu na CPU: zero optimizer steps, zero learner updates,
bez środowiska/kontrolera/nowych eksperymentów. Zamrożone runtime bez zmian.
Poniższe wpisy ACTIVE i zgody na następne piloty są historyczne.

## Świeży pilot actor LR9e-4 uruchomiony — 6 października 12:20 Warsaw

`queue-sd-sac-actorlr0009-20261006`, fresh `tmrl-repair-sd-sac-actorlr0009-s17`.
Start 10:20:15.518330 UTC, SAVE14:10:15.518330 UTC (16:10Warsaw),
HARD14:20:15.518330 UTC (16:20Warsaw), cap14400s bez przedłużenia.
145408 transitions/max11400s, potem 10 greedy/max2100s.
Nowy frozen runtime `C:/Users/szulc/.codex/worktrees/tmrl-sd-sac-actorlr-runtime/AITrackmania`
commit `c72591eda6415bd36ee18e5c4fe8eceddacd4e9c`, nie edytować podczas próby.
Zmiana względem best target.8/beta.0005 wyłącznie actor_learning_rate9e-4;
krytyk i temperatura3e-4, nagroda/model/replay/UTD bez zmian. Domyślny parametr
zachowuje dotychczasowe zachowanie; starych checkpointów nie wznawiano.

33 testy algorytmu, 24 guard/candidate/closure tests, Ruff i validate-only passed.
Actual preflight240ms/33 fields/UID ready. Runner66756 created1791282014.9514704,
launcher65000, guard69604, trainlauncher66748, learner45364, actor64172 registered.
W&B `1rd6oj0m` running potwierdzony API; wcześniejsze `awhtibso`/`la0tyfwk` finished.
Przypięto poprzednią normalną closure, dokładne receipts treningu/oceny i audytu;
QR pins i stare STOP hash/mtime zachowane, nowe STOP zawsze wygrywa.
Full SD BLOCKED do rzeczywistej bramki >=8/10 i wszystkich warunków technicznych.
Nadzór ACTIVE, brak automatycznego pełnego treningu.

## Target 1.2: rzeczywista bramka FAIL; następna hipoteza aktora — 6 października 12:16 Warsaw

Ocena fresh02 zakończyła się normalnie 10:11:54.813 UTC, guard closure forced=[];
wszystkie jej procesy zamknięte. 0/10 finishes, średni postęp 0.673439%, no_progress.
1457/1457 pomiarów valid, max/p99 60ms, bez błędów kontrolera/telemetrii,
skips3120/max6; checkpoint SHA niezmieniony. Wszystkie warunki techniczne bramki
spełnione, jazda FAIL. Wynik target1.2 gorszy niż target.8 beta.0005 (48.4221%).

Audyt CPU exact frozen4706, 1024 replay states, threads2/CUDA-1, bez kontrolera
ani aktualizacji learnera: na starcie aktor wybiera akcję5 (pełny skręt z brake tap)
z prawdopodobieństwem około .969, krytyk akcję41 (prosto z brake tap).
1000 kroków niezależnej kopii głowy przy stałym Q podniosło zgodność aktor/Q
z .340 do .804/.807 dla beta0/.0005. Usunięcie kotwicy daje podobny wynik;
próba nie zapisuje polityki i nie dowodzi poprawności krytyka ani jakości jazdy.

Następna hipoteza: wrócić do lepszego target.8/beta.0005 i przetestować wyłącznie
szybsze dopasowanie aktora (actor LR9e-4 zamiast3e-4). Krytyk i temperatura
zachowają LR3e-4. Dodano opcjonalny parametr SD-SAC z domyślnym zachowaniem
bez zmian; nowa próba wymaga świeżego treningu i osobnego zamrożonego źródła.
Dowody: `artifacts/tmrl-test-comparison/sd-sac-target120-audit-20261006`.
Full SD BLOCKED; QR PASS zachowany. Poniższy ACTIVE fresh02 jest historyczny.

## Target 1.2: nowa ocena zachowanego checkpointu uruchomiona — 6 października 12:09 Warsaw

Po ręcznym przywróceniu wtyczki potwierdzono protokół 2 i właściwy UID.
Świeża, osobna kolejka `queue-sd-sac-target120-eval-fresh02-20261006` ocenia
checkpoint 33859 SHA `a7e7aadb5d6860263c4f7c37b5bfe3f5ab19d6d8aa1d31e95242efa07dced16c`.
Tylko 10 przejazdów/max2100s, bez learnera i bez ponownego treningu.
Stary trening pozostaje zakończony, wygasłe recovery01 pozostaje nieuruchomione;
przypięto jego dowody i pierwotne zamknięcie. Nie przedłużono jego limitu.

Nowy plan ma osobny cap do 3000s: start 10:09:44.076605 UTC,
SAVE 10:49:06.272848 UTC, HARD 10:59:06.272848 UTC.
8 testów i validate-only passed; pierwszy testowy błąd dotyczył ścieżki tymczasowej
pytest, poprawiono wyłącznie miejsce plików testowych poza frozen runtime.
Rzeczywisty preflight 300ms/33 fields/UID ready. Runner 67284, guard 35756,
evaluation launcher 68372; W&B `awhtibso` online. Full SD BLOCKED do wyniku bramki.
QR PASS zachowany, źródła frozen4706 bez zmian, brak automatycznego full training.
Poniższe stany oczekiwania na ręczne przywrócenie wtyczki są historyczne.

## Target 1.2: limit recovery minął bez uruchomienia oceny — 6 października 11:58 Warsaw

Pierwotny HARD 09:48:43.413685 UTC minął. Przygotowana kolejka
`queue-sd-sac-target120-eval-recovery01-20261006` nie została uruchomiona:
brak launch receipt, learnera, kontrolera i oceny. Kontrola po HARD nie wykazała
procesów Python ani nasłuchu telemetrycznego 9000/9001. Nie przedłużono limitu.
To **brak wyniku jazdy**, nie 0/10; checkpoint 33859 i kompletne dowody treningu zachowane.

Pulpit i gra zostały wcześniej przywrócone, oryginalna mapa załadowana w walidacji.
Aktualną blokadą jest odrzucenie TrackmaniaRL_Connect przez Openplanet:
`Plugin is not suitable for the current signature mode. It requires School mode.`
Poproszono o ręczne przywrócenie School mode i załadowanie wtyczki; ustawienia
bezpieczeństwa nie są zmieniane automatycznie. Po tej interwencji potrzebny jest
nowy, uzasadniony i ograniczony plan; wygasłego recovery nie uruchamiać.
Full SD nadal BLOCKED, QR PASS zachowany, nadzór ACTIVE. Dowód:
`artifacts/tmrl-test-comparison/queue-sd-sac-target120-eval-recovery01-20261006/deadline-expiry-receipt.json`.
Poniższe wcześniejsze stany i możliwość uruchomienia recovery przed HARD są historyczne.

## Target 1.2: trening kompletny, ocena jeszcze nie rozpoczęta — 6 października 11:10 Warsaw

Nadrzędny stan: trening `tmrl-repair-sd-sac-target120-s17` zakończył się o 08:50:54 UTC.
145437/145408 transitions, 33859 updates, credit .25, earned=accounted 33859.25;
checkpoint finite/fingerprint/accounted/drained/budget_complete. SHA
`a7e7aadb5d6860263c4f7c37b5bfe3f5ab19d6d8aa1d31e95242efa07dced16c`.
Ocena nie ruszyła: preflight o 08:51:09 UTC odrzucił reset, który nie wrócił do startu.
To **brak wyniku jazdy**, nie 0/10. Normalna closure o 08:51:09.665 UTC, forced_owned_pids=[];
runner/guard/learner/actor zamknięte. Dowody starej kolejki pozostają zachowane.

Przygotowano osobną ocenę tego samego checkpointu:
`artifacts/tmrl-test-comparison/queue-sd-sac-target120-eval-recovery01-20261006`.
Validate-only i 8 testów zabezpieczeń passed; tylko 10 ocen/max2100s, bez learnera.
Przypięto poprzednie receipts/source/checkpoint/QR oraz STOP i pierwotne deadlines:
SAVE 09:38:43.413685 UTC, HARD 09:48:43.413685 UTC. Bez przedłużenia cap.
Recovery **nie uruchomiono**. Gra została zamknięta; ponowne uruchomienie nie utworzyło procesu.
Windows zwrócił `GetCursorPos: access denied 0x80070005`; poproszono użytkownika
odblokowanie pulpitu/zamknięcie ewentualnego okna systemowego. Przyczyna resetu nieudowodniona.
Full SD nadal BLOCKED; QR 29/30 median54.3 PASS zachowany. Monitor ACTIVE.
Poniższe wcześniejsze stany ACTIVE target120 są historyczne.

## SD-SAC beta .0005 nie zdał; świeży target 1.2 — 6 października 07:48 Warsaw

Poprzedni `queue-sd-sac-anchor0005-20261006` zakończył się normalnie o 05:36:38 UTC.
Wszystkie jego owned procesy zamknięto bez force-kill. Trening osiągnął 145429/145408 transitions,
33857 updates, credit .25, earned=accounted 33857.25; kompletny/drained/accounted/finite/fingerprint checkpoint.
SHA `f3b701086fa43071edcdf212ff7c6d4259a7c0fe8f494556e834c6c3caa383a3`.
Dziesięć ocen greedy: **0/10 met**, mean progress48.4221%; siedem prób około41%, trzy około65%.
12140/12140 timing valid, max60/p9950ms, no controller/telemetry errors, skips23928/max6.
W&B train6r1s96ly/evalg3248vg6 finished. Wszystkie techniczne warunki bramki spełnione; jazda FAIL.
Full SD pozostaje **BLOCKED**. Nie wznowiono checkpointu ani starej kolejki.

CPU audit dokładnego checkpointu w frozen4706/root PYTHONPATH/threads2/CUDA hidden, zero learner updates/no controller:
anchor gradient .001220 vs value .011101, cosine +.5765; projected cancellation -.0633.
Poprzednia hipoteza konfliktu anchor nie opisuje już tego checkpointu. Entropymean .720/maxprob .790,
actor/Q greedy agreement .302; te wartości nie dowodzą poprawności krytyka ani jakości jazdy.
Niezależne actor-head copies, frozen Q/encoder,1000 kroków,beta.0005: alpha.004555 vs.01 daje
entropy.8997 vs1.2926/maxprob.6752 vs.5841 przy expectedQ cost .00343. Bez zapisu polityki/checkpointu/jazdy.
Wyższy target1.2 ma sprawdzić hipotezę przedwczesnego skupienia polityki i niedostatecznej eksploracji.
Pierwszy audit z niewłaściwym PYTHONPATH nie przeszedł walidacji spec przed update; zachowano log.
Powtórzenie z exact frozen root/fingerprint/SHA przeszło. Nie użyto wyników z main jako dowodu.

NOWY `queue-sd-sac-target120-20261006`, fresh `tmrl-repair-sd-sac-target120-s17`, frozen4706a06b bez zmian.
Jedyna zmiana learning: target entropy .8→1.2; beta.0005/reward/model/GNN/LR/replay/update ratio unchanged.
145408/max11400s +10greedy/max2100s, całkowity cap14400s od tego świeżego startu; bez przedłużenia poprzednika.
Start05:48:43.413685UTC=07:48Warsaw, SAVE09:38:43.413685UTC=11:38Warsaw,
HARD09:48:43.413685UTC=11:48Warsaw. Launcher47728/runner62204created1791265722.9196036,
guard55592,trainlauncher67920/learner29452/actor66932,W&Bla0tyfwk. Preflight280ms/33fields/właściwy UID ready.
Learner ready, actor registered/collecting; to nie wynik bramki jazdy.
28 guard/strict candidate/previous receipt closure tests passed, validate-only passed. Początkowy błąd temp-root
oraz stary regex komunikatu w skopiowanym teście poprawiono w nowym external helper/test; final28passed.
Zachowano pinned poprzednie receipts/normal closure/QR reservation i plan; QR29/30median54.3PASS nie powtarzaj.
Nowy STOP wygrywa; stare pinned STOP hash/mtime unchanged. Nie ma drugiego kontrolera.

Bramka niezmieniona: >=8/10met oraz complete/drained/accounted/finite/fingerprint/SHA,
10complete validUID/alltimingmax100ms/noerrors/skips. Full/manual readiness nadalBLOCKED.
Nadzór15minACTIVE; podSAVE pozostanie wturnwaitmax60s do closure/HARD, bez procesów learner/controller poHARD.
Autonomiczna uzasadniona diagnoza poFAIL, bez full/miesięcznych autostartów/sharedreward/override/replayreset.
Dowody: artifacts/tmrl-test-comparison/sd-sac-anchor0005-audit-20261006/ oraz obie zachowane kolejki.
Poniższe starsze stany/deadlines są historyczne.

# Przygotowanie trzech seedów — 4 października 2026

## SD-SAC beta .005 nie zdał; świeży beta .0005 — 6 października 04:18 Warsaw

SD recovery01 zakończył się normalnie o 04:09:53 Warsaw; guard zamknął wszystkie owned procesy bez force-kill.
Trening osiągnął 142346/145408 transitions, 32540 updates; credit 546.5, earned=accounted 33086.5.
Checkpoint finite/fingerprint, SHA `0efa7b27cc6aacc5bf67696d2d85ab7a78b4230bab74d896d6b2c5a845259498`.
Budżet niekompletny/niedrained: niekwalifikowany. Dziesięć greedy ocen: **0/10 met**, średni postęp 22.0199%;
6684/6684 timing valid, max/p99 50ms, brak controller/telemetry errors, skips 7930/max6.
To częściowy postęp względem beta .5 (3.5517%), bez dowodu użytecznej jazdy. Full SD pozostaje **BLOCKED**.

CPU audit dokładnego checkpointu w frozen 4706, threads2/CUDA hidden, zero learner updates/no controller:
na 1024 replay stanach anchor gradient .023843 vs value .021992, cosine -.7243, projection cancelled .7849.
Q clipping blokował ~.1–.2% stanów. Lokalny konflikt utrzymuje się, ale audyt nie dowodzi poprawności krytyka.
Niezależne kopie samego actor head/frozen Q po 1000 krokach: beta .005 entropy1.9345/maxprob.4442,
beta .0005 entropy.8965/maxprob.6935, expected Q +.0088. Bez zapisu polityki/checkpointu lub jazdy;
ta hipoteza wymaga rzeczywistego pilota i ocen.

Na podstawie bezpośredniej autoryzacji dalszych autonomicznych napraw uruchomiono NOWY świeży test
`queue-sd-sac-anchor0005-20261006`, run `tmrl-repair-sd-sac-anchor0005-s17`, frozen 4706a06b bez zmian.
Jedyna zmiana learning: beta .005→.0005; target .8, reward/model/GNN/LR/replay/update ratio bez zmian.
145408 transitions; osobny limit train11400s zamiast10800s uzasadnia zaobserwowane ucięcie poprzednika;
10greedy/max2100s; CAŁY cap14400s, SAVE600s przed HARD. Starego pilota nie przedłużono ani nie wznowiono.
Start02:18:21.154620UTC=04:18Warsaw, SAVE06:08:21.154620UTC=08:08Warsaw,
HARD06:18:21.154620UTC=08:18Warsaw. Actual preflight250ms/33fields/właściwy UID ready.
Runner58428/launcher54820, guard52900, trainlauncher67476, learner67128/actor63900.
26 testów guard/strict candidate limits/previous closure passed; validate-only passed.
QR nadal PASS29/30/median54.3; pinned QR reservation/plan i receipts pozostają niezmienione.
Nowy STOP wygrywa; stare superseded STOP mają te same przypięte hash/mtime. Żadnego drugiego kontrolera.

Bramka SD niezmieniona: pełny/drained/accounted/finite/fingerprint/SHA checkpoint, 10 ważnych ocen,
>=8/10 met, timing wszystkich max100ms, brak errors i raport skips. Do tego czasu full/manual readiness BLOCKED.
Nadzór ACTIVE15min; dalsza uzasadniona naprawa po FAIL, bez automatycznych miesięcznych startów.
Dowody: artifacts/tmrl-test-comparison/sd-sac-iterative-repair-20261006/{checkpoint-audit.json,actor-anchor-probe.json,next-hypothesis-decision.json}
oraz oba zachowane queue receipts. Poniższe starsze stany są historyczne.


## QR bramka zaliczona, SD-SAC recovery — 6 października 01:03 Warsaw

QR retry02 ukończył 145448 przejść /33862 aktualizacje, earned=accounted=33862,
credit0, finite, fingerprint i potwierdzony SHA
6530a914dfe24037ce967241b4eaa72ea2b8609d6be467a87744376e317af5ab.
Kompletne 30 greedy: 29/30 met, median54.30 s, mean54.233103 s, best52.95 s,
32288/32288 ważnych pomiarów, max60/p9950 ms, bez błędów controller/telemetry,
skips46685/max6. qr-comparison-decision.json: gate_passed=true wobec baseline
26/30 i median56.915 s. To wynik jednego seeda/polityki; nie dowodzi globalnej
optymalności ani ukończenia miesięcznego treningu lub kwalifikacji trzech seedów.
Normalne zamknięcie kolejki 22:50:35 UTC, guard closed normally/forced[].

Pierwsze przejęcie SD-SAC retry02 nie rozpoczęło learnera: confirm_ready odrzucił
player_not_ready. Odczyt okna wykazał końcowy ekran ostatniej walidacji QR.
Po potwierdzeniu ekranu i wybraniu widocznego Improve przywrócono local player
na starcie. Nie zmieniono mapy, nagrody ani frozen sources; stara kolejka,
failure i normalne guard closure zachowane. Read-only/checks i bounded probe
potwierdziły UID oqIJ5rQDRrNwLPTh9H2p_W4tLof, 33 pola i race290 ms.

NOWY recovery queue-sd-sac-anchor005-recovery01-20261006 prowadzi tylko pozostałe
SD-SAC145408/max10800 +10greedy/max2100 na4706a06b, beta.005/target.8.
Fresh run tmrl-repair-sd-sac-anchor005-s17-retry02-recovery01, bez starego cp.
Runner38704, launcher58684, learner43504, actor28068, guard66512.
Start23:03:00 UTC (01:03 Warsaw), rzeczywisty preflight210 ms/33fields/UIDready.
ZACHOWANO pierwotny SAVE2026-10-06T02:40:37.901224UTC (04:40 Warsaw) i
HARD02:50:37.901224UTC (04:50 Warsaw), nie odnowiono czterogodzinnego cap.
Budżet recovery13657.145471 s; helper Budget przyjmuje wcześniejszy deadline,
odrzuca zmianę deadline oraz recovery już uruchomionego learnera/force closure.
24 testy guardów i nowych ograniczeń przeszły; validate-only passed.

SD-SAC nadal BLOCKED: bramka wymaga pełnego/drained/accounted/finite cp,
fingerprint/SHA, 10 kompletnych ocen z prawidłowym UID/timing max100 ms,
bez błędów, raport skips i >=8/10 met. Aktualny start nie jest wynikiem jazdy.
PPO historyczne3/10 nadal ogranicza pełny start. Monitor ACTIVE15min;
po SD FAIL kontynuować uzasadnioną autonomiczną diagnozę świeżym bounded pilotem,
bez automatycznych miesięcznych startów; każdy nowy STOP nadrzędny.


## Autonomiczne wznowienie po pauzie — 5 października 22:12 Warsaw

Użytkownik odwołał pauzę: „Ogarnij wszystko sam, ja idę spać”, następnie
„Nawet jak będzie problem to sam to ogarnij”. Nowe ograniczone próby retry02
zachowują checkpoint zatrzymanego QR retry01 i wszystkie stare STOP jako dowody.
RESUME-AUTHORIZATION przypina ich hash i mtime; każdy nowy lub dotknięty STOP zatrzymuje.
Nie wznowiono starego runnera ani checkpointu na innym kodzie.

Po ponownym uruchomieniu gry przywrócono obraz i istniejącą wtyczkę telemetrii.
Pierwsze otwarcie kopii mapy w edytorze zgłaszało brak startu; bez zapisywania
zmian zamknięto ją i otwarto oryginalny zapis My Maps/tmrl-test.Map.Gbx.
Gra pokazuje VALIDATED i start, session protocol potwierdza oczekiwany UID
oqIJ5rQDRrNwLPTh9H2p_W4tLof oraz ready. Dokładnej przyczyny pierwszego otwarcia
nie udowodniono. Nie zmieniono geometrii, nagrody ani zamrożonych źródeł.

QR queue-qr-repair-retry02-20261005 rozpoczęty 20:12:16 UTC (22:12 Warsaw),
SAVE 23:32:16 UTC (01:32 dnia 6 października), HARD 23:42:16 UTC (01:42).
Cap 12600 s, 145408/max8100 + 30 greedy/max2700, źródła c4ec36a0.
Bramka pozostaje >=29/30, median<=56.915 s, complete/drained/accounted/finite/
fingerprint/SHA i valid timing max100/noerrors/skips. Nie ma jeszcze oceny jazdy.
31 testów nadzoru QR przeszło w tym wznowieniu; plany QR i SD validate-only przeszły.

SD queue-sd-sac-anchor005-retry02-20261005 czeka pasywnie, bez drugiego kontrolera,
na normalne zamknięcie obu etapów nowego QR, kompletny checkpoint/30 ocen z matching
SHA, normalny guard, zamknięte procesy, wolny mutex i brak nowego STOP.
Pasywny limit 23:47:16 UTC (01:47 Warsaw). Potem fresh beta.005/target.8 seed17,
4706a06b, 145408/max10800 +10greedy/max2100, cap14400 s od faktycznego startu,
SAVE600 przed HARD. >=8/10 plus kompletne bramki wymagane; full SD pozostaje BLOCKED.
PPO ostatnio 3/10 i SD beta.5 0/10 nadal są ograniczeniami, nie nowymi sukcesami.

Monitor ACTIVE15min kontynuuje autonomiczne diagnozy i uzasadnione ograniczone
nowe piloty do użytecznej jazdy SD-SAC albo nowego STOP. Żadnych automatycznych
miesięcznych startów, zmian wspólnej nagrody lub obejścia fingerprintu.


## Nowe ograniczone QR → SD-SAC uruchomione — 5 października 17:04 Warsaw

Użytkownik potwierdził: **„Tak, nowe ograniczone QR → SD-SAC”** po pytaniu
o nowy pilot QR do 3,5 godziny i osobny SD-SAC do 4 godzin. To nowa próba,
bez przedłużania lub restartowania kolejek zakończonych awarią.
Poprzednie awarie i wyniki PPO 3/10 oraz SD-SAC 0/10 pozostają zachowane.

Lokalny gracz i telemetria wróciły. Preflight wymaga gotowego gracza przed wejściem,
pomija bezwarunkowe potwierdzenie finish-screen, łączy telemetrię po resecie
i odrzuca brak powrotu na start (race_time > 2500 ms). Osobny test live potwierdził
reset do 210 ms; właściwy QR preflight do **240 ms**, właściwy UID i 33 pola.
Nie udowodniono, który element dawnego menu spowodował utratę gracza.
Zmiany dotyczą wyłącznie zewnętrznych helperów; zamrożone algorytmy są niezmienione.
Pięć testów kontraktu preflight, 31 testów nadzoru QR i 20 SD-SAC przeszło;
oba plany przeszły validate-only.

QR działa w `queue-qr-repair-retry01-20261005`: launcher 43888 / runner 51928,
train launcher 58120, learner 53772, actor 53428; W&B **5x4ydddg online**.
Fresh run `tmrl-repair-qr-loss-sum-s17-retry01` zbiera rollouts, bez gotowego wyniku.
Start **17:03:16 Warsaw**, SAVE **20:23:16**, HARD **20:33:16**, cap 12600 s.
Plan zachowuje 145408 kroków / max 8100 s i 30 greedy / max 2700 s,
zamrożony kod `c4ec36a0`. Bramka pozostaje >=29/30 i mediana <=56,915 s,
pełny/drained/accounted checkpoint, finite/fingerprint/SHA, ważny timing max100 ms,
brak błędów i raport pominiętych ramek. Naprawa matematyczna nie gwarantuje jazdy.

Nowy SD-SAC `queue-sd-sac-anchor005-retry01-20261005`: launcher 58956 /
waiter 30764, created_time 1791212631.7032077; czeka **bez kontrolera i launch.json**.
Przypina nowy QR plan/rezerwację, wymaga obu ukończonych etapów, pełnego finite/drained
checkpointu, 30 ocen z matching SHA, normalnego zamknięcia guardu/procesów,
braku STOP i wolnego mutexu. Pasywne oczekiwanie kończy się **20:38:16 Warsaw**.
Po rzeczywistym przejęciu fresh beta0,005 seed17, 145408/max10800 s + 10 greedy/max2100 s,
cap14400 s, SAVE600 s przed HARD. Źródła `4706a06b` są zamrożone i niezmienione.
Wymagane >=8/10 i pozostałe pełne bramki; full SD-SAC nadal BLOCKED.

Aktualne wskaźniki active-queue/pending-queue/sd-sac-pending-queue opisują te nowe próby;
czytaj faktyczne statusy i launch. Nadzór pozostaje ACTIVE co15 min do zakończenia
autoryzowanego celu SD-SAC albo jawnego STOP. Żadnych automatycznych miesięcznych
startów, zmian nagrody, fingerprint override ani wznawiania starych checkpointów.


## Aktualny wynik i blokada środowiska — 5 października 15:29 Warsaw

Ta aktualizacja zastępuje poniższe historyczne informacje o działającym PPO
i oczekujących procesach. PPO ukończył diagnostyczny uczony prefix cp70:
143360 kroków / 70 aktualizacji, finite i zgodny fingerprint; to nie jest pełny trening.
Dziesięć ocen dało **3/10 met**, medianę 55,34 s wyłącznie ukończonych prób.
Wszystkie pomiary timingu 6681/6681 są ważne, max 60 / p99 50 ms, bez błędów;
10778 pominiętych ramek, max 5. Timing treningu poprawił się do max 70 ms.
Skuteczność jazdy nadal nie przechodzi progu 8/10; krótszy czas trzech finisherów
nie dowodzi poprawy polityki względem wcześniejszego 8/10. Nie odblokowano pełnego startu.

QR przejął kolejkę o 15:07:14, lecz preflight zakończył się o 15:07:31:
po resecie nie otrzymano ramki telemetrii w 10 s. **Nie powstał learner, checkpoint
ani katalog treningu QR.** Jego procesy i guard zamknęły się normalnie, bez forcekill.
Oczekujący SD-SAC beta 0,005 prawidłowo przerwał przejęcie po awarii QR i również
nie rozpoczął treningu. Poprzedni SD-SAC pozostaje 0/10 i full BLOCKED.

Osobny ograniczony probe połączenia dopiero po resecie też zakończył się timeoutem.
Pięć testów kontraktu z atrapami przeszło; nie są dowodem usunięcia rzeczywistej awarii.
OpenPlanet potwierdza właściwy UID, ale `confirm_ready` zwraca `player_not_ready`.
Okno gry pokazuje przejazd bez HUD-u; dostępne próby wejścia do menu nie przywróciły
lokalnego gracza. Poproszono użytkownika o ponowne uruchomienie walidacji mapy.

Dowody są w `artifacts/tmrl-test-comparison/qr-preflight-recovery-20261005/`
(`recovery-status.json`, `session-state.json`, `probe.log`, archiwum pierwotnej awarii).
Wskaźnik active-queue nadal opisuje zakończoną awarią QR, a nie żywy kontroler;
zawsze czytaj status kolejki. Wszystkie obserwowane procesy zamknięto.
Nie uruchamiaj ponownie użytych runnerów ani zakończonych etapów. Ewentualne wznowienie
wyłącznie pozostałego QR wymaga osobnego jawnego recovery, gotowej telemetrii,
braku nowego STOP i wolnego mutexu, z zachowaniem pierwotnego **SAVE 18:27:14 /
HARD 18:37:14 Warsaw**. Limit nie zaczyna się ponownie od recovery.
Nowa rezerwacja SD-SAC musi wskazywać rzeczywistego poprawnie zakończonego poprzednika.
Cel dalszej naprawy SD-SAC i monitor ACTIVE pozostają obowiązujące; żadnych automatycznych
miesięcznych startów, zmian wspólnej nagrody ani aktualizacji zamrożonych źródeł.


## Dalsza naprawa SD-SAC — 5 października 13:33 Warsaw

Użytkownik polecił: „No to poprawiaj SD SAC tak długo aż będzie sensowny”.
Nadzór pozostaje ACTIVE również po samym PPO/QR, jeśli SD-SAC nie spełni bramki.
Ostatni SD-SAC nadal ma **0/10 met** i niepełny budżet; pełne treningi są zablokowane.

Audyt CPU zgodnego checkpointu na 1024 stanach wykazał, że przy beta 0,5 kara
utrzymująca entropię zachowania znosi niemal cały gradient preferencji wartości
(cosinus −0,9295; udział zniesiony 1,0059). To lokalna diagnoza, nie dowód błędu
matematycznego całego algorytmu. Nowa hipoteza zmienia tylko beta **0,5 → 0,005**;
wspólna nagroda, model, target entropy 0,8 i budżet 145408 pozostają takie same.
Kod dodaje diagnostykę prawdopodobieństw akcji i zgodności z krytykami oraz
usuwa zbędny graf krytyków z kroku aktora, zachowując cel i gradient aktora.

Nowy pilot faktycznie oczekuje w `queue-sd-sac-anchor005-20261005`:
**launcher 48368 / waiter 57588**, created_time 1791199959.0079837, bez kontrolera
i `launch.json`. `sd-sac-pending-queue.json` wskazuje rezerwację; właściwy stan
odczytuje się z kolejki. Frozen runtime `tmrl-sd-sac-anchor-runtime` ma źródła
`4706a06b0eea5ba3fe0d2c6aa1ad641ed29aca84`, opublikowane atomowo na obu branchach.

Kolejność pozostaje **PPO → QR → nowy SD-SAC**. PPO działał o 13:33 online:
49152 kroki / 24 pełne aktualizacje rolloutu, finite, bez awarii.
Terminy PPO i QR oraz ich zamrożone źródła są bez zmian. SD-SAC przejmuje dopiero
po normalnym końcu QR, pełnym rozliczonym checkpointcie QR, 30 ocenach, zamknięciu
procesów i guardu, braku STOP i wolnym mutexie. Pasywne oczekiwanie ma limit
**20:09 Warsaw**; nie uruchamia learnera. Po faktycznym starcie SD-SAC:
trening maks. 10800 s, 10 greedy maks. 2100 s, całe zadanie maks. **4 h**,
SAVE 600 s przed HARD. Właściwe daty pojawią się w jego `launch.json`.

Przeszło **106 testów uczenia, 20 testów nadzoru, 42 schematy**, Ruff, mypy
zmienionego źródła i standardowy CPU update/zstd roundtrip rzeczywistej kompozycji
GNN / 78 akcji. To dowody poprawności technicznej, nie nowy wynik jazdy.
Bramka pozostaje: pełny budżet, drained/accounted credit, finite/fingerprint/SHA,
10 kompletnych ważnych prób, **co najmniej 8/10 met**, max 100 ms, brak błędów,
raport pominiętych ramek. Nieudany wynik zachowujemy i diagnozujemy kolejną
uzasadnioną hipotezę w osobnym ograniczonym świeżym pilocie; bez automatycznych
pełnych treningów i bez deklaracji globalnej optymalności lub kwalifikacji 3 seedów.

Szczegóły i ograniczenia: [SD_SAC_ITERATIVE_REPAIR_20261005.md](SD_SAC_ITERATIVE_REPAIR_20261005.md).
Poniższa sekcja z 12:50 zachowuje wcześniejszy wynik i dowody.

## Nowy wynik SD-SAC i przejęcie QR — 5 października 12:50 Warsaw

Świeży SD-SAC target0,8 / beta0,5 / behavior nie zaliczył bramki: **0/10 met**,
średnio **3,552% trasy**. Wszystkie 10 prób mają ważny timing, 1961/1961 pomiarów,
max/p99 50 ms, brak błędów kontrolera i telemetrii; 2440 pominiętych ramek,
maksimum 5. Nie ma potwierdzonej poprawy skuteczności jazdy. Pełny SD-SAC
pozostaje zablokowany, generator i konfiguracje pełnych treningów bez promocji.

Pilot zakończono przez limit czasu o 12:44:29 Warsaw: **135406/145408 kroków**,
30775 aktualizacji, credit576,5, earned=accounted31351,5. Checkpoint jest finite
z potwierdzonym fingerprintem/SHA, ale budżet jest niepełny i credit nie został
wyczerpany. To ograniczenie testu, nie pełny ukończony trening. SHA checkpointu:
`6fdab80112a7e27a5bfd65cd98e01eb20297f647b4b7258045ea726e82154175`.
Dowody: `queue-ppo-sd-sac-repair-20261005/sd-sac-completion.json`,
`eval-sd-sac-evaluation.json`, `sd-sac-repair-decision.json` oraz
`tmrl-repair-diag-sd-sac-s17-benchmark-20261005T104448532426/evaluation.json`.

PPO rozpoczął świeży prefix o **12:47:13 Warsaw**, PID54168, W&B `yygy62p7` online.
Oceniamy zachowany pełny uczony cp70/143360 i 10 przejazdów zgodnie z protokołem.
SAVE16:19/HARD16:29 poprzedniej fazy pozostają bez zmian.

Przed rozpoczęciem QR wykryto dodatkowy błąd pomocnika przekazania kolejki:
czytał `sd-sac-evaluation.json` / `ppo-evaluation.json`, choć poprzednik zapisuje
`eval-sd-sac-evaluation.json` / `eval-ppo-evaluation.json`. Zatrzymano wyłącznie
pasywny pomocnik, przed jakimkolwiek etapem QR, zachowano snapshot w
`passive-wait-handoff-repair-02` i poprawiono nazwy odczytu. Nowa regresja
odtworzyła błąd; po poprawce **31 testów nadzoru** przeszło, w tym rzeczywisty
kontrakt nazw i odrzucanie niepełnego prefixu / brakujących prób / niefinite cp.
Ruff helperów i validate-only planu przeszły.

QR ponownie oczekuje: **launcher8384 / waiter55608**, bez kontrolera i bez
`launch.json` QR. `passive-handoff-recovery.json` i `pending-queue.json` wskazują
nową tożsamość. Źródła QR c4ec36a0 i SD-SAC/PPO266e2629 bez zmian; poprzednik nie
był restartowany. Warunki przejęcia, deadline oczekiwania16:34, własny limitQR
3h30 od rzeczywistego startu i bramki29/30 / mediana≤56,915s pozostają niezmienione.
Matematyczna poprawka QR nadal wymaga świeżego wyniku w grze. Monitor ACTIVE.

Poniższe sekcje zachowują wcześniejsze stany i dowody.

## Przywrócone oczekiwanie QR — 5 października 12:21 Warsaw

Pomocnik oczekujący QR zakończył się o 12:09 z `PermissionError` podczas odczytu
`status.json` poprzedniej kolejki. QR nie otworzył kontrolera, nie rozpoczął
treningu ani oceny. Zachowano pierwotny błąd, logi, rezerwację i helpery w
`queue-qr-repair-20261005/passive-wait-recovery-01`; zapis w
`passive-recovery.json` potwierdza zamknięcie starego pomocnika i brak etapów QR.

Poprawiono tylko odczyty zewnętrznego pomocnika QR i jego przyszłego guardu:
ponawianie trwa najwyżej 5 s, a trwały błąd nadal przerywa pracę. Wszystkie
26 testów CPU nadzoru przeszło, w tym 5 nowych regresji: rzeczywista wyłączna
blokada pliku Windows, ograniczony trwały błąd uprawnień, przerwa podmiany pliku,
odrzucenie uszkodzonego JSON i niedozwolonego długiego ponawiania. Walidacja
planu i Ruff helperów przeszły. Poprzednie 32 testy źródeł i próba QR64 update /
zstd roundtrip pozostają zachowanymi dowodami, nie nowym wynikiem jazdy.

Oczekiwanie przywrócono: launcher **60600**, runner **60380**,
`waiting_for_predecessor`, pusty `runner.err`, bez `launch.json` QR.
`pending-queue.json` wskazuje nową tożsamość; aktywna kolejka pozostaje SD-SAC/PPO
z runnerem 61144 i guardem 3356. Nie zmieniono jej źródeł, helperów ani etapów.
Zamrożone runtime QR c4ec36a0 oraz SD-SAC/PPO 266e2629 pozostały bez zmian.

Zakres i warunki przejęcia są niezmienione: wszystkie cztery etapy poprzednika,
pełny uczony prefix PPO i 10+10 ocen, zamknięte procesy, wolna wspólna blokada,
brak nowego STOP. Oczekiwanie kończy się najpóźniej o 16:34 Warsaw. QR nadal ma
własne 3 godz. 30 min od rzeczywistego startu, SAVE 10 min przed HARD i wcześniej
zapisane bramki 29/30 met / mediana ≤56,915 s. Wynik nowego QR w grze nadal
oczekiwany. Nadzór pozostaje ACTIVE; pełnych treningów nie uruchomiono.

Poniższe sekcje zachowują wcześniejsze stany i dowody.

## Aktualny stan — QR-DQN dołączony po PPO, 5 października 12:05 Warsaw

Użytkownik polecił poprawić QR-DQN i dodać go do kolejki, następnie zezwolił
na późniejsze zakończenie testów, także po19:00, przed prawdziwymi treningami.
Poprawka QR jest opublikowana: **c4ec36a0b6d2eaf948c9eb24a783e344b81c1314**.
Nowy QR używa osobnego frozenruntime `tmrl-qr-repair-runtime`; aktualne
SD-SAC/PPO266e2629 oraz starsze runtime pozostają bez zmian.

Faktycznie dodany etap oczekujący: `queue-qr-repair-20261005`, launcher28636,
runner61196, status `waiting_for_predecessor`, bez kontrolera i bez `launch.json`.
`pending-queue.json` wskazuje QR; `active-queue.json` do zakończenia PPO wskazuje
bieżącą kolejkę SD-SAC/PPO. QR przejmie wskaźnik dopiero po wszystkich czterech
etapach, potwierdzonych wynikach, zamknięciu ich procesów/guardu i sprawdzeniu
wspólnej blokady. Nie uruchamia drugiego kontrolera ani nie ponawia poprzednika.

QR: świeży145408/seed17, max8100s, potem30greedy/max2700s. Nowa faza ma osobny
limit3h30 od faktycznego startu QR i SAVE10min przed HARD. Konkretne godziny
będą dopiero w QR `launch.json`/`effective-deadline.json`;16:29 pozostaje limitem
poprzedniej fazy. Jej prawidłowe zamknięcie nie kończy jeszcze autoryzowanego QR.
Nowy STOP, awaria lub niepełne PPO zatrzymują automatyczne przejęcie.

32testy źródeł +21testów CPU nadzoru i bramki przeszły, Ruff i mypy zmienionego
źródła przeszły. Pełna konfiguracja kompozycji QR64 przeszła standardowy validator
na CPU: syntetyczny update i zstd checkpoint roundtrip, bez gry, kontrolera i
nowego W&B. Bramka QR jest z góry: pełny/drained/finite/fingerprint/SHA pilot,
30ważnychprób/max100ms/noerrors/skips, >=29/30met i mediana<=56,915s. Wynik
jazdy nadal oczekiwany; matematyczna poprawka nie gwarantuje poprawy polityki.

Monitor ACTIVEco15min obejmuje również oczekujący QR. Raport końcowy i PAUSED
nastąpią po wszystkich autoryzowanych etapach lub ich zatrzymaniu/awarii/limicie.
Pełnych treningów nie uruchamiamy; QR, PPO i SD-SAC nie otrzymują deklaracji
optymalnych HP lub gotowości trzech seedów bez rzeczywistych wyników.
Protokół, analiza ograniczeń i punkt odniesienia:
[QR_REPAIR_20261005.md](QR_REPAIR_20261005.md).

Poniższe sekcje zachowują wcześniejsze stany i dowody.

Aktualna oddzielna kolejka naprawcza wystartowała 5 października10:29Warsaw: `queue-ppo-sd-sac-repair-20261005`, nowy frozenruntime266e2629. Świeży SD-SAC target0,8/beta0,5/behavior145408 +10greedy, następnie PPO.01 prefix143360uczonych (STOP145408, LRhorizon2048000) +10greedy. Nowa faza ma limit6h, SAVE16:19/HARD16:29Warsaw; te godziny są nadrzędne dla nowych testów po bezpośrednim przedłużeniu przez użytkownika. Właściwy wskaźnik: `artifacts/tmrl-test-comparison/active-queue.json`; runner61144, niezależny guard3356, W&B `wcsd62q4` online. 203testy kodu +21testów guardówCPU przeszły. Stary fixturepoczątkowegoplanu odłożono: historyczna retencja usunęła cp659; jego nowy start jest zabroniony. Pełny SD-SAC nadal zablokowany, wspólna nagroda bez zmian, bez automatycznych miesięcznych treningów. Live wynik nowych poprawek jest jeszcze w toku.


## Zakończenie nocy i naprawy po przedłużeniu — 5 października

Nocna kolejka ukończyła wszystkie etapy 09:26:36 Warsaw; ostatni QR **26/30 met**,
mediana56,915s/średnia57,124s, max50ms, poprawny timing i brak błędów.
Użytkownik następnie przedłużył pracę poleceniem „Możesz robić dłużej niż do 10
 tylko spraw by PPO i DSAC poprawnie działało”. Wcześniejsze terminy są historią
zakończonej kolejki; nowe ograniczone testy mają własny jawny limit.
Reset nowego kodu zaliczony w grze; poprawki rozgrzewki/timingów PPO i stabilizacji
SD-SAC są testowane przed świeżymi pilotami. Pełny SD-SAC pozostaje zablokowany,
PPO zachowuje entropy0,01. Nagroda pozostaje wspólna; automatycznych miesięcznych
treningów nie uruchamiamy. Pełne wyniki, dowody, ograniczenia i nowy protokół:
[REPAIR_PPO_SD_SAC_20261005.md](REPAIR_PPO_SD_SAC_20261005.md).

Starsze sekcje poniżej opisują kolejne historyczne stany; powyższy stan jest bieżący.

## SAC 30 prób — 5 października, 09:10 Europe/Warsaw

Zachowany stary cp33854 na runtime d0dfe645 ukończył ocenę: **30/30 met**,
średnia **44,366 s**, mediana **44,500 s**, najlepsza **42,500 s**.
Wszystkie 30 prób mają ważny timing: 26 627/26 627 pomiarów, max/p99 **50 ms**,
brak błędów telemetrii/kontrolera. Pominięte ramki **8 021**, maksimum 5.
Zachowujemy pozytywny baseline SAC. To ocena jednej polityki seed17, nie dowód
powtarzalności treningu na trzech seedach ani osiągnięcia celu 37 s.
Dowód: `tmrl-overnight-diag-sac-s17-benchmark-20261005T063406463907/evaluation.json`;
SHA cp `620f9b1aaabb1afda5355b82ec7a1b71412262df6299e37de0bb3aae76f0ea3e`.
W&B `wjlsmn59` finished. Ostatnia ocena QR rozpoczęła 08:57:38 Warsaw, PID56000;
09:10 ukończono 13 prób, 12 met — wynik częściowy. Kampania trwa.
SAVE 10:24 / HARD 10:34 bez zmian; pełnych treningów nie uruchomiono.

## TQC 30 prób — 5 października, 08:40 Europe/Warsaw

Zachowany stary cp33858 na runtime d0dfe645 ukończył ocenę:
**29/30 met**, średnia **44,384 s**, mediana **44,300 s**, najlepsza **43,790 s**.
Wszystkie 30 prób mają ważny timing: 26 222/26 222 pomiarów, max **60 ms**,
p99 **50 ms**, brak błędów telemetrii/kontrolera. Pominięte ramki **8 080**,
maksimum 6. Zachowujemy pozytywny baseline TQC. Ocena potwierdza tę politykę
seeda17, nie powtarzalność treningu na trzech seedach ani osiągnięcie celu 37 s.
Dowód: `tmrl-overnight-diag-tqc-s17-benchmark-20261005T061051859378/evaluation.json`;
SHA cp `6dda21d2d3d5ee119919d66741517f608b2b9e12e63f8764fe4d3ce8aff29a7f`.
W&B `4twuxhg7` finished. SAC rozpoczął 08:34:03 Warsaw, PID52608,
W&B `wjlsmn59`; QR pozostaje w kolejce. Kampania trwa.
SAVE 10:24 / HARD 10:34 bez zmian; pełnych treningów nie uruchomiono.

## Powtarzalność IQN — 5 października, 08:11 Europe/Warsaw

Ocena zachowanego starego cp33854, na zamrożonym runtime d0dfe645, ukończona:
**29/30 met**, średnia ukończonych **49,452 s**, mediana **49,400 s**,
najlepsza **48,370 s**. Wszystkie 30 prób mają ważny timing:
29 587/29 587 pomiarów, max/p99 **50 ms**, brak błędów telemetrii/kontrolera.
Pominięte ramki **15 564**, maksimum 5. Wynik wspiera powtarzalność tej polityki
na seed17; nie dowodzi powtarzalności uczenia na trzech seedach ani celu 37 s.
Nie zmieniamy IQN arbitralnie po dawnym wyniku 1/2: baseline pozostaje.
Pełne nowe starty nadal wymagają wspólnego nowego commita, kalibracji lokalnej,
smoke i 30 końcowych prób dla każdego z seedów 17/29/43.
Dowód: `tmrl-overnight-diag-iqn-s17-benchmark-20261005T054442205511/evaluation.json`;
SHA checkpointu `ad4fabf38558c893ed54546143a47418eb3bc589925d828a95968f5a932d4f95`.
Istniejący runner rozpoczął TQC 08:10:48 Warsaw; SAC i QR pozostają w kolejce.
Kampania nadal trwa, SAVE 10:24 / HARD 10:34 bez zmian.

## Decyzja PPO — 5 października, 07:55 Europe/Warsaw

Porównanie obu wariantów zakończone na wspólnym cp70: **143 360 uczonych kroków /
70 aktualizacji**, pełne prefixy diagnostyczne, skończone checkpointy,
zgodność fingerprint i SHA potwierdzona. Pełnych miesięcznych treningów nie wykonano.

| Współczynnik entropii | Mety | Mediana ukończonych | Średnia ukończonych | Max/p99 kroku | Pominięte ramki |
| --- | --- | --- | --- | --- | --- |
| 0,01 baseline | 8/10 | 56,445 s | 56,585 s | 50/50 ms | 18 648 (max 6) |
| 0 | 7/10 | 57,880 s | 57,673 s | 50/50 ms | 18 173 (max 6) |

Wszystkie 20 prób mają ważny timing i brak błędów kontrolera/telemetrii.
Wariant 0 nie osiągnął minimum 8/10 met, miał niższy odsetek met oraz medianę
około 2,54% wolniejszą, zamiast wymaganych przynajmniej 5% poprawy.
**Zachowujemy baseline 0,01 w generatorze i pełnych konfiguracjach 17/29/43.**
Nie jest to dowód globalnej optymalności ani ukończenie końcowych 30 prób na seed.
Skoki race-clock treningu ponad 100 ms nadal stanowią ograniczenie lokalne.
Dowód decyzji: `queue-overnight-tuning-20261005/ppo-comparison-decision.json`.
Ocena wariantu 0: `tmrl-overnight-diag-ppo-entropy0-s17-benchmark-20261005T053439289662/evaluation.json`,
SHA cp70 `1646e5648265e9b5af3503c40feb4f9b0eb6ea256d2efaf30c05b5d244c29b83`.
Kampania trwa: istniejący runner rozpoczął 30 prób IQN 07:44:38 Warsaw.
SAVE 10:24 / HARD 10:34 bez zmian; pełny SD-SAC nadal zablokowany.

## PPO baseline — 5 października, 05:25 Europe/Warsaw

Świeży PPO entropy coefficient 0,01 zakończył krótki prefix diagnostyczny.
Do porównania wybrano potwierdzony, skończony cp70: **143 360 uczonych kroków,
70 aktualizacji**, processed_transitions=transitions; prefix_complete=true,
full_training_complete=false, wspólny harmonogram LR 2 048 000.
SHA256: `250767107f53a5917d9cc54fb8426539c8d234fd83a8dd01d09faf9bc8660bbb`.
Dziesięć prób greedy: **8/10 met**, średnia ukończonych 56,585 s,
mediana **56,445 s**, najlepsza 55,520 s. Dwie porażki przy około 81,20% postępu.
Wszystkie próby mają ważny timing; 11 068/11 068 pomiarów, max/p99 **50 ms**,
brak błędów kontrolera i telemetrii. Pominięte ramki: **18 648**, maksimum 6.
W treningu występowały skoki race-clock powyżej 100 ms (maksimum 530 ms);
poprawna ocena nie usuwa tego ograniczenia lokalnego treningu.
To wynik jednego seeda i 10 prób diagnostycznych, nie końcowe 30 prób na seed
ani dowód globalnej optymalności. Baseline 0,01 pozostaje ustawieniem domyślnym;
wybór 0 wymaga pełnego porównania zgodnego z ustalonym wcześniej protokołem.
Wariant entropy0 już działa od 05:21:32 Warsaw, W&B `yei92q7a`;
baseline `hymfpseo` potwierdzony finished. Kampania nadal trwa.
Dowody: `queue-overnight-tuning-20261005/ppo-baseline-completion.json` i
`tmrl-overnight-diag-ppo-baseline-s17-benchmark-20261005T031138975953/evaluation.json`.
SAVE 10:24 / HARD 10:34 pozostają nadrzędne; pełnych treningów nie uruchomiono.

## Wynik SD-SAC — 5 października, 03:10 Europe/Warsaw

Pilot celu entropii 2,0 / beta 0 zakończył **145 487 kroków / 33 871 aktualizacji**.
Końcowy cp33871 ma skończone tensory, zgodny fingerprint i rozliczony kredyt
**0,75** (`earned = accounted = 33 871,75`). Ocena zakończona: **0/10 met**,
postęp 4,24–5,17%, średnio 4,42%. Wszystkie pomiary czasu ważne, max/p99 50 ms,
brak błędów telemetrii/kontrolera; 651 pominiętych ramek, maksimum 5.
Warunek minimum 8/10 met NIE został spełniony: **pełny SD-SAC nadal zablokowany**,
hipoteza diagnostyczna nie zostaje promowana do konfiguracji seedów 17/29/43.
Poprawny zapis i sprawny pomiar nie oznaczają skutecznej polityki greedy.
Dowody: `queue-overnight-tuning-20261005/dsac-completion.json` oraz
`tmrl-overnight-diag-dsac-e200-b0-s17-benchmark-20261005T005546181914/evaluation.json`
w `artifacts/tmrl-test-comparison`. W&B trening `2fslk8zh`, ocena `vepn4pyj`.

To wynik częściowy kampanii. Świeży PPO baseline ruszył 02:58:36 Warsaw;
pozostałe etapy prowadzi istniejący runner. Nagroda i zamrożone runtime bez zmian.
Termin zapisu 10:24, twardy koniec 10:34 pozostają nadrzędne.


## Wznowienie nocne — 5 października, 00:50 Europe/Warsaw

Użytkownik odwołał pauzę i zatwierdził **10 godzin łącznie testów i nadzoru**.
Aktualna kolejka to `queue-overnight-tuning-20261005`, wskazana przez
`artifacts/tmrl-test-comparison/active-queue.json`. Start **00:50:18**, zapis
do **10:24**, twardy koniec **10:34 5 października**. Tej granicy nie
wydłużamy. Pełne eksperymenty pozostają do ręcznego uruchomienia.
Limit liczymy od rozpoczęcia pracy o 00:34, włącznie z przygotowaniem i testami
CPU. `effective-deadline.json` oraz osobny watchdog terminu skracają wewnętrzny
limit runnera liczony od uruchomienia kolejki. Watchdog nie otwiera kontrolera.

Przerwana kolejka z 4 października pozostaje zatrzymana; jej STOP i dane
zachowano. Nowa kolejka wznawia SD-SAC z `distributed-update-00000659.pt`:
**12 658 kroków, 659 aktualizacji, kredyt 5,5**, skończone tensory i zgodny
fingerprint potwierdzone. Odtwarza pełny stan, bez resetowania replay.
Nowy segment W&B: [2fslk8zh](https://wandb.ai/dsc-pjatk-warsaw/my-trackmania-agent/runs/2fslk8zh).
Audytowy W&B `dwz2e073` miał wyłącznie setup i zero kroków — nie jest pilotem.

Plan: dokończenie SD-SAC celu 2,0 / beta 0 oraz 10 ocen; świeże PPO baseline
entropii 0,01 i osobno 0,0, każde z maksymalnie 2 h 15 min jazdy i 10 ocenami;
następnie po 30 przejazdów starych końcowych IQN, TQC, SAC i QR, jeśli mieszczą
się we wspólnym terminie. Globalny termin ma pierwszeństwo przed sumą etapów.
Pominięte lub niepełne oceny muszą zostać wskazane w raporcie końcowym.

Oba PPO badają **krótki prefix 145 408 kroków z horyzontem LR 2 048 000**,
nie pełny trening. Pozostałe parametry, seed 17, nagroda i model są identyczne.
Zatrzymanie następuje po potwierdzonym zapisie prefixu lub limicie czasu.
Jeśli zamknięcie zapisze częściowy, nieuczony ostatni rollout, narzędzie jawnie
odrzuca go i wybiera zachowany kompletny checkpoint z równymi licznikami
`processed_transitions` i `transitions`. Przed startem obu PPO ustalono ocenę
na **wspólnym zapisie 70 pełnych rolloutów = 143 360 uczonych kroków**.
Zatrzymanie kolekcji pozostaje przy 145 408; konserwatywna ocena wcześniejszego
zapisu chroni przed nadpisaniem ostatniego checkpointu podczas domykania.
Nie zmieniamy liczników ani zawartości zapisu. Receipt rozróżnia kompletność
wspólnego prefixu od pełnego eksperymentu; `full_training_complete` zawsze false.

SD-SAC zachowuje wcześniej ustalony warunek: cały budżet 145 408, rozliczony
kredyt, finite checkpoint, 10 kompletnych prób, minimum 8 met, ważny timing
wszystkich, max kroku ≤100 ms i brak błędów. Bez spełnienia pozostaje zablokowany.
PPO entropii 0 może być wybrane wyłącznie prowizorycznie, gdy oba prefixy są
kompletne przy wspólnym uczonym budżecie, wszystkie pomiary ważne, brak błędów,
max kroku ≤100 ms, co najmniej 8/10 met, liczba met nie niższa od baseline
i mediana czasu co najmniej 5% lepsza. W przeciwnym razie baseline zostaje;
jeżeli oba warianty zawodzą, nie deklarujemy gotowości i nie ukrywamy wyniku.
Jedno ziarno i 10 prób nie dowodzą optymalności ani przewagi na trzech seedach.

Zabezpieczenia nowego runnera: 19 testów CPU, mutex jednej instancji sterowania,
STOP z zapisem, watchdog twardego terminu, atomowy JSON z ponawianiem,
kontrola zamrożonego commita i potwierdzonego checkpointu. Dwa CPU update /
checkpoint roundtrip PPO przeszły na syntetycznych sekwencjach 64; jazda
używa 2048. Dalsze wyniki będą zapisywane w receiptach i raporcie końcowym.

Podczas preflight wykryto i naprawiono w edytowalnym kodzie pomocnicze problemy:
syntetyczny validator PPO dodawał chronione demonstracje poza pojemność replay
2048; teraz używa jednej pełnej sekwencji on-policy. Fingerprint pakietów
namespace deduplikuje korzenie importów, żeby cwd/PYTHONPATH nie zmieniały
tożsamości tych samych plików. To zmiana nowego kodu zespołu: istniejące piloty
pozostają na zamrożonym `8f440075`, stare checkpointy na `d0dfe645`; żadnych
wznowień na nowym kodzie ani obchodzenia fingerprintu. Nagroda bez zmian.
Regresje po poprawkach: **48 testów przeszło**, Ruff i mypy bez błędów.
Dwa testy PowerShell wywołujące launcher pominięto w ponownym przebiegu,
ponieważ prawdziwy nocny runner posiada globalny mutex; ich pierwsza próba
została prawidłowo zablokowana przez działającą kampanię. Guardów nie osłabiano.

Pełne starty od zera: jeden końcowy wspólny commit, seedy **17/29/43**, po
**2 048 000 kroków i 30 greedy prób/seed**, kalibracja i lokalny smoke.
Dobór SD-SAC i PPO oraz końcowa gotowość pozostają otwarte do wyników tej nocy.

## Historia: kolejka z 4 października 17:28, zatrzymana na polecenie użytkownika

Kod zespołu opublikowany na obu branchach: **8f440075**. Nowy pilot korzysta
z osobnego, czystego i zamrożonego `tmrl-tuning-runtime` na tym commicie.
Stare IQN/PPO będą oceniane na zachowanym `tmrl-algorithm-runs` (`d0dfe645`,
źródła Python `378f9c6a`), bez przenoszenia checkpointów między wersjami.

Kolejka `queue-tuning-20261004` wystartowała **17:27:56 Europe/Warsaw**.
Termin zapisu **20:17:56**, twardy koniec **20:27:56**. To zatwierdzone nowe
3 godziny łącznie, nie przedłużanie poprzedniej kampanii. Wskaźnik
`artifacts/tmrl-test-comparison/active-queue.json` wskazuje tę kolejkę.
SD-SAC `tmrl-tuning-dsac-entropy200-beta000-s17` działa, W&B
[o3aknhmz](https://wandb.ai/dsc-pjatk-warsaw/my-trackmania-agent/runs/o3aknhmz)
potwierdzony online/running. O 17:30 zapisano 1943 kroki w fazie warmup,
ważny timing ostatniego epizodu (p99 50 ms, max 60 ms), brak błędu telemetrii.
To kontrola uruchomienia, nie wynik uczenia ani kwalifikacja SD-SAC.

Następnie planowane są 10 ocen SD-SAC, 10 IQN i 5 PPO w pozostałym czasie.
Warunki promocji SD-SAC oraz sprawdzone ustawienia opisuje [TUNING.md](TUNING.md).
Monitor działa co 5 minut i ma zakończyć pracę po raporcie tej kampanii.
Pełnych treningów nie uruchomiono. Końcowy dobór SD-SAC pozostaje otwarty do
wyniku jazdy; jego pełne uruchomienia są nadal zablokowane.


Użytkownik zatwierdził osobną kampanię po zakończeniu nocnych pilotów:
**maksymalnie 3 godziny łącznie testów w grze**, od rzeczywistego startu nowej
kolejki. Dawna granica 16:15 dotyczyła zakończonej kampanii. Nowe testy nie
uruchamiają automatycznie pełnych treningów. Nagroda, mapa i przestrzenie akcji
pozostają wspólne i niezmienione.

## Decyzje o ustawieniach

Wybieramy ustawienia poparte dotychczasowymi wynikami, bez deklaracji globalnego
optimum. Dwa przejazdy na jednym seedzie nie wystarczą do tuningu wszystkich
algorytmów. Dokładne wartości są zapisane w generatorze oraz YAML-ach.

| Algorytm | Ustawienia do pełnych prób | Status |
| --- | --- | --- |
| IQN | LR 1e-4, gamma .995, batch 256, UTD .25, target co 1000 aktualizacji, kwantyle 64/64/32 | Bez zmian; dodatkowe 10 ocen starego pilota sprawdzi powtarzalność |
| QR | LR 1e-4, gamma .995, batch 256, UTD .25, target co 1000, 64 kwantyle | Zachować pozytywny baseline |
| TQC | LR 3e-4, tau .005, alpha początkowe .2 uczone, cel entropii -3, 2 krytyków × 25 kwantyli, odcięcie po 2 | Zachować pozytywny baseline |
| SAC | LR 3e-4, tau .005, alpha początkowe .2 uczone, cel entropii -3, 2 krytyków | Zachować pozytywny baseline |
| PPO | LR 3e-4 z liniowym spadkiem do końca 2 048 000 kroków, rollout 2048, 10 epok, minibatch 256, GAE .95, entropy .01, clips .2, value .5, grad .5, KL .02 | Naprawiona inicjalizacja adapterów; hiperparametry bez niezweryfikowanych zmian |
| SD-SAC | Pełne ustawienia jeszcze niewybrane; dotychczasowe pliki full nie są zakwalifikowane | Blokada pełnego startu do wyniku nowego pilota |

Off-policy: uniform replay 1 mln, warmup 10 tys., batch 256, n-step 1,
UTD .25 ze ścisłym budżetem aktualizacji. Wszystkie pełne próby od zera,
seedy **17/29/43**, po **2 048 000 kroków**, **30 ocen bez eksploracji na seed**.
Nie przenosimy wag ani replayu pilota. Każdy komputer wymaga kalibracji,
sprawdzenia UID, CUDA, W&B i timingu. PPO zachowuje wyłączoną dodatkową
normalizację obserwacji i nagrody.

## Kontrolowany test SD-SAC

Główna hipoteza: cel entropii **2.0**, kara kotwicząca **beta 0.0**,
seed 17, **145 408 kroków**, limit treningu **8100 sekund**. Model, nagroda,
przestrzeń 78 akcji, LR, alpha, Q-clip i budżet aktualizacji bez zmian względem
poprzedniej próby beta 0. Pomijamy obliczanie nieużywanej kary przy beta 0;
nie zmienia to skończonej funkcji celu. Plik
`configs/diagnostic/sd-sac-entropy200-beta000-s17.yaml` pozostaje hipotezą,
nie automatycznym wyborem pełnych ustawień.

Po treningu planowane jest 10 ocen SD-SAC, następnie 10 ocen IQN i 5 PPO ze
starych końcowych checkpointów, w granicach wspólnego limitu. Częściowe oceny
trzeba oznaczyć; nie uznawać brakujących prób za ukończone. Stare checkpointy
oceniamy na starym zamrożonym kodzie. To test powtarzalności IQN i timingu PPO,
nie test długiego treningu nowej inicjalizacji PPO.

Porównanie SD-SAC obejmuje równy budżet kroków, mety, postęp, ważność timingu,
błędy, finite losses/checkpoint, alpha i entropię oraz kredyt aktualizacji.
Nie wybieramy parametrów na podstawie samego spadku lossu. Nowe ustawienie
można przenieść do trzech pełnych seedów dopiero po ocenie wyniku jazdy;
w razie niepowodzenia blokada pozostaje. Końcowe 30 prób na seed jest osobną
oceną po pełnym treningu, nie zbiorem używanym do wyboru tej hipotezy.

Przed uruchomieniem ustalamy warunek dopuszczenia SD-SAC: cały budżet 145 408
kroków z rozliczonym kredytem i poprawnym checkpointem, komplet 10 ocen,
co najmniej **8/10 met**, ważne pomiary timingu we wszystkich próbach,
brak błędów kontrolera/telemetrii i maksymalny odstęp decyzji **100 ms**.
Pominięte ramki raportujemy, nie ukrywamy. Ten warunek ogranicza ryzyko
zmarnowania trzech pełnych treningów; nie dowodzi optymalności parametrów.
Przy krótszym zapisie lub niepełnej ocenie hipoteza pozostaje niezakwalifikowana.

## Naprawy i wykonanie

- PPO zachowuje celowo zerowe projekcje trzech adapterów enkodera.
  Pozostałe wagi i zużycie RNG nie zmieniają się. Stare wagi po załadowaniu
  dają te same akcje; wpływ nowej inicjalizacji na naukę wymaga osobnego testu.
- STOP trenera obejmuje także okresową i końcową ewaluację PPO. Przerwanie
  zachowuje checkpoint i nie raportuje niepełnej ewaluacji jako sukcesu.
- PPO zamyka środowisko treningowe przed utworzeniem środowiska oceny.
  Po ocenie okresowej tworzy nowe środowisko, zachowując rosnące identyfikatory
  epizodów. Zamknięcie gamepada usuwa callback i zwalnia urządzenie, także
  po błędzie; ponowne zamknięcie jest bezpieczne.
- Ręczny launcher prowadzi każdy seed przez trening i 30 ocen końcowego
  modelu. Weryfikuje ukończenie budżetu, zapis i fingerprint. PPO używa
  istniejącej końcowej oceny. Niespełniony próg osiągów pozostaje widoczny;
  STOP, błąd i niekompletne dane zatrzymują dalsze uruchomienia.
- Nowa kolejka ma osobny katalog `queue-tuning-20261004`, jeden kontroler,
  atomowy zapis statusu z retry oraz globalny termin zapisu i zamknięcia.
  Launcher i kolejka trzymają wspólną blokadę Windows już przed przygotowaniem
  mapy, aby równoczesne uruchomienie nie tworzyło dwóch kontrolerów.
  Dokładny start, termin i zamrożony commit są zapisywane przy uruchomieniu.
- Wszyscy używają jednego nowego commita zespołu z osobnych checkoutów.
  Runtime pilota zostaje zamrożony na czas uczenia i późniejszych ocen.
  Nie omijamy fingerprintów ani nie aktualizujemy kodu używanego przez proces.

Wyniki poprzedniej kampanii: [READINESS.md](READINESS.md). Audyt matematyki,
ograniczenia i źródła: [ALGORITHM_AUDIT.md](ALGORITHM_AUDIT.md).

## Weryfikacja przed zamrożeniem kodu

- 32 testy cyklu trening/ewaluacja i kontrolera, 25 testów launchera oraz
  10 testów algorytmów dyskretnych przeszły. Testy PPO i adapterów opisuje audyt.
- 14 testów zewnętrznej kolejki, w tym rzeczywista blokada pliku Windows,
  współdzielony mutex, zachowanie STOP i koniec czasu dla procesu testowego,
  przeszło bez sterowania grą.
- 36 YAML-i odpowiada generatorowi. Dla sześciu algorytmów konfiguracje
  pełne różnią się między seedami tylko seedem i identyfikatorem.
- Dziewięć standardowych konfiguracji oraz osobna hipoteza SD-SAC przeszły
  syntetyczną aktualizację CPU i zapis/odczyt checkpointu. Ruff, mypy dla
  zmienionych modułów oraz kontrola różnic przeszły.
- Weryfikator końcowego checkpointu sprawdzono także na rzeczywistych starych
  plikach IQN i PPO, na ich oryginalnym kodzie i konfiguracji, z kontrolą
  fingerprintu. Przerwanego przed ostatnią aktualizacją PPO nie uznaje za pełny.

Powyższe testy nie zastępują nowego pilota jazdy ani lokalnego sprawdzenia
sprzętu zespołu. Ograniczenie nowej inicjalizacji PPO pozostaje jawne.
